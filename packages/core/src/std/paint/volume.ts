/**
 * std/paint/volume — volumetric interiors.
 *
 * The words for shading THROUGH a body instead of across its surface: a chord ray into
 * the shape, and a front-to-back emission/absorption march along it. The medium itself
 * (what glows, what absorbs, at each sample) is the look — it stays algebra at the call
 * site; the integrator owns the compositing so self-shadowing and depth occlusion are
 * structural, not re-derived per effect.
 */
import type {Expr} from '../../gpu/contract'
import {add, exp, local, mul, neg, splat3, vec3} from '../math'
import {volumeNoiseAt} from './materials'
import type {SurfaceFrame, SurfaceField} from './materials'
import {opticalThickness} from './materials'

export interface InteriorRay {
    /** Chord entry depth along the view (volumetric taps carry it in `.w`; flat shapes enter at 0). */
    entry: Expr
    /** Chord length (the shared optical-thickness reading, scaled by the material's factor). */
    length: Expr
    /** The in-plane interior position at chord depth `z` — origin sheared along the view. */
    at(z: Expr): {x: Expr; y: Expr}
}

/**
 * The ray a shape's interior is sampled along: entry depth + chord length, and `at(z)` —
 * the position any interior content (gas samples, star planes, backdrops) lives at for a
 * given depth. `shear` sets how strongly deeper samples slide along the view ray's
 * in-plane direction (the perspective parallax of looking INTO a solid).
 */
export function interiorRay(
    field: SurfaceField,
    frame: SurfaceFrame,
    opts: {origin: {x: Expr; y: Expr}; view?: Expr; shear?: number; lengthScale?: number},
): InteriorRay {
    const entry = local(mul(field.s0.member('w'), frame.volFlag), 'rayEntry')
    const length = local(
        opts.lengthScale === undefined
            ? opticalThickness(field, frame)
            : mul(opticalThickness(field, frame), opts.lengthScale),
        'rayLen',
    )
    const shearX = opts.view ? local(mul(opts.view.member('x'), opts.shear ?? 1), 'rayShearX') : undefined
    const shearY = opts.view ? local(mul(opts.view.member('y'), opts.shear ?? 1), 'rayShearY') : undefined
    return {
        entry,
        length,
        at(z: Expr) {
            return {
                x: shearX ? add(opts.origin.x, mul(shearX, z)) : opts.origin.x,
                y: shearY ? add(opts.origin.y, mul(shearY, z)) : opts.origin.y,
            }
        },
    }
}

/**
 * A content plane at a fixed depth along an interior ray, counter-panned by a rotation
 * sensor so it parallaxes like scenery behind a window: rotating the shape pans the
 * plane, deeper planes drifting further (`panRate`).
 */
export function parallaxPlane(
    ray: InteriorRay,
    opts: {depth: number; pan?: {x: Expr; y: Expr}; panRate?: number; hint?: string},
): {x: Expr; y: Expr; z: Expr} {
    const hint = opts.hint ?? 'plane'
    const z = local(add(ray.entry, mul(ray.length, opts.depth)), `${hint}Z`)
    const base = ray.at(z)
    if (!opts.pan) return {x: base.x, y: base.y, z}
    const rate = opts.panRate ?? 1
    return {
        x: add(base.x, mul(opts.pan.x, rate)),
        y: add(base.y, mul(opts.pan.y, rate)),
        z,
    }
}

/**
 * The house turbulence: a billowing 3D cloud density at an interior point — one-octave
 * domain warp (the folds) over a 3-octave fbm, drifting on `drift` and decorrelated per
 * axis by `seed`. Returns the composed `density` plus the raw `mid`/`fine` octaves, which
 * double as free detail fields (dust bands, ridge filaments) at zero extra noise cost.
 */
export function turbulentMedium(
    p: {x: Expr; y: Expr; z: Expr},
    opts: {frequency: Expr; drift?: Expr; seed?: Expr; billow?: Expr; hint?: string},
): {density: Expr; mid: Expr; fine: Expr} {
    const h = opts.hint ?? ''
    const drift = opts.drift
    const seed = opts.seed
    const axis = (v: Expr, freqScale: number | undefined, driftScale: number, seedScale: number): Expr => {
        let e: Expr = freqScale === undefined ? mul(v, opts.frequency) : mul(v, mul(opts.frequency, freqScale))
        if (drift) e = add(e, driftScale === 1 ? drift : mul(drift, driftScale))
        if (seed) e = add(e, seedScale === 1 ? seed : mul(seed, seedScale))
        return e
    }
    const q = local(vec3(
        axis(p.x, undefined, 1, 1),
        axis(p.y, undefined, -0.7, 0.53),
        axis(p.z, 1.25, 1.3, 0.79),
    ), `q${h}`)
    const warp = local(
        volumeNoiseAt(drift ? add(mul(q, 0.5), vec3(mul(drift, 1.4), 0, mul(drift, -0.9))) : mul(q, 0.5)),
        `warp${h}`,
    )
    const qw = opts.billow
        ? local(add(q, mul(vec3(warp, mul(warp, -0.73), mul(warp, 0.41)), opts.billow)), `qw${h}`)
        : q
    const mid = local(volumeNoiseAt(add(mul(qw, 2.13), vec3(5.2, 1.3, 8.4))), `mid${h}`)
    const fine = local(volumeNoiseAt(add(mul(qw, 4.31), vec3(9.1, 3.7, 1.2))), `fine${h}`)
    const density = local(mul(add(add(
        volumeNoiseAt(qw),
        mul(mid, 0.5)),
        mul(fine, 0.27)),
        0.57), `fbm${h}`)
    return {density, mid, fine}
}

export interface VolumeMarchStep {
    /** Step index (build-time). */
    i: number
    /** Chord fraction at the step centre, 0..1 (build-time). */
    t: number
    /** Depth along the chord (entry + length·(t + jitter)), locals-bound. */
    z: Expr
}

/**
 * Front-to-back emission/absorption march: `steps` samples along the chord, each
 * contributing `emit · transmittance (· emissionGain)` and attenuating the transmittance
 * by `exp(−absorb)`. Absorption is per-channel (a vec3 `absorb` tints what lies behind —
 * interstellar reddening, colored glass, murky water). The unrolled loop costs exactly
 * what the sample callback costs; banding is bought off with `jitter` (a per-pixel hash
 * offset in chord-fraction units), not more steps.
 *
 * Returns the accumulated color and the surviving transmittance — multiply anything
 * BEHIND the medium (backgrounds, star planes) by the latter.
 */
export function volumeMarch(
    opts: {
        steps: number
        entry: Expr
        length: Expr
        /** Per-pixel offset added to each step's chord fraction (banding → grain). */
        jitter?: Expr
        /** Per-step emission normalizer (typically `k / steps` so tiers match in brightness). */
        emissionGain?: Expr
        hint?: string
    },
    sample: (step: VolumeMarchStep) => {emit: Expr; absorb: Expr},
): {color: Expr; transmittance: Expr} {
    let transmit: Expr = splat3(1)
    let color: Expr = vec3(0, 0, 0)
    for (let i = 0; i < opts.steps; i++) {
        const t = (i + 0.5) / opts.steps
        const z = local(
            add(opts.entry, mul(opts.length, opts.jitter ? add(t, opts.jitter) : t)),
            `z${i}`,
        )
        const {emit, absorb} = sample({i, t, z})
        const contribution = mul(emit, transmit)
        color = add(color, opts.emissionGain ? mul(contribution, splat3(opts.emissionGain)) : contribution)
        transmit = local(mul(transmit, exp(neg(absorb))), `transmit${i}`)
    }
    return {color: local(color, opts.hint ?? 'marchAcc'), transmittance: transmit}
}

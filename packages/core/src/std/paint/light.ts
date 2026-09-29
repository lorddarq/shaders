/**
 * std/paint/light — the light-part vocabulary.
 *
 * Mid-level parts every light effect is built from: frames (the local coordinate system a
 * recipe evaluates in), framed parts (ray lobes, feather masks, noise rays), beam-glow parts
 * (drift, bloom, streaks, dither), the lens-flare stack, heat ramps, vignette masks, and the
 * light composites. Part constructors are slot-taking — they bind `p()` prop refs (or scalar
 * graphs / literals) and lower them against the composition params — so a shader file states
 * its recipe as visible composition: a frame first, parts over it, one composite at the end.
 * The GPU bodies live in the kit (`kit/lightfields` and friends).
 */
import type {Expr, GpuFragmentParams} from '../../gpu/contract'
import {call, floatE, vec4, mixExpr, asLocal} from '../../gpu/composer'
import {add, div, max, mul, normalize, pow, reflect, sin, smoothstep, sub, vec3} from '../math'
import {animatedTime} from '../../gpu/porters'
import {lightfields, gradientPaints, shapePaints} from '../../gpu/kit/index'
import type {FilterParams} from '../../gpu/scaffolds/pointwiseFilter'
import type {PointwiseEffect} from '../types'
import type {PropRef} from '../values'
import {Scalar} from '../values'
import {uniformOf, resolveScalar, type ArgSpec} from '../invoke'

/** A lowered light value — evaluate against the composition params. */
export type LightValue = (params: GpuFragmentParams) => Expr

// ── Concept words (Expr-level — no slots, compose anywhere) ──────────────────────────────

/**
 * `rayBands` — banded shimmer along an axis: a traveling sine over `along`, lifted to
 * [0, 1] and sharpened by a power curve. Higher `sharpness` narrows the bright bands.
 * Feed a noise-warped `along` for organic rays (aurora curtains, godray shimmer, fabric).
 */
export function rayBands(opts: {
    along: Expr
    frequency: Expr | number
    time: Expr | number
    speed: Expr | number
    phase?: Expr | number
    sharpness: Expr | number
}): Expr {
    let arg: Expr = add(mul(opts.along, opts.frequency), mul(opts.time, opts.speed))
    if (opts.phase !== undefined) arg = add(arg, opts.phase)
    return pow(add(mul(sin(arg), 0.5), 0.5), opts.sharpness)
}

/**
 * `emitterFalloff` — how light spilling from an emitter dims with distance `d` from it:
 * `(radius / (radius + d))^power` — 1 at the emitter, softened over `radius`, then a 1/d
 * line-emitter tail at power 1 tightening toward a 1/d² point-emitter tail at power 2. The
 * long-tailed cousin of the gaussian {@link glowSpot}: real irradiance, not a soft blob.
 */
export function emitterFalloff(d: Expr, opts: {radius: Expr | number; power: Expr | number}): Expr {
    return pow(div(opts.radius, add(opts.radius, max(d, 0))), opts.power)
}

/**
 * `glowAt` — a round gaussian glow at a position in aspect-corrected UV space:
 * `exp(−0.5·dist²/sizeSq)`. Takes the SQUARED radius (callers usually drive several glows from
 * CPU-computed data). The positional cousin of {@link glowSpot}.
 */
export function glowAt(opts: {uv: Expr; aspect: Expr | number; x: Expr | number; y: Expr | number; sizeSq: Expr | number}): Expr {
    const e = (v: Expr | number): Expr => (typeof v === 'number' ? floatE(v) : v)
    return call(lightfields.radialGaussianGlow, 'glowAt', [opts.uv, e(opts.aspect), e(opts.x), e(opts.y), e(opts.sizeSq)])
}

/**
 * `domeNormal` — a fake surface: the normal field of a flat disc treated as a bulging dome
 * ("make it look puffy" — the Bevel & Emboss family's opening move). Geometry only; light
 * it with {@link shine}. `bulge.tilt` sets how far normals lean outward at the rim,
 * `bulge.drop` how much the dome flattens toward the edge. Defaults are the house dome.
 */
export function domeNormal(opts: {
    delta: Expr
    dist: Expr
    radius: Expr | number
    bulge?: {tilt?: number; drop?: number}
}): Expr {
    const tilt = opts.bulge?.tilt ?? 0.2
    const drop = opts.bulge?.drop ?? 0.1
    const slope = asLocal(smoothstep(0, opts.radius, opts.dist), 'domeSlope')
    const leaned = add(slope, tilt)
    return normalize(vec3(
        mul(opts.delta.member('x'), leaned),
        mul(opts.delta.member('y'), leaned),
        sub(1, mul(slope, drop)),
    ))
}

/**
 * `shine` — light a normal field: the specular lobe of a unit light direction against
 * `normal`, viewed along +z (the 2.5D worldview). Returns the RAW lobe — scale by
 * intensity × coverage at the call site. Shines anything that yields a normal: a
 * {@link domeNormal}, a bevel, a heightfield slope, a material surface.
 */
export function shine(opts: {normal: Expr; light: Expr; gloss: Expr | number}): Expr {
    return pow(max(0, reflect(mul(opts.light, -1), opts.normal).member('z')), opts.gloss)
}

/**
 * A light part over a frame's local coordinates. Radial-frame parts read
 * `vec2(dist, angle)`; beam-frame parts read `vec2(u, v)`.
 */
export type LightPart = (frame: Expr, params: GpuFragmentParams) => Expr

// ── Shared resolution helpers ───────────────────────────────────────────────────────────

/** Resolve a scalar-ish slot (prop, scalar graph, ctx token, or number literal) to an Expr. */
function arg(spec: ArgSpec, params: GpuFragmentParams): Expr {
    if (typeof spec === 'number') return floatE(spec)
    if (spec instanceof Scalar || spec.kind !== 'ctx') return resolveScalar(spec, params as never)
    return params.ctx[spec.name]
}

/** The composed UV + effective viewport every generator paint evaluates against. */
function paintFrame(params: GpuFragmentParams): {uv: Expr; viewport: Expr} {
    return {
        uv: params.uvContext ?? params.ctx.uv,
        viewport: params.effectiveViewportSize ?? params.ctx.viewportSize,
    }
}

// ── Frames ──────────────────────────────────────────────────────────────────────────────

/**
 * Distance + angle (`vec2(dist, angle)`) around an unflipped centre prop in aspect-corrected
 * space — the radial frame of a burst, halo or sweep.
 */
export function radialFrame(center: PropRef): LightValue {
    return (params) => {
        const {uv, viewport} = paintFrame(params)
        return call(lightfields.radialBurstFrame, 'radialFrame', [uv, viewport, uniformOf(center, params)])
    }
}

/**
 * The seam-free radial frame volumetric rays march in — distance plus a wrap-free angular
 * coordinate around `center`.
 */
export function seamlessRadialFrame(center: PropRef): LightValue {
    return (params) => {
        const {uv, viewport} = paintFrame(params)
        return call(gradientPaints.godraysFrame, 'godraysFrame', [uv, viewport, uniformOf(center, params)])
    }
}

/**
 * The local frame of a beam anchored at a point and aimed at the canvas centre:
 * `vec2(u, v)` — distance along the beam and lateral offset — in aspect-corrected space.
 * Reads the raw filter context (a pointwise effect's frame, not a generator's).
 */
export function beamFrame(anchor: PropRef): LightValue {
    return (params) => call(lightfields.beamLocalFrame, 'beamFrame', [params.ctx.uv, params.ctx.aspect, uniformOf(anchor, params)])
}

/**
 * Evaluate a light part in a frame — the composition root of a framed light recipe. A `hint`
 * names the stage and binds the frame as a WGSL local (required when the part reads the frame
 * more than once).
 */
export function withFrame(frame: LightValue, part: LightPart, hint?: string): LightValue {
    return (params) => {
        const f = hint ? asLocal(frame(params), hint) : frame(params)
        return part(f, params)
    }
}

/** Multiply light parts over one frame (a mask over a pattern). */
export function modulate(first: LightPart, ...rest: LightPart[]): LightPart {
    return (frame, params) => rest.reduce((acc, part) => acc.mul(part(frame, params)), first(frame, params))
}

// ── Radial-frame parts ──────────────────────────────────────────────────────────────────

/** Rotating sine ray lobes around the frame angle — `count` lobes, spun by the animated clock × `spin`, narrowed by `sharpness`. */
export function rayLobes(slots: {count: ArgSpec; spin: number; sharpness: ArgSpec}): LightPart {
    return (frame, params) => call(lightfields.angularSineLobes, 'rayLobes', [
        frame.member('y'), arg(slots.count, params), animatedTime(params).mul(slots.spin), arg(slots.sharpness, params),
    ])
}

/**
 * {@link rayLobes} driven by a designer-facing softness slider (softness inverts into the lobe
 * exponent, so soft rays widen without moving their centres). `spin` scales the animated clock
 * (negative = clockwise).
 */
export function softRayLobes(slots: {count: ArgSpec; softness: ArgSpec; spin: number}): LightPart {
    return (frame, params) => call(lightfields.softRayLobes, 'softRayLobes', [
        frame.member('y'), arg(slots.count, params), animatedTime(params).mul(slots.spin), arg(slots.softness, params),
    ])
}

/** Diffraction-style spikes with hard nulls — pass `blades / 2` as `halfCount`; high power thins. */
export function raySpikes(slots: {halfCount: ArgSpec; power: ArgSpec}): LightPart {
    return (frame, params) => call(lightfields.angularCosineSpikes, 'raySpikes', [
        frame.member('y'), arg(slots.halfCount, params), arg(slots.power, params),
    ])
}

/** Three concentric chromatic ring bands (R outside, B inside for positive spread) as an RGB. */
export function ringBand(slots: {center: ArgSpec; spread: ArgSpec; width: ArgSpec}): LightPart {
    return (frame, params) => call(lightfields.chromaticRingBand, 'ringBand', [
        frame.member('x'), arg(slots.center, params), arg(slots.spread, params), arg(slots.width, params),
    ])
}

/** An isotropic Gaussian falloff over the frame distance: `exp(-(dist² · sharpness))`. */
export function glowPoint(slots: {sharpness: ArgSpec}): LightPart {
    return (frame, params) => call(lightfields.radialGaussianFalloff, 'glowPoint', [frame.member('x'), arg(slots.sharpness, params)])
}

/** The outer fade of a bounded radial light: 1 inside `radius`, fading over the last `feather` fraction. */
export function featherMask(slots: {radius: ArgSpec; feather: ArgSpec}): LightPart {
    return (frame, params) => call(lightfields.radialFeatherMask, 'featherMask', [
        frame.member('x'), arg(slots.radius, params), arg(slots.feather, params),
    ])
}

/**
 * Two seam-free noise layers accumulated into a volumetric ray coverage over the frame,
 * animated on the definition's clock: `density` sets sector frequency, `intensity` ray
 * visibility, `spotty` the spot density along each ray.
 */
export function noiseRays(slots: {density: ArgSpec; intensity: ArgSpec; spotty: ArgSpec}): LightPart {
    return (frame, params) => call(gradientPaints.godraysRayStack, 'godraysRayStack', [
        frame, animatedTime(params),
        arg(slots.density, params), arg(slots.intensity, params), arg(slots.spotty, params),
    ])
}

// ── Beam-glow parts (a light leak's stages) ─────────────────────────────────────────────

/**
 * The slow drift/breathing signal pack of a bleeding light: breath gain in `.x`, streak drift
 * offsets in the rest. `flicker` sets the breathing depth, `seed` re-rolls the pattern.
 */
export function slowDrift(slots: {seed: ArgSpec; flicker: ArgSpec}): (clock: Expr, params: GpuFragmentParams) => Expr {
    return (clock, params) => call(gradientPaints.leakDrift, 'leakDrift', [
        clock, arg(slots.seed, params), arg(slots.flicker, params),
    ])
}

/** The main anisotropic glow pool (`.x`) and its wider shoulder (`.y`) along a beam frame. */
export function beamBloom(slots: {spread: ArgSpec}): (frame: Expr, drift: Expr, params: GpuFragmentParams) => Expr {
    return (frame, drift, params) => call(gradientPaints.leakBloom, 'leakBloom', [
        frame.member('x'), frame.member('y'), arg(slots.spread, params), drift,
    ])
}

/** Drifting parallel streak bands further into the frame along the beam. */
export function beamStreaks(slots: {spread: ArgSpec; strength: ArgSpec}): (frame: Expr, drift: Expr, params: GpuFragmentParams) => Expr {
    return (frame, drift, params) => call(gradientPaints.leakStreaks, 'leakStreaks', [
        frame.member('x'), frame.member('y'), arg(slots.spread, params), arg(slots.strength, params), drift,
    ])
}

/** Hash dither over a heat field to hide ramp banding. */
export function dithered(heat: Expr, uv: Expr, clock: Expr): Expr {
    return call(gradientPaints.heatDither, 'heatDither', [heat, uv, clock])
}

// ── Expr-level primitives (for bespoke recipes over hand-built coordinates) ─────────────

/**
 * An elliptical Gaussian glow over a pre-normalized 2D offset (each axis already divided by its
 * own radius): `exp(-(du² + dv²) · sharpness)`. The core of a bloom, a glare, or a soft pool.
 */
export function glowSpot(du: Expr, dv: Expr, sharpness: Expr): Expr {
    return call(lightfields.anisotropicGaussianSpot, 'glowSpot', [du, dv, sharpness])
}

/** One soft Gaussian band across a 1D coordinate, peaking at `center` with the given `width`. */
export function streakBand(u: Expr, center: Expr, width: Expr): Expr {
    return call(lightfields.streakBand, 'streakBand', [u, center, width])
}

/** Project a point onto a segment: `vec2(t, distance)` with `t` clamped to [0,1]. */
export function segmentProject(p: Expr, a: Expr, b: Expr): Expr {
    return call(lightfields.pointToSegment, 'segmentProject', [p, a, b])
}

/**
 * The glow cross-section of a tapered beam: thickness/softness interpolate end-to-end along the
 * segment parameter; returns `vec2(colorT, alpha)`.
 */
export function lightBeam(
    dist: Expr, t: Expr,
    thickness: {start: Expr; end: Expr},
    softness: {start: Expr; end: Expr},
): Expr {
    return call(lightfields.taperedSegmentGlow, 'lightBeam', [
        dist, t, thickness.start, thickness.end, softness.start, softness.end,
    ])
}

// ── Lens-flare stack parts ──────────────────────────────────────────────────────────────
//
// Each part reads the shared flare frame — a struct local with members `aspect`, `lightPos`,
// `flareAxis`, `lightAngle`, `lightDist`, `masterFade` — bound once by the shader file.

/**
 * The shared frame of a camera flare: light position, flare axis, master fade (intensity ×
 * edge fade × shimmer), and the pixel's polar coordinates about the light.
 */
export function flareFrame(slots: {position: PropRef; intensity: ArgSpec; edgeFade: ArgSpec}): LightValue {
    return (params) => {
        const {uv, viewport} = paintFrame(params)
        return call(shapePaints.flareFrame, 'flareFrame', [
            uv, viewport, animatedTime(params), uniformOf(slots.position, params),
            arg(slots.intensity, params), arg(slots.edgeFade, params),
        ])
    }
}

/**
 * One internal-reflection ghost disc along the flare axis: `disc` = offset/size/bright/hollow,
 * `tint` = r/g/b/gPhase.
 */
export function lensGhost(slots: {intensity: ArgSpec; spread: ArgSpec; chroma: ArgSpec}) {
    return (flare: Expr, disc: [number, number, number, number], tint: [number, number, number, number], params: GpuFragmentParams): Expr => {
        const {uv} = paintFrame(params)
        return call(shapePaints.lensFlareGhost, 'lensFlareGhost', [
            uv, flare.member('aspect'), flare.member('lightPos'), flare.member('flareAxis'), animatedTime(params),
            arg(slots.spread, params), arg(slots.chroma, params), arg(slots.intensity, params),
            vec4(...disc), vec4(...tint),
        ])
    }
}

/** The chromatic halo ring about the light. */
export function flareHalo(slots: {intensity: ArgSpec; radius: ArgSpec; chroma: ArgSpec; softness: ArgSpec}) {
    return (flare: Expr, params: GpuFragmentParams): Expr => {
        const {uv} = paintFrame(params)
        return call(shapePaints.flareHalo, 'flareHalo', [
            uv, flare.member('aspect'), flare.member('lightPos'), animatedTime(params),
            arg(slots.intensity, params), arg(slots.radius, params),
            arg(slots.chroma, params), arg(slots.softness, params),
        ])
    }
}

/** The diffraction-spike starburst about the light. */
export function flareStarburst(slots: {intensity: ArgSpec; points: ArgSpec}) {
    return (flare: Expr, params: GpuFragmentParams): Expr =>
        call(shapePaints.flareStarburst, 'flareStarburst', [
            flare.member('lightAngle'), flare.member('flareAxis'), flare.member('aspect'), animatedTime(params),
            flare.member('lightDist'), arg(slots.points, params), arg(slots.intensity, params),
        ])
}

/** The anamorphic horizontal streak through the light. */
export function flareStreak(slots: {intensity: ArgSpec; length: ArgSpec}) {
    return (flare: Expr, params: GpuFragmentParams): Expr => {
        const {uv} = paintFrame(params)
        return call(shapePaints.flareStreak, 'flareStreak', [
            uv, flare.member('lightPos'), arg(slots.length, params), arg(slots.intensity, params),
        ])
    }
}

/** The broad veiling glare about the light. */
export function flareGlare(slots: {intensity: ArgSpec; size: ArgSpec}) {
    return (flare: Expr, params: GpuFragmentParams): Expr =>
        call(shapePaints.flareGlare, 'flareGlare', [
            flare.member('lightDist'), arg(slots.size, params), arg(slots.intensity, params),
        ])
}

/** The bright core at the light itself. */
export function flareCore(slots: {intensity: ArgSpec}) {
    return (flare: Expr, params: GpuFragmentParams): Expr =>
        call(shapePaints.flareCore, 'flareCore', [flare.member('lightDist'), arg(slots.intensity, params)])
}

/** Finish an additive light stack: scale by the frame's master fade with a luminance-derived alpha. */
export function flareComposite(total: Expr, flare: Expr): Expr {
    return call(shapePaints.flareComposite, 'flareComposite', [total, flare.member('masterFade')])
}

// ── Masks, ramps, composites ────────────────────────────────────────────────────────────

/**
 * Quadratic corner-darkening mask about the frame centre, aspect-corrected from the effective
 * viewport. `strength` 0 disables it; `scale` remaps a designer-range slider into mask units.
 */
export function vignetteMask(slots: {strength: ArgSpec; scale?: number}): LightValue {
    return (params) => {
        const {uv, viewport} = paintFrame(params)
        let strength = arg(slots.strength, params)
        if (slots.scale !== undefined) strength = strength.mul(slots.scale)
        return call(lightfields.vignetteMask, 'vignetteMask', [uv, viewport, strength])
    }
}

/** Sum additive light contributions: `a + b + c + …` in the given order. */
export function additive(first: Expr, ...rest: Expr[]): Expr {
    return rest.reduce((acc, part) => acc.add(part), first)
}

/**
 * A slot-bound overexposure ramp (fringe → mid → hot, with chromatic per-channel taps): binds the
 * three ramp color props and yields the RGB of any heat field. The visible "color story" part
 * of a light leak or glow.
 */
export function heatRamp(colors: {hot: PropRef; mid: PropRef; fringe: PropRef}) {
    return {
        kind: 'heatRamp' as const,
        /** The chromatic RGB of a heat field (three offset ramp taps). */
        taps: (heat: Expr, params: {uniforms: Record<string, Expr>}): Expr =>
            call(lightfields.chromaticHeatTaps, 'chromaticHeatTaps', [
                heat,
                uniformOf(colors.hot, params).member('rgb'),
                uniformOf(colors.mid, params).member('rgb'),
                uniformOf(colors.fringe, params).member('rgb'),
            ]),
    }
}
export type HeatRamp = ReturnType<typeof heatRamp>

/**
 * Mix `color` over `background` component-wise (rgb AND alpha) by a coverage field; `hint`
 * names the coverage stage's WGSL local.
 */
export function coverageMix(slots: {color: PropRef; background: PropRef; coverage: LightValue; hint?: string}): LightValue {
    return (params) => {
        const alpha = slots.hint ? asLocal(slots.coverage(params), slots.hint) : slots.coverage(params)
        const col = uniformOf(slots.color, params)
        const bg = uniformOf(slots.background, params)
        const rgb = mixExpr(bg.member('rgb'), col.member('rgb'), alpha)
        const a = mixExpr(bg.member('a'), col.member('a'), alpha)
        return vec4(rgb, a)
    }
}

/** Straight-alpha composite of `color` over `background` by a coverage field (linear RGB). */
export function coverageOver(slots: {color: PropRef; background: PropRef; coverage: LightValue}): LightValue {
    return (params) => call(gradientPaints.coverageOverComposite, 'coverageOver', [
        uniformOf(slots.color, params), uniformOf(slots.background, params), slots.coverage(params),
    ])
}

/**
 * Screen-composite an emitted glow RGB over the child, exposure-style: screen blend +
 * overexposure push + a slight halation lift; emitted light raises alpha over transparent
 * areas too. The pointwise composition root of a light-bleed filter.
 */
export function screenGlow(glow: (params: FilterParams) => Expr): PointwiseEffect {
    return {
        kind: 'pointwise',
        body: {fn: lightfields.screenGlowComposite, hint: 'screenGlowComposite'},
        args: (params) => [glow(params)],
    }
}

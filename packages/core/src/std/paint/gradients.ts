/**
 * std — gradient paint nouns. Built from the kit's gradient bodies
 * (`kit/gradientPaints.ts`): each noun binds prop slots and returns a paint closure
 * `(params) => Expr` for a generator definition's `paint:` field.
 * Metric gradients (linear/radial/conic/diamond/spiral) and noise-field paints are
 * COMPOSITIONS — see `std/paint/fields.ts` for the field pipeline they are written in;
 * light recipes (bursts, godrays, leaks) compose the parts in `std/paint/light.ts`.
 *
 * Conventions every noun owns (so the shader file doesn't):
 *  - The UV idiom — standalone renders read `ctx.uv`; wrapped by a UV-propagating parent
 *    the composed `uvContext` wins. Aspect uses the effective viewport (resize-fit box)
 *    when present, else the canvas viewport.
 *  - Structural slots (`space`, `mode`) are compile-time props: the noun reads their CPU
 *    values and JS-branches, emitting only the selected WGSL path.
 */
import type {Expr, GpuFragmentParams} from '../../gpu/contract'
import {call, floatE, vec4} from '../../gpu/composer'
import {animatedTime} from '../../gpu/porters'
import {d, gradientPaints, colorMixing} from '../../gpu/kit/index'
import type {PropRef} from '../values'
import {Scalar} from '../values'
import {local} from '../math'
import {uniformOf, resolveScalar, type ArgSpec} from '../invoke'

/** A generator paint: the composition builder a `role: 'generator'` definition runs. */
export type Paint = (params: GpuFragmentParams) => Expr

// ── Shared resolution helpers ───────────────────────────────────────────────────────────

/** Resolve a scalar-ish slot (prop, scalar graph, or number literal) to an Expr. */
function arg(spec: ArgSpec, params: GpuFragmentParams): Expr {
    if (typeof spec === 'number') return floatE(spec)
    if (spec instanceof Scalar || spec.kind !== 'ctx') return resolveScalar(spec, params as never)
    return params.ctx[spec.name]
}

/** The composed UV + effective viewport every gradient paint evaluates against. */
function paintFrame(params: GpuFragmentParams): {uv: Expr; viewport: Expr} {
    return {
        uv: params.uvContext ?? params.ctx.uv,
        viewport: params.effectiveViewportSize ?? params.ctx.viewportSize,
    }
}

/** Read a structural (compile-time) slot's CPU value. */
function structural(ref: PropRef, params: GpuFragmentParams): unknown {
    return params.propValues[ref.name]
}

// ── Paint nouns ─────────────────────────────────────────────────────────────────────────

/**
 * Fill with a single color. The bound prop is already a P3-linear rgba uniform, so the
 * paint is a direct read; alpha passes through, so a half-transparent fill composites
 * correctly.
 */
export function solidColor(color: PropRef): Paint {
    return (params) => uniformOf(color, params)
}

/**
 * A directional gradient that cycles through colors over time (the definition's
 * `animatedTime` clock). `mode` (structural) selects the procedural rainbow spectrum or a
 * custom three-color loop `a`→`b`→`c`→`a`, each transition mixed in the structural
 * `space` (rainbow generates colors, so `space` does not apply there). `direction`
 * (degrees) spins the gradient in place about the canvas centre; `scale` sets how many
 * cycles span the viewport.
 */
export function colorWheel(slots: {
    mode: PropRef
    direction: ArgSpec
    scale: ArgSpec
    palette: {a: PropRef; b: PropRef; c: PropRef}
    space: PropRef
}): Paint {
    return (params) => {
        const {uv, viewport} = paintFrame(params)
        const animTime = animatedTime(params)
        const t = call(gradientPaints.colorWheelT, 'colorWheelT', [
            uv, viewport, arg(slots.direction, params), arg(slots.scale, params), animTime,
        ])

        // `mode` is a structural string prop → JS-branch the whole paint (recomposes on change).
        const mode = (structural(slots.mode, params) as string) ?? 'rainbow'
        if (mode === 'rainbow') {
            return call(gradientPaints.colorWheelRainbow, 'colorWheelRainbow', [t])
        }
        // Custom 3-color cycle: phase → (segmentMix, segmentIndex), the three per-segment
        // blends mixed at builder level in the structural color space, then the segment pick.
        const phase = local(call(gradientPaints.colorWheelPhase, 'colorWheelPhase', [t]), 'wheelPhase')
        const si = phase.member('x')
        const floorT3 = phase.member('y')
        const spaceMode = (structural(slots.space, params) as number) ?? 0
        const variant = colorMixing.mixColorsVariants[spaceMode as keyof typeof colorMixing.mixColorsVariants] ?? colorMixing.mixColorsLinear
        const c01 = call(variant, 'mixColors', [uniformOf(slots.palette.a, params), uniformOf(slots.palette.b, params), si])
        const c12 = call(variant, 'mixColors', [uniformOf(slots.palette.b, params), uniformOf(slots.palette.c, params), si])
        const c20 = call(variant, 'mixColors', [uniformOf(slots.palette.c, params), uniformOf(slots.palette.a, params), si])
        return call(gradientPaints.colorWheelSelect, 'colorWheelSelect', [c01, c12, c20, floorT3])
    }
}

/**
 * Five individually placed color points blended by inverse-distance proximity. The kit
 * body returns the four incremental running-weighted-average factors; the paint folds the
 * five colors through them, each mix in the structural `space`. `smoothness` inverts into
 * the distance power (higher = blobs spread further). Exactly five points.
 */
export function pointCloudGradient(slots: {
    points: {color: PropRef; position: PropRef}[]
    smoothness: ArgSpec
    space: PropRef
}): Paint {
    if (slots.points.length !== 5) throw new Error('pointCloudGradient expects exactly 5 points')
    return (params) => {
        const {uv, viewport} = paintFrame(params)
        const factors = local(call(gradientPaints.mpgFactors, 'mpgFactors', [
            uv, viewport,
            ...slots.points.map((pt) => uniformOf(pt.position, params)),
            arg(slots.smoothness, params),
        ]), 'mpgFactors')
        const spaceMode = (structural(slots.space, params) as number) ?? 0
        const variant = colorMixing.mixColorsVariants[spaceMode as keyof typeof colorMixing.mixColorsVariants] ?? colorMixing.mixColorsLinear
        let acc: Expr = uniformOf(slots.points[0].color, params)
        const channels = ['x', 'y', 'z', 'w'] as const
        for (let i = 1; i < 5; i++) {
            acc = call(variant, 'mixColors', [acc, uniformOf(slots.points[i].color, params), factors.member(channels[i - 1])])
        }
        return acc
    }
}

/**
 * The `extraFields` the {@link beam} paint drives: for a non-linear color space the
 * forward P3→working-space conversion of the two endpoint colors is pixel-invariant, so
 * it is computed ONCE per frame on the CPU (dirty-keyed) into these vec3 fields and read
 * on the GPU by `mixPreconvertedVariants[mode]`. Declare on the definition alongside the
 * paint.
 */
export const beamPreconvertedFields = {
    convA: {schema: d.vec3f, initial: [0, 0, 0]},
    convB: {schema: d.vec3f, initial: [0, 0, 0]},
}

/**
 * A beam of light from `from` to `to`: the pixel projects onto the segment, thickness and
 * softness taper between the `start`/`end` values along it, and the cross-section shades
 * `colors.inside` → `colors.outside` with a glow alpha. The structural `space` selects the
 * mix: linear mixes per pixel; a non-linear space preconverts both endpoints on the CPU
 * each frame into {@link beamPreconvertedFields} and pays only the weighted mix +
 * back-conversion per pixel. Alpha is the endpoint average scaled by the glow.
 */
export function beam(slots: {
    from: PropRef
    to: PropRef
    thickness: {start: ArgSpec; end: ArgSpec}
    softness: {start: ArgSpec; end: ArgSpec}
    colors: {inside: PropRef; outside: PropRef}
    space: PropRef
}): Paint {
    return (params) => {
        const {uv, viewport} = paintFrame(params)
        const mode = (structural(slots.space, params) as number) ?? 0

        const field = local(call(gradientPaints.beamField, 'beamField', [
            uv, viewport, uniformOf(slots.from, params), uniformOf(slots.to, params),
            arg(slots.thickness.start, params), arg(slots.thickness.end, params),
            arg(slots.softness.start, params), arg(slots.softness.end, params),
        ]), 'beamField')
        const colorT = field.member('x')
        const alpha = field.member('y')
        const inside = uniformOf(slots.colors.inside, params)
        const outside = uniformOf(slots.colors.outside, params)

        let beamColorRGB: Expr
        if (mode !== 0) {
            let lastKey = ''
            params.onBeforeRender(() => {
                const a = params.getCpuValue(slots.colors.inside.name) as {x: number; y: number; z: number} | undefined
                const b = params.getCpuValue(slots.colors.outside.name) as {x: number; y: number; z: number} | undefined
                if (!a || !b) return
                const key = `${a.x},${a.y},${a.z}|${b.x},${b.y},${b.z}`
                if (key === lastKey) return
                lastKey = key
                const ca = colorMixing.convertP3ToMixSpaceCPU(a.x, a.y, a.z, mode)
                const cb = colorMixing.convertP3ToMixSpaceCPU(b.x, b.y, b.z, mode)
                params.setExtraField('convA', [ca[0], ca[1], ca[2]])
                params.setExtraField('convB', [cb[0], cb[1], cb[2]])
            })
            const variant = colorMixing.mixPreconvertedVariants[mode as keyof typeof colorMixing.mixPreconvertedVariants] ?? colorMixing.mixPreconvertedLinear
            const mixed = call(variant, 'mixPreconvertedColors', [
                params.uniforms.convA, params.uniforms.convB, inside.member('a'), outside.member('a'), colorT,
            ])
            beamColorRGB = mixed.member('rgb')
        } else {
            const mixed = call(colorMixing.mixColorsLinear, 'mixColors', [inside, outside, colorT])
            beamColorRGB = mixed.member('rgb')
        }

        // avgAlpha = (inside.a + outside.a) / 2, then scaled by the glow alpha.
        const avgAlpha = inside.member('a').add(outside.member('a')).mul(0.5)
        return vec4(beamColorRGB, avgAlpha.mul(alpha))
    }
}

/**
 * std — drawn-figure paint nouns. Built from the kit's figure/stroke bodies
 * (`kit/shapePaints.ts`): each noun binds prop slots and returns a paint closure
 * `(params) => Expr` for a generator definition's `paint:` field.
 *
 * Only genuinely generic drawing figures live here (a stroked segment, a sine band). LOOKS do
 * not: Blob's recipe lives in its shader file, composed from std words (the language rule —
 * std is how we draw, the shader file is the stylesheet).
 *
 * Conventions every noun owns (so the shader file doesn't):
 *  - The UV idiom — standalone renders read `ctx.uv`; wrapped by a UV-propagating parent
 *    the composed `uvContext` wins; strokes use the effective viewport (resize-fit box).
 *  - The single-color strokes end in the shared color tail
 *    `vec4(color.rgb, color.a * mask)`.
 *  - Animated nouns read the per-node accumulated time themselves; the `animatedTime:`
 *    declaration stays on the definition.
 */
import type {Expr, GpuFragmentParams} from '../../gpu/contract'
import {call, vec4, floatE, animatedTime} from '../../gpu/porters'
import {shapePaints} from '../../gpu/kit/index'
import type {PropRef} from '../values'
import {Scalar} from '../values'
import {uniformOf, resolveScalar, paintFrame, type ArgSpec} from '../invoke'

/** A generator paint: the composition builder a `role: 'generator'` definition runs. */
export type Paint = (params: GpuFragmentParams) => Expr

// ── Shared resolution helpers ───────────────────────────────────────────────────────────

/** Resolve a scalar-ish slot (prop, scalar graph, ctx token, or number literal) to an Expr. */
function arg(spec: ArgSpec, params: GpuFragmentParams): Expr {
    if (typeof spec === 'number') return floatE(spec)
    if (spec instanceof Scalar || spec.kind !== 'ctx') return resolveScalar(spec, params as never)
    return params.ctx[spec.name]
}

/** The single-color stroke tail: the stroke color with alpha scaled by the coverage mask. */
function maskedColor(color: PropRef, mask: Expr, params: GpuFragmentParams): Expr {
    const c = uniformOf(color, params)
    return vec4(c.member('rgb'), c.member('a').mul(mask))
}

// ── Stroked segment ─────────────────────────────────────────────────────────────────────

/** The {@link strokedSegment} noun's slots. */
export interface StrokedSegmentSlots {
    /** The segment endpoints (position props). */
    from: PropRef
    to: PropRef
    /** Stroke thickness in canvas-height units. */
    width: ArgSpec
    /** Line style as a runtime mode number (solid 0 / dashed 1 / dotted 2). */
    style: ArgSpec
    /** Dash length (including caps) and the gap between dashes / spacing between dots. */
    dashLength: ArgSpec
    gapLength: ArgSpec
    /** Cap shape per end as a runtime mode number (square 0 / rounded 1). */
    capStart: ArgSpec
    capEnd: ArgSpec
    /** The stroke color prop. */
    color: PropRef
}

/**
 * A straight stroked segment between two points: per-end square/rounded caps, and dash/dot
 * patterns rescaled to land exactly on both endpoints. Antialiased with a ~1px device-space
 * feather at any thickness; degenerate (coincident) endpoints render a single cap-shaped point.
 */
export function strokedSegment(slots: StrokedSegmentSlots): Paint {
    return (params) => {
        const {uv, viewport} = paintFrame(params)
        const mask = call(shapePaints.lineMask, 'lineMask', [
            uv,
            viewport,
            uniformOf(slots.from, params),
            uniformOf(slots.to, params),
            arg(slots.width, params),
            arg(slots.style, params),
            arg(slots.dashLength, params),
            arg(slots.gapLength, params),
            arg(slots.capStart, params),
            arg(slots.capEnd, params),
        ])
        return maskedColor(slots.color, mask, params)
    }
}

// ── Sine stroke ─────────────────────────────────────────────────────────────────────────

/** The {@link sineStroke} noun's slots. */
export interface SineStrokeSlots {
    /** The wave's center (a position prop) and rotation in degrees. */
    position: PropRef
    angle: ArgSpec
    /** Cycle count across the frame and wave height. */
    frequency: ArgSpec
    amplitude: ArgSpec
    /** Band thickness and edge softness of the stroke. */
    thickness: ArgSpec
    softness: ArgSpec
    /** The stroke color prop. */
    color: PropRef
}

/**
 * A traveling sine-wave stroke: a soft band of `thickness`/`softness` around an animated sine
 * centered on `position` and rotated by `angle`. Reads the definition's per-node animated time
 * (declare `animatedTime: {speed: ...}` on the definition).
 */
export function sineStroke(slots: SineStrokeSlots): Paint {
    return (params) => {
        const {uv, viewport} = paintFrame(params)
        const mask = call(shapePaints.sineWaveMask, 'sineWaveMask', [
            uv,
            viewport,
            arg(slots.angle, params),
            uniformOf(slots.position, params),
            arg(slots.frequency, params),
            arg(slots.amplitude, params),
            arg(slots.thickness, params),
            arg(slots.softness, params),
            animatedTime(params),
        ])
        return maskedColor(slots.color, mask, params)
    }
}

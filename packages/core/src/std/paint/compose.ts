/**
 * std/paint/compose — stacking paints.
 *
 * `layers` is the declarative composite: a list of contributions read top-of-file to
 * bottom-of-stack, each optionally weighted (`opacity`) and occluded (`behind` — a
 * transmittance the layer is seen through). Additive is the default (emissive light);
 * `screen` soft-saturates instead of summing. The alternative — a hand-balanced
 * `add(add(add(…)))` tree — hides the scene's structure from the next reader.
 */
import type {Expr, GpuFragmentParams} from '../../gpu/contract'
import {call} from '../../gpu/composer'
import {animatedTime} from '../../gpu/porters'
import {tone as toneKit} from '../../gpu/kit/index'
import {paintFrame} from '../invoke'
import {add, clamp, div, local, max, mul, sub, vec4} from '../math'

export type Layer =
    | Expr
    | {
        paint: Expr
        /** 'add' (default) sums light; 'screen' folds it in with 1−(1−a)(1−b). */
        blend?: 'add' | 'screen'
        /** Scales the layer's contribution. */
        opacity?: Expr | number
        /** A transmittance the layer is seen through (content behind a medium). */
        behind?: Expr
    }

/**
 * Weighted layering: build N variations of the same thing from a plain-data layer list and
 * accumulate them — `weightedSum = Σ value·weight` and `totalWeight = Σ weight`, both in
 * list order. The caller owns the blend (usually a guarded weighted mean:
 * `weightedSum / (totalWeight + ε)`), because the guard and the remap are look decisions.
 * Cheap depth by stacking: aurora curtains, cloud banks, echo/ghost trails. Compile-time
 * layer counts are just `list.slice(0, n)` — disabled layers aren't emitted.
 */
export function layered<T>(
    layerData: readonly T[],
    build: (layer: T, index: number) => {value: Expr; weight: Expr},
): {weightedSum: Expr; totalWeight: Expr} {
    let weightedSum: Expr | undefined
    let totalWeight: Expr | undefined
    layerData.forEach((layer, index) => {
        const {value, weight} = build(layer, index)
        const contribution = mul(value, weight)
        weightedSum = weightedSum === undefined ? contribution : add(weightedSum, contribution)
        totalWeight = totalWeight === undefined ? weight : add(totalWeight, weight)
    })
    if (!weightedSum || !totalWeight) throw new Error('std: layered([]) needs at least one layer')
    return {weightedSum, totalWeight}
}

/** Composite a stack of paint contributions, first entry at the back. */
export function layers(entries: Layer[]): Expr {
    let acc: Expr | undefined
    for (const entry of entries) {
        const spec = entry instanceof Object && 'paint' in entry ? entry : {paint: entry as Expr}
        let term = spec.paint
        if (spec.opacity !== undefined) term = mul(term, spec.opacity)
        if (spec.behind) term = mul(term, spec.behind)
        acc = acc === undefined
            ? term
            : spec.blend === 'screen'
                ? sub(add(acc, term), mul(acc, term))
                : add(acc, term)
    }
    if (!acc) throw new Error('std: layers([]) needs at least one layer')
    return acc
}

/**
 * `emissiveAlpha` — light as straight-alpha rgba: alpha is the peak channel and the color is
 * the light divided by it, so the layer composites over black exactly as the additive light it
 * is, and over anything else as a translucent tint. The output step of any emissive paint drawn
 * over transparency (glows, spills, flares) — never for opaque bodies.
 */
export function emissiveAlpha(rgb: Expr, hint = 'emissive'): Expr {
    const c = local(rgb, hint)
    const a = local(clamp(max(max(c.member('x'), c.member('y')), c.member('z')), 0, 1), `${hint}A`)
    return vec4(div(c, max(a, 0.0001)), a)
}

/**
 * `dithered` — the always-on ±0.002 output dither on a final rgba (hides banding on long soft
 * ramps). Apply to the COLOR, never to a ramp coordinate. Reads the composed UV and the
 * definition's clock.
 */
export function dithered(color: Expr, params: GpuFragmentParams): Expr {
    const {uv, viewport} = paintFrame(params)
    return call(toneKit.rgbDither, 'rgbDither', [color, uv, viewport, animatedTime(params)])
}

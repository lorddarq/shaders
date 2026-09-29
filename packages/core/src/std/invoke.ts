/**
 * std — the generic invocation layer.
 *
 * Vocabulary modules (std/effects/*, std/paint/*, std/warps/*) build named nouns on top
 * of these helpers: an `ArgSpec` is declarative data describing one GPU-body argument
 * (a prop binding, a scalar value graph, a context token, or a number literal), and
 * `pointwiseOp` wires a kit body into the pointwise filter species from specs alone —
 * so a noun like `saturate(p('intensity'))` is one line and carries no plumbing.
 */
import type {Expr, GpuFragmentParams} from '../gpu/contract'
import {call, expr, floatE} from '../gpu/composer'
import {fields} from '../gpu/kit/index'
import type {FilterParams} from '../gpu/scaffolds/pointwiseFilter'
import type {PointwiseEffect} from './types'
import {Scalar, type PropRef, type ScalarSource, type SignalSlotParams} from './values'

/** A read of one of the system context values. Build via {@link ctx}. */
export interface CtxToken {
    readonly kind: 'ctx'
    readonly name: 'uv' | 'aspect' | 'time' | 'viewportSize' | 'logicalViewportSize' | 'pointer'
}

/** Context tokens for arg specs: `ctx.uv`, `ctx.aspect`, … */
export const ctx = {
    uv: {kind: 'ctx', name: 'uv'} as CtxToken,
    aspect: {kind: 'ctx', name: 'aspect'} as CtxToken,
    time: {kind: 'ctx', name: 'time'} as CtxToken,
    viewportSize: {kind: 'ctx', name: 'viewportSize'} as CtxToken,
    logicalViewportSize: {kind: 'ctx', name: 'logicalViewportSize'} as CtxToken,
    pointer: {kind: 'ctx', name: 'pointer'} as CtxToken,
} as const

/** One declarative GPU-body argument. */
export type ArgSpec = PropRef | Scalar | CtxToken | number

/** The minimal host surface slots resolve against (warp hooks carry no `ctx`). */
export type SlotParams = SignalSlotParams

/** Resolve a prop binding to its uniform accessor (throws on a dangling name). */
/**
 * The uvContext idiom every generator paint shares: evaluate against the composed UV/viewport
 * when a UV-propagating parent supplies one, else the raw canvas frame.
 */
export function paintFrame(params: GpuFragmentParams): {uv: Expr; viewport: Expr} {
    return {
        uv: params.uvContext ?? params.ctx.uv,
        viewport: params.effectiveViewportSize ?? params.ctx.viewportSize,
    }
}

export function uniformOf(ref: PropRef, params: {uniforms: Record<string, Expr>}): Expr {
    const accessor = params.uniforms[ref.name]
    if (!accessor) throw new Error(`std: effect binds unknown prop '${ref.name}'`)
    return accessor
}

/** Resolve a scalar value-graph node to an Expr against the composition params. */
export function resolveScalar(input: Scalar | ScalarSource, params: SlotParams): Expr {
    const node: ScalarSource = input instanceof Scalar ? input.node : input
    switch (node.kind) {
        case 'prop':
            return uniformOf(node, params)
        case 'mul':
            return resolveScalar(node.a, params).mul(resolveScalar(node.b, params))
        case 'radialMask':
            if (!params.ctx) throw new Error('std: radialMask needs a fragment host (warp slots have no ctx)')
            return call(fields.radialFalloffMask, 'radialFalloffMask', [
                params.ctx.uv, params.ctx.aspect,
                uniformOf(node.center, params), uniformOf(node.radius, params), uniformOf(node.falloff, params),
            ])
        case 'signal':
            return node.build(params)
    }
}

/** Resolve any ArgSpec to an Expr against the composition params. */
export function resolveArg(spec: ArgSpec, params: FilterParams): Expr {
    return resolveArgIn(spec, params)
}

/** {@link resolveArg} against any slot host (warp hooks included). */
export function resolveArgIn(spec: ArgSpec, params: SlotParams): Expr {
    if (typeof spec === 'number') return floatE(spec)
    if (spec instanceof Scalar) return resolveScalar(spec, params)
    switch (spec.kind) {
        case 'prop':
            return uniformOf(spec, params)
        case 'ctx':
            if (!params.ctx) throw new Error(`std: ctx.${spec.name} needs a fragment host (warp slots have no ctx)`)
            return params.ctx[spec.name]
        default:
            return resolveScalar(spec, params)
    }
}

/** Never-used guard so `expr` stays available to vocabulary modules re-exporting it. */
export {expr as rawExpr}

/**
 * Build a pointwise filter effect from a kit body + declarative arg specs. The child
 * color is always the body's first argument; `args` follow in order.
 */
export function pointwiseOp(
    fn: unknown,
    hint: string,
    args: ArgSpec[],
    extra?: {compose?: PointwiseEffect['compose']; setup?: PointwiseEffect['setup']},
): PointwiseEffect {
    return {
        kind: 'pointwise',
        body: {fn, hint},
        args: (params) => args.map((spec) => resolveArg(spec, params)),
        compose: extra?.compose,
        setup: extra?.setup,
    }
}

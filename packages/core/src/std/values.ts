/**
 * std — the value graph.
 *
 * Nouns and slots reference values through these nodes, never through Exprs or uniforms
 * directly — the authored file stays strict-literal data, and the lowering (`lower.ts`)
 * turns nodes into Exprs against the composition params. Phase 0 slice: prop bindings and
 * scalar-producing nouns with `.times()` composition; the full ScalarInput union
 * (lengths, signals, field-valued params) lands with later phases.
 */

/** A binding to one of the definition's own props. */
export interface PropRef {
    readonly kind: 'prop'
    readonly name: string
}

/** Bind a prop by name. Identity-checked against the definition's props at lowering. */
export function p(name: string): PropRef {
    return {kind: 'prop', name}
}

/** A scalar-valued node: a prop, a product, a scalar-producing noun, or a live signal. */
export type ScalarSource =
    | PropRef
    | {readonly kind: 'mul'; readonly a: ScalarSource; readonly b: ScalarSource}
    | {readonly kind: 'radialMask'; readonly center: PropRef; readonly radius: PropRef; readonly falloff: PropRef}
    | {readonly kind: 'signal'; readonly build: (params: SignalSlotParams) => import('../gpu/contract').Expr}

/** What a signal build receives: any slot host has props+uniforms; `ctx` only in fragment hosts. */
export interface SignalSlotParams {
    props: import('../gpu/contract').Expr
    uniforms: Record<string, import('../gpu/contract').Expr>
    ctx?: import('../gpu/contract').GpuFragmentParams['ctx']
}

/**
 * A scalar value in the authoring surface. Wraps the data node and carries the fluent
 * combinators (`.times()`); serialization reads `.node`.
 */
export class Scalar {
    constructor(readonly node: ScalarSource) {}

    /** Multiply by another scalar (a prop or another Scalar). */
    times(other: Scalar | PropRef): Scalar {
        return new Scalar({kind: 'mul', a: this.node, b: other instanceof Scalar ? other.node : other})
    }
}

/** What scalar-typed slots accept. */
export type ScalarInput = Scalar | PropRef

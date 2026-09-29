/**
 * std/lists — reading LIST props (the generic array-valued prop, `utilities/listProps`) in a
 * recipe, and looping over them.
 *
 * A list prop `lights` with item fields `{position, color, intensity}` arrives in the fragment as
 * `uniforms.lights_position[i]` / `uniforms.lights_color[i]` / `uniforms.lights_intensity[i]`
 * (vec4 lanes, see the packing note in `listProps`) plus `uniforms.lightsCount`. `listOf` names
 * those; `accumulate` is the loop — a real WGSL `for` over the RUNTIME count (adding an item never
 * recompiles), whose body may hoist `local()`s freely (they land inside the loop, the `guarded`
 * rule: anything shared with the enclosing recipe goes through `deps`).
 */
import type {EmitContext} from '../gpu/contract'
import {Expr} from '../gpu/contract'
import {listCountName, listFieldName} from '../utilities/listProps'

let listCounter = 0

const raw = (name: string): Expr => new Expr(() => name)

/** One item's field accessors at a loop index (the packed vec4 lane, decoded per kind). */
export interface ListItemAt {
    /** `position` fields: vec2 in the stored (x, 1−y) convention — what a position uniform holds. */
    position: (field: string) => Expr
    /** `color` fields: linear rgba vec4. */
    color: (field: string) => Expr
    /** `number` / `boolean` fields: f32 (booleans are ±1). */
    number: (field: string) => Expr
}

export interface ListRef {
    /** The live item count (f32 uniform). */
    count: Expr
    /** Field accessors for item `i` (an Expr indexing the lane arrays). */
    at: (i: Expr) => ListItemAt
}

/** Bind a list prop's uniforms. */
export function listOf(params: {uniforms: Record<string, Expr>}, prop: string): ListRef {
    const lane = (field: string, i: Expr): Expr => {
        const arr = params.uniforms[listFieldName(prop, field)]
        if (!arr) throw new Error(`std: list '${prop}' has no field '${field}'`)
        return new Expr((ctx) => `${arr._emit(ctx)}[${i._emit(ctx)}]`)
    }
    const count = params.uniforms[listCountName(prop)]
    if (!count) throw new Error(`std: '${prop}' is not a list prop`)
    return {
        count,
        at: (i) => ({
            position: (field) => lane(field, i).member('xy'),
            color: (field) => lane(field, i),
            number: (field) => lane(field, i).member('x'),
        }),
    }
}

/**
 * Sum a per-item expression over a list: `var acc = zero; for (i < count) { acc += body(i) }`.
 * `zero` sets the accumulator type (`'f32'` or `'vec3f'`). Statement machinery — see the module
 * header for the hoisting rule.
 */
export function accumulate(
    list: ListRef,
    body: (i: Expr) => Expr,
    opts: {zero?: 'f32' | 'vec3f'; deps?: Expr[]; hint?: string} = {},
): Expr {
    const id = listCounter++
    const zero = opts.zero === 'vec3f' ? 'vec3f(0.0)' : '0.0'
    const hint = opts.hint ?? 'sum'
    return new Expr((ctx) =>
        ctx.memo(`accumulate:${id}`, () => {
            for (const dep of opts.deps ?? []) dep._emit(ctx)
            const acc = ctx.freshLocal(hint)
            const i = ctx.freshLocal('item')
            const countText = list.count._emit(ctx)
            const stmts: string[] = []
            const scoped: EmitContext = {
                external: (v, h) => ctx.external(v, h),
                statement: (w) => stmts.push(w),
                freshLocal: (h) => ctx.freshLocal(h),
                memo: (k, f) => ctx.memo(k, f),
            }
            const term = body(raw(i))._emit(scoped)
            ctx.statement(`var ${acc} = ${zero};`)
            ctx.statement([
                `for (var ${i} = 0u; ${i} < u32(${countText}); ${i}++) {`,
                ...stmts.map((w) => `  ${w}`),
                `  ${acc} += ${term};`,
                `}`,
            ].join('\n'))
            return acc
        }),
    )
}

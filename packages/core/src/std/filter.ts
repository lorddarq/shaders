/**
 * std — filter effect constructors.
 *
 * Two tiers per species:
 *  - Noun effects (`tintToward`, `displaceBy`) — pure vocabulary; the GPU body is a kit
 *    primitive the lowering wires up, so the authored file contains no GPU code.
 *  - `pointwise(...)` / `gather(...)` — the L1 tier for filters whose body is a blessed
 *    bespoke `'use gpu'` fn or tap chain. Same species, same engine mechanics.
 *
 * Constructors capture data only; all mechanics (child guards, identity bypass with
 * driver refusal, call wiring, the per-species alpha discipline) belong to the lowering.
 */
import {call} from '../gpu/composer'
import {blend} from '../gpu/kit/index'
import type {Paint} from './paint/fields'
import {local, mul, vec4} from './math'
import type {GatherEffect, PointwiseEffect} from './types'
import type {PropRef, ScalarInput} from './values'

/** Mix the child's rgb toward a color by a scalar amount, preserving alpha (linear-RGB). */
export interface TintTowardEffect {
    readonly kind: 'tintToward'
    readonly color: PropRef
    readonly amount: ScalarInput
}

export function tintToward(color: PropRef, opts: {amount: ScalarInput}): TintTowardEffect {
    return {kind: 'tintToward', color, amount: opts.amount}
}

/**
 * Displace the child's sample coordinates by a vector field (a simulation output), with a
 * chromatic R/B split around the displaced tap and edge handling per tap. A gather effect:
 * the child renders to a texture, taps are premultiplied, and the result is straight
 * alpha — the alpha discipline is the species', never the author's.
 */
export interface DisplaceByEffect {
    readonly kind: 'displaceBy'
    readonly field: import('./sim').SimOutputRef
    readonly strength: PropRef
    readonly chromatic: PropRef
    /** Binds a structural edge-mode prop ('stretch' | 'transparent' | 'mirror' | 'wrap'). */
    readonly edges: PropRef
}

export function displaceBy(
    field: import('./sim').SimOutputRef,
    opts: {strength: PropRef; chromatic: PropRef; edges: PropRef},
): DisplaceByEffect {
    return {kind: 'displaceBy', field, strength: opts.strength, chromatic: opts.chromatic, edges: opts.edges}
}

/** L1 tier: a pointwise filter around a blessed bespoke `'use gpu'` body. */
export function pointwise(config: Omit<PointwiseEffect, 'kind'>): PointwiseEffect {
    return {kind: 'pointwise', ...config}
}

/** L1 tier: a gather filter around a blessed bespoke tap chain over the child texture. */
export function gather(config: Omit<GatherEffect, 'kind'>): GatherEffect {
    return {kind: 'gather', ...config}
}

/**
 * A paint shown through the child — the CSS `mask-image` of the language, inverted:
 * the child is the mask, the paint is the content. `by: 'alpha'` (default) reveals the
 * paint wherever the child has coverage (gradient-through-text); `by: 'luminance'`
 * reveals it by the child's brightness. The paint's own alpha still applies.
 */
export function paintThrough(paint: Paint, opts: {by?: 'alpha' | 'luminance'} = {}): PointwiseEffect {
    return {
        kind: 'pointwise',
        build: (params) => {
            const painted = local(paint(params), 'painted')
            const mask = opts.by === 'luminance'
                ? call(blend.luminance, 'luminance', [params.childNode.member('rgb')])
                : params.childNode.member('a')
            return vec4(painted.member('rgb'), mul(painted.member('a'), mask))
        },
    }
}

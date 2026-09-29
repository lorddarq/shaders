/**
 * std — declarative slot/identity rule constructors. These replace function-valued
 * escape hatches with data: `crosses(v)` replaces a hand-written `compileTimeWhen`
 * predicate, and the identity rules replace hand-written `identity.when` closures.
 * The lowerings live in `lower.ts`.
 */
import type {IdentityRule, RecompileRule} from './types'

/** Recompose only when the prop value crosses `value` (identity ↔ effect boundary). */
export function crosses(value: number): RecompileRule {
    return {kind: 'crosses', value}
}

/** Recompose when a bespoke predicate flips — the recompile escape hatch. */
// The loose fn signature mirrors PropConfig.compileTimeWhen.
export function recompileWhen(predicate: (prev: any, next: any) => boolean): RecompileRule {
    return {kind: 'custom', predicate}
}

/** The effect is a provable no-op when `prop` (or its default when unset) is exactly 0. */
export function isZero(prop: string): IdentityRule {
    return {kind: 'isZero', prop}
}

/** The effect is a provable no-op when `prop` (or its default when unset) equals `value`. */
export function isValue(prop: string, value: unknown): IdentityRule {
    return {kind: 'isValue', prop, value}
}

/** The effect is a no-op only when EVERY rule holds (multi-prop identities). */
export function allOf(...rules: IdentityRule[]): IdentityRule {
    return {kind: 'allOf', rules}
}

/** Identity escape hatch: a bespoke predicate over the listed props' CPU values. */
export function identityWhenever(props: string[], when: (values: Record<string, unknown>) => boolean): IdentityRule {
    return {kind: 'custom', props, when}
}

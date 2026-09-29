/**
 * std — the expression algebra.
 *
 * Per-pixel math written in the language instead of `'use gpu'` bodies: every function
 * here builds a GPU expression (`Expr`) from expressions and numbers, lowering through
 * the composer's emitter exactly like every other noun. `local()` binds a shared
 * sub-expression to one WGSL local so multi-consumer values evaluate once.
 *
 * Numbers format as f32 literals; every compound operand is parenthesized, so composed
 * expressions never re-associate under WGSL precedence.
 */
import {Expr, type EmitContext} from '../gpu/contract'
import {floatE, vec4 as vec4E, mixExpr, asLocal} from '../gpu/composer'

type Val = Expr | number

const toE = (v: Val): Expr => (typeof v === 'number' ? floatE(v) : v)
const emit = (v: Val, ctx: EmitContext): string => (typeof v === 'number' ? floatE(v)._emit(ctx) : v._emit(ctx))

/** Emit `name(args…)` for a WGSL builtin. */
function callB(name: string, ...args: Val[]): Expr {
    return new Expr((ctx) => `${name}(${args.map((a) => emit(a, ctx)).join(', ')})`)
}

function binary(op: string, a: Val, b: Val): Expr {
    return new Expr((ctx) => `(${emit(a, ctx)} ${op} ${emit(b, ctx)})`)
}

// ── Arithmetic ──────────────────────────────────────────────────────────────────────────

export const add = (a: Val, b: Val): Expr => binary('+', a, b)
export const sub = (a: Val, b: Val): Expr => binary('-', a, b)
export const mul = (a: Val, b: Val): Expr => binary('*', a, b)
export const div = (a: Val, b: Val): Expr => binary('/', a, b)
export const neg = (a: Val): Expr => new Expr((ctx) => `(-(${emit(a, ctx)}))`)

// ── WGSL builtins ───────────────────────────────────────────────────────────────────────

export const abs = (x: Val): Expr => callB('abs', x)
export const floor = (x: Val): Expr => callB('floor', x)
export const ceil = (x: Val): Expr => callB('ceil', x)
export const fract = (x: Val): Expr => callB('fract', x)
export const sqrt = (x: Val): Expr => callB('sqrt', x)
export const pow = (x: Val, y: Val): Expr => callB('pow', x, y)
export const exp = (x: Val): Expr => callB('exp', x)
export const log = (x: Val): Expr => callB('log', x)
export const sin = (x: Val): Expr => callB('sin', x)
export const cos = (x: Val): Expr => callB('cos', x)
export const tan = (x: Val): Expr => callB('tan', x)
export const atan2 = (y: Val, x: Val): Expr => callB('atan2', y, x)
export const min = (a: Val, b: Val): Expr => callB('min', a, b)
export const max = (a: Val, b: Val): Expr => callB('max', a, b)
export const clamp = (x: Val, lo: Val, hi: Val): Expr => callB('clamp', x, lo, hi)
export const mix = (a: Val, b: Val, t: Val): Expr => mixExpr(toE(a), toE(b), typeof t === 'number' ? t : t)
export const step = (edge: Val, x: Val): Expr => callB('step', edge, x)
export const smoothstep = (e0: Val, e1: Val, x: Val): Expr => callB('smoothstep', e0, e1, x)
export const sign = (x: Val): Expr => callB('sign', x)
export const length = (v: Val): Expr => callB('length', v)
export const distance = (a: Val, b: Val): Expr => callB('distance', a, b)
export const dot = (a: Val, b: Val): Expr => callB('dot', a, b)
export const normalize = (v: Val): Expr => callB('normalize', v)
export const cross = (a: Val, b: Val): Expr => callB('cross', a, b)
/** Rotate a 2D point about the origin by a precomputed cos/sin pair (inline algebra). */
export const rotate2 = (p: Val, cosA: Val, sinA: Val): Expr =>
    vec2(sub(mul(member2(p, 'x'), cosA), mul(member2(p, 'y'), sinA)),
        add(mul(member2(p, 'x'), sinA), mul(member2(p, 'y'), cosA)))
const member2 = (v: Val, m: 'x' | 'y'): Val => (typeof v === 'number' ? v : v.member(m))

export const reflect = (i: Val, n: Val): Expr => callB('reflect', i, n)
export const refract = (i: Val, n: Val, eta: Val): Expr => callB('refract', i, n, eta)
export const exp2 = (x: Val): Expr => callB('exp2', x)

/**
 * A seeded random phase in [0, 2π) — `fract(sin(seed·k)·big)·τ`, the standard cheap
 * decorrelator for sine schedules (each independent phase gets its own `salt`). The salt is
 * the raw `(k, big)` pair so ported schedules keep their exact historical constants.
 */
export const hashPhase = (seed: Val, salt: {k: number; big: number}): Expr =>
    mul(fract(mul(sin(mul(seed, salt.k)), salt.big)), 6.283185307179586)

// ── Comparison & selection ──────────────────────────────────────────────────────────────

export const lt = (a: Val, b: Val): Expr => binary('<', a, b)
export const le = (a: Val, b: Val): Expr => binary('<=', a, b)
export const gt = (a: Val, b: Val): Expr => binary('>', a, b)
export const ge = (a: Val, b: Val): Expr => binary('>=', a, b)
/** `cond ? ifTrue : ifFalse` (WGSL `select` takes false-value first — handled here). */
export const select = (cond: Expr, ifTrue: Val, ifFalse: Val): Expr =>
    new Expr((ctx) => `select(${emit(ifFalse, ctx)}, ${emit(ifTrue, ctx)}, ${emit(cond, ctx)})`)

// ── Constructors & structure ────────────────────────────────────────────────────────────

export const float = (n: number): Expr => floatE(n)
export const vec2 = (x: Val, y: Val): Expr => callB('vec2f', x, y)
export const vec3 = (x: Val, y: Val, z: Val): Expr => callB('vec3f', x, y, z)
export const vec4 = (...parts: Val[]): Expr => vec4E(...parts.map(toE))
/** Broadcast a scalar into a vec3 (the common vector-mix third argument). */
export const splat3 = (x: Val): Expr => callB('vec3f', x)

/** Bind an expression to one WGSL local so every consumer reads the same evaluation. */
export const local = (e: Expr, hint: string): Expr => asLocal(e, hint)

// ── Derived (general shapes used across the catalog) ───────────────────────────────────

/** Smooth relu: 0 below zero, →x above, blended over `k` — the standard soft elbow. */
export const softPlus = (x: Expr, k: Val): Expr => {
    const xe = local(x, 'sp')
    return mul(add(xe, sqrt(add(mul(xe, xe), mul(k, k)))), 0.5)
}

/** Gaussian bell over a normalized coordinate: `exp(−k·x²)`. */
export const gaussBell = (x: Expr, k: Val): Expr => exp(neg(mul(mul(x, x), k)))

/**
 * std — motion: shaping time.
 *
 * Two tiers. The EXPRESSION tier turns a clock Expr into rhythm — cycle phases, pulse
 * trains, oscillations, easings, per-cycle random seeds — for use anywhere algebra goes.
 * The SLOT tier (`oscillating`, `pulsing`) wraps the same shapes as `Scalar` signals, so
 * any ArgSpec slot on a warp or effect noun can be driven by time directly:
 *
 *     map: twirl({center: p('center'), intensity: oscillating({rate: 0.5, min: 0.4, max: 1.6})})
 *
 * Slot signals read the global clock by default; `clock: 'node'` reads the definition's
 * `animatedTime` clock instead (declare `animatedTime: {speed: '...'}` — that makes the
 * signal speed-controllable and pausable from the panel).
 */
import type {Expr} from '../gpu/contract'
import {animatedTime} from '../gpu/porters'
import {add, div, floor, fract, local, mul, smoothstep, sin, sub} from './math'
import {Scalar, type SignalSlotParams} from './values'

const TWO_PI = 6.283185307179586

/** Position within a repeating cycle, 0..1 (`fract(t / period)`). */
export function phase(t: Expr, period: Expr | number): Expr {
    return fract(div(t, period))
}

/**
 * A repeating cycle: `progress` (0..1 within the cycle) and `index` (which cycle this
 * is) — the "every N seconds" word. Feed `index` to {@link cycleSeed} for something
 * random-but-stable per pass (a meteor's angle, a glitch's offset).
 */
export function cycle(t: Expr, period: Expr | number): {progress: Expr; index: Expr} {
    const cycles = local(div(t, period), 'cycles')
    return {progress: local(fract(cycles), 'cycleT'), index: local(floor(cycles), 'cycleI')}
}

/** A random-but-stable 0..1 value per cycle index (change `salt` for independent draws). */
export function cycleSeed(index: Expr, salt = 0): Expr {
    return fract(mul(sin(add(mul(index, 12.9898), salt * 78.233 + 4.1414)), 43758.5453))
}

/**
 * An on/off rhythm: 1 for the first `duty` fraction of each cycle, 0 for the rest.
 * `soften` (cycle-fraction units) eases both edges; 0 is a hard blink.
 */
export function pulseTrain(t: Expr, opts: {period: Expr | number; duty: Expr | number; soften?: number}): Expr {
    const p = local(phase(t, opts.period), 'pulseT')
    const s = opts.soften ?? 0
    if (s <= 0) return sub(1, smoothstep(opts.duty, add(opts.duty, 0.0001), p))
    const rise = smoothstep(0, s, p)
    const fall = sub(1, smoothstep(sub(opts.duty, s), opts.duty, p))
    return mul(rise, fall)
}

/** A sine sweep between `min` and `max` at `rate` cycles per second. */
export function oscillate(t: Expr, opts: {rate?: number; min?: Expr | number; max?: Expr | number; offset?: number}): Expr {
    const wave = add(mul(sin(add(mul(t, (opts.rate ?? 1) * TWO_PI), opts.offset ?? 0)), 0.5), 0.5)
    const lo = opts.min ?? 0
    const hi = opts.max ?? 1
    return add(lo, mul(wave, sub(hi, lo)))
}

// ── Easings (unit in → unit out) ─────────────────────────────────────────────────────────

export const easeIn = (x: Expr): Expr => mul(x, x)
export const easeOut = (x: Expr): Expr => sub(1, mul(sub(1, x), sub(1, x)))
export const easeInOut = (x: Expr): Expr => smoothstep(0, 1, x)

// ── Slot signals — drive any ArgSpec slot with time ──────────────────────────────────────

type SignalClock = 'global' | 'node'

function clockOf(clock: SignalClock | undefined, params: SignalSlotParams): Expr {
    // Warp hooks carry no ctx — signals there always ride the node's animated clock.
    if (clock === 'node' || !params.ctx) return animatedTime(params)
    return params.ctx.time
}

/** {@link oscillate} as a slot signal. */
export function oscillating(opts: {rate?: number; min?: Expr | number; max?: Expr | number; offset?: number; clock?: SignalClock}): Scalar {
    return new Scalar({kind: 'signal', build: (params) => oscillate(clockOf(opts.clock, params), opts)})
}

/** {@link pulseTrain} as a slot signal. */
export function pulsing(opts: {period: number; duty: number; soften?: number; clock?: SignalClock}): Scalar {
    return new Scalar({kind: 'signal', build: (params) => pulseTrain(clockOf(opts.clock, params), opts)})
}

/**
 * std — simulation nouns. Declared state + step operators; the engine owns
 * ping-pong, init, scheduling, rest/settle, and device lifecycle (the lowering wires the
 * kit's runtime harnesses — `kit/waves.ts` is the first).
 *
 * Phase 0 slice: `simulate.grid` with the wave-field op set (`op.wave` + `op.splat` at the
 * pointer, `derive: {…: op.gradient()}`). The op vocabulary grows per plan Phase 4; a
 * configuration the lowering doesn't recognize throws at definition time, never silently
 * degrades.
 */
import type {PropRef} from './values'
import type {PointerSignal, PointerSpeedSignal} from './signal'

// ── Ops ─────────────────────────────────────────────────────────────────────────────────

/** Damped `avg − prev` wave propagation. Needs `history: 2` (reads t−1 AND t−2). */
export interface WaveOp {
    readonly kind: 'op.wave'
    readonly damping: PropRef
}

/** Gaussian brush injection at a positioned signal, scaled by an amount signal. */
export interface SplatOp {
    readonly kind: 'op.splat'
    readonly at: PointerSignal
    readonly amount: PointerSpeedSignal
    readonly radius: PropRef
}

/** Central-difference gradient of the field → an RG vector-field texture (a derive op). */
export interface GradientOp {
    readonly kind: 'op.gradient'
}

export type GridStepOp = WaveOp | SplatOp
export type GridDeriveOp = GradientOp

export const op = {
    wave(config: {damping: PropRef}): WaveOp {
        return {kind: 'op.wave', damping: config.damping}
    },
    splat(config: {at: PointerSignal; amount: PointerSpeedSignal; radius: PropRef}): SplatOp {
        return {kind: 'op.splat', at: config.at, amount: config.amount, radius: config.radius}
    },
    gradient(): GradientOp {
        return {kind: 'op.gradient'}
    },
} as const

// ── Grid simulation ─────────────────────────────────────────────────────────────────────

export interface GridSimConfig {
    /** Square grid resolution (state texels per side). */
    resolution: number
    /** History levels the step ops read (the wave equation needs 2: t−1 and t−2). */
    history: number
    /** Ordered per-texel step ops, run once per frame. */
    step: GridStepOp[]
    /** Named derived outputs, consumable as fields by effects (`sim.output(name)`). */
    derive: Record<string, GridDeriveOp>
    /** Rest semantics: when the engine may stop dispatching. */
    rest?: {settlesWhen: 'derived-from-damping'}
}

/** A reference to one of a simulation's derived outputs (feeds effect slots). */
export interface SimOutputRef {
    readonly kind: 'simOutput'
    readonly sim: GridSim
    readonly output: string
}

/** A declared grid simulation. Reference its outputs by identity: `sim.output('name')`. */
export class GridSim {
    constructor(readonly config: GridSimConfig) {}

    output(name: string): SimOutputRef {
        if (!(name in this.config.derive)) {
            throw new Error(`std: simulation has no derived output '${name}' (declared: ${Object.keys(this.config.derive).join(', ')})`)
        }
        return {kind: 'simOutput', sim: this, output: name}
    }
}

export const simulate = {
    grid(config: GridSimConfig): GridSim {
        return new GridSim(config)
    },
} as const

/**
 * std/sim — the GRID simulation vocabulary.
 *
 * A grid sim is a fixed-resolution cell field advanced once per frame by an ordered program of
 * stages: host bookkeeping (pointer velocities, uniform writes), settle/idle gates, one-shot
 * (re)seeding, iterated step passes (ping-pong reaction steps, parity-alternating sort passes),
 * and a publish pass into the texture the fragment samples. GridDistortion, Liquify, PixelSort and
 * ReactionDiffusion each hand-wrote that frame program; `gridSim` owns the skeleton ONCE and each
 * shader declares its program as named `op.*` stages — the kernels, layouts and constants stay in
 * the shader file (the constants ARE the look):
 *
 *   ...gridSim((params, root) => ({
 *       outputs: {displacement},
 *       clampDt: 0.016,
 *       stages: [
 *           op.host('cursorVelocity', (f) => {...}),
 *           op.settle({activeWhen: () => active, settleMs: () => ...}),
 *           op.values('write', (f) => paramsU.write({...})),
 *           op.pass(update),
 *           op.publish(output),
 *       ],
 *   }))
 *
 * Frame-program phases (identical to every hand-written consumer):
 *   ready gates → dt clock → host ticks → settle skips → emits (writes + dispatches), in stage order.
 *
 * ChromaFlow and PixelThrow are the CPU members of the family: their fields are advanced by host JS
 * loops and uploaded as a data texture from inside the fragment builder (no compute hook). Their
 * loops stay on the CPU — moving them to WGSL would re-derive the arithmetic, not relocate it — but
 * the frame program becomes the same visible ordered stage list via `hostGridProgram` +
 * `hostFieldTexture`.
 *
 * ## Compile-time vs runtime (C5)
 * Everything in a config is STRUCTURAL (pipelines, stage order — recompose to change); the values
 * each stage writes per frame are the runtime surface, exactly as before. The build callback runs
 * once per component instance, so stage state (parity ticks, settle clocks, velocity trackers)
 * created inside it is per-instance.
 */
import type {GpuComputeNode, GpuFragmentParams} from '../../gpu/contract'
import type {ComputeStep} from '../../gpu/compute'

/** The TypeGPU root, via the contract (direct `typegpu` imports are restricted to src/gpu/). */
export type GridRoot = NonNullable<GpuFragmentParams['gpu']>['root']

/** The renderer's per-frame params, as the grid consumers read them. */
export interface GridFrameParams {
    pointer?: {x: number; y: number}
    deltaTime?: number
    dimensions?: {width: number; height: number}
}

/** One frame of the simulation, handed to every stage. */
export interface GridFrame {
    /** `fp.deltaTime ?? 0`, clamped to `clampDt`. */
    dt: number
    /** `Date.now()` this frame — the settle gates' clock. */
    now: number
    num(key: string, fallback: number): number
    getCpuValue(key: string): unknown
    frameParams: GridFrameParams
}

/** One stage of the frame program. Phases run in order: ready → tick → skip → emit. */
export interface GridStage {
    /** Frame gate: false → the whole frame returns null (a late-bound input is still pending). */
    ready?(): boolean
    /** Pre-gate bookkeeping (pointer smoothing, activity flags). */
    tick?(f: GridFrame): void
    /** Settle gate: true → skip the frame entirely (the last published texture persists). */
    skip?(f: GridFrame): boolean
    /** Push this stage's dispatches / uniform writes. */
    emit?(f: GridFrame, nodes: ComputeStep[]): void
}

export interface GridSimConfig {
    /** Compute outputs the fragment samples (state/display textures, child RTT handles). */
    outputs: Record<string, unknown>
    /** Late input binding (child RTT → luma prepass, child-modulated kernels). Spread only if set. */
    bindInputs?: (resolve: (key: string) => {texture: unknown} | undefined) => void
    /** Frame-delta clamp (seconds). */
    clampDt: number
    /** Frames with `dt <= 0`: run anyway (default) or skip (Liquify's guard). */
    zeroDt?: 'run' | 'skip'
    stages: GridStage[]
}

/**
 * The grid-simulation noun. Spread it into a definition: `...gridSim((params, root) => ({...}))`.
 * The build callback runs at compute-node creation (per instance, after the no-device bail); it may
 * return null to opt out entirely (a required child is missing).
 */
export function gridSim(
    build: (params: GpuFragmentParams, root: GridRoot) => GridSimConfig | null,
): {compute: GpuComputeNode} {
    return {
        compute: (params: GpuFragmentParams) => {
            const {gpu, getCpuValue} = params
            const root = gpu?.root
            if (!root) return null // GPU-free / no device: fragment falls back to its passthrough.
            const cfg = build(params, root)
            if (!cfg) return null

            const num = (key: string, fallback: number): number => {
                const v = getCpuValue(key)
                return typeof v === 'number' ? v : fallback
            }

            return {
                outputs: cfg.outputs as NonNullable<ReturnType<GpuComputeNode>>['outputs'],
                ...(cfg.bindInputs ? {bindInputs: cfg.bindInputs} : {}),
                getComputeNodes: (frameParams: unknown): ComputeStep[] | null => {
                    for (const s of cfg.stages) if (s.ready && !s.ready()) return null
                    const fp = frameParams as GridFrameParams
                    const dt = Math.min(fp.deltaTime ?? 0, cfg.clampDt)
                    if (cfg.zeroDt === 'skip' && dt <= 0) return null
                    const f: GridFrame = {dt, now: Date.now(), num, getCpuValue, frameParams: fp}
                    const nodes: ComputeStep[] = []
                    for (const s of cfg.stages) s.tick?.(f)
                    for (const s of cfg.stages) if (s.skip?.(f)) return null
                    for (const s of cfg.stages) s.emit?.(f, nodes)
                    return nodes
                },
            }
        },
    }
}

// ── Stage (op) factories ───────────────────────────────────────────────────────────────────────

/** A parity-alternating sort chain's handle: which ping-pong side is current after this frame. */
export interface SortPassStage extends GridStage {
    side(): 'A' | 'B'
}

export const op = {
    /** Named pre-gate host step (pointer smoothing, activity flags). Runs in the tick phase. */
    host(name: string, run: (f: GridFrame) => void): GridStage {
        void name
        return {tick: run}
    },

    /** Named emit-phase host step (per-frame uniform writes / prop derivation). */
    values(name: string, run: (f: GridFrame) => void): GridStage {
        void name
        return {emit: (f) => run(f)}
    },

    /**
     * Settle gate: once nothing has driven the sim for `settleMs` of SIMULATED time, skip frames
     * entirely — the last published texture persists on screen, so a settled sim costs zero GPU time.
     *
     * Simulated, not wall-clock: a hidden tab stops rAF, so no frames step the field while the user
     * is away. Measured on the wall clock the gate would freeze the sim the moment the tab came back
     * with the field still fully visible (it never decayed). Only frames that actually dispatch
     * count toward the settle budget.
     */
    settle(opts: {activeWhen: (f: GridFrame) => boolean; settleMs: (f: GridFrame) => number}): GridStage {
        /** Simulated ms stepped since the last active frame. */
        let simSinceActive = 0
        return {
            tick(f) {
                if (opts.activeWhen(f)) simSinceActive = 0
            },
            skip(f) {
                const skip = simSinceActive > opts.settleMs(f)
                if (!skip) simSinceActive += f.dt * 1000
                return skip
            },
        }
    },

    /** Frame gate on a late-bound input (child RTT arriving via `bindInputs`). */
    readyWhen(fn: () => boolean): GridStage {
        return {ready: fn}
    },

    /** One dispatch of a fixed pass. */
    pass(step: ComputeStep): GridStage {
        return {emit: (_f, nodes) => nodes.push(step)}
    },

    /** A per-frame cache prepass, resolved per frame (PixelSort's late-rebound child-luma snapshot). */
    cache(step: (f: GridFrame) => ComputeStep): GridStage {
        return {emit: (f, nodes) => nodes.push(step(f))}
    },

    /** The publish tail: write the sim state into the texture the fragment samples. Resolved per
     *  frame so it can follow a ping-pong side (`() => output.with(side() === 'A' ? bgA : bgB)`). */
    publish(step: (f: GridFrame) => ComputeStep): GridStage {
        return {emit: (f, nodes) => nodes.push(step(f))}
    },

    /** One-shot (re)initialisation ahead of the step loop, latched on `staleWhen`. */
    seedOnce(opts: {pass: ComputeStep; staleWhen: (f: GridFrame) => boolean; onSeed?: () => void}): GridStage {
        return {
            emit(f, nodes) {
                if (!opts.staleWhen(f)) return
                opts.onSeed?.()
                nodes.push(opts.pass)
            },
        }
    },

    /** N ordered iterations of a step (the caller's `step` handles its own ping-pong swap). */
    iterate(opts: {count: (f: GridFrame) => number; step: (i: number, nodes: ComputeStep[], f: GridFrame) => void}): GridStage {
        return {
            emit(f, nodes) {
                const n = opts.count(f)
                for (let i = 0; i < n; i++) opts.step(i, nodes, f)
            },
        }
    },

    /**
     * Odd-even transposition sort chain: `passes` compare-swap dispatches, alternating parity and
     * ping-pong side per pass. The parity tick PERSISTS across frames (harness policy — a new frame
     * continues the transposition sequence where the last one stopped, which is what makes the sort
     * converge "bit by bit"). `side()` reports the current buffer for the publish pass.
     */
    sortPass(opts: {passes: (f: GridFrame) => number; pass: (parity: 0 | 1, side: 'A' | 'B') => ComputeStep}): SortPassStage {
        let cur: 'A' | 'B' = 'A'
        let tick = 0
        return {
            emit(f, nodes) {
                const n = opts.passes(f)
                for (let i = 0; i < n; i++) {
                    nodes.push(opts.pass((tick % 2) as 0 | 1, cur))
                    cur = cur === 'A' ? 'B' : 'A'
                    tick++
                }
            },
            side: () => cur,
        }
    },
} as const

// ── Shared host parts ──────────────────────────────────────────────────────────────────────────

/**
 * Viewport dimensions with the standard resize-tracked fallback: `tracked()` is the last
 * `onResize` value (what a texture-space prepass wants), `safe(fp)` prefers this frame's reported
 * dimensions and floors both at 1 (what an aspect computation wants).
 */
export function trackedViewport(params: GpuFragmentParams): {
    tracked(): {width: number; height: number}
    safe(fp: GridFrameParams): {width: number; height: number}
} {
    let curW = Math.max(1, Math.round(params.dimensions.width))
    let curH = Math.max(1, Math.round(params.dimensions.height))
    params.onResize(({width, height}) => {
        curW = Math.max(1, Math.round(width))
        curH = Math.max(1, Math.round(height))
    })
    return {
        tracked: () => ({width: curW, height: curH}),
        safe: (fp) => ({
            width: Math.max(1, fp.dimensions?.width ?? curW),
            height: Math.max(1, fp.dimensions?.height ?? curH),
        }),
    }
}

// ── CPU grid sims (ChromaFlow, PixelThrow) ─────────────────────────────────────────────────────
//
// These fields are advanced by host JS loops (the loop arithmetic is the shader's own — relocated,
// never re-derived) and uploaded as an rgba16float data texture the fragment samples. The program
// runner gives the loops the same visible named-stage structure the compute sims have.

/** One frame of a CPU grid program. */
export interface HostFrame {
    /** Wall-clock delta since the last frame, clamped (these sims ignore the renderer's deltaTime). */
    dt: number
    now: number
    pointer: {x: number; y: number}
    frameParams: unknown
}

/** A named CPU step. Return `'skip'` to end the frame (idle gate). */
export interface HostStep {
    name: string
    run(f: HostFrame): void | 'skip'
}

export const hostStep = (name: string, run: (f: HostFrame) => void | 'skip'): HostStep => ({name, run})

/**
 * An ordered CPU frame program for `onBeforeRender`: wall-clock dt (clamped), then each named step
 * in order until one skips.
 */
export function hostGridProgram(opts: {clampDt: number; steps: HostStep[]}): (fp: unknown) => void {
    let lastTime = Date.now()
    return (fp) => {
        const {pointer} = fp as {pointer: {x: number; y: number}}
        const now = Date.now()
        const dt = Math.min((now - lastTime) / 1000, opts.clampDt)
        lastTime = now
        const f: HostFrame = {dt, now, pointer, frameParams: fp}
        for (const s of opts.steps) if (s.run(f) === 'skip') return
    }
}

/**
 * The CPU field's GPU face: an rgba16float data texture (filterable — a filtering sampler rejects
 * r32float, so cells are half-encoded before upload) registered as a media texture the fragment
 * samples. `texData` is the upload staging buffer the publish step packs into.
 */
export function hostFieldTexture(
    params: GpuFragmentParams,
    opts: {size: number; label: string},
): {texData: Uint16Array; texture: ReturnType<GpuFragmentParams['registerMediaTexture']>; upload(): void} {
    const texData = new Uint16Array(opts.size * opts.size * 4)
    const tex = params.createDataTexture({width: opts.size, height: opts.size, format: 'rgba16float', data: texData, label: opts.label})
    params.onCleanup(() => tex.destroy())
    const texture = params.registerMediaTexture(() => tex.texture)
    return {texData, texture, upload: () => tex.write(texData)}
}

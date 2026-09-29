/**
 * std/sim — the STABLE-FLUIDS simulation vocabulary.
 *
 * A fluid shader is a Stable-Fluids solve (scaffolds/fluids owns the kernels) wrapped in a frame
 * program that was copy-pasted across every consumer: allocate the six state buffers, publish an
 * output texture, write the per-frame params, run emission, run the solve chain, run the output
 * pass. `fluidSim` owns that program ONCE; the shader keeps what is genuinely its own — its params
 * struct, its emission/force/restore kernels (the constants ARE the look), and its `values`
 * derivation — and declares them as named parts, so the definition reads as the pipeline it is:
 *
 *   ...fluidSim((params, root) => ({
 *       resolution: N, layout, outputLayout, paramsSchema,
 *       solver: {kernels, jacobiIters},
 *       output: {kernel: outputKernel, key: 'smokeTexture'},
 *       init: seededFieldInit({...}),            // Fog: fBm seed + silent warm-up
 *       inject: [splat({kernel: splatKernel})],  // or cursorRibbon({...}) for stroke emitters
 *       solve: [ambientForce(forceKernel), restoreToward(colorRestoreKernel)],
 *       values: (f) => ({dt: f.dt, ...}),
 *   }))
 *
 * Frame-program order (identical to every hand-written consumer):
 *   [container pre-march] → [init + warm-up when stale] → emitter ticks → idle-gate skip →
 *   params write → [container mask pass] → emitter dispatches → pre-solve stages →
 *   curl…advect chain → post-solve stages → output.
 *
 * ## Compile-time vs runtime (C5)
 * Everything in the config is STRUCTURAL (kernels, layouts, resolution — recompose to change).
 * The `values` object is the only runtime surface, exactly as before.
 *
 * The build callback runs once per component instance, so parts created inside it (trackers,
 * color cycles, init latches) are per-instance state — never hoist a part to module scope.
 */
import {d} from '../../gpu/kit/index'
import type {GpuComputeNode, GpuFragmentParams} from '../../gpu/contract'
import type {ComputeStep, KitComputePipeline} from '../../gpu/compute'
import {createStateBuffer} from '../../gpu/compute'
import {
    createFluidKernelPass, createStableFluidsPasses, type StableFluidsKernels,
} from '../../gpu/scaffolds/fluids'
import {
    createIdleGate, createPointerVelocityTracker, pathStampRibbon,
    type PointerFrame, type PointerLike, type PointerVelocityTrackerOptions,
} from '../../gpu/kit/host/pointer'
/** The TypeGPU root, via the contract (direct `typegpu` imports are restricted to src/gpu/). */
export type FluidRoot = NonNullable<GpuFragmentParams['gpu']>['root']

/** A 2D-dispatched TGSL kernel over the N×N grid. */
export type FluidKernel = (cx: number, cy: number) => void

/** The renderer's per-frame params, as the fluid consumers read them. */
export interface FluidFrameParams {
    pointer?: PointerLike
    deltaTime?: number
    dimensions?: {width: number; height: number}
}

/** One frame of the simulation, handed to every part and to `values`. */
export interface FluidFrame {
    /** Frame delta, clamped to `maxDeltaTime` (0.033 s default). */
    dt: number
    /** `Date.now()` this frame (wall-clock bookkeeping; the idle gates run on simulated `dt`). */
    now: number
    /** Local clock: sum of clamped deltas. An init part may reset it (Fog's warm-up). */
    readonly elapsed: number
    setElapsed(t: number): void
    /** The noun-owned tracker's sample (config `pointer`), or null. Ribbons own their own. */
    ptr: PointerFrame | null
    num(key: string, fallback: number): number
    getCpuValue(key: string): unknown
    frameParams: FluidFrameParams
    /** Write the params uniform directly (per-stamp rewrites inside ribbon thunks). */
    writeParams(v: unknown): void
}

/** What parts get to build their pipelines against at composition time. */
export interface FluidSetupCtx {
    params: GpuFragmentParams
    root: FluidRoot
    /** Grid resolution per axis. */
    n: number
    buffers: {
        velA: unknown; velB: unknown; dyeA: unknown; dyeB: unknown
        pressure: unknown; divergence: unknown; maskBuf?: unknown
    }
    /** The params uniform's buffer (for a part's own bind group). */
    paramsBuffer: unknown
    /** The solve bind group every solver pass runs on. */
    fluidBindGroup: unknown
    /** One kernel as a guarded pass — on the fluid bind group unless another is given. */
    pass(kernel: FluidKernel, bindGroup?: unknown): KitComputePipeline
}

/** An emission part: pre-gate bookkeeping, an optional idle gate, and this frame's dispatches. */
export interface FluidEmitter<TParams = unknown> {
    setup?(ctx: FluidSetupCtx): void
    /** Pre-gate bookkeeping (pointer sampling, color-cycle drift, markActive). */
    tick?(f: FluidFrame): void
    /** Idle gate: true → the whole frame is skipped (the settled field persists on screen). */
    skip?(f: FluidFrame): boolean
    /** Push this frame's emission dispatches (they run before the solve chain). */
    emit?(f: FluidFrame, nodes: ComputeStep[], values: TParams): void
}

/** A per-SOLVE stage: runs inside every solve, including an init part's warm-up steps. */
export interface FluidSolveStage {
    stage: 'pre' | 'post'
    kernel: FluidKernel
}

/** Ambient body force (Fog's turbulence field) — dispatched before every solve chain. */
export const ambientForce = (kernel: FluidKernel): FluidSolveStage => ({stage: 'pre', kernel})

/** Restoration pass (Fog's color re-seed against numerical diffusion) — after every solve chain. */
export const restoreToward = (kernel: FluidKernel): FluidSolveStage => ({stage: 'post', kernel})

/** What an init part gets to append with. `writeParams` returns a QUEUED thunk (device.queue order). */
export interface FluidInitIo<TParams> {
    nodes: ComputeStep[]
    writeParams(v: TParams): ComputeStep
    solve(nodes: ComputeStep[], jacobiIters: number): void
}

export interface FluidInitPart<TParams = unknown> {
    setup?(ctx: FluidSetupCtx): void
    stale(f: FluidFrame): boolean
    run(f: FluidFrame, io: FluidInitIo<TParams>): void
}

/** What a container part (SmokeFill's confining shape) contributes. */
export interface FluidContainerParts {
    /** The mask pass, dispatched after the params write and before emission. */
    maskPass: KitComputePipeline
    /** Extra compute outputs to expose (the volumetric field texture). */
    outputs?: Record<string, unknown>
    /**
     * Per-frame pre-nodes (the volumetric field pre-march). They run FIRST, and they are what a
     * sub-millisecond frame still returns — a pending march is never dropped.
     */
    preFrame?(frameParams: unknown): ComputeStep[] | null
}

export interface FluidSimConfig<TParams> {
    /** Grid resolution per axis (structural — the kernels are baked against it). */
    resolution: number
    /** The shader's fluid bind-group layout (the scaffolds/fluids entry-name contract). */
    layout: unknown
    /** The output pass's layout (`dyeA` read-only + `outTex`). */
    outputLayout: unknown
    /** The shader's per-frame params struct. */
    paramsSchema: unknown
    solver: {kernels: StableFluidsKernels; jacobiIters: number}
    /** The dye→texture publish pass and the `outputs` key the fragment samples it under. */
    output: {kernel: FluidKernel; key: string}
    /** How the per-frame params reach the GPU: one direct write (default) or a queued thunk. */
    write?: 'direct' | 'thunk'
    /** Noun-owned pointer tracker for cursor-force consumers; ribbon emitters own their own. */
    pointer?: PointerVelocityTrackerOptions
    /** Confining shape container (allocates `maskBuf`, dispatches its mask pass every frame). */
    container?: {setup(ctx: FluidSetupCtx): FluidContainerParts}
    init?: FluidInitPart<TParams>
    inject?: FluidEmitter<TParams>[]
    solve?: FluidSolveStage[]
    /** The per-frame params derivation — the one runtime surface. */
    values(f: FluidFrame): TParams
    /** Upper bound on the frame delta. Default 0.033 s (every current consumer's clamp). */
    maxDeltaTime?: number
}

const STATE_FORMAT = 'rgba16float' as const

/**
 * The fluid-simulation noun. Spread it into a definition: `...fluidSim((params, root) => ({...}))`.
 *
 * The build callback runs at COMPUTE-NODE creation (per instance, after the no-device bail), so
 * parts and closures created inside it are instance state.
 */
export function fluidSim<TParams>(
    build: (params: GpuFragmentParams, root: FluidRoot) => FluidSimConfig<TParams>,
): {compute: GpuComputeNode} {
    return {
        compute: (params: GpuFragmentParams) => {
            const {gpu, getCpuValue, registerComputeTexture, onCleanup} = params
            const root = gpu?.root
            if (!root) return null // GPU-free / no device: the fragment falls back to transparent.

            const cfg = build(params, root)
            const n = cfg.resolution
            const count = n * n

            const velA = createStateBuffer(root, d.vec4f, count)
            const velB = createStateBuffer(root, d.vec4f, count)
            const dyeA = createStateBuffer(root, d.vec4f, count)
            const dyeB = createStateBuffer(root, d.vec4f, count)
            const pressure = createStateBuffer(root, d.f32, count)
            const divergence = createStateBuffer(root, d.f32, count)
            const maskBuf = cfg.container ? createStateBuffer(root, d.vec4f, count) : undefined

            const outTex = root.createTexture({size: [n, n], format: STATE_FORMAT}).$usage('storage', 'sampled')
            onCleanup(() => outTex.destroy())
            const outputTexture = registerComputeTexture(outTex)

            const paramsU = root.createUniform(cfg.paramsSchema as never) as {buffer: unknown; write: (v: never) => void}
            const entries: Record<string, unknown> = {velA, velB, dyeA, dyeB, pressure, divergence, params: paramsU.buffer}
            if (maskBuf) entries.maskBuf = maskBuf
            const fluidBg = root.createBindGroup(cfg.layout as never, entries as never)
            const outputBg = root.createBindGroup(cfg.outputLayout as never, {dyeA, outTex} as never)

            const ctx: FluidSetupCtx = {
                params, root, n,
                buffers: {velA, velB, dyeA, dyeB, pressure, divergence, maskBuf},
                paramsBuffer: paramsU.buffer,
                fluidBindGroup: fluidBg,
                pass: (kernel, bindGroup = fluidBg) => createFluidKernelPass(root, kernel, {n, bindGroup}),
            }

            const containerParts = cfg.container?.setup(ctx)
            cfg.init?.setup?.(ctx)
            const emitters = cfg.inject ?? []
            for (const e of emitters) e.setup?.(ctx)
            const solvePre = (cfg.solve ?? []).filter((s) => s.stage === 'pre').map((s) => ctx.pass(s.kernel))
            const solvePost = (cfg.solve ?? []).filter((s) => s.stage === 'post').map((s) => ctx.pass(s.kernel))
            const solver = createStableFluidsPasses(root, cfg.solver.kernels, {n, bindGroup: fluidBg})
            const outputPass = ctx.pass(cfg.output.kernel, outputBg)

            const solveInto = (nodes: ComputeStep[], jacobiIters: number) => {
                nodes.push(...solvePre, ...solver.solveSteps({jacobiIters}), ...solvePost)
            }

            const tracker = cfg.pointer ? createPointerVelocityTracker(cfg.pointer) : null
            const maxDt = cfg.maxDeltaTime ?? 0.033
            let lastTime = Date.now()
            let elapsed = 0
            const num = (key: string, fallback: number): number => {
                const v = getCpuValue(key)
                return typeof v === 'number' ? v : fallback
            }
            const writeParams = (v: unknown) => paramsU.write(v as never)

            return {
                outputs: {[cfg.output.key]: outputTexture, ...(containerParts?.outputs ?? {})} as NonNullable<ReturnType<GpuComputeNode>>['outputs'],
                getComputeNodes: (frameParams: unknown): ComputeStep[] | null => {
                    const fp = frameParams as FluidFrameParams
                    const preNodes = containerParts?.preFrame ? containerParts.preFrame(frameParams) : null
                    const now = Date.now()
                    const dt = Math.min(fp.deltaTime ?? (now - lastTime) / 1000, maxDt)
                    lastTime = now
                    if (dt < 0.001) return preNodes
                    elapsed += dt

                    const f: FluidFrame = {
                        dt, now, frameParams: fp, num, getCpuValue, writeParams,
                        ptr: tracker ? tracker.update(fp.pointer, dt) : null,
                        get elapsed() { return elapsed },
                        setElapsed(t) { elapsed = t },
                    }

                    const nodes: ComputeStep[] = preNodes ? [...preNodes] : []
                    let ranInit = false
                    if (cfg.init?.stale(f)) {
                        ranInit = true
                        cfg.init.run(f, {
                            nodes,
                            writeParams: (v) => () => paramsU.write(v as never),
                            solve: solveInto,
                        })
                    }
                    for (const e of emitters) e.tick?.(f)
                    // An idle-gated frame must still submit a freshly-run init chain: the init
                    // part latched itself as seeded, so dropping the nodes here would leave the
                    // field permanently unseeded (an idle gate can skip the very first frame).
                    for (const e of emitters) if (e.skip?.(f)) return ranInit ? nodes : null

                    const v = cfg.values(f)
                    if ((cfg.write ?? 'direct') === 'thunk') nodes.push(() => paramsU.write(v as never))
                    else paramsU.write(v as never)
                    if (containerParts) nodes.push(containerParts.maskPass)
                    for (const e of emitters) e.emit?.(f, nodes, v)
                    solveInto(nodes, cfg.solver.jacobiIters)
                    nodes.push(outputPass)
                    return nodes
                },
            }
        },
    }
}

// ── Emitter parts ──────────────────────────────────────────────────────────────────────────────

/** A fixed-source emission pass (Smoke's cone, SmokeFill's confined cone): one dispatch per frame. */
export function splat<TParams = unknown>(
    opts: {kernel: FluidKernel} | {setup: (ctx: FluidSetupCtx) => KitComputePipeline},
): FluidEmitter<TParams> {
    let pass: KitComputePipeline | null = null
    return {
        setup(ctx) {
            pass = 'kernel' in opts ? ctx.pass(opts.kernel) : opts.setup(ctx)
        },
        emit(_f, nodes) {
            if (pass) nodes.push(pass)
        },
    }
}

/** One ribbon's stamp geometry + payload, resolved per frame. */
export interface CursorRibbonSpec<TPrepared> {
    /** Spacing between stamps, in viewport UV. */
    stepSize: number
    /** Hard cap on stamps per frame. */
    maxSteps: number
    /** Per-stamp payload advanced at BUILD time in stamp order (InkFlow's color cycle). */
    prepare?: (t: number) => TPrepared
    /** Write the stamp's uniform — runs in a thunk right before that stamp's dispatch. */
    write: (posX: number, posY: number, t: number, prepared: TPrepared) => void
}

/**
 * Stroke emission: a pointer-velocity tracker (teleport/drag policy as data), an idle gate that
 * freezes the sim once the field has faded after the last stroke, and a per-frame ribbon of stamps
 * interpolated along the drag path (`pathStampRibbon` — no dotted gaps on a fast flick).
 *
 * The part owns its tracker; read the frame's sample back via `.ptr()` / `.active()` (SmokeFlow's
 * `values` derives the cursor fields from them).
 */
export function cursorRibbon<TParams = unknown, TPrepared = undefined>(opts: {
    tracker?: PointerVelocityTrackerOptions
    /** When emission is live this frame — a live frame marks the idle gate active. */
    activeWhen: (ptr: PointerFrame, f: FluidFrame) => boolean
    /** Idle window: seconds after the last activity before the sim may freeze. */
    fadeSeconds: (f: FluidFrame) => number
    /** Extra pre-gate per-frame bookkeeping (InkFlow's color-cycle time drift). */
    onFrame?: (f: FluidFrame, ptr: PointerFrame) => void
    /** The stamp pass — on the fluid bind group, or a part-owned pipeline via `setup`. */
    pass: {kernel: FluidKernel} | {setup: (ctx: FluidSetupCtx) => KitComputePipeline}
    /** This frame's ribbon geometry + payload writer. */
    ribbon: (f: FluidFrame, ptr: PointerFrame, values: TParams) => CursorRibbonSpec<TPrepared>
}): FluidEmitter<TParams> & {ptr(): PointerFrame | null; active(): boolean} {
    const tracker = createPointerVelocityTracker(opts.tracker)
    const idle = createIdleGate()
    let pass: KitComputePipeline | null = null
    let ptr: PointerFrame | null = null
    let scale = 1
    let isActive = false
    return {
        setup(ctx) {
            scale = ctx.n
            pass = 'kernel' in opts.pass ? ctx.pass(opts.pass.kernel) : opts.pass.setup(ctx)
        },
        tick(f) {
            ptr = tracker.update(f.frameParams.pointer, f.dt)
            opts.onFrame?.(f, ptr)
            isActive = opts.activeWhen(ptr, f)
            if (isActive) idle.markActive()
        },
        skip(f) {
            const skip = idle.shouldSkip(opts.fadeSeconds(f))
            // Only frames that actually step the field count toward its decay budget.
            if (!skip) idle.tickFrame(f.dt)
            return skip
        },
        emit(f, nodes, values) {
            if (!isActive || !ptr || !pass) return
            const spec = opts.ribbon(f, ptr, values)
            pathStampRibbon(nodes, {
                fromX: ptr.prevX, fromY: ptr.prevY, dx: ptr.dx, dy: ptr.dy,
                dragDist: ptr.dragDist, stepSize: spec.stepSize, maxSteps: spec.maxSteps,
                scale, prepare: spec.prepare, write: spec.write, pass,
            })
        },
        ptr: () => ptr,
        active: () => isActive,
    }
}

// ── Init parts ─────────────────────────────────────────────────────────────────────────────────

/**
 * Deterministic field seeding + silent warm-up (Fog): when the seed key changes (or on first
 * frame), dispatch the init kernel and run the whole solve `warm.steps` times with time-advancing
 * param thunks, then hand the warm clock to the frame clock so ambient forces continue seamlessly.
 */
export function seededFieldInit<TParams>(opts: {
    kernel: FluidKernel
    seed: (f: FluidFrame) => number
    warm: {
        steps: number
        jacobiIters: number
        /** Simulated seconds per warm step. */
        dt: number
        /** The warm clock's starting value (Fog randomizes it so two instances differ). */
        startTime: () => number
        /** The full params object for a warm step at simulated time `t` (t = 0 is the init write). */
        values: (t: number, f: FluidFrame) => TParams
    }
}): FluidInitPart<TParams> {
    let pass: KitComputePipeline | null = null
    let initialized = false
    let lastSeed = -1
    return {
        setup(ctx) {
            pass = ctx.pass(opts.kernel)
        },
        stale(f) {
            return !initialized || opts.seed(f) !== lastSeed
        },
        run(f, io) {
            initialized = true
            lastSeed = opts.seed(f)
            io.nodes.push(io.writeParams(opts.warm.values(0, f)))
            if (pass) io.nodes.push(pass)
            let warmTime = opts.warm.startTime()
            for (let w = 0; w < opts.warm.steps; w++) {
                const wt = warmTime
                warmTime += opts.warm.dt
                io.nodes.push(io.writeParams(opts.warm.values(wt, f)))
                io.solve(io.nodes, opts.warm.jacobiIters)
            }
            f.setElapsed(warmTime)
        },
    }
}

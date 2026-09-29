/**
 * std/sim — the FEEDBACK simulation vocabulary.
 *
 * A feedback shader advances a fixed-resolution state texture one rule-step per frame: read last
 * frame's state, apply the shader's own kernel, write the new state (ping-pong) plus a display
 * copy the fragment samples. The mechanics (ping-pong pair, display copy, late-bound child RTT,
 * clamped speed-scaled clock) live in `scaffolds/feedbackSim` and are NOT re-implemented here —
 * this noun owns the frame program that every consumer hand-wrote around them:
 *
 *   ...feedbackSim((params, root) => ({
 *       size: STATE_RES, format: 'rgba16float', speedProp: 'speed',
 *       paramsSchema: MoshParams,
 *       bindGroups: (ctx, read, write) => root.createBindGroup(layout, {...}),
 *       step: moshKernel,                    // the compile-time-selected state-advance kernel
 *       skip?: (t) => boolean,               // idle-gate policy as data
 *       values: (t) => ({time: t.localTime, dt: t.dt, ...}),   // the one runtime surface
 *   }))
 *
 * Frame-program order (identical to every hand-written consumer): scaffold tick (clock + swap
 * bookkeeping) → skip gate → params write → one step dispatch on this orientation's bind groups.
 *
 * ## Compile-time vs runtime (C5)
 * Everything in the config is STRUCTURAL (kernel variant, layout wiring, resolution — recompose to
 * change; the build callback re-runs). `values` is the only runtime surface, exactly as before.
 *
 * The build callback runs once per component instance, so anything created inside it (samplers,
 * uniforms, compile-time kernel selection) is per-instance state — never hoist it to module scope.
 */
import type {GpuComputeNode, GpuFragmentParams} from '../../gpu/contract'
import type {ComputeStep} from '../../gpu/compute'
import {createFeedbackTrailSim, type SimBindContext, type SimSize, type SimSlot, type SimTick} from '../../gpu/scaffolds/feedbackSim'
import {createFluidKernelPass} from '../../gpu/scaffolds/fluids'

/** The TypeGPU root, via the contract (direct `typegpu` imports are restricted to src/gpu/). */
export type FeedbackRoot = NonNullable<GpuFragmentParams['gpu']>['root']

/** A 2D-dispatched TGSL kernel over the state grid. */
export type FeedbackKernel = (cx: number, cy: number) => void

/** What the bind-group builders get: the scaffold's context plus the params uniform's buffer. */
export type FeedbackBindContext = SimBindContext & {paramsBuffer: unknown}

export interface FeedbackSimConfig<TParams, TGroups> {
    /** State + display resolution per axis (square — both current consumers' shape). */
    size: number
    format: 'rgba16float' | 'rgba32float'
    /** Extra texture pairs swapping in lockstep with the state (TimeTrail's prev-live copy). */
    extraSlots?: Record<string, SimSize>
    /** Orientation-independent extra textures. */
    textures?: Record<string, SimSize>
    /** Prop whose value scales `dt`. */
    speedProp?: string
    maxDeltaTime?: number
    /** The shader's per-frame params struct — the noun owns its uniform. */
    paramsSchema: unknown
    /** Build one orientation's bind groups (called twice, late, by the scaffold). */
    bindGroups: (ctx: FeedbackBindContext, read: SimSlot, write: SimSlot) => TGroups
    /** Build orientation-independent bind groups (called once, late). */
    staticGroups?: (ctx: FeedbackBindContext) => unknown
    /**
     * The state-advance kernel — ONE kernel per step, compile-time variant selected in the build
     * callback. Splitting a step into several kernels is a different simulation (and different
     * WGSL); a shader that needs a multi-dispatch step should not ride this noun's default frame.
     */
    step: FeedbackKernel
    /** Idle-gate policy: true → skip the frame entirely (the scaffold then skips the swap too). */
    skip?: (t: SimTick<TGroups, unknown>) => boolean
    /** The per-frame params derivation — the one runtime surface. Written before the dispatch. */
    values: (t: SimTick<TGroups, unknown>) => TParams
    /** Extra `outputs` entries beyond the scaffold's childTexture/display. */
    outputs?: Record<string, unknown>
}

/**
 * The feedback-simulation noun. Spread it into a definition: `...feedbackSim((params, root) => ({...}))`.
 *
 * Requires a child (these are all child-consuming echo/decoder effects): bails to `null` (fragment
 * passthrough) without one, without a device, or when the scaffold cannot allocate.
 */
export function feedbackSim<TParams, TGroups = unknown>(
    build: (params: GpuFragmentParams, root: FeedbackRoot) => FeedbackSimConfig<TParams, TGroups>,
): {compute: GpuComputeNode} {
    return {
        compute: (params: GpuFragmentParams) => {
            const {childNode, gpu} = params
            if (!childNode) return null
            const root = gpu?.root
            if (!root) return null // GPU-free resolve/tests: fragment falls back to passthrough.

            const cfg = build(params, root)
            const paramsU = root.createUniform(cfg.paramsSchema as never) as {buffer: unknown; write: (v: never) => void}

            const sim = createFeedbackTrailSim<TGroups, unknown>(params, {
                size: cfg.size,
                format: cfg.format,
                extraSlots: cfg.extraSlots,
                textures: cfg.textures,
                speedProp: cfg.speedProp,
                maxDeltaTime: cfg.maxDeltaTime,
                bindGroups: (ctx, read, write) => cfg.bindGroups({...ctx, paramsBuffer: paramsU.buffer}, read, write),
                staticGroups: cfg.staticGroups
                    ? (ctx) => cfg.staticGroups!({...ctx, paramsBuffer: paramsU.buffer})
                    : undefined,
            })
            if (!sim) return null

            // The step pipeline binds its groups per frame (`.with(groups)`), so it is built bare.
            const pipeline = createFluidKernelPass(root, cfg.step, {n: cfg.size, bindGroup: undefined})

            return {
                outputs: {
                    childTexture: sim.childTexture, display: sim.display, ...(cfg.outputs ?? {}),
                } as NonNullable<ReturnType<GpuComputeNode>>['outputs'],
                bindInputs: sim.bindInputs,
                getComputeNodes: (frameParams: unknown): ComputeStep[] | null =>
                    sim.tick(frameParams, (t) => {
                        if (cfg.skip?.(t)) return null
                        paramsU.write(cfg.values(t) as never)
                        return [pipeline.with(t.groups as never)]
                    }),
            }
        },
    }
}

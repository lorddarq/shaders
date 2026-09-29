/**
 * std/sim — `agentSim`: the agent-simulation noun.
 *
 * The declarative spread a `species: 'custom'` agent shader composes its compute/fragment
 * halves from (the gaussianBlur/ComputeBackedEffect precedent): the shader file keeps its
 * layout, uniform struct and constants visible, declares its physics as `force`/`torque`
 * parts folded by an `integrator` factory, its render as a `renderAgents` variant, and its
 * per-frame CPU math as named `agentFrame` recipes — and spreads
 *
 *     ...agentSim({layout, params, maxAgents, output, bake, frame, fragment})
 *
 * into the definition. The noun owns the plumbing: device guards, `createAgentSystem` wiring
 * (buffers, output texture, guarded pipelines, init latch), the late child binding, and the
 * standard full-canvas bilinear fragment with its GPU-free fallback.
 */
import type {Expr, GpuComputeNode, GpuFragmentParams, KitTexture} from '../../gpu/contract'
import type {d} from '../../gpu/kit/index'

type AnyWgslStruct = d.AnyWgslStruct
import {ZERO} from '../../gpu/composer'
import {
    createAgentSystem,
    type AgentPipelineSpec, type AgentSystem, type AgentSystemConfig, type DeviceTier,
} from '../../gpu/scaffolds/agentSystem'
import type {ComputeStep} from '../../gpu/compute'
import {
    splat, resolve,
    type OrientedWorldSplatConfig, type PointWorldSplatConfig, type VolumeSplatConfig,
    type ReliefSplatConfig, type RampResolveConfig,
    type WorldSplatLayout, type PointWorldSplatLayout, type RampResolveLayout,
} from './agentRender'

// ── renderAgents ──────────────────────────────────────────────────────────────────────────

/** The two render pipelines every agent shader ends its frame program with. */
export interface RenderAgentsPipelines {
    splat: AgentPipelineSpec
    resolve: AgentPipelineSpec
}

/**
 * The render subsystem as one declared part: a splat variant + its matching resolve, returned
 * as ready pipeline specs for the frame program. Kernel factories live in `agentRender`.
 */
export const renderAgents = {
    /** Heading-oriented world-space agents with the rest→excited color ramp (Boids,
     *  MagneticFilings, ParticleFlow). */
    orientedWorld(
        layout: WorldSplatLayout & RampResolveLayout,
        cfg: Omit<OrientedWorldSplatConfig, 'name'> & {
            ramp: Omit<RampResolveConfig, 'name' | 'res' | 'invGain'> & {invGain?: number}
            names: {splat: string; resolve: string}
            /** Chain the extra bind group onto the splat (a shape family's own resources). */
            extraOnSplat?: boolean
        },
    ): RenderAgentsPipelines {
        const {ramp, names, extraOnSplat, ...splatCfg} = cfg
        return {
            splat: {
                kernel: splat.orientedWorld(layout, {...splatCfg, name: names.splat}),
                threads: 'agents',
                ...(extraOnSplat ? {extra: true} : {}),
            },
            resolve: {
                kernel: resolve.ramp(layout, {
                    invGain: 1 / 255,
                    ...ramp,
                    res: cfg.res,
                    name: names.resolve,
                }),
                threads: 'fixed',
                size: [cfg.res, cfg.res],
            },
        }
    },

    /** Un-oriented point motes with a live softness and a single tint (FloatingParticles). */
    pointWorld(
        layout: PointWorldSplatLayout & Parameters<typeof resolve.tint>[0],
        cfg: Omit<PointWorldSplatConfig, 'name'> & {names: {splat: string; resolve: string}},
    ): RenderAgentsPipelines {
        const {names, ...splatCfg} = cfg
        return {
            splat: {kernel: splat.pointWorld(layout, {...splatCfg, name: names.splat}), threads: 'agents'},
            resolve: {
                kernel: resolve.tint(layout, {res: cfg.res, name: names.resolve}),
                threads: 'fixed',
                size: [cfg.res, cfg.res],
            },
        }
    },

    /** Perspective-projected 3D swarm with the speed-driven color ramp (Particles). */
    volume(
        layout: Parameters<typeof splat.volume>[0] & RampResolveLayout,
        cfg: Omit<VolumeSplatConfig, 'names'> & {
            ramp: Omit<RampResolveConfig, 'name' | 'res' | 'invGain'>
            names: VolumeSplatConfig['names'] & {resolve: string}
            extraOnSplat?: boolean
        },
    ): RenderAgentsPipelines {
        const {ramp, names, extraOnSplat, ...splatCfg} = cfg
        return {
            splat: {
                kernel: splat.volume(layout, {...splatCfg, names}),
                threads: 'agents',
                ...(extraOnSplat ? {extra: true} : {}),
            },
            resolve: {
                kernel: resolve.ramp(layout, {invGain: 1 / 256, ...ramp, res: cfg.outRes, name: names.resolve}),
                threads: 'fixed',
                size: [cfg.outRes, cfg.outRes],
            },
        }
    },

    /** Camera-projected image relief with depth-weighted color averaging (ParticleField). */
    relief(
        layout: Parameters<typeof splat.relief>[0],
        cfg: Omit<ReliefSplatConfig, 'name'> & {
            out: [number, number]
            weightedColor: {alphaK: number}
            names: {splat: string; resolve: string}
        },
    ): RenderAgentsPipelines {
        const {out, weightedColor, names, ...splatCfg} = cfg
        return {
            splat: {kernel: splat.relief(layout, {...splatCfg, name: names.splat}), threads: 'grid'},
            resolve: {
                kernel: resolve.weightedColor(layout, {fp: cfg.fp, alphaK: weightedColor.alphaK, name: names.resolve}),
                threads: 'fixed',
                size: out,
            },
        }
    },
} as const

// ── agentSim ──────────────────────────────────────────────────────────────────────────────

/** What `bake` assembles at composition time from the declared parts. */
export interface AgentSimBaked {
    pipelines: Record<string, AgentPipelineSpec>
    /** Per-frame execution order (keys of `pipelines`). */
    program: readonly string[]
    /** One-shot spawn step, prepended on the first frame and on a reseed. */
    initStep?: string
    /** Override the declared output extent (an aspect-fitted target decided per composition). */
    outputSize?: [number, number]
    /** A second bind group chained onto every pipeline marked `extra` (shape-family SDFs). */
    extraBindGroup?: unknown
    /** Resources already available at composition to bind onto `externalKeys` immediately. */
    bindNow?: Record<string, unknown>
    /** A composition payload handed to `frame` (fluid solvers, per-shape CPU resolvers…). */
    setup?: unknown
}

export interface AgentSimFragmentConfig {
    /** The compute output the fragment bilinear-samples full-canvas (rgba16f is filterable). */
    output: string
    /** GPU-free fallback: transparent (generators) or the raw child (child-driven effects). */
    fallback: 'transparent' | 'child'
    /** Optional compose over the sampled field (e.g. straight-alpha over the child). */
    compose?: (field: Expr, params: GpuFragmentParams) => Expr
}

export interface AgentSimConfig<TParams extends AnyWgslStruct> {
    layout: AgentSystemConfig<TParams>['layout']
    params: TParams
    maxAgents: number
    /** Effective count ceiling per device tier (the prop's declared range is unchanged). */
    countCap?: DeviceTier
    output: AgentSystemConfig<TParams>['output']
    /** Layout keys bound late or via `bake.bindNow` (a child RTT, a solver handoff texture). */
    externalKeys?: readonly string[]
    /** Read the child as a texture and late-bind it onto this external key (ParticleField). */
    childTexture?: {externalKey: string}
    /** Composition-time baking: assemble the declared parts. Null = no simulation composes. */
    bake: (params: GpuFragmentParams) => AgentSimBaked | null
    /** The per-frame CPU program: write uniforms, return the frame's steps (usually via
     *  `sys.frame(...)`, prepending any extra passes the shader owns). */
    frame: (
        sys: AgentSystem<TParams>,
        params: GpuFragmentParams,
        setup: unknown,
    ) => (frameParams: unknown) => ComputeStep[] | null
    fragment: AgentSimFragmentConfig
}

/** The two definition halves the spread contributes (the ComputeBackedEffect shape). */
export interface AgentSimSpread {
    compute: GpuComputeNode
    gpu: {fragment: (params: GpuFragmentParams) => Expr}
}

/**
 * Compose an agent simulation: `bake` the declared parts into kernels, wire them through the
 * `createAgentSystem` harness, and sample the resolved field in the fragment.
 */
export function agentSim<TParams extends AnyWgslStruct>(cfg: AgentSimConfig<TParams>): AgentSimSpread {
    return {
        compute: (params: GpuFragmentParams) => {
            if (cfg.childTexture && !params.childNode) return null
            if (!params.gpu?.root) return null // GPU-free (no device): fragment falls back.

            const childTexture = cfg.childTexture ? params.convertToTexture(params.childNode!) : null

            const baked = cfg.bake(params)
            if (!baked) return null

            const sys = createAgentSystem(params, {
                layout: cfg.layout,
                params: cfg.params,
                maxAgents: cfg.maxAgents,
                output: baked.outputSize ? {...cfg.output, size: baked.outputSize} : cfg.output,
                ...(cfg.countCap ? {countCap: cfg.countCap} : {}),
                ...(cfg.externalKeys ? {externalKeys: cfg.externalKeys} : {}),
                ...(baked.extraBindGroup ? {extraBindGroup: baked.extraBindGroup} : {}),
                pipelines: baked.pipelines,
                program: baked.program,
                ...(baked.initStep ? {initStep: baked.initStep} : {}),
            })
            if (!sys) return null
            if (baked.bindNow) sys.bindExternal(baked.bindNow)

            const tick = cfg.frame(sys, params, baked.setup)

            return {
                outputs: childTexture ? {childTexture, ...sys.outputs} : sys.outputs,
                ...(childTexture && cfg.childTexture
                    ? {
                        bindInputs: (resolveInput: (key: string) => {texture: unknown} | undefined) => {
                            const src = resolveInput(childTexture.key)
                            if (!src) return
                            sys.bindExternal({[cfg.childTexture!.externalKey]: src.texture})
                        },
                    }
                    : {}),
                getComputeNodes: tick,
            }
        },
        gpu: {
            fragment: (params: GpuFragmentParams): Expr => {
                const {ctx, childNode, computeOutputs} = params
                const field = computeOutputs?.[cfg.fragment.output] as KitTexture | undefined
                if (!field) return cfg.fragment.fallback === 'child' ? (childNode ?? ZERO) : ZERO
                const sampled = field.sample(ctx.uv, 'linearClamp')
                return cfg.fragment.compose ? cfg.fragment.compose(sampled, params) : sampled
            },
        },
    }
}

// ── Re-exports: the full agent-simulation vocabulary from one import surface ──────────────

export {force, torque, field, drift, channel, integrator, composeForces3, composeSteer2, composeTorques, restOrientationAngle} from './agentForces'
export type {
    AgentForce3, AgentSteer2, AgentTorque, AgentFieldAt, AgentHome2, AgentDrift, AgentChannel, ShapeField3,
} from './agentForces'
export {splat, resolve, pointSlot} from './agentRender'
export {shapeField} from './shapeFields'
export * as agentFrame from './agentFrame'

/**
 * `@coreroot/std` — the standard authoring layer for shader definitions.
 *
 * A shader definition authored on std is declarative data built from these constructors;
 * `defineStd` lowers it onto the engine scaffolds and kit primitives and returns an
 * ordinary `GpuShaderDefinition` — registry generation, framework components, editor
 * metadata and presets are all untouched. GPU math lives in the kit behind the nouns; the
 * per-species L1 tiers (`pointwise`, `gather`, a warp's map fn, a generator's paint
 * builder) carry blessed bespoke bodies where the vocabulary does not reach yet.
 */
export {defineStd} from './lower'
export {tintToward, pointwise, gather, displaceBy, paintThrough} from './filter'
export {radialMask} from './mask'
export {simulate, op, GridSim} from './sim'
export {pointer, pointerSpeed} from './signal'
export {p, Scalar} from './values'
export {crosses, recompileWhen, isZero, isValue, allOf, identityWhenever} from './slots'
export {ctx, pointwiseOp, resolveArg, resolveScalar, uniformOf} from './invoke'

// WGSL data schemas for `extraFields` declarations (the one GPU-typed surface a
// definition legitimately carries).
export {d as schema} from '../gpu/kit/index'
export type {ArgSpec, CtxToken} from './invoke'

// Warp map combinators (coordinate-space partial application and hard piecewise selection).
export {lerpToIdentity, selectMap} from '../gpu/scaffolds/uvRemapShader'
export type {UvMap, UvMapResult, UvMapSource, UvRemapHookParams} from '../gpu/scaffolds/uvRemapShader'
export type {FilterParams} from '../gpu/scaffolds/pointwiseFilter'
export type {RttFilterParams} from '../gpu/scaffolds/rttFilter'
export type {SdfShapeShaderSpec, SdfShapeBounds} from '../gpu/scaffolds/sdfShape'

export type {
    StdDefinition,
    StdDefinitionBase,
    StdPointwiseFilterDefinition,
    StdGatherFilterDefinition,
    StdWarpDefinition,
    StdShapeDefinition,
    StdGeneratorDefinition,
    StdCustomDefinition,
    StdProps,
    StdPropConfig,
    PointwiseEffect,
    GatherEffect,
    RecompileRule,
    IdentityRule,
} from './types'
export type {TintTowardEffect, DisplaceByEffect} from './filter'
export type {PropRef, ScalarInput, ScalarSource} from './values'
export type {GridSimConfig, SimOutputRef, GridStepOp, GridDeriveOp} from './sim'
export type {PointerSignal, PointerSpeedSignal} from './signal'

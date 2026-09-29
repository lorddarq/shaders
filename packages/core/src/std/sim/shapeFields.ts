/**
 * std/sim — `shapeField`: baked 3D signed-distance FIELD parts for the volume-swarm family.
 *
 * A containment force (see `force.containment`) needs one thing from the shape: a
 * `ShapeField3` — signed distance at any shape-local 3D point. This module bakes that field
 * for each shape source a swarm can be given: a flat analytic 2D shape or an SVG outline
 * extruded in z, a true analytic 3D volume, or an SVG lifted into 3D — so ONE integrator
 * serves every shape kind. Layout contracts follow the family convention: factories take the
 * consumer's layouts and reference entries by REQUIRED NAMES (documented per factory).
 */
import {tgpu, d, std, sdf3d} from '../../gpu/kit/index'
import type {ShapeField3} from './agentForces'

/** Rounded z-extrusion of a flat signed distance (the standard opExtrusion). */
const extrudeZ = tgpu.fn([d.f32, d.f32], d.f32)((d2, wz) => {
    'use gpu'
    const ox = std.max(d2, 0.0)
    const oz = std.max(wz, 0.0)
    return std.min(std.max(d2, wz), 0.0) + std.sqrt(ox * ox + oz * oz)
}).$name('shapeFieldExtrudeZ')

/** The analytic 2D SDF menu's call shape (`sdf.buildAnalyticSdfFn`). */
type AnalyticSdf2d = (
    uv: d.v2f, radius: number, sides: number, rounding: number, innerRatio: number,
    rotation: number, height: number, offset: number, aperture: number,
) => {readonly x: number}

/** Params view for the extruded flat-shape fields: the CPU-resolved analytic sub-prop bundle
 *  (the driveAnalyticSubProps convention) plus the extrusion half-depth. */
interface ExtrudedAnalyticParamsView {
    readonly saRadius: number
    readonly saSides: number
    readonly saRounding: number
    readonly saInnerRatio: number
    readonly saRotation: number
    readonly saHeight: number
    readonly saOffset: number
    readonly saAperture: number
    readonly halfDepth: number
}

/** The per-axis sin/cos rotation bundle the sdf3d MarchParams carry. */
interface RotSinCosView {
    readonly cx: number
    readonly sx: number
    readonly cy: number
    readonly sy: number
    readonly cz: number
    readonly sz: number
}

/** A layout whose MarchParams uniform carries the shape rotation (the sdf3d setups). */
interface RotatedFieldLayout {
    readonly $: {readonly params: {readonly rot: RotSinCosView}}
}

/**
 * Flat analytic shape extruded in z: the baked 2D SDF evaluated in shape-local uv (y up →
 * v down), sub-props read live from `params.sa*`, rounded-extruded over `params.halfDepth`.
 */
function analyticExtruded(
    layout: {readonly $: {readonly params: ExtrudedAnalyticParamsView}},
    sdf2d: AnalyticSdf2d,
): ShapeField3 {
    return tgpu.fn([d.vec3f], d.f32)((p3) => {
        'use gpu'
        const prm = layout.$.params
        const uv2 = d.vec2f(p3.x + 0.5, 0.5 - p3.y)
        const d2 = sdf2d(uv2, prm.saRadius, prm.saSides, prm.saRounding, prm.saInnerRatio, prm.saRotation, prm.saHeight, prm.saOffset, prm.saAperture).x
        return extrudeZ(d2, std.abs(p3.z) - prm.halfDepth)
    }).$name('shapeFieldAnalyticExtruded') as ShapeField3
}

/**
 * Flat SVG outline extruded in z (nearest texel read — plenty for containment forces).
 * Beyond the SDF texture's ±0.5 footprint the clamped field is flat (or, under a feature
 * touching the border, even reads INSIDE) — which strands strays and draws plumb-line
 * columns — so the field is extended with the exact distance to the footprint box, keeping
 * magnitude + direction valid everywhere the swarm can roam.
 */
function svgExtruded(
    layout: {readonly $: {readonly params: {readonly halfDepth: number}}},
    svgLayout: {readonly $: {readonly sdfSource: d.Infer<d.WgslTexture2d<d.F32>>}},
    cfg: {size: number},
): ShapeField3 {
    const SIZE = cfg.size
    return tgpu.fn([d.vec3f], d.f32)((p3) => {
        'use gpu'
        const prm = layout.$.params
        const sizeM1 = SIZE - 1
        const tx = std.clamp(d.i32((p3.x + 0.5) * SIZE), 0, sizeM1)
        const ty = std.clamp(d.i32((0.5 - p3.y) * SIZE), 0, sizeM1)
        const d2 = std.textureLoad(svgLayout.$.sdfSource, d.vec2u(d.u32(tx), d.u32(ty)), 0).x
        const exX = std.max(std.abs(p3.x) - 0.5, 0.0)
        const exY = std.max(std.abs(p3.y) - 0.5, 0.0)
        const d2e = d2 + std.sqrt(exX * exX + exY * exY)
        return extrudeZ(d2e, std.abs(p3.z) - prm.halfDepth)
    }).$name('shapeFieldSvgExtruded') as ShapeField3
}

/**
 * True analytic 3D volume: the setup's baked SDF sampled at the rotated point — rotation
 * (and any CPU-animated sub-props) arrive through the setup's MarchParams uniform, so the
 * field moves under the swarm exactly like the marched render would.
 */
function analytic3d(layout: RotatedFieldLayout, baked: ShapeField3): ShapeField3 {
    return tgpu.fn([d.vec3f], d.f32)((p3) => {
        'use gpu'
        return baked(sdf3d.rotateVec3(p3, layout.$.params.rot))
    }).$name('shapeFieldAnalytic3d') as ShapeField3
}

/**
 * SVG lifted into 3D (the bevel-extruding SVG-3D setup's SDF), rotated by MarchParams and —
 * like the flat SVG field — extended beyond the SDF texture's footprint with the exact
 * distance to the footprint box (the clamped field is flat out there: zero gradient, and a
 * border-touching feature can even read inside).
 */
function svgLifted3d(layout: RotatedFieldLayout, baked: ShapeField3): ShapeField3 {
    return tgpu.fn([d.vec3f], d.f32)((p3) => {
        'use gpu'
        const ps = sdf3d.rotateVec3(p3, layout.$.params.rot)
        const exX = std.max(std.abs(ps.x) - 0.5, 0.0)
        const exY = std.max(std.abs(ps.y) - 0.5, 0.0)
        return baked(ps) + std.sqrt(exX * exX + exY * exY)
    }).$name('shapeFieldSvgLifted3d') as ShapeField3
}

export const shapeField = {
    analyticExtruded,
    svgExtruded,
    analytic3d,
    svgLifted3d,
} as const

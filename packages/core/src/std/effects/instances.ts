/**
 * std — instance layouts: repeat a child layer in grid / radial / line placements with
 * per-instance variation and an optional per-instance hue rotation, as one declarative
 * recipe. Part bodies live in kit/instanceParts; this module resolves `p()` slots and
 * assembles the loop.
 *
 * The source rect follows the child layer's bounds when the engine provides them (falling
 * back to the full canvas), and the placement field follows the consumer's own bounding
 * box the same way — declare `wantsBoundsParams` on the definition.
 */
import {Expr, type EmitContext, type GpuFragmentParams} from '../../gpu/contract'
import {call, expr, ZERO} from '../../gpu/composer'
import {blend, instanceParts} from '../../gpu/kit/index'
import type {PropRef} from '../values'

const {formatFloat} = instanceParts

// ── Placement specs ─────────────────────────────────────────────────────────────────────

export interface GridSlots {
    columns: PropRef
    rows: PropRef
    gapX: PropRef
    gapY: PropRef
    stagger: PropRef
    /** Structural: alternate-mirror mode ('none' | flip-x | flip-y | both). */
    flip: PropRef
}
export interface RadialSlots {
    count: PropRef
    radius: PropRef
    startAngle: PropRef
    sweep: PropRef
    faceCenter: PropRef
}
export interface LineSlots {
    count: PropRef
    direction: PropRef
    spacing: PropRef
}

export type PlacementSpec =
    | {kind: 'grid'; slots: GridSlots}
    | {kind: 'radial'; slots: RadialSlots}
    | {kind: 'line'; slots: LineSlots}

export const grid = (slots: GridSlots): PlacementSpec => ({kind: 'grid', slots})
export const radial = (slots: RadialSlots): PlacementSpec => ({kind: 'radial', slots})
export const line = (slots: LineSlots): PlacementSpec => ({kind: 'line', slots})

/** Pick a placement member by a structural mode prop — only the active member emits. */
export interface PlacementPick {
    mode: PropRef
    members: {grid: PlacementSpec; radial: PlacementSpec; linear: PlacementSpec}
}
export const byMode = (mode: PropRef, members: PlacementPick['members']): PlacementPick => ({mode, members})

// ── Variation & recipe ──────────────────────────────────────────────────────────────────

export interface VariationSlots {
    scale: PropRef
    rotation: PropRef
    opacity: PropRef
    jitter: {position: PropRef; rotation: PropRef; scale: PropRef; opacity: PropRef}
    seed: PropRef
    /** Seamless layout animation phase (wraps at 1). */
    phase: PropRef
}

export interface RepeatInstancesRecipe {
    /** Crop insets applied to the sampled source rect. */
    source: {crop: {left: PropRef; right: PropRef; top: PropRef; bottom: PropRef}}
    placement: PlacementPick | PlacementSpec
    variation: VariationSlots
    /** Per-instance hue rotation (radians/instance); active when the prop is non-zero. */
    hueShift?: PropRef
    /** Structural stacking order ('forward' | 'backward'). */
    order: PropRef
}

const MODE_MAP: Record<string, number> = {grid: 0, radial: 1, linear: 2}
const FLIP_MAP: Record<string, number> = {'none': 0, 'alternate-flip-x': 1, 'alternate-flip-y': 2, 'both': 3}

/**
 * Build the instance-repeat fragment: placement frame → per-instance variation
 * (→ hue-shift stage) → premultiplied over-accumulation of the child sampled per instance.
 */
export function repeatInstances(recipe: RepeatInstancesRecipe) {
    return (params: GpuFragmentParams): Expr => {
        const {uniforms, childNode, ctx, propValues, childBoundsParams, ownBoundsParams, convertToTexture} = params
        if (!childNode) return ZERO

        const childTex = convertToTexture(childNode)
        const texKey = childTex.key

        // Structural reads: only the active placement member / stages are emitted.
        const structural = (ref: PropRef, map: Record<string, number>): number => {
            const v = propValues[ref.name]
            return typeof v === 'number' ? v : (map[v as string] ?? 0)
        }
        const spec: PlacementSpec = 'members' in recipe.placement
            ? [recipe.placement.members.grid, recipe.placement.members.radial, recipe.placement.members.linear][
                structural(recipe.placement.mode, MODE_MAP)] ?? recipe.placement.members.grid
            : recipe.placement
        const zBack = recipe.order ? (propValues[recipe.order.name] === 'backward' || propValues[recipe.order.name] === 1) : false
        const hueActive = recipe.hueShift ? ((propValues[recipe.hueShift.name] as number) ?? 0) !== 0 : false

        return new Expr((ec: EmitContext) => {
            const f: instanceParts.InstanceEmitFrame = {p: ec.freshLocal('rep'), L: formatFloat}
            const {p, L} = f
            const R = (ref: PropRef): string => {
                const accessor = uniforms[ref.name]
                if (!accessor) throw new Error(`std: repeatInstances binds unknown prop '${ref.name}'`)
                return accessor._emit(ec)
            }
            const stmts: string[] = []

            stmts.push(`let ${p}_uv = ${ctx.uv._emit(ec)};`)
            stmts.push(`let ${p}_aspect = ${ctx.aspect._emit(ec)};`)
            stmts.push(`let ${p}_pix = vec2f(${p}_uv.x * ${p}_aspect, ${p}_uv.y);`)

            // Source rect: the child layer bounds when available, else the full canvas.
            const src = {
                cx: childBoundsParams ? childBoundsParams.centerX._emit(ec) : L(0.5),
                cy: childBoundsParams ? childBoundsParams.centerY._emit(ec) : L(0.5),
                hw: childBoundsParams ? childBoundsParams.halfWidth._emit(ec) : L(0.5),
                hh: childBoundsParams ? childBoundsParams.halfHeight._emit(ec) : L(0.5),
            }
            stmts.push(...instanceParts.sourceRect(f, src, {
                left: R(recipe.source.crop.left), right: R(recipe.source.crop.right),
                top: R(recipe.source.crop.top), bottom: R(recipe.source.crop.bottom),
            }))

            // Placement field: the consumer's own bounding box when active, else the canvas.
            stmts.push(...instanceParts.placementField(f, {
                cx: ownBoundsParams ? ownBoundsParams.centerX._emit(ec) : L(0.5),
                cy: ownBoundsParams ? ownBoundsParams.centerY._emit(ec) : L(0.5),
                hw: ownBoundsParams ? ownBoundsParams.halfWidth._emit(ec) : L(0.5),
                hh: ownBoundsParams ? ownBoundsParams.halfHeight._emit(ec) : L(0.5),
            }))

            stmts.push(`let ${p}_phase = ${R(recipe.variation.phase)};`)
            stmts.push(`let ${p}_seed = ${R(recipe.variation.seed)};`)

            // Radial / line placements anchor to the child layer position (fallback: field centre).
            const anchorSq = childBoundsParams
                ? `vec2f((${childBoundsParams.centerX._emit(ec)}) * ${p}_aspect, ${childBoundsParams.centerY._emit(ec)})`
                : `${p}_fieldCenterSq`
            const placement: instanceParts.PlacementPart =
                spec.kind === 'grid'
                    ? instanceParts.gridPlacement({
                        columns: R(spec.slots.columns), rows: R(spec.slots.rows),
                        gapX: R(spec.slots.gapX), gapY: R(spec.slots.gapY), stagger: R(spec.slots.stagger),
                    }, structural(spec.slots.flip, FLIP_MAP))
                    : spec.kind === 'radial'
                        ? instanceParts.radialPlacement(anchorSq, {
                            count: R(spec.slots.count), sweep: R(spec.slots.sweep),
                            startAngle: R(spec.slots.startAngle), radius: R(spec.slots.radius),
                            faceCenter: R(spec.slots.faceCenter),
                        })
                        : instanceParts.linePlacement(anchorSq, {
                            count: R(spec.slots.count), direction: R(spec.slots.direction), spacing: R(spec.slots.spacing),
                        })

            stmts.push(...instanceParts.accumulateInstances(f, {
                placement,
                variation: instanceParts.instanceVariation(f, {
                    scale: R(recipe.variation.scale), rotation: R(recipe.variation.rotation),
                    opacity: R(recipe.variation.opacity),
                    jitterPosition: R(recipe.variation.jitter.position), jitterRotation: R(recipe.variation.jitter.rotation),
                    jitterScale: R(recipe.variation.jitter.scale), jitterOpacity: R(recipe.variation.jitter.opacity),
                }),
                hue: hueActive && recipe.hueShift ? instanceParts.instanceHueShift(f, R(recipe.hueShift)) : [],
                zBack,
                texKey,
            }))

            for (const s of stmts) ec.statement(s)

            // The accumulator is premultiplied → straight alpha, like every RTT-sampling filter.
            return call(blend.unpremultiplyAlpha, 'unpremultiplyAlpha', [expr(`${p}_accum`)])._emit(ec)
        })
    }
}

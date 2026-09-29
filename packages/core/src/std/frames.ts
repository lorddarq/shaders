/**
 * std — coordinate frames.
 *
 * A frame turns the raw surface coordinate into named, ready-to-use fields (`along`,
 * `across`, `length`…), all bound to WGSL locals so downstream algebra reads one
 * evaluation. Frames own the house conventions: the `uvContext ?? ctx.uv` generator
 * idiom, aspect correction, and the transformPosition storage convention (positions
 * arrive as `(x, 1−y)`; frames recover authored y).
 */
import type {Expr, GpuFragmentParams} from '../gpu/contract'
import type {PropRef} from './values'
import {uniformOf} from './invoke'
import {add, atan2, cos, div, dot, fract, local, max, min, mul, neg, sin, sqrt, sub, vec2, length as vlen} from './math'
import {constants} from '../gpu/kit/index'

const DEG_TO_RAD = constants.DEG_TO_RAD

/** The generator surface: composed UV, effective viewport, and aspect — locals-bound. */
export function surfaceOf(params: GpuFragmentParams): {uv: Expr; viewport: Expr; aspect: Expr} {
    const uv = params.uvContext ?? params.ctx.uv
    const viewport = params.effectiveViewportSize ?? params.ctx.viewportSize
    const aspect = local(div(viewport.member('x'), viewport.member('y')), 'aspect')
    return {uv, viewport, aspect}
}

/**
 * The generator surface against the RAW canvas viewport (no resize-fit box) — for paints whose
 * pattern is anchored to the canvas rather than a fitted box (Blob's organic disc). Same
 * `uvContext ?? ctx.uv` idiom as {@link surfaceOf}.
 */
export function rawSurfaceOf(params: GpuFragmentParams): {uv: Expr; viewport: Expr; aspect: Expr} {
    const uv = params.uvContext ?? params.ctx.uv
    const viewport = params.ctx.viewportSize
    const aspect = local(div(viewport.member('x'), viewport.member('y')), 'aspect')
    return {uv, viewport, aspect}
}

export interface CentredFrame {
    /** Vector from the centre to the pixel, aspect-corrected (centre-scaled, D-1). */
    delta: Expr
    /** Radial distance from the centre. */
    dist: Expr
}

/**
 * An aspect-corrected frame centred on a position prop: the pixel's `delta` from the centre and
 * its radial `dist` — the opening of every centred field. The centre is a transformPosition
 * value (stored `(x, 1−y)`); the centre is scaled, not the UV alone (D-1 centre-scaled rule).
 */
export function centredFrame(slots: {center: PropRef}) {
    return (params: GpuFragmentParams, surface: {uv: Expr; aspect: Expr}): CentredFrame => {
        const center = uniformOf(slots.center, params)
        const {uv, aspect} = surface
        const delta = local(vec2(
            sub(mul(uv.member('x'), aspect), mul(center.member('x'), aspect)),
            sub(uv.member('y'), sub(1, center.member('y'))),
        ), 'centred')
        const dist = local(vlen(delta), 'centredDist')
        return {delta, dist}
    }
}

export interface SegmentFrame {
    /** Distance travelled along the from→to axis (0 at `from`). */
    along: Expr
    /** Signed perpendicular distance from the axis. */
    across: Expr
    /** Length of the from→to segment (≥ 1e-4). */
    length: Expr
}

/**
 * An aspect-corrected frame along the segment between two position props: the pixel's
 * `along`/`across` coordinates plus the segment `length`. Positions are transformPosition
 * values (stored `(x, 1−y)`).
 */
export function segmentFrame(slots: {from: PropRef; to: PropRef}) {
    return (params: GpuFragmentParams, surface: {uv: Expr; aspect: Expr}): SegmentFrame => {
        const from = uniformOf(slots.from, params)
        const to = uniformOf(slots.to, params)
        const {uv, aspect} = surface

        const cx = mul(from.member('x'), aspect)
        const cy = sub(1, from.member('y'))
        const rel = local(vec2(sub(mul(uv.member('x'), aspect), cx), sub(uv.member('y'), cy)), 'rel')
        const toRel = local(
            vec2(sub(mul(to.member('x'), aspect), cx), sub(sub(1, to.member('y')), cy)),
            'toRel',
        )
        const len = local(max(vlen(toRel), 0.0001), 'segLen')
        const dir = local(div(toRel, len), 'segDir')

        const along = local(dot(rel, dir), 'along')
        const across = local(
            add(mul(rel.member('x'), neg(dir.member('y'))), mul(rel.member('y'), dir.member('x'))),
            'across',
        )
        return {along, across, length: len}
    }
}

/** A degrees prop as a unit vector — `(cos θ, sin θ)`, bound to one local. */
export function direction(angleDeg: Expr | number, hint = 'dir'): Expr {
    const rad = mul(angleDeg, DEG_TO_RAD)
    return local(vec2(cos(rad), sin(rad)), hint)
}

export interface DirectionFrame {
    /** Unit vector along the frame's direction. */
    tangent: Expr
    /** Unit vector perpendicular to it (90° counter-clockwise). */
    perp: Expr
    /** Project a 2D point into the frame: distance along / across the direction. */
    coordsOf(p: Expr): {along: Expr; across: Expr}
}

/**
 * A rotated 2D basis from a degrees prop — the frame a directional surface (brush grain, weave,
 * streaks, flow) lives in. `coordsOf` gives grain-space coordinates; `tangent`/`perp` are the
 * axes for smearing or tilting along it.
 */
export function directionFrame(angleDeg: Expr | number, hint = 'grain'): DirectionFrame {
    const tangent = direction(angleDeg, `${hint}T`)
    const perp = local(vec2(neg(tangent.member('y')), tangent.member('x')), `${hint}P`)
    return {
        tangent,
        perp,
        coordsOf(p: Expr) {
            return {
                along: local(dot(p, tangent), 'along'),
                across: local(dot(p, perp), 'across'),
            }
        },
    }
}

/**
 * Fold a point around a centre into one angular sector (mirrored at the sector's half) —
 * the kaleidoscope domain fold. Returns the folded point in polar-recovered coordinates.
 */
export function sectorFold(x: Expr, y: Expr, sectorAngle: Expr, center = 0.5): Expr {
    const PI = 3.141592653589793
    const dx = local(sub(x, center), 'kfDx')
    const dy = local(sub(y, center), 'kfDy')
    const r = local(sqrt(add(mul(dx, dx), mul(dy, dy))), 'kfR')
    const theta = add(atan2(dy, dx), PI)
    const inSector = local(mul(fract(div(theta, sectorAngle)), sectorAngle), 'kfInSector')
    const folded = local(min(inSector, sub(sectorAngle, inSector)), 'kfFolded')
    return vec2(mul(r, cos(folded)), mul(r, sin(folded)))
}

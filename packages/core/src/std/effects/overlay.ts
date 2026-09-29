/**
 * std/effects — the HUD-drawing vocabulary shared by the overlay-role shaders (ObjectTracker,
 * KeyFrames): premultiplied "over" compositing, round-rect signed distance, the corner-bracket
 * stroke mask, and the premultiplied→straight output convention. The part BODIES live in
 * kit/overlayParts (statement-level string builders — see the rationale there); this module is
 * the std-facing surface shader definitions import.
 *
 * Only multi-consumer vocabulary lives here (the pass-4 rule: single-consumer stages stay in
 * their shader file as named parts).
 */
export {overPremul, roundRectSD, cornerBracketMask, straightFromPremul} from '../../gpu/kit/overlayParts'

// ── Detection overlay recipe ────────────────────────────────────────────────────────────────
import type {Expr as ExprT, GpuFragmentParams, EmitContext} from '../../gpu/contract'
import {Expr, formatFloat, ZERO} from '../../gpu/porters'
import {blend, overlayParts} from '../../gpu/kit/index'
import {straightFromPremul} from '../../gpu/kit/overlayParts'
import type {PropRef} from '../values'

const num = (v: unknown, f: number): number => (typeof v === 'number' ? v : f)
const DETECTION_REFERENCE_HEIGHT = 1080.0

/** The prop roles the detection-overlay recipe binds. */
export interface DetectionOverlaySlots {
    detectionMode: PropRef
    threshold: PropRef
    layout: PropRef
    cellSize: PropRef
    maxDepth: PropRef
    boxStyle: PropRef
    lineWidth: PropRef
    cornerRadius: PropRef
    strokeColor: PropRef
    fillColor: PropRef
    labelColor: PropRef
    labelBackgroundColor: PropRef
    labelMode: PropRef
    labelPosition: PropRef
    labelInset: PropRef
    labelRadius: PropRef
    fontFamily: PropRef
    fontWeight: PropRef
    fontSize: PropRef
    letterSpacing: PropRef
}

/**
 * The detection-overlay recipe: each pixel walks a spatial partition of the canvas to its leaf
 * cell, scans that leaf for content (coverage + a content-tight bbox), and — if the leaf holds
 * an object — composites a bounding box + an optional glyph-pill label over the source. One
 * raw-WGSL builder over the kit's detection parts: the walk is JS-unrolled over compile-time
 * `maxDepth`, each cell scan a JS-unrolled N×N textureSampleLevel loop, bbox accumulation via
 * `select`-folds (uniform control flow, AA stays valid). Compile-time branches (detectionMode /
 * layout / maxDepth / boxStyle / labelMode / labelPosition) read propValues.
 */
export function detectionOverlay(slots: DetectionOverlaySlots): (params: GpuFragmentParams) => ExprT {
    return (params) => {
        const {uniforms, ctx, childNode, convertToTexture, propValues} = params
        if (!childNode) return ZERO
        const childTex = convertToTexture(childNode)
        const texKey = childTex.key

        // ── Compile-time options ──
        const modeMap: Record<string, number> = {alpha: 0, bright: 1, dark: 2, red: 3, green: 4, blue: 5}
        const layoutMap: Record<string, number> = {grid: 0, quadtree: 1, mosaic: 2}
        const labelMap: Record<string, number> = {none: 0, dimensions: 1, percentage: 2}
        const posMap: Record<string, number> = {'bottom-left': 0, 'bottom-right': 1, 'top-left': 2, 'top-right': 3}
        const rawMode = propValues[slots.detectionMode.name]
        const modeStr = (['alpha', 'bright', 'dark', 'red', 'green', 'blue'] as const)[
            typeof rawMode === 'number' ? rawMode : (modeMap[rawMode as string] ?? 1)
        ]
        const rawLayout = propValues[slots.layout.name]
        const layoutStr = (['grid', 'quadtree', 'mosaic'] as const)[
            typeof rawLayout === 'number' ? rawLayout : (layoutMap[rawLayout as string] ?? 2)
        ]
        let D = Math.round(num(propValues[slots.maxDepth.name], 2))
        if (!Number.isFinite(D)) D = 2
        D = Math.max(0, Math.min(5, D))
        if (layoutStr === 'grid') D = 0
        const rawBox = propValues[slots.boxStyle.name]
        const cornerStyle = (typeof rawBox === 'string' ? rawBox : 'corners') === 'corners'
        const rawLabel = propValues[slots.labelMode.name]
        const labelStr = (['none', 'dimensions', 'percentage'] as const)[
            typeof rawLabel === 'number' ? rawLabel : (labelMap[rawLabel as string] ?? 0)
        ]
        const labelsOn = labelStr !== 'none'
        const rawPos = propValues[slots.labelPosition.name]
        const labelPos = typeof rawPos === 'number' ? rawPos : (posMap[rawPos as string] ?? 1)

        // ── Label glyph atlas (only when labels on) — the host recipe part ──
        const atlasKey = labelsOn
            ? overlayParts.createGlyphStripAtlas(params, {
                label: 'DetectionOverlay:atlas',
                familyProp: slots.fontFamily.name, weightProp: slots.fontWeight.name,
            }).key
            : undefined

        return new Expr((ec: EmitContext) => {
            const p = ec.freshLocal('ot')
            const s: string[] = []
            const f: overlayParts.DetectionEmitFrame = {
                p,
                U: (name: string): string => uniforms[slots[name as keyof DetectionOverlaySlots].name]._emit(ec),
                L: (n: number): string => formatFloat(n),
                s,
                sampleChild: (uvWgsl: string): string => `textureSampleLevel(tex.$.${texKey}, samp.$.linearClamp, ${uvWgsl}, 0.0)`,
                UN: ec.external(blend.unpremultiplyAlpha, 'unpremultiplyAlpha'),
                modeStr,
            }
            const {U, L} = f

            // ── Shared HUD frame: logical-pixel coords, AA width, scaled line/box metrics ──
            s.push(`let ${p}_res = ${ctx.logicalViewportSize._emit(ec)};`)
            s.push(`let ${p}_scaleFactor = ${p}_res.y / ${L(DETECTION_REFERENCE_HEIGHT)};`)
            const uvE = ctx.uv._emit(ec)
            s.push(`let ${p}_fragPx = vec2f((${uvE}).x * ${p}_res.x, (${uvE}).y * ${p}_res.y);`)
            s.push(`let ${p}_aa = max(max(fwidth(${p}_fragPx.x), fwidth(${p}_fragPx.y)), ${L(0.0001)});`)
            s.push(`let ${p}_cellPx = max(${U('cellSize')} * ${p}_scaleFactor, ${L(4)});`)
            s.push(`let ${p}_minSize = ${L(3)} * ${p}_scaleFactor;`)
            s.push(`let ${p}_lineHalf = max(${U('lineWidth')} * ${p}_scaleFactor * ${L(0.5)}, ${L(0.5)});`)
            s.push(`let ${p}_cornerRadiusPx = ${U('cornerRadius')} * ${p}_scaleFactor;`)
            s.push(`let ${p}_boxInset = ${p}_lineHalf + ${p}_scaleFactor * ${L(1.5)};`)
            s.push(`let ${p}_thr = ${U('threshold')};`)

            // ── Analysis core (atomic — see the irreducibility note on the parts) ──
            overlayParts.partitionWalkStmts(f, layoutStr, D)
            overlayParts.contentBBoxStmts(f)

            s.push(`var ${p}_dst = ${f.sampleChild(`(${uvE})`)};`)

            // ── Boxes / labels drawn by parts ──
            overlayParts.detectionBoxStmts(f, cornerStyle)
            if (labelsOn && atlasKey) overlayParts.glyphPillLabelStmts(f, {labelStr, labelPos, atlasKey})

            for (const line of s) ec.statement(line)
            // Premultiplied → straight for output.
            return straightFromPremul(`${p}_dst`)
        })
    }
}

// ── Motion-tracker HUD recipe ───────────────────────────────────────────────────────────────
import type {KitTexture} from '../../gpu/contract'
import {trackerSim} from '../../gpu/kit/index'
import {diamondStamp, pathSegmentStamp, ringTrailStmts, bracketGizmo, type HudEmitFrame} from '../../gpu/kit/overlayParts'

/** The prop roles the motion-tracker HUD recipe binds. */
export interface MotionTrackerHudSlots {
    trackers: PropRef
    trail: PropRef
    markerSize: PropRef
    lineWidth: PropRef
    markerColor: PropRef
    keyframeColor: PropRef
    pathColor: PropRef
}

/**
 * The motion-tracker HUD fragment: ONE copy of the draw body in a runtime loop over the tracker
 * count, reading the pursuit sim's state rows (`trackState` compute output). Per pixel each
 * tracker costs 2 cache-resident loads (head + trail AABB) unless the pixel is inside its padded
 * box, where the trail-ring stamps (keyframe diamond + path segment) and the bracket gizmo run.
 * The AABB cull is an atomic raw-WGSL core: a runtime-count loop whose draw body sits behind a
 * per-tracker early-out, which no Expr-graph fold can express inside emitted control flow.
 */
export function motionTrackerHud(slots: MotionTrackerHudSlots): (params: GpuFragmentParams) => ExprT {
    return (params) => {
        const {uniforms, ctx, childNode, convertToTexture, computeOutputs} = params
        if (!childNode) return ZERO
        const childTex = (computeOutputs?.childTexture as KitTexture | undefined) ?? convertToTexture(childNode)
        const texKey = childTex.key
        const stateKit = computeOutputs?.trackState as KitTexture | undefined

        return new Expr((ec: EmitContext) => {
            const p = ec.freshLocal('kft')
            const s: string[] = []
            const f: HudEmitFrame = {
                p,
                U: (name: string): string => uniforms[slots[name as keyof MotionTrackerHudSlots].name]._emit(ec),
                L: (n: number): string => formatFloat(n),
                s,
            }
            const {U, L} = f
            const uvE = ctx.uv._emit(ec)

            // ── Shared HUD frame: logical-pixel coords, AA width, scaled stroke/gizmo metrics ──
            s.push(`let ${p}_res = ${ctx.logicalViewportSize._emit(ec)};`)
            s.push(`let ${p}_sf = ${p}_res.y / ${L(DETECTION_REFERENCE_HEIGHT)};`)
            s.push(`let ${p}_px = vec2f((${uvE}).x * ${p}_res.x, (${uvE}).y * ${p}_res.y);`)
            s.push(`let ${p}_aa = max(max(fwidth(${p}_px.x), fwidth(${p}_px.y)), ${L(0.0001)});`)
            s.push(`let ${p}_lw = max(${U('lineWidth')} * ${p}_sf, ${L(0.5)});`)
            s.push(`let ${p}_ms = max(${U('markerSize')} * ${p}_sf, ${L(4)});`)
            s.push(`var ${p}_dst = textureSampleLevel(tex.$.${texKey}, samp.$.linearClamp, (${uvE}), 0.0);`)

            if (stateKit) {
                const stateKey = stateKit.key
                const load = (col: string): string => `textureLoad(tex.$.${stateKey}, vec2u(${col}, ti), 0)`

                // Runtime tracker loop with the per-tracker AABB early-out.
                s.push(`let ${p}_n = u32(clamp(${U('trackers')}, ${L(1)}, ${L(trackerSim.TRACKER_MAX)}));`)
                s.push(`for (var ti = 0u; ti < ${p}_n; ti = ti + 1u) {`)
                s.push(`let head = ${load('0u')};`)
                s.push(`if (head.w < 0.5) { continue; }`)
                // Per-pixel early-out on the kernel-published trail AABB, padded by gizmo reach.
                s.push(`let bb = ${load(`${trackerSim.TRACKER_AABB_COL}u`)};`)
                s.push(`let pad = ${p}_ms + ${p}_lw * ${L(4)} + ${p}_aa * ${L(2)};`)
                s.push(`let bbMin = vec2f(bb.x * ${p}_res.x, bb.y * ${p}_res.y) - vec2f(pad);`)
                s.push(`let bbMax = vec2f(bb.z * ${p}_res.x, bb.w * ${p}_res.y) + vec2f(pad);`)
                s.push(`if (all(${p}_px >= bbMin) && all(${p}_px <= bbMax)) {`)
                // NOTE: raw-WGSL locals must dodge WGSL reserved words (`meta`, `half`, …).
                s.push(`let mta = ${load('1u')};`)
                s.push(`let tp = vec2f(head.x * ${p}_res.x, head.y * ${p}_res.y);`)
                s.push(`let searching = select(${L(1)}, ${L(0.45)}, mta.x > ${L(4)});`)

                // ── trail ring frame → diamond + path-segment stamps → bracket gizmo ──
                ringTrailStmts(f, {
                    len: trackerSim.TRACKER_TRAIL_LEN, every: trackerSim.TRACKER_TRAIL_EVERY,
                    load, stages: [diamondStamp, pathSegmentStamp(trackerSim.TRACKER_TRAIL_EVERY)],
                })
                bracketGizmo(f)

                s.push(`}`) // AABB early-out
                s.push(`}`) // for ti
            }

            for (const line of s) ec.statement(line)
            // Premultiplied → straight for output.
            return straightFromPremul(`${p}_dst`)
        })
    }
}

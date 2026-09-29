/**
 * std/effects — the painterly stylize parts (watercolor washes, chalk sketching, ASCII glyph
 * screens). Each part takes a slot object of prop bindings (`p('name')`) or other parts, and the
 * shader definition file composes them into the effect's recipe — a shared frame (a wobbled tap
 * position, a character grid) is built once and handed to every stage that reads it.
 *
 * Parts whose result is read by more than one consumer are memoised per composition so the
 * readers share one lowered Expr (exactly a `const` local in a fused builder). The gather parts
 * own the species' alpha discipline; the ASCII parts are custom-tier stages over
 * `GpuFragmentParams` because their glyph atlas is rasterised on the CPU — the host lifecycle
 * stays with the definition and is handed in through the `atlas` slot.
 */
import type {Expr, GpuFragmentParams, KitTexture} from '../../gpu/contract'
import {call, vec4, floatE, expr, ZERO, asLocal} from '../../gpu/porters'
import {stylizePaints, blend, tgpu, d, std, constants, noise} from '../../gpu/kit/index'
import type {RttFilterParams} from '../../gpu/scaffolds/rttFilter'
import type {GatherEffect} from '../types'
import {gather} from '../filter'
import type {PropRef} from '../values'
import {uniformOf} from '../invoke'

/** One gather stage: a value derived from the child RTT. */
export type GatherStage = (params: RttFilterParams) => Expr

/**
 * Memoise a stage per composition so every reader shares ONE lowered value — the slot-graph
 * equivalent of a `const` local in a fused builder.
 */
function shared<P extends object, V>(build: (params: P) => V): (params: P) => V {
    const cache = new WeakMap<P, V>()
    return (params) => {
        const hit = cache.get(params)
        if (hit !== undefined) return hit
        const v = build(params)
        cache.set(params, v)
        return v
    }
}

// ═══ Watercolor parts ═══════════════════════════════════════════════════════════════════════════

/**
 * The bleed frame: the tap position with a time-wobbled edge break, shared by the original
 * sample and every brush tap. `amount` at 0 compiles the wobble away (pair with
 * `recompile: crosses(0)` on the definition); the wobble is authored in LOGICAL pixels.
 */
export function bleedWarp(slots: {amount: PropRef}): GatherStage {
    return shared((params) => {
        const amount = (params.propValues[slots.amount.name] as number) ?? 0
        if (amount === 0) return params.ctx.uv
        return asLocal(call(stylizePaints.watercolorUvBase, 'watercolorUvBase', [
            params.ctx.uv, params.ctx.logicalViewportSize, uniformOf(slots.amount, params), params.ctx.time,
        ]), 'wcUvBase')
    })
}

/** The four overlapping Kuwahara quadrants: premultiplied sum + rgb sum² each, plus the tap weight. */
export type KuwaharaBrush = (params: RttFilterParams) => {quadrants: {sum: Expr; sq: Expr}[]; invN: number}

/**
 * The Kuwahara brush: 4 overlapping quadrants of taps around the `at` frame, accumulating
 * premultiplied sum + rgb sum² per quadrant at the builder level. `radius` is a compile-time
 * brush size (the tap loop is unrolled in JS, stride-2 taps at texel-pair midpoints so one
 * bilinear fetch covers a 2×2 block); every tap is hoisted into a local so it samples exactly
 * once. The footprint is authored in LOGICAL pixels so it stays the same physical size at any
 * DPR.
 */
export function kuwahara(slots: {at: GatherStage; radius: PropRef}): KuwaharaBrush {
    return shared((params) => {
        // compileTime brush radius (clamps round(radius) to [1,8], with a `?? 4` fallback).
        const R = Math.max(1, Math.min(8, Math.round((params.propValues[slots.radius.name] as number) ?? 4)))
        // At least 2 taps per axis — a 1-tap quadrant has zero variance by construction, so all
        // four sigmas tie and the Kuwahara pick degenerates into a fixed diagonal offset sample.
        const taps = Math.max(2, Math.ceil((R + 1) / 2))
        const invN = 1 / (taps * taps)
        const uvBase = slots.at(params)
        const lvp = params.ctx.logicalViewportSize

        const quadrant = (dx: number, dy: number): {sum: Expr; sq: Expr} => {
            let sum: Expr = expr('vec4f(0.0, 0.0, 0.0, 0.0)')
            let sq: Expr = expr('vec3f(0.0, 0.0, 0.0)')
            for (let j = 0; j < taps; j++) {
                for (let i = 0; i < taps; i++) {
                    const uv = call(stylizePaints.watercolorTapUV, 'watercolorTapUV',
                        [uvBase, lvp, floatE((2 * i + 0.5) * dx), floatE((2 * j + 0.5) * dy)])
                    const c = asLocal(params.texture.sample(uv), 'wcTap')
                    const rgb = c.member('rgb')
                    sum = sum.add(c)
                    sq = sq.add(rgb.mul(rgb))
                }
            }
            return {sum, sq}
        }
        return {quadrants: [quadrant(-1, -1), quadrant(1, -1), quadrant(-1, 1), quadrant(1, 1)], invN}
    })
}

/**
 * The paper-grain stage: screen-space (device-pixel) value noise. `amount` at 0 compiles the
 * noise away (pair with `recompile: crosses(0)` on the definition).
 */
export function paperGrain(slots: {amount: PropRef}): GatherStage {
    return (params) => {
        const amount = (params.propValues[slots.amount.name] as number) ?? 0
        return amount === 0 ? floatE(0) : call(stylizePaints.watercolorGrain, 'watercolorGrain', [params.ctx.uv, params.ctx.viewportSize])
    }
}

/**
 * The wash compose gather recipe: the winning-quadrant Kuwahara mean (lowest variance, mixed
 * back toward the original by `strength`) over the `at` frame, broken by the paper grain toward
 * `paperColor`. The gather runs in PREMULTIPLIED space and unpremultiplies once in the compose,
 * so the result is straight alpha. `strength` at 0 compiles the whole gather away (pair with
 * `recompile: crosses(0)` on the definition).
 */
export function wash(slots: {
    at: GatherStage
    brush: KuwaharaBrush
    grain: GatherStage
    paper: PropRef
    paperColor: PropRef
    strength: PropRef
}): GatherEffect {
    return gather({
        resultAlpha: 'straight',
        build: (params): Expr => {
            const u = (ref: PropRef) => uniformOf(ref, params)
            const original = asLocal(params.sampleStraight(slots.at(params)), 'wcOriginal')

            // strength=0 → compose mixes entirely back to the original, so skip the gather outright.
            const strength = (params.propValues[slots.strength.name] as number) ?? 1
            if (strength === 0) return original

            const {quadrants: [q0, q1, q2, q3], invN} = slots.brush(params)
            const grain = slots.grain(params)
            const finalRGB = call(stylizePaints.watercolorCompose, 'watercolorCompose', [
                q0.sum, q1.sum, q2.sum, q3.sum, q0.sq, q1.sq, q2.sq, q3.sq,
                floatE(invN), original, u(slots.paperColor), u(slots.paper), u(slots.strength), grain,
            ])
            // Output is straight alpha (the winning quadrant mean is unpremultiplied in compose).
            return vec4(finalRGB, original.member('a'))
        },
    })
}

// ═══ Chalk parts ════════════════════════════════════════════════════════════════════════════════

/** The Sobel ring: the two 4-luma bundles around the centre tap, plus the centre sample itself. */
export type SobelRing = (params: RttFilterParams) => {sobelA: Expr; sobelB: Expr; centre: Expr; centreLum: Expr}

/**
 * The Sobel tap ring: 8 straight-alpha tap luminances at `spacing` around the fragment (packed
 * `[tl, t, tr, l]` / `[r, bl, b, br]`), plus the centre sample and its luminance. Multi-tap → a
 * true gather, not a uvRemap candidate.
 */
export function sobelTaps(slots: {spacing: PropRef}): SobelRing {
    return shared((params) => {
        const {sampleStraight, ctx} = params
        const viewport = ctx.viewportSize
        const lumAt = (ox: number, oy: number): Expr => {
            const uv = call(stylizePaints.chalkOffsetUV, 'chalkOffsetUV', [ctx.uv, viewport, uniformOf(slots.spacing, params), floatE(ox), floatE(oy)])
            return call(stylizePaints.chalkLuma, 'chalkLuma', [sampleStraight(uv)])
        }
        const tl = lumAt(-1, -1)
        const t = lumAt(0, -1)
        const tr = lumAt(1, -1)
        const l = lumAt(-1, 0)
        const r = lumAt(1, 0)
        const bl = lumAt(-1, 1)
        const bo = lumAt(0, 1)
        const br = lumAt(1, 1)
        const centre = sampleStraight(ctx.uv)
        const centreLum = call(stylizePaints.chalkLuma, 'chalkLuma', [centre])
        return {sobelA: vec4(tl, t, tr, l), sobelB: vec4(r, bl, bo, br), centre, centreLum}
    })
}

/**
 * The chalk-sketch compose gather recipe: the Sobel edge magnitude from the ring becomes the
 * stroke outline, cross-hatch line families fill darker regions, and value-noise dust breaks up
 * the strokes over the board color. Straight alpha throughout (every tap is unpremultiplied).
 */
export function chalkSketch(slots: {
    edges: SobelRing
    sensitivity: PropRef
    hatchScale: PropRef
    shading: PropRef
    grain: PropRef
    board: PropRef
    chalk: PropRef
}): GatherEffect {
    return gather({
        resultAlpha: 'straight',
        build: (params): Expr => {
            const u = (ref: PropRef) => uniformOf(ref, params)
            const {sobelA, sobelB, centre, centreLum} = slots.edges(params)
            const misc = vec4(centreLum, centre.member('a'), 0, 0)
            const tuning = vec4(u(slots.sensitivity), u(slots.hatchScale), u(slots.shading), u(slots.grain))
            return call(stylizePaints.chalkboardCompose, 'chalkboardCompose', [
                sobelA, sobelB, params.ctx.uv, params.ctx.viewportSize, misc, tuning, u(slots.board), u(slots.chalk),
            ])
        },
    })
}

// ═══ ASCII parts (custom tier — CPU-rasterised glyph atlas) ═════════════════════════════════════

/** One custom-tier fragment stage. */
export type FragmentStage = (params: GpuFragmentParams) => Expr

/**
 * The character grid frame: the cell struct (`cellCenter`, `cellUV`, `isOutside`) shared by the
 * cell sample, the glyph lookup, and the tint.
 */
export function charGrid(slots: {cellSize: PropRef; spacing: PropRef}): FragmentStage {
    return shared((params) => call(stylizePaints.asciiGrid, 'asciiGrid', [
        params.ctx.uv, params.ctx.viewportSize, uniformOf(slots.cellSize, params), uniformOf(slots.spacing, params),
    ]))
}

/** The per-cell child color: the child sampled once at the cell centre, unpremultiplied. */
export function cellSample(slots: {grid: FragmentStage}): FragmentStage {
    return shared((params) => {
        const child = params.convertToTexture(params.childNode!)
        return call(blend.unpremultiplyAlpha, 'unpremultiplyAlpha', [child.sample(slots.grid(params).member('cellCenter'))])
    })
}

/**
 * The glyph lookup: the cell brightness (gamma-curved) picks a glyph from the CPU-rasterised
 * atlas, sampled nearest/clamped at the cell-local UV. Requires the
 * `_charCount`/`_atlasScale`/`_atlasSize` extraFields on the definition; the `atlas` slot runs
 * the host raster lifecycle at composition time and returns the glyph texture.
 */
export function glyphFor(slots: {
    grid: FragmentStage
    cell: FragmentStage
    gamma: PropRef
    atlas: (params: GpuFragmentParams) => KitTexture
}): FragmentStage {
    return (params) => {
        const atlasKit = slots.atlas(params)
        const atlasUV = call(stylizePaints.asciiAtlasUV, 'asciiAtlasUV', [
            slots.cell(params).member('rgb'), uniformOf(slots.gamma, params),
            params.uniforms._charCount, params.uniforms._atlasSize, params.uniforms._atlasScale,
            slots.grid(params).member('cellUV'),
        ])
        return atlasKit.sample(atlasUV, 'nearestClamp')
    }
}

/**
 * The glyph tint compose: the glyph shaped by the cell color, with background / out-of-bounds /
 * below-threshold pixels turned transparent. Returns the fragment closure for the custom tier's
 * `gpu.fragment`.
 */
export function glyphTint(slots: {
    grid: FragmentStage
    cell: FragmentStage
    glyph: FragmentStage
    alphaThreshold: PropRef
    preserveAlpha: PropRef
}): FragmentStage {
    return (params) => {
        if (!params.childNode) return ZERO
        const cellColor = slots.cell(params)
        const glyph = slots.glyph(params)
        return call(stylizePaints.asciiCompose, 'asciiCompose', [
            glyph.member('rgb'), cellColor, slots.grid(params).member('isOutside'),
            uniformOf(slots.alphaThreshold, params), uniformOf(slots.preserveAlpha, params),
        ])
    }
}

// ═══ Placed-box frame ═════════════════════════════════════════════════════════════════════════

const OBF_DEG_TO_RAD = constants.DEG_TO_RAD

/**
 * Oriented-box placement field: aspect-corrected screen UV → rotated box-local UV + inside mask.
 * `center` is the TRANSFORMED position prop (transformPosition stores (x, 1-y)), so `1 - center.y`
 * recovers the authored y. Returns vec3(localU, localV, insideMask); the consumer samples its
 * content texture at `.xy` and gates by `.z`. No V-flip — media textures written via
 * `copyExternalImageToTexture` are top-left origin, matching the local V directly.
 */
export const orientedBoxField = tgpu.fn([d.vec2f, d.f32, d.vec2f, d.f32, d.f32, d.f32], d.vec3f)(
    (uv, aspect, center, rotationDeg, halfW, halfH) => {
        'use gpu'
        const aspectUV = d.vec2f(uv.x * aspect, uv.y)
        const centerPos = d.vec2f(center.x * aspect, 1.0 - center.y)
        const dx = aspectUV.x - centerPos.x
        const dy = aspectUV.y - centerPos.y
        const rotRad = rotationDeg * OBF_DEG_TO_RAD
        const cosR = std.cos(rotRad)
        const sinR = std.sin(rotRad)
        const rdx = dx * cosR + dy * sinR
        const rdy = dy * cosR - dx * sinR
        const localU = rdx / (halfW * 2.0) + 0.5
        const localV = rdy / (halfH * 2.0) + 0.5
        const insideMask = std.step(0.0, localU) * std.step(localU, 1.0) * std.step(0.0, localV) * std.step(localV, 1.0)
        return d.vec3f(localU, localV, insideMask)
    },
)

// ═══ Line-engraving parts ═════════════════════════════════════════════════════════════════════

const HP_TAU = constants.TAU
const HP_DEG_TO_RAD = constants.DEG_TO_RAD

/**
 * One hatched line plate. Lines run along `angleDeg`; the coordinate perpendicular to them
 * carries a cosine wave whose crests are inked. `level` (0 = black, 1 = white) slides the ink
 * threshold across the wave, so darker tone → wider ink until the lines merge to solid;
 * `reliefPhase` shifts the wave with image brightness — the classic engraved "lines climb over
 * the form" look. `aa` is the analytic per-pixel wave footprint used for the smoothstep edge.
 */
export const hatchPlate = tgpu.fn([d.vec2f, d.f32, d.f32, d.f32, d.f32, d.f32], d.f32)(
    (p, angleDeg, frequency, reliefPhase, level, aa) => {
        'use gpu'
        const angleRad = angleDeg * HP_DEG_TO_RAD
        const c = std.cos(angleRad)
        const s = std.sin(angleRad)
        // Coordinate perpendicular to the line direction (lines run along (c, s)).
        const w = p.x * (s * -1.0) + p.y * c
        const phase = w * frequency * HP_TAU + reliefPhase
        const v = std.cos(phase)
        // level 0 (black) → threshold below the wave → solid ink; level 1 (white) → above → none.
        const thr = std.mix(-1.15, 1.15, std.clamp(level, 0.0, 1.0))
        return std.smoothstep(thr - aa, thr + aa, v)
    })

/**
 * The spiral plate: an Archimedean spiral around `c` — rings spaced 1/frequency apart, each
 * advancing one full spacing per turn, so the line spirals continuously outward from the centre
 * (the classic guilloché portrait cut). Same tonal threshold + relief displacement as the
 * straight plate; the angular term's footprint grows toward the centre, so the spiral naturally
 * tightens into a dense knot there.
 */
export const spiralPlate = tgpu.fn([d.vec2f, d.vec2f, d.f32, d.f32, d.f32, d.f32], d.f32)(
    (p, c, frequency, reliefPhase, level, aa) => {
        'use gpu'
        const dv = p.sub(c)
        const r = std.length(dv)
        const theta = std.atan2(dv.y, dv.x)
        const phase = (r * frequency - theta * (1.0 / HP_TAU)) * HP_TAU + reliefPhase
        const v = std.cos(phase)
        const thr = std.mix(-1.15, 1.15, std.clamp(level, 0.0, 1.0))
        return std.smoothstep(thr - aa, thr + aa, v)
    })

/**
 * The linework composite (factory — `style` is a compile-time JS branch: 0 = single line plate,
 * 1 = cross-hatch (base + a +72° plate engaging in the shadows + a −38° plate in the deepest
 * blacks — the classic copper-plate build-up), 2 = one continuous spiral cut around `center`).
 *
 * Shared tonal front-end: pivot-0.5 contrast curve on luminance, wavy burin-stroke domain warp,
 * analytic AA, brightness→phase relief — then the style's plate stack inked between `paper`
 * and `ink`. p0 = (angleDeg, frequency, relief, waviness); p1 = (contrast, viewportY, 0, 0).
 */
export function makeLineworkComposite(style: number) {
    return tgpu.fn(
        [d.vec4f, d.vec2f, d.f32, d.vec4f, d.vec4f, d.vec2f, d.vec4f, d.vec4f], d.vec4f)(
        (childColor, uv, aspect, paper, ink, center, p0, p1) => {
            'use gpu'
            const angleDeg = p0.x
            const frequency = p0.y
            const relief = p0.z
            const waviness = p0.w
            const contrast = p1.x
            const viewportY = p1.y

            // Tonal level: luminance through a pivot-0.5 contrast curve.
            const lum = std.dot(d.vec3f(childColor.x, childColor.y, childColor.z), d.vec3f(0.299, 0.587, 0.114))
            const level = std.clamp(0.5 + (lum - 0.5) * contrast, 0.0, 1.0)

            // Wavy domain: two decorrelated noise channels bend the line work organically.
            const pa = d.vec2f(uv.x * aspect, uv.y)
            const nx = noise.mxNoiseFloat2(pa.mul(3.1))
            const ny = noise.mxNoiseFloat2(pa.mul(3.1).add(d.vec2f(7.31, 3.77)))
            const p = pa.add(d.vec2f(nx, ny).mul(waviness * 0.06))

            // Analytic AA: the wave's max per-pixel footprint (phase slope / viewport height).
            const aa = std.clamp(frequency * HP_TAU / std.max(viewportY, 1.0), 0.02, 1.0)

            // Brightness shifts the wave phase by up to ~1.5 periods (frequency-independent relief).
            const reliefPhase = relief * level * 9.42

            let cover = d.f32(0)
            if (style === 2) {
                // Spiral cut around the configurable centre (transformPosition stores (x, 1 − y)).
                const sc = d.vec2f(center.x * aspect, 1.0 - center.y)
                cover = spiralPlate(p, sc, frequency, reliefPhase, level, aa)
            } else if (style === 1) {
                // Base plate + shadow cross plate (+72°) + deep-black plate (−38°).
                const ink1 = hatchPlate(p, angleDeg, frequency, reliefPhase, level, aa)
                const level2 = std.clamp(level / 0.55, 0.0, 1.0)
                const ink2 = hatchPlate(p, angleDeg + 72.0, frequency * 0.92, reliefPhase * 0.7, level2, aa)
                const level3 = std.clamp(level / 0.28, 0.0, 1.0)
                const ink3 = hatchPlate(p, angleDeg - 38.0, frequency * 1.13, reliefPhase * 0.5, level3, aa)
                cover = std.max(ink1, std.max(ink2, ink3))
            } else {
                cover = hatchPlate(p, angleDeg, frequency, reliefPhase, level, aa)
            }

            const rgb = std.mix(d.vec3f(paper.x, paper.y, paper.z), d.vec3f(ink.x, ink.y, ink.z), d.vec3f(cover))
            const alpha = childColor.w * std.mix(paper.w, ink.w, cover)
            return d.vec4f(rgb.x, rgb.y, rgb.z, alpha)
        }).$name(`lineworkComposite_${style}`)
}

/** Style names → plate-stack modes, matching the linework `style` prop transform. */
const LINEWORK_MODES: Record<string, number> = {line: 0, crosshatch: 1, spiral: 2}

/**
 * Line-engraving gather noun: sample the composed child once (unpremultiplied) and redraw it as
 * luminance-displaced line work between `paper` and `ink`. `style` binds a compile-time prop —
 * exactly one plate stack is emitted per composition. The composite works on and returns
 * STRAIGHT alpha, so no unpremultiply tail is appended.
 */
export function linework(slots: {
    style: PropRef
    frequency: PropRef
    angle: PropRef
    center: PropRef
    relief: PropRef
    waviness: PropRef
    contrast: PropRef
    ink: PropRef
    paper: PropRef
}): GatherEffect {
    return gather({
        resultAlpha: 'straight',
        build: ({sampleStraight, ctx, uniforms, propValues}): Expr => {
            const childColor = sampleStraight(ctx.uv)
            const raw = propValues[slots.style.name]
            const style = typeof raw === 'number' ? raw : (LINEWORK_MODES[String(raw)] ?? 1)
            const composite = makeLineworkComposite(style)
            const p0 = vec4(uniforms[slots.angle.name], uniforms[slots.frequency.name], uniforms[slots.relief.name], uniforms[slots.waviness.name])
            const p1 = vec4(uniforms[slots.contrast.name], ctx.viewportSize.member('y'), floatE(0), floatE(0))
            return call(composite, `lineworkComposite_${style}`, [
                childColor, ctx.uv, ctx.aspect, uniforms[slots.paper.name], uniforms[slots.ink.name], uniforms[slots.center.name], p0, p1,
            ])
        },
    })
}

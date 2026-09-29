/**
 * std/paint — the pattern parts: lattice frames, per-lattice cell fields, per-cell variation,
 * stroke-over-fill mixing, the simple band/ring masks, and the print-screen stages (halftone
 * plates, dither grids). Each part takes a slot object of prop bindings (`p('name')`) or other
 * parts, and the shader definition file composes them into the effect's recipe — the composition
 * lives in the shader file, the parts live here.
 *
 * A paint part is `(params) => Expr`; parts whose result is read by more than one consumer are
 * memoised per composition so the readers share one lowered Expr (exactly a `const` local in a
 * fused builder). Parts own the UV-context idiom (a UV-propagating parent supplies a distorted
 * `uvContext` + `effectiveViewportSize`; standalone falls back to `ctx.uv` / `ctx.viewportSize`)
 * and their own animated-time reads (`animatedTime: {speed}` stays declared on the definition).
 */
import type {Expr, GpuFragmentParams, GpuMapSampleUVs} from '../../gpu/contract'
import {call, vec4, mixExpr, animatedTime, arrayExpr, floatE} from '../../gpu/porters'
import {patternPaints, cells as cellKit} from '../../gpu/kit/index'
import type {RttFilterParams} from '../../gpu/scaffolds/rttFilter'
import type {GatherEffect} from '../types'
import {gather} from '../filter'
import type {PropRef} from '../values'
import {uniformOf, paintFrame} from '../invoke'
import {mixColorsIn} from './fields'
import {local} from '../math'

/** A generator paint part — what `StdGeneratorDefinition.paint` accepts. */
export type PatternPaint = (params: GpuFragmentParams) => Expr

/** What a paint slot accepts: a prop binding or another paint part. */
export type PaintSlot = PropRef | PatternPaint

function resolveSlot(slot: PaintSlot, params: GpuFragmentParams): Expr {
    return typeof slot === 'function' ? slot(params) : uniformOf(slot, params)
}

/**
 * Memoise a part per composition so every reader shares ONE lowered Expr object — the
 * slot-graph equivalent of a `const` local in a fused builder.
 */
function shared<P extends object>(build: (params: P) => Expr): (params: P) => Expr {
    const cache = new WeakMap<P, Expr>()
    return (params) => {
        const hit = cache.get(params)
        if (hit) return hit
        const e = build(params)
        cache.set(params, e)
        return e
    }
}

// ═══ Lattice parts ══════════════════════════════════════════════════════════════════════════════

/**
 * A lattice frame's coordinate convention. `'flippedY'` = image-style Y-down, positive rotation
 * counter-clockwise; `'clockwise'` = Y as-is, positive rotation clockwise; `'plain'` = Y as-is,
 * positive rotation counter-clockwise. The convention is part of each pattern's look — the
 * shader file states it alongside the cell count.
 */
export type CellFrameConvention = 'flippedY' | 'clockwise' | 'plain'

const FRAME_BODIES = {
    flippedY: {fn: () => patternPaints.latticeFrameFlipY, hint: 'latticeFrameFlipY'},
    clockwise: {fn: () => patternPaints.latticeFrameCW, hint: 'latticeFrameCW'},
    plain: {fn: () => patternPaints.latticeFrame, hint: 'latticeFrame'},
} as const

/**
 * The lattice frame part: maps the paint UV (uvContext-aware) into rotated, aspect-corrected
 * lattice space, scaled to `cells` lattice units. Everything drawn IN the cells composes on top.
 */
export function cellFrame(slots: {cells: PropRef; rotation: PropRef; convention: CellFrameConvention}): PatternPaint {
    const body = FRAME_BODIES[slots.convention]
    return shared((params) => {
        const {uv, viewport} = paintFrame(params)
        return call(body.fn(), body.hint, [uv, viewport, uniformOf(slots.cells, params), uniformOf(slots.rotation, params)])
    })
}

/** The per-cell brightness-jitter part: scale a fill's RGB by a variation factor (alpha untouched). */
export function vary(color: PaintSlot, factor: PaintSlot): PatternPaint {
    return (params) => {
        const c = resolveSlot(color, params)
        return vec4(c.member('rgb').mul(resolveSlot(factor, params)), c.member('a'))
    }
}

/** The stroke-over-fill part: mix `fill` → `stroke` by the stroke mask in the `space` color space. */
export function strokeOver(layers: {fill: PaintSlot; stroke: PaintSlot; mask: PaintSlot; space?: PropRef}): PatternPaint {
    return shared((params) => call(mixColorsIn(layers.space, params), 'mixColors', [
        resolveSlot(layers.fill, params), resolveSlot(layers.stroke, params), resolveSlot(layers.mask, params),
    ]))
}

/**
 * Sample the given mapped props at the CELL CENTRE rather than the fragment position, so a
 * mapped size/thickness transitions as whole cells instead of being clipped at map-source
 * boundaries. `rotation` (degrees) rotates the lattice about the canvas centre before snapping.
 * Uses the raw canvas UV/viewport (the global, not effectiveViewportSize) — declare it on the
 * definition's `mapSampleUVs` field.
 */
export function sampleMapsAtCellCentres(slots: {
    cells: PropRef
    rotation?: PropRef
    props: string[]
}): GpuMapSampleUVs {
    return ({uniforms, ctx}) => {
        const params = {uniforms}
        const cellCenter = slots.rotation
            ? call(patternPaints.gridCellCenterUV, 'gridCellCenterUV', [
                ctx.uv, ctx.viewportSize, uniformOf(slots.cells, params), uniformOf(slots.rotation, params),
            ])
            : call(cellKit.cellCentreUV, 'cellCentreUV', [ctx.uv, ctx.viewportSize, uniformOf(slots.cells, params)])
        const out: Record<string, Expr> = {}
        for (const prop of slots.props) out[prop] = cellCenter
        return out
    }
}

// ═══ Compose parts ═════════════════════════════════════════════════════════════════════════════

/** Pair a color's RGB with a separately built alpha. */
export function withAlpha(color: PaintSlot, alpha: PaintSlot): PatternPaint {
    return (params) => {
        const c = resolveSlot(color, params)
        return vec4(c.member('rgb'), resolveSlot(alpha, params))
    }
}

/** Read a color's alpha channel. */
export function alphaOf(color: PaintSlot): PatternPaint {
    return (params) => resolveSlot(color, params).member('a')
}

/** Read a color's RGB channels. */
export function rgbOf(color: PaintSlot): PatternPaint {
    return (params) => resolveSlot(color, params).member('rgb')
}

/** Multiply two slot values. */
export function times(a: PaintSlot, b: PaintSlot): PatternPaint {
    return (params) => resolveSlot(a, params).mul(resolveSlot(b, params))
}

/** Linear mix `from` → `to` by `amount` (component-wise, no color-space handling). */
export function mixOf(from: PaintSlot, to: PaintSlot, amount: PaintSlot): PatternPaint {
    return (params) => mixExpr(resolveSlot(from, params), resolveSlot(to, params), resolveSlot(amount, params))
}

/** Finish an RGB value as a fully opaque color. */
export function opaque(rgb: PaintSlot): PatternPaint {
    return (params) => vec4(resolveSlot(rgb, params), 1.0)
}

// ═══ Mask and field parts ══════════════════════════════════════════════════════════════════════

/**
 * Checker cells: the analytically anti-aliased checker blend factor (Quilez 2D filter over the
 * screen-space footprint). `cells` counts cells along the canvas height (square cells).
 */
export function checkerCells(slots: {cells: PropRef; softness: PropRef}): PatternPaint {
    return (params) => {
        const {uv, viewport} = paintFrame(params)
        return call(patternPaints.checkerBlend, 'checkerBlend', [
            uv, viewport, uniformOf(slots.cells, params), uniformOf(slots.softness, params),
        ])
    }
}

/**
 * Stripe bands: the analytically anti-aliased directional stripe mask (Quilez 1D filter,
 * `balance` as the duty threshold), scrolled by the node's accumulated animation time plus
 * `offset`. Requires `animatedTime: {speed}` on the definition.
 */
export function stripeBands(slots: {
    angle: PropRef
    density: PropRef
    balance: PropRef
    softness: PropRef
    offset: PropRef
}): PatternPaint {
    return (params) => {
        const {uv, viewport} = paintFrame(params)
        return call(patternPaints.stripesMask, 'stripesMask', [
            uv,
            viewport,
            uniformOf(slots.angle, params),
            uniformOf(slots.density, params),
            animatedTime(params),
            uniformOf(slots.offset, params),
            uniformOf(slots.balance, params),
            uniformOf(slots.softness, params),
        ])
    }
}

/**
 * Zigzag bands: the fwidth-anti-aliased chevron stripe mask, scrolled by the node's accumulated
 * animation time plus `offset`. Requires `animatedTime: {speed}` on the definition.
 */
export function zigzagBands(slots: {
    count: PropRef
    angle: PropRef
    balance: PropRef
    softness: PropRef
    offset: PropRef
}): PatternPaint {
    return shared((params) => {
        const {uv, viewport} = paintFrame(params)
        return call(patternPaints.chevronMask, 'chevronMask', [
            uv, viewport,
            uniformOf(slots.count, params),
            uniformOf(slots.angle, params),
            uniformOf(slots.balance, params),
            uniformOf(slots.softness, params),
            uniformOf(slots.offset, params),
            animatedTime(params),
        ])
    })
}

/**
 * Ring waves: the mask of concentric rings emanating from `center`, animated by the node's
 * accumulated time plus `phase`. Requires `animatedTime: {speed}` on the definition.
 */
export function ringWaves(slots: {
    center: PropRef
    frequency: PropRef
    thickness: PropRef
    softness: PropRef
    phase: PropRef
}): PatternPaint {
    return (params) => {
        const {uv, viewport} = paintFrame(params)
        return call(patternPaints.ripplesMask, 'ripplesMask', [
            uniformOf(slots.center, params),
            uniformOf(slots.frequency, params),
            uniformOf(slots.softness, params),
            uniformOf(slots.thickness, params),
            uniformOf(slots.phase, params),
            animatedTime(params),
            uv,
            viewport,
        ])
    }
}

/**
 * Falling streaks: directional streaks with per-column random speed/phase and a rounded leading
 * cap, driven by the node's accumulated animation time. `mask` is the streak coverage, `fade`
 * the lead→trail position within a streak. Requires `animatedTime: {speed}` on the definition.
 */
export function fallingStreaks(slots: {
    angle: PropRef
    density: PropRef
    speedVariance: PropRef
    trailLength: PropRef
    strokeWidth: PropRef
    rounding: PropRef
    balance: PropRef
}): {mask: PatternPaint; fade: PatternPaint} {
    const field = shared<GpuFragmentParams>((params) => {
        const {uv, viewport} = paintFrame(params)
        return local(call(patternPaints.fallingLinesField, 'fallingLinesField', [
            uv, viewport,
            uniformOf(slots.angle, params),
            uniformOf(slots.density, params),
            uniformOf(slots.speedVariance, params),
            uniformOf(slots.trailLength, params),
            uniformOf(slots.strokeWidth, params),
            uniformOf(slots.rounding, params),
            uniformOf(slots.balance, params),
            animatedTime(params),
        ]), 'fallingField')
    })
    return {mask: (params) => field(params).member('x'), fade: (params) => field(params).member('y')}
}

/**
 * Dot lattice: the coverage of a square lattice of anti-aliased discs with optional brick-style
 * row stagger, per-row animated drift (the node's accumulated time × per-row random speed), and
 * a per-dot twinkle on the global clock. Requires `animatedTime: {speed}` on the definition.
 */
export function dotLattice(slots: {
    density: PropRef
    dotSize: PropRef
    offset: PropRef
    speedVariance: PropRef
    twinkle: PropRef
}): PatternPaint {
    return (params) => {
        const {uv, viewport} = paintFrame(params)
        return call(patternPaints.dotGridAlpha, 'dotGridAlpha', [
            uv,
            viewport,
            uniformOf(slots.density, params),
            uniformOf(slots.dotSize, params),
            uniformOf(slots.offset, params),
            uniformOf(slots.speedVariance, params),
            uniformOf(slots.twinkle, params),
            animatedTime(params),
            params.ctx.time,
        ])
    }
}

/**
 * Grid lines drawn in a lattice frame: Quilez axis integrals over the dpdx/dpdy footprint.
 * `lines` is the line mask, `shade` the per-cell variation factor for the cell fill.
 */
export function gridLines(slots: {
    frame: PatternPaint
    thickness: PropRef
    softness: PropRef
    variation: PropRef
}): {lines: PatternPaint; shade: PatternPaint} {
    const field = shared<GpuFragmentParams>((params) => local(call(patternPaints.gridField, 'gridField', [
        slots.frame(params), uniformOf(slots.thickness, params), uniformOf(slots.softness, params), uniformOf(slots.variation, params),
    ]), 'gridField'))
    return {lines: (params) => field(params).member('x'), shade: (params) => field(params).member('y')}
}

/**
 * Honeycomb lines drawn in a lattice frame (pointy-top hexagons). `lines` is the line mask,
 * `shade` the per-cell variation factor for the cell fill.
 */
export function hexLines(slots: {
    frame: PatternPaint
    thickness: PropRef
    softness: PropRef
    variation: PropRef
}): {lines: PatternPaint; shade: PatternPaint} {
    const field = shared<GpuFragmentParams>((params) => local(call(patternPaints.hexGridField, 'hexGridField', [
        slots.frame(params), uniformOf(slots.thickness, params), uniformOf(slots.softness, params), uniformOf(slots.variation, params),
    ]), 'hexField'))
    return {lines: (params) => field(params).member('x'), shade: (params) => field(params).member('y')}
}

/**
 * Skewed equilateral-triangle lines drawn in a lattice frame, with per-row animated drift.
 * `lines` is the line mask, `shade` the per-cell variation factor. Requires
 * `animatedTime: {speed}` on the definition.
 */
export function triangleLines(slots: {
    frame: PatternPaint
    thickness: PropRef
    softness: PropRef
    variation: PropRef
    speedVariance: PropRef
}): {lines: PatternPaint; shade: PatternPaint} {
    const field = shared<GpuFragmentParams>((params) => local(call(patternPaints.triangularGridField, 'triangularGridField', [
        slots.frame(params),
        uniformOf(slots.thickness, params),
        uniformOf(slots.softness, params),
        uniformOf(slots.variation, params),
        uniformOf(slots.speedVariance, params),
        animatedTime(params),
    ]), 'triField'))
    return {lines: (params) => field(params).member('x'), shade: (params) => field(params).member('y')}
}

/**
 * Brick courses: staggered rows with mortar gaps, optional rotation, static offset, and per-row
 * animated drift (the node's accumulated time). `bricks` is the brick coverage mask, `shade`
 * the per-brick variation factor. Brick keeps its OWN frame part (`patternPaints.brickFrame`)
 * rather than the shared {@link cellFrame} slot — its raw-UV, non-uniform-cell framing is not
 * byte-equivalent to any lattice frame convention (Gate-C candidate; see the note on
 * `brickFrame`). Requires `animatedTime: {speed}` on the definition.
 */
export function brickCourses(slots: {
    cellsX: PropRef
    cellsY: PropRef
    mortar: PropRef
    softness: PropRef
    variation: PropRef
    rotation: PropRef
    offset: PropRef
    speedVariance: PropRef
    seed: PropRef
}): {bricks: PatternPaint; shade: PatternPaint} {
    const field = shared<GpuFragmentParams>((params) => {
        const {uv, viewport} = paintFrame(params)
        return local(call(patternPaints.brickField, 'brickField', [
            uv, viewport,
            uniformOf(slots.cellsX, params),
            uniformOf(slots.cellsY, params),
            uniformOf(slots.mortar, params),
            uniformOf(slots.softness, params),
            uniformOf(slots.variation, params),
            uniformOf(slots.rotation, params),
            uniformOf(slots.offset, params),
            uniformOf(slots.speedVariance, params),
            uniformOf(slots.seed, params),
            animatedTime(params),
        ]), 'brickField')
    })
    return {bricks: (params) => field(params).member('x'), shade: (params) => field(params).member('y')}
}

/**
 * Truchet arcs drawn in a lattice frame: quarter-circle arc tiles, orientation hashed per tile
 * (`seed` reshuffles the maze). Returns the arc line mask.
 */
export function truchetArcs(slots: {
    frame: PatternPaint
    thickness: PropRef
    softness: PropRef
    seed: PropRef
}): PatternPaint {
    return (params) => call(patternPaints.truchetField, 'truchetField', [
        slots.frame(params), uniformOf(slots.thickness, params), uniformOf(slots.softness, params), uniformOf(slots.seed, params),
    ])
}

/**
 * Weave threads drawn in a lattice frame: interlaced horizontal/vertical thread bands with a
 * checkerboard over-under rule, compositing `colors[0]` (horizontal) and `colors[1]` (vertical)
 * by per-thread alpha weight — not a color-space mix, so there is no `space` slot.
 */
export function weaveThreads(slots: {
    frame: PatternPaint
    gap: PropRef
    /** `[horizontal, vertical]` thread colors. */
    colors: [PropRef, PropRef]
}): PatternPaint {
    return (params) => call(patternPaints.weaveColor, 'weaveColor', [
        slots.frame(params),
        uniformOf(slots.gap, params),
        uniformOf(slots.colors[0], params),
        uniformOf(slots.colors[1], params),
    ])
}

/**
 * Rhombille (tumbling-blocks) faces drawn in a lattice frame — three shaded rhombus faces per
 * hexagon read as a 3D cube. `random` is the per-cube color hash, `tone` the face shading
 * factor, `wire` the edge-line mask.
 */
export function isoCubeFaces(slots: {
    frame: PatternPaint
    thickness: PropRef
    softness: PropRef
}): {random: PatternPaint; tone: PatternPaint; wire: PatternPaint} {
    const field = shared<GpuFragmentParams>((params) => local(call(patternPaints.isoCubeField, 'isoCubeField', [
        slots.frame(params), uniformOf(slots.thickness, params), uniformOf(slots.softness, params),
    ]), 'isoCubeField'))
    return {
        random: (params) => field(params).member('x'),
        tone: (params) => field(params).member('y'),
        wire: (params) => field(params).member('z'),
    }
}

// ═══ Print-screen parts (halftone) ═════════════════════════════════════════════════════════════

/**
 * Halftone plate part: the rotated dot-screen coverage for one ink plate — dot size follows
 * `intensity` (the plate's ink value at this pixel), spaced by `frequency` at the plate's screen
 * `angle`.
 */
export function dotScreenMask(uv: Expr, aspect: Expr, angle: Expr, intensity: Expr, frequency: Expr): Expr {
    return call(patternPaints.halftonePlateGrid, 'halftonePlateGrid', [uv, aspect, angle, intensity, frequency])
}

/**
 * Halftone plate part: subtractive ink lay-down — white where the plate leaves paper bare,
 * fading toward the ink color (× ink alpha) where the dot covers. Plates multiply together.
 */
export function inkTransmission(inkColor: Expr, inkMask: Expr): Expr {
    return call(patternPaints.halftoneTransmission, 'halftoneTransmission', [inkColor, inkMask])
}

/**
 * Classic halftone gather recipe: the child sampled once (straight alpha), its brightness
 * modulating a single rotated dot plate.
 */
export function dotScreen(slots: {angle: PropRef; frequency: PropRef}): GatherEffect {
    return gather({
        resultAlpha: 'straight',
        build: ({sampleStraight, ctx, uniforms}): Expr => {
            const childColor = sampleStraight(ctx.uv)
            return call(patternPaints.halftoneClassic, 'halftoneClassic', [
                childColor, ctx.uv, ctx.aspect, uniformOf(slots.angle, {uniforms}), uniformOf(slots.frequency, {uniforms}),
            ])
        },
    })
}

/** One ink plate of a {@link cmykPress}: which CMYK channel it prints, at which screen angle, in which ink. */
export interface InkPlateSpec {
    readonly channel: 'cyan' | 'magenta' | 'yellow' | 'black'
    readonly screenAngle: PropRef
    readonly ink: PropRef
}

export function inkPlate(spec: InkPlateSpec): InkPlateSpec {
    return spec
}

const PLATE_CHANNELS = {
    cyan: {fn: () => patternPaints.halftoneChannelC, hint: 'halftoneChannelC'},
    magenta: {fn: () => patternPaints.halftoneChannelM, hint: 'halftoneChannelM'},
    yellow: {fn: () => patternPaints.halftoneChannelY, hint: 'halftoneChannelY'},
    black: {fn: () => patternPaints.halftoneChannelK, hint: 'halftoneChannelK'},
} as const

/**
 * CMYK press gather recipe: one subtractive plate per ink, laid down in order. Each plate samples
 * the child at its own registration offset around `misprintAngle` (quarter turns per plate index;
 * mis-registration shows as color fringing), reads its CMYK channel, screens it through
 * `dotScreenMask`, and multiplies its `inkTransmission` into the paper. Alpha follows the
 * un-offset centre sample; every tap is straight-alpha.
 */
export function cmykPress(slots: {
    paper: PropRef
    frequency: PropRef
    misprint: PropRef
    misprintAngle: PropRef
    plates: InkPlateSpec[]
}): GatherEffect {
    return gather({
        resultAlpha: 'straight',
        build: ({sampleStraight, ctx, uniforms}): Expr => {
            const u = (ref: PropRef) => uniformOf(ref, {uniforms})
            let printed = u(slots.paper).member('rgb')
            slots.plates.forEach((plate, index) => {
                const sample = sampleStraight(call(patternPaints.halftonePlateUV, 'halftonePlateUV', [
                    ctx.uv, u(slots.misprintAngle), floatE(index * 90), u(slots.misprint), ctx.aspect,
                ]))
                const channel = PLATE_CHANNELS[plate.channel]
                const intensity = call(channel.fn(), channel.hint, [sample])
                printed = printed.mul(inkTransmission(u(plate.ink), dotScreenMask(ctx.uv, ctx.aspect, u(plate.screenAngle), intensity, u(slots.frequency))))
            })
            return vec4(printed, sampleStraight(ctx.uv).member('a'))
        },
    })
}

/**
 * Compile-time recipe switch: pick a whole gather recipe from a structural enum prop. `read`
 * maps the raw compile-time value (robust to a pre-transform string) to a case key. All cases
 * must share their alpha discipline and carry no setup hooks.
 */
export function chosenBy<K extends string>(prop: PropRef, read: (raw: unknown) => K, recipes: Record<K, GatherEffect>): GatherEffect {
    const cases = Object.values(recipes) as GatherEffect[]
    const resultAlpha = cases[0]?.resultAlpha
    for (const c of cases) {
        if (c.resultAlpha !== resultAlpha || c.setup) throw new Error('std: chosenBy recipes must share resultAlpha and carry no setup hooks')
    }
    return gather({
        resultAlpha,
        build: (params): Expr => recipes[read(params.propValues[prop.name])].build(params),
    })
}

// ═══ Print-screen parts (dither) ═══════════════════════════════════════════════════════════════

// Compile-time enum readers — robust to a raw string (a preset loaded before the prop transform
// ran) as well as the bridge-mapped number.
const ditherPatternOf = (raw: unknown): number => {
    if (typeof raw === 'number') return raw
    const patterns: Record<string, number> = {bayer2: 0, bayer4: 1, bayer8: 2, clusteredDot: 3, blueNoise: 4, whiteNoise: 5, floydSteinberg: 6}
    return patterns[raw as string] ?? 1
}
const ditherColorModeOf = (raw: unknown): number => (typeof raw === 'number' ? raw : raw === 'source' ? 1 : 0)

/** One post-sample gather stage: a value derived from the child RTT. */
export type GatherStage = (params: RttFilterParams) => Expr

/**
 * The dither grid frame: the shared cell geometry of a pixel-grid screen. `coord` is the dither
 * cell coordinate, `source` the child sampled once per cell (the pixellated source color). The
 * grid is sized against the LOGICAL (authored-frame) resolution so the dot count stays constant
 * under infinite-canvas resolution scaling.
 */
export interface PixelGrid {
    readonly size: PropRef
    readonly coord: GatherStage
    readonly source: GatherStage
}

export function pixelGrid(slots: {size: PropRef}): PixelGrid {
    const coord = shared<RttFilterParams>((params) =>
        local(call(patternPaints.ditherCoord, 'ditherCoord', [params.ctx.uv, params.ctx.logicalViewportSize, uniformOf(slots.size, params)]), 'ditherCoord'))
    const source = shared<RttFilterParams>((params) =>
        local(params.sampleStraight(call(patternPaints.ditherPixUV, 'ditherPixUV', [coord(params), uniformOf(slots.size, params), params.ctx.logicalViewportSize])), 'ditherSource'))
    return {size: slots.size, coord, source}
}

/**
 * Ordered-dither part: the threshold FIELD for a compile-time pattern mode — the closed-form
 * Bayer 2/4/8 or clustered-dot lattice values, or the blue/white-noise hashes, at the dither
 * cell coordinate. Feed it to `ditherOrderedResult` to quantise a luminance against it.
 * (Floyd–Steinberg is not a threshold field — its quantisation diffuses error serially.)
 */
export function orderedThreshold(pattern: number, coord: Expr): Expr {
    if (pattern === 4) return call(patternPaints.ditherBlueNoise, 'ditherBlueNoise', [coord])
    if (pattern === 5 || pattern > 6 || pattern < 0) return call(patternPaints.ditherWhiteNoise, 'ditherWhiteNoise', [coord])
    const periodic = call(patternPaints.ditherPeriodic, 'ditherPeriodic', [coord])
    switch (pattern) {
        case 0: return periodic.member('x') // bayer2
        case 1: return periodic.member('y') // bayer4
        case 2: return periodic.member('z') // bayer8
        default: return periodic.member('w') // clusteredDot
    }
}

/**
 * The quantise stage: the 0..1 dither result for the grid cell. The ordered modes compose an
 * `orderedThreshold` field with the cell luminance; the compile-time Floyd–Steinberg mode has no
 * threshold field — it samples the 64 block-cell luminances and runs its tile-confined serpentine
 * error diffusion, reading off this fragment's cell.
 */
export function quantise(slots: {grid: PixelGrid; pattern: PropRef; threshold: PropRef; spread: PropRef}): GatherStage {
    return (params) => {
        const u = (ref: PropRef) => uniformOf(ref, params)
        const threshold = u(slots.threshold)
        const spread = u(slots.spread)
        const coord = slots.grid.coord(params)
        const pattern = ditherPatternOf(params.propValues[slots.pattern.name])
        if (pattern === 6) {
            const gridRes = params.ctx.logicalViewportSize
            const pixelSize = u(slots.grid.size)
            const blockOrigin = call(patternPaints.ditherBlockOrigin, 'ditherBlockOrigin', [coord])
            const lic = call(patternPaints.ditherLocalCellIndex, 'ditherLocalCellIndex', [coord, blockOrigin])
            const lums: Expr[] = []
            for (let ly = 0; ly < 8; ly++) {
                for (let lx = 0; lx < 8; lx++) {
                    const cellUV = call(patternPaints.ditherCellUV, 'ditherCellUV', [blockOrigin, floatE(lx), floatE(ly), pixelSize, gridRes])
                    lums.push(call(patternPaints.ditherLuma, 'ditherLuma', [params.sampleStraight(cellUV)]))
                }
            }
            return call(patternPaints.ditherFloydSteinberg, 'ditherFloydSteinberg', [arrayExpr('f32', lums), lic, threshold, spread])
        }
        const luminance = call(patternPaints.ditherLuma, 'ditherLuma', [slots.grid.source(params)])
        return call(patternPaints.ditherOrderedResult, 'ditherOrderedResult', [orderedThreshold(pattern, coord), luminance, threshold, spread])
    }
}

/**
 * The dither compose gather recipe: color the quantised `levels`. The compile-time `mode`
 * branches — custom mixes `colors[0]`→`colors[1]`, source darkens/brightens the grid's own
 * pixellated child color. Every tap is straight-alpha.
 */
export function ditherInks(slots: {
    grid: PixelGrid
    levels: GatherStage
    mode: PropRef
    colors: [PropRef, PropRef]
}): GatherEffect {
    return gather({
        resultAlpha: 'straight',
        build: (params): Expr => {
            const u = (ref: PropRef) => uniformOf(ref, params)
            const levels = slots.levels(params)
            if (ditherColorModeOf(params.propValues[slots.mode.name]) === 0) {
                return call(patternPaints.ditherComposeCustom, 'ditherComposeCustom', [u(slots.colors[0]), u(slots.colors[1]), levels])
            }
            return call(patternPaints.ditherComposeSource, 'ditherComposeSource', [slots.grid.source(params), levels])
        },
    })
}

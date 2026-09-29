/**
 * std/paint — the noise-texture vocabulary.
 *
 * Every noise generator is a visible composition over the field pipeline (`./fields`):
 * a basis ({@link noiseField} from the shared registry, or one of the parameter-taking
 * producers below) sampled through a domain part ({@link seededPlane}, {@link pixelGrid},
 * {@link evolving}), shaped by a tone part ({@link noiseTone}, {@link signedTone},
 * {@link gainTone}), and ramped by a palette (`stops`, `pair`, {@link linearPair}). The
 * fractal and worley recipes live in the nouns at the bottom; the voronoi cell parts
 * compose in their shader definitions (Marble's veining is pure algebra in its file).
 *
 * Animated parts read the definition's per-node accumulated time themselves; the
 * `animatedTime:` declaration stays on the definition.
 */
import type {Expr, GpuFragmentParams} from '../../gpu/contract'
import {call, floatE, vec4, asLocal} from '../../gpu/composer'
import {mixExpr, animatedTime} from '../../gpu/porters'
import {colorMixing, colorStops as colorStopsKit, fields as fieldsKit, noise as noiseKit, noiseColor, noisePaints, noiseStylize, tone as toneKit} from '../../gpu/kit/index'
import {colorSpaceModeOf, rampOver, stops, type Field, type Paint, type Palette} from './fields'
import type {PropRef} from '../values'
import {uniformOf, paintFrame} from '../invoke'
import {add, mul, sin} from '../math'

// ── Waves — the organic sine-interference engine ─────────────────────────────────────────

/**
 * One wave: coefficients on the named axes (frequency along x/y, drift speed along t) plus
 * an optional phase. Every field is serializable data — a wave is `{x: 3.2, t: 0.8}`:
 * "a wave across x at frequency 3.2, drifting at speed 0.8". Negative coefficients reverse
 * direction. Coefficient `1` emits the bare axis (no `× 1`).
 */
export interface Wave {
    x?: number
    y?: number
    t?: number
    /** Radians, or a `hashPhase(...)` Expr for seeded decorrelation. */
    phase?: Expr | number
}

/**
 * One term of a schedule: a single wave, or a product of waves (the "organic" move — an
 * x-wave × a y-wave interferes instead of striping). `weight` scales the term (default 1,
 * emitted only when ≠ 1).
 */
export interface WaveTerm {
    waves: Wave[]
    weight?: number
}

/**
 * `waves` — overlapping traveling sine waves, schedule as data. The engine under every
 * "organic wobble": evaluates `Σ weightᵢ · Π sin(x·kx + y·ky + t·kt + phase)` over the
 * named axes, SIGNED output (tone words shape it; `warped` coords warp it — both stay
 * outside, deliberately). Emission is deterministic — terms sum in schedule order, waves
 * multiply in wave order, axes emit x → y → t → phase — so a ported schedule reproduces
 * its original arithmetic. Schedules are the caller's look; `organicWaves` generates one
 * from intent when nobody has hand-tuned constants to preserve.
 */
export function waves(axes: {x: Expr | number; y?: Expr | number; t?: Expr | number}, schedule: WaveTerm[]): Expr {
    const axisTerm = (axis: Expr | number | undefined, k: number | undefined, name: string): Expr | number | undefined => {
        if (k === undefined) return undefined
        if (axis === undefined) throw new Error(`std: waves() schedule uses axis '${name}' but the axes object doesn't provide it`)
        return k === 1 ? axis : mul(axis, k)
    }
    const waveExpr = (w: Wave): Expr => {
        const parts = [
            axisTerm(axes.x, w.x, 'x'),
            axisTerm(axes.y, w.y, 'y'),
            axisTerm(axes.t, w.t, 't'),
            w.phase,
        ].filter((part): part is Expr | number => part !== undefined)
        if (parts.length === 0) throw new Error('std: a wave needs at least one axis coefficient or phase')
        return sin(parts.reduce((a, b) => add(a, b)))
    }
    const terms = schedule.map((term) => {
        if (term.waves.length === 0) throw new Error('std: a waves() term needs at least one wave')
        const product = term.waves.map(waveExpr).reduce((a, b) => mul(a, b))
        return term.weight === undefined || term.weight === 1 ? product : mul(product, term.weight)
    })
    if (terms.length === 0) throw new Error('std: waves() needs at least one term')
    return terms.reduce((a, b) => add(a, b))
}

/**
 * `wavyLine` — an animated organic curve: the offset of a line at coordinate `along`, as a
 * weighted stack of traveling sines. Sugar over {@link waves} on one axis; give each wave a
 * `hashPhase` to decorrelate a family of lines. Curtain paths, water horizons, wavy
 * baselines, cloth edges.
 */
export function wavyLine(opts: {
    along: Expr | number
    time: Expr | number
    waves: {freq: number; speed: number; amount: number; phase?: Expr | number}[]
}): Expr {
    return waves({x: opts.along, t: opts.time}, opts.waves.map((w) => ({
        waves: [{x: w.freq, t: w.speed, phase: w.phase}],
        weight: w.amount,
    })))
}

/**
 * `organicWaves` — generate a {@link waves} schedule from intent, for authors (human or AI)
 * who don't have hand-tuned constants to preserve. Three knobs + seed, deliberately
 * (knobs grow under pressure from real use, never speculatively): `frequency` sets the base
 * scale, `detail` the number of product-pair layers (each ~1.8× finer, half the weight),
 * `drift` the animation speed. Deterministic per seed. TUNING CANDIDATE — constants pending
 * an eye pass; treat the output as a starting point, not a spec.
 */
export function organicWaves(opts: {frequency: number; detail?: number; drift?: number; seed?: number}): WaveTerm[] {
    const detail = Math.max(1, Math.round(opts.detail ?? 3))
    const drift = opts.drift ?? 0.5
    let state = ((opts.seed ?? 1) >>> 0) || 1
    const rand = () => {
        state = (state * 1664525 + 1013904223) >>> 0
        return state / 2 ** 32
    }
    const jitter = () => 0.8 + rand() * 0.4
    const terms: WaveTerm[] = []
    let weight = 1
    let totalWeight = 0
    for (let i = 0; i < detail; i++) {
        const f = opts.frequency * 1.8 ** i
        terms.push({
            waves: [
                {x: f * jitter(), t: drift * jitter() * (rand() < 0.5 ? -1 : 1)},
                {y: f * jitter(), t: drift * jitter()},
            ],
            weight,
        })
        // A diagonal single every other layer breaks up the grid feel of pure x×y products.
        if (i % 2 === 0) {
            terms.push({
                waves: [{x: f * 1.5 * jitter(), y: -f * 1.1 * jitter(), t: -drift * jitter()}],
                weight: weight * 0.8,
            })
            totalWeight += weight * 0.8
        }
        totalWeight += weight
        weight *= 0.5
    }
    // Normalize so the summed weight is 1 — amplitude is the caller's knob.
    return terms.map((t) => ({...t, weight: (t.weight ?? 1) / totalWeight}))
}

// ── Domain parts ────────────────────────────────────────────────────────────────────────

/**
 * The exponential-scale noise framing: aspect-correct the UV, scale by `exp(scale)` (each +1
 * on the slider doubles-ish the frequency), offset by `seed`. Hands the inner field a vec2
 * pattern position.
 */
export function seededPlane(field: Field, slots: {scale: PropRef; seed: PropRef}): Field {
    return (params, uv) => {
        const {viewport} = paintFrame(params)
        return field(params, call(fieldsKit.aspectScaledDomain, 'aspectScaledDomain', [
            uv, viewport, uniformOf(slots.scale, params), uniformOf(slots.seed, params),
        ]))
    }
}

/**
 * The RAW-divide exponential-scale framing (Scratches' historical domain — the unguarded
 * aspect divide is part of its exact arithmetic).
 */
export function rawPlane(field: Field, slots: {scale: PropRef; seed: PropRef}): Field {
    return (params, uv) => {
        const {viewport} = paintFrame(params)
        return field(params, call(noisePaints.rawAspectScaledDomain, 'rawAspectScaledDomain', [
            uv, viewport, uniformOf(slots.scale, params), uniformOf(slots.seed, params),
        ]))
    }
}

/**
 * The per-pixel framing: floor the device-pixel position into `grain`-sized cells, offset by
 * `seed` — for patterns defined on the pixel grid (blue noise) rather than in UV space.
 */
export function pixelGrid(field: Field, slots: {grain: PropRef; seed: PropRef}): Field {
    return (params, uv) => {
        const {viewport} = paintFrame(params)
        return field(params, call(fieldsKit.pixelGridDomain, 'pixelGridDomain', [
            uv, viewport, uniformOf(slots.grain, params), uniformOf(slots.seed, params),
        ]))
    }
}

/**
 * The linear-scale noise framing: aspect-correct the UV and multiply by `scale` directly (no
 * exponential remap), bound as a local — for recipes whose parts read the same plane position
 * several times.
 */
export function scaledPlane(field: Field, slots: {scale: PropRef}): Field {
    return (params, uv) => {
        const {viewport} = paintFrame(params)
        const pos = asLocal(
            call(noisePaints.aspectPlane, 'aspectPlane', [uv, viewport]).mul(uniformOf(slots.scale, params)),
            'plane',
        )
        return field(params, pos)
    }
}

/**
 * Lift a vec2 pattern position into 3D by walking the third axis with time — the difference
 * between noise that MORPHS in place (this) and noise that slides. `rate` scales the
 * evolution independently of the node's `speed` prop.
 */
export function evolving(field: Field, opts: {rate: number}): Field {
    return (params, coord) => field(params, call(fieldsKit.timeAxisDomain, 'timeAxisDomain', [
        coord, animatedTime(params), floatE(opts.rate),
    ]))
}

// ── Basis producers the NOISE_BASES registry shape doesn't fit ──────────────────────────
// (Registry bases are `(vec3) → f32`; these take a phase, an extra shape parameter, or a
// vec2 pixel-grid coordinate. The basis math itself stays atomic in kit/noise.ts.)

/** Oriented sine grains at `frequency` waves per cell, phase-driven by the node's clock. Signed. */
export function gaborGrains(slots: {frequency: PropRef}): Field {
    return (params, coord) => call(noiseKit.gabor12, 'gabor12', [
        coord, uniformOf(slots.frequency, params), animatedTime(params),
    ])
}

/** Rotating banded wavelets; `detail` is the per-octave frequency ratio. Phase-animated, signed. */
export function waveletBands(slots: {detail: PropRef}): Field {
    return (params, coord) => call(noiseKit.wavelet12, 'wavelet12', [
        coord, animatedTime(params), uniformOf(slots.detail, params),
    ])
}

/** Spatial-high-pass blue-noise speckle over a pixel-grid coordinate. Static, unit range. */
export function blueSpeckle(): Field {
    return (_params, coord) => call(noiseKit.blue12, 'blue12', [coord])
}

/** Branching hydraulic-erosion ridge height (the `.x` of the erosion field). Static, signed. */
export function erosionRidges(): Field {
    return (_params, coord) => call(noiseKit.erosion12, 'erosion12', [coord]).member('x')
}

/** Swirling curl-flow speed (√2-normalized magnitude), morphing in place at `rate` × the clock. */
export function curlSpeed(opts: {rate: number}): Field {
    return (params, coord) => call(noisePaints.curlMagnitude, 'curlMagnitude', [
        coord, animatedTime(params).mul(opts.rate),
    ])
}

/** Fine hairline streaks at `thickness`, flickering on the node's clock. Fragment-only (fwidth). */
export function scratchStreaks(slots: {thickness: PropRef}): Field {
    return (params, coord) => call(noiseKit.scratches12, 'scratches12', [
        coord, animatedTime(params), uniformOf(slots.thickness, params),
    ])
}

// ── Tone parts ──────────────────────────────────────────────────────────────────────────

/** Remap a signed [-1,1] field to the unit range (`v · 0.5 + 0.5`). */
export function unitized(field: Field): Field {
    return (params, coord) => call(toneKit.signedToUnit, 'signedToUnit', [field(params, coord)])
}

/**
 * The shared noise tone tail: additive contrast about mid-grey + balance shift, inverted so a
 * ramp's colorA reads as the LOW end. `contrast`/`balance` omitted = 0 (identity) for
 * textures without tone controls.
 */
export function noiseTone(field: Field, slots?: {contrast?: PropRef; balance?: PropRef}): Field {
    return (params, coord) => call(noiseColor.noiseToneKColor, 'noiseToneKColor', [
        field(params, coord),
        slots?.contrast ? uniformOf(slots.contrast, params) : floatE(0),
        slots?.balance ? uniformOf(slots.balance, params) : floatE(0),
    ])
}

/**
 * The signed tone tail: contrast/balance applied to the RAW [-1,1] field BEFORE the squash to
 * [0,1] (then inverted), so high contrast rides further before clipping.
 */
export function signedTone(field: Field, slots: {contrast: PropRef; balance: PropRef}): Field {
    return (params, coord) => call(toneKit.toneSignedInverted, 'noiseToneSigned', [
        field(params, coord), uniformOf(slots.contrast, params), uniformOf(slots.balance, params),
    ])
}

/** The multiplicative tone tail (contrast is a gain, 1 = identity), non-inverted — Worley's shape. */
export function gainTone(field: Field, slots: {contrast: PropRef; balance: PropRef}): Field {
    return (params, coord) => call(noisePaints.worleyTone, 'worleyTone', [
        field(params, coord), uniformOf(slots.contrast, params), uniformOf(slots.balance, params),
    ])
}

// ── Palettes ────────────────────────────────────────────────────────────────────────────

/** Two endpoint color props mixed in LINEAR space — for definitions without a color-space prop. */
export function linearPair(a: PropRef, b: PropRef): Palette {
    return (t, params) => call(colorMixing.mixColorsLinear, 'mixColors', [
        uniformOf(a, params), uniformOf(b, params), t,
    ])
}

// ── Structured-texture recipes ──────────────────────────────────────────────────────────

/**
 * Multi-octave fractal Brownian motion: rotated centred plane → gated 8-octave golden-angle
 * fBm over MaterialX simplex → weight-normalized unit field → stop ramp. `octaves` gates the
 * fixed loop at runtime, so the slider does not recompile.
 */
export function fractalNoise(slots: {
    angle: PropRef
    detail: PropRef
    contrast: PropRef
    octaves: PropRef
    seed: PropRef
    space: PropRef
}): Paint {
    const ridges: Field = (params, uv) => {
        const {viewport} = paintFrame(params)
        const pos = call(noisePaints.rotatedCenteredDomain, 'rotatedCenteredDomain', [
            uv, viewport, uniformOf(slots.angle, params),
        ])
        const sum = call(noisePaints.fractalSum, 'fractalSum', [
            pos, uniformOf(slots.detail, params), uniformOf(slots.contrast, params),
            animatedTime(params), uniformOf(slots.seed, params), uniformOf(slots.octaves, params),
        ])
        return call(noisePaints.fbmNormalize, 'fbmNormalize', [sum])
    }
    return rampOver(ridges, stops(slots.space))
}

/**
 * Cellular (Worley) noise: aspect plane → gated 4-octave cellular sum (nearest-2 distances
 * under a selectable metric, reduced per `mode`) → per-mode normalization → gain tone → ramp.
 * `mode`/`distance`/`octaves` are compile-time props read as CPU values and folded to literals.
 */
export function worleyNoise(slots: {
    scale: PropRef
    jitter: PropRef
    lacunarity: PropRef
    persistence: PropRef
    contrast: PropRef
    balance: PropRef
    seed: PropRef
    /** Compile-time: the field reduction (f1/f2/…), read from prop values. */
    mode: PropRef
    /** Compile-time: the distance metric, read from prop values. */
    distance: PropRef
    /** Compile-time: the octave gate (1–4), read from prop values. */
    octaves: PropRef
    space: PropRef
}): Paint {
    const cells: Field = (params, uv) => {
        const {viewport} = paintFrame(params)
        const modeNum = noisePaints.worleyModeNum(params.propValues[slots.mode.name])
        const distNum = noisePaints.worleyDistNum(params.propValues[slots.distance.name])
        const octavesNum = Math.max(1, Math.min(4, Math.round((params.propValues[slots.octaves.name] as number) ?? 1)))
        const pos = call(noisePaints.aspectPlane, 'aspectPlane', [uv, viewport])
        const sum = call(noisePaints.worleyCells, 'worleyCells', [
            pos, uniformOf(slots.scale, params), uniformOf(slots.seed, params),
            uniformOf(slots.jitter, params), uniformOf(slots.lacunarity, params),
            uniformOf(slots.persistence, params), animatedTime(params),
            floatE(octavesNum), floatE(distNum), floatE(modeNum),
        ])
        return call(noisePaints.worleyNormalize, 'worleyNormalize', [
            sum, floatE(noisePaints.worleyModeScale(modeNum, distNum)),
        ])
    }
    // No `stops` prop on the definition → the ramp always takes its two-color branch; the
    // palette is here for the compile-time colorSpace dispatch.
    return rampOver(gainTone(cells, {contrast: slots.contrast, balance: slots.balance}), stops(slots.space))
}

// ── Voronoi cell parts ──────────────────────────────────────────────────────────────────

/**
 * Nearest-two cellular distances `vec2(d1, d2)` over the aspect plane at `scale`, the cell
 * points drifting on the node's clock. Wrap in `share` when a fill and a border mask read
 * the same evaluation.
 */
export function cellDistances(slots: {scale: PropRef; seed: PropRef}): Field {
    return (params, uv) => {
        const {viewport} = paintFrame(params)
        const pos = call(noisePaints.aspectPlane, 'aspectPlane', [uv, viewport]).mul(uniformOf(slots.scale, params))
        return call(noisePaints.voronoiCells, 'voronoiNearest2', [pos, animatedTime(params), uniformOf(slots.seed, params)])
    }
}

/** The F1/F2 fill gradient: `edgeIntensity` sets how far the edge color reaches into each cell. */
export function cellFill(cells: Field, slots: {edgeIntensity: PropRef}): Field {
    return (params, coord) => call(noisePaints.cellFill, 'cellFill', [
        cells(params, coord), uniformOf(slots.edgeIntensity, params),
    ])
}

/**
 * The cell boundary-line mask (0 on a boundary, 1 inside a cell); `scale` compensates the
 * line width for the cell count.
 */
export function cellBorders(cells: Field, slots: {softness: PropRef; scale: PropRef}): Field {
    return (params, coord) => call(noisePaints.cellBorders, 'cellBorders', [
        cells(params, coord), uniformOf(slots.softness, params), uniformOf(slots.scale, params),
    ])
}

/**
 * Overlay `color` on a paint's rgb where `mask` falls to 0; the base paint keeps its alpha.
 * The base binds as a local: both the overlay mix and the alpha read it.
 */
export function borderOverlay(paint: Paint, mask: Field, slots: {color: PropRef}): Paint {
    return (params) => {
        const {uv} = paintFrame(params)
        const base = asLocal(paint(params), 'base')
        const rgb = mixExpr(uniformOf(slots.color, params).member('rgb'), base.member('rgb'), mask(params, uv))
        return vec4(rgb, base.member('a'))
    }
}

// ── Strands parts ───────────────────────────────────────────────────────────────────────

/** An rgb-only stage for {@link overRgb} — maps a paint's rgb; alpha passes through. */
export type RgbStage = (rgb: Expr, params: GpuFragmentParams) => Expr

/** Apply rgb-only stages over a paint, threading the base alpha through unchanged. */
export function overRgb(paint: Paint, ...stages: RgbStage[]): Paint {
    return (params) => {
        const base = paint(params)
        let rgb = base.member('rgb')
        for (const stage of stages) rgb = stage(rgb, params)
        return vec4(rgb, base.member('a'))
    }
}

/** The closing working-space → P3-linear back-conversion at the compile-time `space` mode. */
export function backToP3(slots: {space?: PropRef}): RgbStage {
    return (rgb, params) => {
        const mode = slots.space ? colorSpaceModeOf(params.propValues[slots.space.name]) : 0
        return colorStopsKit.backConvertToP3(rgb, mode)
    }
}

/** The `pow(rgb, 0.85)` tone lift. */
export function tonePow(): RgbStage {
    return (rgb) => call(noisePaints.toneLift, 'toneLift', [rgb])
}

/**
 * `ribbons` — flowing, gradient-colored ribbons between two anchors (the ribbons genre
 * primitive): an ATOMIC runtime-count reduce — each ribbon's working-space gradient lookup
 * lives inside the loop — returning working-space rgba. Close it with {@link backToP3} (and
 * {@link tonePow}) via {@link overRgb}. Reads BOTH of the definition's clocks: the main one
 * drives the wave motion, the `color` extra clock
 * (`extraAnimatedTimes: {color: ...}` stays declared on the definition) scrolls the colors.
 */
export function ribbons(slots: {
    from: PropRef
    to: PropRef
    count: PropRef
    width: PropRef
    amplitude: PropRef
    frequency: PropRef
    softness: PropRef
    spread: PropRef
    pinEdges: PropRef
    colorScale: PropRef
    colorVariance: PropRef
}): Paint {
    return (params) => {
        const {uniforms} = params
        const {uv, viewport} = paintFrame(params)
        const waveT = animatedTime(params)
        const colorT = animatedTime(params, undefined, '_animTime_color')

        // Pack scalars into vec4 params (ribbonsField is over the 15-arg tgpu.fn cap otherwise).
        const packed1 = vec4(
            uniformOf(slots.amplitude, params), uniformOf(slots.frequency, params),
            uniformOf(slots.count, params), uniformOf(slots.width, params),
        )
        const packed2 = vec4(
            uniformOf(slots.softness, params), uniformOf(slots.spread, params),
            uniformOf(slots.colorScale, params), uniformOf(slots.colorVariance, params),
        )
        const packed3 = vec4(uniformOf(slots.pinEdges, params), waveT, colorT, uniforms.stopCount)

        return call(noisePaints.ribbonsField, 'ribbonsField', [
            uv, viewport, uniformOf(slots.from, params), uniformOf(slots.to, params),
            packed1, packed2, packed3,
            uniforms.colorsArray, uniforms.positionsArray, uniforms.convertedColorsArray,
        ])
    }
}

// ── Relief + paper filters ──────────────────────────────────────────────────────────────

/** One entry of {@link reliefBases}: a kit height-field basis for {@link noiseRelief}. */
export interface ReliefBasis {
    /** The `(vec2f) → f32` height field sampled by the relief filter. */
    readonly fn: unknown
    /** The emitted WGSL height-fn name. */
    readonly hint: string
}

/** The height fields {@link noiseRelief} accepts. */
export const reliefBases = {
    /** Marbled stone height field. */
    stone: {fn: noiseKit.stone12, hint: 'stone12'},
    /** Interwoven fibrous fabric height field. */
    wool: {fn: noiseKit.wool12, hint: 'wool12'},
} as const satisfies Record<string, ReliefBasis>

/**
 * Relief filter over child content: sample the child texture through a Perlin-gradient surface
 * distortion, then modulate its brightness by the `basis` height field (an RTT filter — the
 * definition keeps `requiresRTT`/`requiresChild` and the shared `reliefStylizeProps` block).
 */
export function noiseRelief(basis: ReliefBasis): (params: GpuFragmentParams) => Expr {
    return (params) => noiseStylize.applyNoiseReliefExpr(params, basis.fn, basis.hint)
}

/** The Worley prop enum transforms + the historical hash alias, for the shader definition. */
export const transformWorleyMode = noisePaints.transformWorleyMode
export const transformWorleyDistance = noisePaints.transformWorleyDistance
export const worleyHash = noisePaints.worleyHash

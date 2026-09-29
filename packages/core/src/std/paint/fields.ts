/**
 * std — the field pipeline. A {@link Field} is a scalar function of a coordinate,
 * `(params, coord) => Expr`: metric-distance producers ({@link dist}) map a screen UV to a
 * gradient parameter, combinators reshape it ({@link rings}, {@link tiles}, {@link tone},
 * {@link threshold}) or its coordinate ({@link warped}, {@link scaledVolume}), and
 * {@link rampOver} turns a field plus a {@link Palette} into a generator paint. Effects are
 * written as visible compositions of these parts; the GPU bodies live in the kit
 * (`kit/gradientPaints.ts` and friends).
 *
 * Conventions the pipeline owns:
 *  - {@link rampOver} owns the UV idiom (`uvContext ?? ctx.uv`) and the effective-viewport
 *    aspect, and hands the field a vec2 UV coordinate. Coordinate-changing combinators
 *    (e.g. {@link scaledVolume}) may hand their inner field a different coordinate type.
 *  - Structural slots (`edges`, a palette's `space`) are compile-time props: read as CPU
 *    values, JS-branched, only the selected WGSL path is emitted.
 *  - Fields are pure per composition; wrap one in {@link share} when two consumers need the
 *    same evaluation (it binds a WGSL local instead of re-deriving the pattern).
 */
import type {Expr, GpuFragmentParams} from '../../gpu/contract'
import {abs, clamp, fract, local, mul, sub, vec2, vec3} from '../math'
import {call, floatE, vec4, asLocal} from '../../gpu/composer'
import {animatedTime} from '../../gpu/porters'
import {gradientPaints, colorStops, colorMixing, fields as fieldsKit, noise as noiseKit, noiseColor} from '../../gpu/kit/index'
import type {PropRef} from '../values'
import {Scalar} from '../values'
import {resolveScalar, uniformOf, type ArgSpec} from '../invoke'
import {transformColorSpace} from '../../utilities/transformations'
import {packScatterAnchors} from '../../utilities/scatterAnchors'

/** A generator paint — what a `role: 'generator'` definition's `paint:` field takes. */
export type Paint = (params: GpuFragmentParams) => Expr

/** A scalar field over a coordinate. Metric fields take a vec2 UV; domain combinators may pass vec3. */
export type Field = (params: GpuFragmentParams, coord: Expr) => Expr

/** A color lookup at a field parameter t ∈ [0,1]. Build via {@link pair} or {@link stops}. */
export type Palette = (t: Expr, params: GpuFragmentParams) => Expr

/**
 * A scalar slot: a prop/scalar/number, optionally affine-mapped — `{base, add, mul}`
 * resolves to `(base + add) * mul` (for props whose UI range needs remapping into the
 * field's own units, e.g. a 0–100 balance slider to a ±0.3 bias).
 */
export type FieldScalar = ArgSpec | {base: ArgSpec; add?: number; mul?: number}

// ── Resolution helpers ──────────────────────────────────────────────────────────────────

function baseArg(spec: ArgSpec, params: GpuFragmentParams): Expr {
    if (typeof spec === 'number') return floatE(spec)
    if (spec instanceof Scalar || spec.kind !== 'ctx') return resolveScalar(spec, params as never)
    return params.ctx[spec.name]
}

function fArg(spec: FieldScalar, params: GpuFragmentParams): Expr {
    if (typeof spec === 'object' && spec !== null && 'base' in spec) {
        let e = baseArg(spec.base, params)
        if (spec.add !== undefined) e = e.add(spec.add)
        if (spec.mul !== undefined) e = e.mul(spec.mul)
        return e
    }
    return baseArg(spec, params)
}

/** The composed UV + effective viewport every paint evaluates against. */
function frame(params: GpuFragmentParams): {uv: Expr; viewport: Expr} {
    return {
        uv: params.uvContext ?? params.ctx.uv,
        viewport: params.effectiveViewportSize ?? params.ctx.viewportSize,
    }
}

function structural(ref: PropRef, params: GpuFragmentParams): unknown {
    return params.propValues[ref.name]
}

// ── Palettes ────────────────────────────────────────────────────────────────────────────

/**
 * Compile-time color-space mode reader — robust to a raw string (a preset loaded before the
 * prop transform ran) as well as the bridge-mapped number.
 */
export function colorSpaceModeOf(raw: unknown): number {
    if (typeof raw === 'number') return raw
    if (typeof raw === 'string') return transformColorSpace(raw)
    return 0
}

/**
 * Pick the color-mix body for the compile-time color space bound by `space` (linear when
 * absent). The mode is a compile-time prop value, so it JS-branches to a specialised variant
 * and participates in the structural hash.
 */
export function mixColorsIn(space: PropRef | undefined, params: GpuFragmentParams) {
    const mode = space ? colorSpaceModeOf(params.propValues[space.name]) : 0
    return colorMixing.mixColorsVariants[mode as keyof typeof colorMixing.mixColorsVariants] ?? colorMixing.mixColorsLinear
}

/**
 * A 3-color ladder A → B (by `t1`) → C (by `t2`) in the compile-time color `space`, chaining
 * the space's mix variant.
 */
export function colorLadder3(slots: {a: PropRef; b: PropRef; c: PropRef; space: PropRef}) {
    return (t1: Expr, t2: Expr, params: GpuFragmentParams): Expr => {
        const variant = mixColorsIn(slots.space, params)
        const cAB = call(variant, 'mixColors', [uniformOf(slots.a, params), uniformOf(slots.b, params), t1])
        return call(variant, 'mixColors', [cAB, uniformOf(slots.c, params), t2])
    }
}

/** Two endpoint color props mixed in the structural `space`. */
export function pair(a: PropRef, b: PropRef, space: PropRef): Palette {
    return (t, params) => {
        const spaceMode = colorSpaceModeOf(structural(space, params))
        const variant = colorMixing.mixColorsVariants[spaceMode as keyof typeof colorMixing.mixColorsVariants] ?? colorMixing.mixColorsLinear
        return call(variant, 'mixColors', [uniformOf(a, params), uniformOf(b, params), t])
    }
}

/**
 * The standard stop ramp: when the `stops` prop packs >1 active stops (the fixed
 * `colorsArray`/`positionsArray`/`convertedColorsArray`/`stopCount` uniforms from
 * `colorStopsPropConfig`), the working-space accumulation loop runs; otherwise it falls
 * back to the `colorA`/`colorB` pair. Both in the structural `space`.
 */
export function stops(space: PropRef): Palette {
    return (t, params) => {
        const spaceMode = colorSpaceModeOf(structural(space, params))
        const stopCount = (params.propValues.stopCount as number) ?? 0
        if (stopCount > 1) {
            return colorStops.mixColorStopsRuntime(
                t,
                {
                    colorsArray: params.uniforms.colorsArray,
                    positionsArray: params.uniforms.positionsArray,
                    convertedColorsArray: params.uniforms.convertedColorsArray,
                    stopCount: params.uniforms.stopCount,
                },
                spaceMode,
            )
        }
        const variant = colorMixing.mixColorsVariants[spaceMode as keyof typeof colorMixing.mixColorsVariants] ?? colorMixing.mixColorsLinear
        return call(variant, 'mixColors', [params.uniforms.colorA, params.uniforms.colorB, t])
    }
}

// ── rampOver ────────────────────────────────────────────────────────────────────────────

// Edge mode constants (mirror transformEdges: 0=stretch, 1=transparent, 2=mirror, 3=wrap).
const TRANSPARENT = 1
const MIRROR = 2
const WRAP = 3

/**
 * The generator paint: evaluate `field` at the composed UV and look its parameter up in
 * `palette`. With an `edges` slot (a structural prop) the parameter is edge-handled first —
 * stretch (clamp), transparent (alpha fades to 0 outside the [0,1] band), mirror, or wrap —
 * and clamped for the color lookup. Without one, the field's parameter feeds the palette
 * raw (fields that bound themselves — rings, sweeps, tone — need no clamp).
 */
export function rampOver(field: Field, palette: Palette, opts?: {edges?: PropRef}): Paint {
    return (params) => {
        const {uv} = frame(params)
        const t = field(params, uv)

        if (!opts?.edges) return palette(t, params)

        // Edge-mode finalT (compile-time branch), then clamp for the color lookup.
        const edgeMode = (structural(opts.edges, params) as number) ?? 0
        let finalT: Expr
        if (edgeMode === MIRROR) finalT = call(gradientPaints.gradientEdgeMirror, 'gradientEdgeMirror', [t])
        else if (edgeMode === WRAP) finalT = call(gradientPaints.gradientEdgeWrap, 'gradientEdgeWrap', [t])
        else if (edgeMode === TRANSPARENT) finalT = t
        else finalT = call(gradientPaints.gradientClamp01, 'gradientClamp01', [t])
        const clampedT = call(gradientPaints.gradientClamp01, 'gradientClamp01', [finalT])

        const color = palette(clampedT, params)

        // Transparent edge mode fades alpha to 0 outside the [0,1] band (RGB unchanged).
        if (edgeMode === TRANSPARENT) {
            const cut = call(gradientPaints.gradientTransparentAlpha, 'gradientTransparentAlpha', [t])
            return vec4(color.member('rgb'), color.member('a').mul(cut))
        }
        return color
    }
}

// ── Metric-distance fields ──────────────────────────────────────────────────────────────

/** Metric-distance field producers — each maps the screen UV to a raw gradient parameter. */
export const dist = {
    /** Projection onto the `from`→`to` axis, rotated by `angle` (degrees) about the midpoint. */
    linear(slots: {from: PropRef; to: PropRef; angle: FieldScalar}): Field {
        return (params, uv) => {
            const {viewport} = frame(params)
            return call(gradientPaints.gradientRawT, 'gradientRawT', [
                uniformOf(slots.from, params), uniformOf(slots.to, params), fArg(slots.angle, params), uv, viewport,
            ])
        }
    },

    /** Elliptical radial distance from `center`, normalized by `radius`; `aspect` stretches, `skew` rotates the axis. */
    radial(slots: {center: PropRef; radius: FieldScalar; aspect: FieldScalar; skew: FieldScalar}): Field {
        return (params, uv) => {
            const {viewport} = frame(params)
            return call(gradientPaints.radialDist, 'radialDist', [
                uniformOf(slots.center, params), fArg(slots.skew, params), fArg(slots.aspect, params), fArg(slots.radius, params), uv, viewport,
            ])
        }
    },

    /** Angular sweep around `center`, CCW from 3-o'clock, shifted by `rotation` (degrees). */
    conic(slots: {center: PropRef; rotation: FieldScalar}): Field {
        return (params, uv) => {
            const {viewport} = frame(params)
            return call(gradientPaints.conicSweep, 'conicSweep', [
                uniformOf(slots.center, params), fArg(slots.rotation, params), uv, viewport,
            ])
        }
    },

    /** Diamond (L1 → L∞ via `roundness`) distance from `center`, tilted by `rotation`, normalized by `size`. */
    diamond(slots: {center: PropRef; size: FieldScalar; rotation: FieldScalar; roundness: FieldScalar}): Field {
        return (params, uv) => {
            const {viewport} = frame(params)
            return call(gradientPaints.diamondDist, 'diamondDist', [
                uniformOf(slots.center, params), fArg(slots.size, params), fArg(slots.rotation, params), fArg(slots.roundness, params), uv, viewport,
            ])
        }
    },

    /**
     * Animated spiral stripe coverage around `center` (0 = background, 1 = stroke), spun by
     * the definition's `animatedTime` clock. Stays a fused kit body: its anti-aliasing reads
     * `fwidth` of the spiral parameter, so the offset must be derived inside the same
     * fragment-scope function the derivatives sample.
     */
    spiral(slots: {center: PropRef; scale: FieldScalar; width: FieldScalar; falloff: FieldScalar; softness: FieldScalar}): Field {
        return (params, uv) => {
            const {viewport} = frame(params)
            return call(gradientPaints.spiralMask, 'spiralMask', [
                uniformOf(slots.center, params),
                fArg(slots.scale, params),
                fArg(slots.width, params),
                fArg(slots.falloff, params),
                fArg(slots.softness, params),
                animatedTime(params),
                uv,
                viewport,
            ])
        }
    },
}

// ── Field combinators ───────────────────────────────────────────────────────────────────

/** Concentric repetition: `count` ≤ 1 clamps, above 1 tiles rings via fract (a runtime select). */
export function rings(field: Field, count: FieldScalar): Field {
    return (params, coord) => call(gradientPaints.repeatRings, 'repeatRings', [field(params, coord), fArg(count, params)])
}

/** Wrapped repetition: every cycle tiles via fract (angular sweeps, starbursts). */
export function tiles(field: Field, count: FieldScalar): Field {
    return (params, coord) => call(gradientPaints.repeatWrap, 'repeatWrap', [field(params, coord), fArg(count, params)])
}

/**
 * Tone-shape a unit field: `glow` spreads via pow, `contrast` is multiplicative around
 * mid-grey, `balance` is a percent-centred (0–100) shift, `invert` flips the result so a
 * multi-stop ramp's stop0…stopN reads in the expected direction. Currently the glow-gamma
 * inverted shape (the one the kit tone library implements for unit fields).
 */
export function tone(field: Field, opts: {glow: FieldScalar; contrast: FieldScalar; balance: FieldScalar; invert: true}): Field {
    return (params, coord) => call(gradientPaints.plasmaTone, 'toneRemap', [
        field(params, coord), fArg(opts.glow, params), fArg(opts.contrast, params), fArg(opts.balance, params),
    ])
}

/** Soft threshold over a signed [-1,1] field: normalize to [0,1], add `bias`, smoothstep `low`→`high`. */
export function threshold(field: Field, opts: {low: number; high: number; bias?: FieldScalar}): Field {
    return (params, coord) => call(gradientPaints.unitThreshold, 'unitThreshold', [
        field(params, coord),
        opts.bias !== undefined ? fArg(opts.bias, params) : floatE(0),
        floatE(opts.low),
        floatE(opts.high),
    ])
}

/**
 * Share one field evaluation between consumers: the first read binds a WGSL local, later
 * reads reuse it (a threshold and a shimmer reading the same flow pattern must not
 * re-derive it). Cached per composition.
 */
export function share(field: Field): Field {
    const cache = new WeakMap<GpuFragmentParams, Expr>()
    return (params, coord) => {
        let bound = cache.get(params)
        if (!bound) {
            bound = asLocal(field(params, coord), 'field')
            cache.set(params, bound)
        }
        return bound
    }
}

// ── Noise fields ────────────────────────────────────────────────────────────────────────

/** Noise bases usable as a {@link noiseField} — unit-range ([0,1]) samples over a vec3 coordinate. */
const NOISE_BASES = {
    /** MaterialX 3D noise, unit-biased. Hash-based (bit-cast) → GPU-only. */
    mx3: {fn: () => gradientPaints.unitNoise3, hint: 'unitNoise3'},
    /** 3D Perlin gradient noise, ~[0,1]. Hash-based (bit-cast) → GPU-only. */
    perlin: {fn: () => noiseKit.perlin13, hint: 'perlin13'},
    /** 3D value noise — soft blocky cells (Quilez permutation), [0,1]. Pure float. */
    value: {fn: () => noiseKit.value13, hint: 'value13'},
    /** RAW MaterialX 3D noise, signed [-1,1] — pair with a signed tone. GPU-only. */
    mx3signed: {fn: () => noiseKit.mxNoiseFloat3, hint: 'mxNoiseFloat3'},
} as const

export type NoiseBasis = keyof typeof NOISE_BASES

/** Sample a noise basis at the incoming (vec3) coordinate. Compose under a domain producer. */
export function noiseField(basis: NoiseBasis): Field {
    const entry = NOISE_BASES[basis]
    return (_params, coord) => call(entry.fn(), entry.hint, [coord])
}

/**
 * Domain-warp the inner field's coordinate (Inigo-Quilez two-level warp over the given
 * noise basis): noise displaces the sampling position before feeding the field, which is
 * what turns smooth noise into flowing plasma. `amount` (optionally pre-scaled by
 * `amountScale`) sets the displacement; the warp evolves on the definition's
 * `animatedTime` clock scaled by `timeScale`.
 */
export function warped(field: Field, opts: {amount: FieldScalar; amountScale?: number; timeScale?: number}): Field {
    const warpFn = fieldsKit.domainWarp2(noiseKit.mxNoiseFloat3, {name: 'warpDomain'})
    return (params, coord) => {
        let amount = fArg(opts.amount, params)
        if (opts.amountScale !== undefined) amount = amount.mul(opts.amountScale)
        let time = animatedTime(params)
        if (opts.timeScale !== undefined) time = time.mul(opts.timeScale)
        return field(params, call(warpFn, 'warpDomain', [coord, time, amount]))
    }
}

/** Lift a vec3-coordinate field onto the screen: aspect-corrected z=0 slab, zoomed exponentially by `scale`. */
export function scaledVolume(field: Field, opts: {scale: FieldScalar}): Field {
    return (params, uv) => {
        const {viewport} = frame(params)
        return field(params, call(gradientPaints.volumeDomain, 'volumeDomain', [uv, viewport, fArg(opts.scale, params)]))
    }
}

// ── Flow pattern + shimmer ──────────────────────────────────────────────────────────────

/**
 * The layered trig flow pattern: three nested sin/cos layers, each domain-warping the
 * previous, combined into one signed [-1,1] field. Animated by the definition's
 * `animatedTime` clock; reads the raw coordinate it is given (no aspect correction — the
 * pattern is viewport-relative by design).
 */
export function flowField(slots: {detail: FieldScalar}): Field {
    return (params, coord) => call(gradientPaints.flowLayers, 'flowLayers', [coord, fArg(slots.detail, params), animatedTime(params)])
}

/** A subtle brightness pulse from a pattern field: `sin(t·speed + pattern·span)·depth + 1`. */
export function pulse(field: Field, opts: {speed: number; span: number; depth: number}): Field {
    return (params, coord) => call(gradientPaints.shimmerPulse, 'shimmerPulse', [
        field(params, coord), animatedTime(params), floatE(opts.speed), floatE(opts.span), floatE(opts.depth),
    ])
}

/**
 * The classic six-segment hue wheel: a cyclic hue coordinate (any real; wraps) → rgb.
 * The rotating-rainbow palette (Prism's fan, ColorWheel-class effects).
 */
export function hueWheel(h: Expr): Expr {
    const x = local(mul(fract(h), 6), 'hue6')
    return vec3(
        clamp(sub(abs(sub(x, 3)), 1), 0, 1),
        clamp(sub(2, abs(sub(x, 2))), 0, 1),
        clamp(sub(2, abs(sub(x, 4))), 0, 1),
    )
}

/** Scale a paint's whole color (rgb AND alpha) by a gain field evaluated at the raw canvas UV. */
export function shimmered(paint: Paint, gain: Field): Paint {
    return (params) => paint(params).mul(gain(params, params.ctx.uv))
}

// ── Scatter fields, ramp wrapping, and the standard palette ─────────────────────────────

/**
 * `scatterField` — a smooth scalar field blending unit values scattered organically across the
 * canvas (the mesh-gradient genre primitive): golden-spiral points, gently drifting, blended by
 * inverse distance. `softness` is the IDW exponent (higher = crisper cells); `count` gates the
 * fixed-8 GPU loop at runtime.
 */
export function scatterField(opts: {
    at: Expr
    count: Expr | number
    seed: Expr | number
    drift: Expr | number
    aspect: Expr | number
    time: Expr | number
    softness: Expr | number
    /**
     * The constellation pre-computed on the CPU (see {@link scatteredAnchors}) — an
     * `array<vec4f, 4>` extraField holding the 8 anchors packed 4-per-vec4. When given, the GPU
     * skips re-deriving every anchor per pixel (`drift`/`aspect`/`time` are then read on the CPU
     * side and ignored here).
     */
    anchors?: Expr
}): Expr {
    const e = (v: Expr | number): Expr => (typeof v === 'number' ? floatE(v) : v)
    if (opts.anchors) {
        return call(fieldsKit.scatterFieldAnchored, 'scatterField', [
            opts.at, e(opts.count), e(opts.seed), opts.anchors, e(opts.softness),
        ])
    }
    return call(fieldsKit.scatterField, 'scatterField', [
        opts.at, e(opts.count), e(opts.seed), e(opts.drift), e(opts.aspect), e(opts.time), e(opts.softness),
    ])
}

/**
 * `scatteredAnchors` — drive an `array<vec4f, 4>` extraField with the {@link scatterField}
 * constellation, computed ONCE per frame on the CPU (the pixel-invariant half of the field:
 * 8 anchor positions from count/seed/drift/aspect/animated time). The definition declares
 * `extraFields: {<field>: {schema: schema.arrayOf(schema.vec4f, 4), initial: 16 zeros}}`; this word
 * registers the per-frame writer and returns the field's GPU accessor to pass as
 * `scatterField({anchors})`. Reads the node's live `_animTime` (the same accumulator the GPU
 * warp reads), so anchors and warp stay on one clock. Dirty-keyed: a static composition (speed 0,
 * no edits) writes nothing. In a GPU-free composition (tests) `_animTime` is unavailable and the
 * field keeps its initial value.
 */
export function scatteredAnchors(params: GpuFragmentParams, slots: {count: PropRef; seed: PropRef; drift: PropRef; field: string}): Expr {
    let aspect = params.dimensions.width / Math.max(params.dimensions.height, 1e-6)
    params.onResize(({width, height}) => { aspect = width / Math.max(height, 1e-6) })
    let lastKey = ''
    params.onBeforeRender(() => {
        const t = params.getCpuValue('_animTime')
        const count = params.getCpuValue(slots.count.name)
        const seed = params.getCpuValue(slots.seed.name)
        const drift = params.getCpuValue(slots.drift.name)
        if (typeof t !== 'number' || typeof count !== 'number' || typeof seed !== 'number' || typeof drift !== 'number') return
        const key = `${count}|${seed}|${drift}|${aspect}|${t}`
        if (key === lastKey) return
        lastKey = key
        params.setExtraField(slots.field, packScatterAnchors(count, seed, drift, aspect, t))
    })
    return params.uniforms[slots.field]
}

/**
 * `wrapRamp` — traverse a palette through `cycles` half-cycles as `t` rises, folding back at the
 * ends (seam-free triangle wave; identity at cycles = 1); sub-pixel bands fade to the palette
 * mid instead of shimmering. Fragment-only (fwidth).
 */
export function wrapRamp(opts: {t: Expr; cycles: Expr | number}): Expr {
    return call(gradientPaints.rampWrap, 'rampWrap', [opts.t, typeof opts.cycles === 'number' ? floatE(opts.cycles) : opts.cycles])
}

/**
 * `foldRamp` — fold an unbounded palette parameter back into [0, 1] as a seam-free triangle
 * wave (0→1→0 every 2 units of `t`). The derivative-free cousin of {@link wrapRamp}: no
 * sub-pixel fade, so it is safe inside a `guarded` branch (non-uniform control flow, where
 * `fwidth` is not). Scale `t` first to set the cycle length.
 */
export function foldRamp(t: Expr): Expr {
    return call(gradientPaints.gradientEdgeMirror, 'gradientEdgeMirror', [t])
}

/**
 * `warpedPoint` — the domain-warped position itself (two Inigo-Quilez warp levels over MaterialX
 * noise), for recipes that need the warped COORDINATE (to feed several fields) rather than one
 * warped field sample (`warped`). Same memoized 'warpDomain' instance as {@link warped}.
 */
export function warpedPoint(opts: {at: Expr; time: Expr | number; amount: Expr | number; levels?: 1 | 2}): Expr {
    const e = (v: Expr | number): Expr => (typeof v === 'number' ? floatE(v) : v)
    if (opts.levels === 1) {
        // The mobile tier: one warp level (2 noise reads instead of 4), same offsets and look data.
        const warp1 = fieldsKit.domainWarp1(noiseKit.mxNoiseFloat3, {name: 'warpDomain1'})
        return call(warp1, 'warpDomain1', [opts.at, e(opts.time), e(opts.amount)])
    }
    const warpFn = fieldsKit.domainWarp2(noiseKit.mxNoiseFloat3, {name: 'warpDomain'})
    return call(warpFn, 'warpDomain', [opts.at, e(opts.time), e(opts.amount)])
}

/**
 * `warpStep` — one chained Inigo-Quilez domain-warp level: displace a position by two
 * decorrelated noise reads (returned as `.z`/`.w` for tail stages to shade with). Chain steps
 * by feeding one step's `.xy` to the next. `offsets` are the two decorrelation offsets — the
 * caller's look data.
 */
export function warpStep(opts: {
    at: Expr
    scale: number
    z: Expr | number
    offsets: [[number, number], [number, number]]
    reach: number
    strength: Expr | number
}): Expr {
    const e = (v: Expr | number): Expr => (typeof v === 'number' ? floatE(v) : v)
    return call(fieldsKit.chainedWarpStep, 'chainedWarpStep', [
        opts.at, floatE(opts.scale), e(opts.z),
        vec2(opts.offsets[0][0], opts.offsets[0][1]),
        vec2(opts.offsets[1][0], opts.offsets[1][1]),
        floatE(opts.reach), e(opts.strength),
    ])
}

/**
 * The standard multi-stop / two-color palette, read from the definition's standard palette
 * props (colorA/colorB/stops/colorSpace). The coordinate is consumed once — the stops loop
 * takes it as a fn parameter.
 */
export function standardPalette(t: Expr, params: GpuFragmentParams): Expr {
    return noiseColor.mixStopsOrColorsExpr(params, t)
}

/**
 * A 2D 4-color blend: row1 = A→B by `t1`, row2 = C→D by `t1`, base = row1→row2 by `t2`, in
 * the compile-time color `space`. For a non-linear space the four endpoint conversions are
 * pixel-invariant, so the word's per-frame driver preconverts them (dirty-keyed) into the
 * `convA`–`convD` vec3 extraFields the definition declares, and the GPU mixes the
 * preconverted values with a single back-conversion.
 */
export function quadBlend(slots: {a: PropRef; b: PropRef; c: PropRef; d: PropRef; space: PropRef}) {
    return (t1: Expr, t2: Expr, params: GpuFragmentParams): Expr => {
        const {uniforms} = params
        const colorSpaceMode = colorSpaceModeOf(params.propValues[slots.space.name])
        const colorA = uniformOf(slots.a, params)
        const colorB = uniformOf(slots.b, params)
        const colorC = uniformOf(slots.c, params)
        const colorD = uniformOf(slots.d, params)
        if (colorSpaceMode !== 0) {
            let lastKey = ''
            params.onBeforeRender(() => {
                const cols = ([slots.a.name, slots.b.name, slots.c.name, slots.d.name]).map(
                    (prop) => params.getCpuValue(prop) as {x: number; y: number; z: number} | undefined,
                )
                if (cols.some((c) => !c)) return
                const key = cols.map((c) => `${c!.x},${c!.y},${c!.z}`).join('|')
                if (key === lastKey) return
                lastKey = key
                const fields = ['convA', 'convB', 'convC', 'convD'] as const
                cols.forEach((c, i) => {
                    const conv = colorMixing.convertP3ToMixSpaceCPU(c!.x, c!.y, c!.z, colorSpaceMode)
                    params.setExtraField(fields[i], [conv[0], conv[1], conv[2]])
                })
            })
            const row1 = asLocal(call(colorMixing.mixPreconvertedInSpace, 'mixInSpace', [
                uniforms.convA, colorA.member('a'), uniforms.convB, colorB.member('a'), t1,
            ]), 'flowRow1')
            const row2 = asLocal(call(colorMixing.mixPreconvertedInSpace, 'mixInSpace', [
                uniforms.convC, colorC.member('a'), uniforms.convD, colorD.member('a'), t1,
            ]), 'flowRow2')
            const variant = colorMixing.mixPreconvertedVariants[colorSpaceMode as keyof typeof colorMixing.mixPreconvertedVariants] ?? colorMixing.mixPreconvertedLinear
            return call(variant, 'mixPreconvertedColors', [
                row1.member('conv'), row2.member('conv'), row1.member('alpha'), row2.member('alpha'), t2,
            ])
        }
        const row1 = asLocal(call(colorMixing.mixColorsLinear, 'mixColors', [colorA, colorB, t1]), 'flowRow1')
        const row2 = asLocal(call(colorMixing.mixColorsLinear, 'mixColors', [colorC, colorD, t1]), 'flowRow2')
        return call(colorMixing.mixColorsLinear, 'mixColors', [row1, row2, t2])
    }
}

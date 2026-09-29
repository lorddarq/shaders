/**
 * std/effects/color — color-filter nouns.
 *
 * Each noun wraps one pointwise color op from `kit/colorOps` (the GPU body) into a
 * `PointwiseEffect`: the authored shader file passes prop bindings, the noun owns the wiring —
 * including builder-level compile-time work (Tint's body pair, the tone filters' color-space
 * mixes, the gradient map's palette branch).
 */
import type {Expr, GpuFragmentParams} from '../../gpu/contract'
import {formatFloat} from '../../gpu/contract'
import {call, expr, vec4, floatE, ZERO} from '../../gpu/composer'
import {colorOps, colorMixing, tgpu, d, std, tonemap} from '../../gpu/kit/index'
import type {PointwiseEffect} from '../types'
import type {PropRef} from '../values'
import {pointwiseOp, resolveArg, uniformOf, type ArgSpec} from '../invoke'
import type {FilterParams} from '../../gpu/scaffolds/pointwiseFilter'

/** Rotate hue by an angle in degrees (Rodrigues rotation around the achromatic axis). */
export function hueRotate(shift: ArgSpec): PointwiseEffect {
    return pointwiseOp(colorOps.hueRotate, 'hueRotate', [shift])
}

/** Scale saturation around grayscale: 0 = grayscale, 1 = unchanged, >1 oversaturates. */
export function saturate(intensity: ArgSpec): PointwiseEffect {
    return pointwiseOp(colorOps.saturate, 'saturate', [intensity])
}

/** Selective saturation adjustment (around 0) that protects already-saturated pixels. */
export function vibrance(intensity: ArgSpec): PointwiseEffect {
    return pointwiseOp(colorOps.vibrance, 'vibrance', [intensity])
}

/** Contrast around the midpoint plus additive brightness; both raw sliders identity at 0. */
export function brightnessContrast(slots: {brightness: ArgSpec; contrast: ArgSpec}): PointwiseEffect {
    return pointwiseOp(colorOps.brightnessContrast, 'brightnessContrast', [slots.brightness, slots.contrast])
}

/** Multiplicative gain — blacks stay black, highlights may exceed 1.0 (unclamped, HDR-safe). */
export function exposure(gain: ArgSpec): PointwiseEffect {
    return pointwiseOp(colorOps.exposure, 'exposure', [gain])
}

/** Luminance-weighted black & white (Rec.709). */
export function grayscale(): PointwiseEffect {
    return pointwiseOp(colorOps.grayscale, 'grayscale', [])
}

/** Invert RGB (1 - rgb), alpha preserved. */
export function invert(): PointwiseEffect {
    return pointwiseOp(colorOps.invert, 'invert', [])
}

/** Invert tones above a luminance threshold, blended in by strength. */
export function solarize(slots: {threshold: ArgSpec; strength: ArgSpec}): PointwiseEffect {
    return pointwiseOp(colorOps.solarize, 'solarize', [slots.threshold, slots.strength])
}

/** Quantise each channel to a step count (banded, poster-like color). */
export function posterize(steps: ArgSpec): PointwiseEffect {
    return pointwiseOp(colorOps.posterize, 'posterize', [steps])
}

/**
 * Mix the child color toward a tint color by amount. `preserveLuminosity` is a compile-time
 * body PAIR — only the chosen variant emits: the preserving body rescales the tinted output back
 * to the original Rec.601 brightness so the tint shifts hue without darkening.
 */
export function tint(slots: {color: PropRef; amount: ArgSpec; preserveLuminosity: PropRef}): PointwiseEffect {
    return {
        kind: 'pointwise',
        body: (propValues) =>
            Number(propValues[slots.preserveLuminosity.name]) > 0
                ? {fn: colorOps.tintPreserveLuma, hint: 'tintPreserveLuma'}
                : {fn: colorOps.tintPlain, hint: 'tintPlain'},
        args: (params) => [resolveArg(slots.color, params), resolveArg(slots.amount, params)],
    }
}

/**
 * Analog film grain weighted toward darker areas. The grain drift is CPU-accumulated into the
 * `animTime` extraField (`animTime += deltaTime * animated * 10`, frozen when `animated` is off) —
 * the definition must declare `extraFields: {animTime: {schema: d.f32, initial: 0}}`; the noun
 * owns the per-frame clock and the GPU read.
 */
export function filmGrain(slots: {strength: ArgSpec; bias: ArgSpec; animated: PropRef}): PointwiseEffect {
    return {
        kind: 'pointwise',
        body: {fn: colorOps.filmGrain, hint: 'filmGrain'},
        args: (params) => [
            params.ctx.uv,
            params.ctx.viewportSize,
            resolveArg(slots.strength, params),
            resolveArg(slots.bias, params),
            params.uniforms.animTime,
        ],
        // `animated` is a boolean → getCpuValue coerced `> 0`. A closure holds the running total
        // (reset on recompose).
        setup: (params) => {
            let animTimeAcc = 0
            params.onBeforeRender(({deltaTime}) => {
                const on = ((params.getCpuValue(slots.animated.name) as number) > 0) ? 1 : 0
                animTimeAcc += deltaTime * on * 10.0
                params.setExtraField('animTime', animTimeAcc)
            })
        },
    }
}

// Compile-time enum reads for the isolines branches (applied by the bridge, so propValues carry
// the mapped number — these stay robust to a raw string).
const sourceModeOf = (raw: unknown): number => (typeof raw === 'number' ? raw : raw === 'alpha' ? 1 : 0)
const colorModeOf = (raw: unknown): number => (typeof raw === 'number' ? raw : raw === 'custom' ? 1 : 0)

/**
 * Topographic contour lines from the child's luminance or alpha (compile-time `source` body
 * pair), banded with fwidth anti-aliasing. Compile-time `colorMode` picks the palette: 'custom'
 * draws lineColor over backgroundColor; 'source' draws the child's own colors (alpha 1) over
 * transparent.
 */
export function isolines(slots: {
    source: PropRef
    levels: ArgSpec
    lineWidth: ArgSpec
    softness: ArgSpec
    gamma: ArgSpec
    invert: ArgSpec
    colorMode: PropRef
    lineColor: PropRef
    backgroundColor: PropRef
}): PointwiseEffect {
    return {
        kind: 'pointwise',
        // Compile-time source channel: luminance dot OR alpha read — only the chosen body emits.
        body: (propValues) =>
            sourceModeOf(propValues[slots.source.name]) === 1
                ? {fn: colorOps.isolinesFromAlpha, hint: 'isolinesFromAlpha'}
                : {fn: colorOps.isolinesFromLuma, hint: 'isolinesFromLuma'},
        args: (params) =>
            [slots.levels, slots.lineWidth, slots.softness, slots.gamma, slots.invert].map((s) => resolveArg(s, params)),
        compose: (lineMask, params) => {
            // Compile-time color mode: source → line=child.rgb@α1 over transparent; custom → props.
            const custom = colorModeOf(params.propValues[slots.colorMode.name]) === 1
            const bg = custom ? uniformOf(slots.backgroundColor, params) : ZERO
            const line = custom ? uniformOf(slots.lineColor, params) : vec4(params.childNode.member('rgb'), floatE(1))
            return call(colorOps.contourCompose, 'contourCompose', [bg, line, lineMask])
        },
    }
}

/** The compile-time color-space mix variant for a colorSpace prop binding. */
function mixVariantFor(colorSpace: PropRef, params: FilterParams) {
    const mode = (params.propValues[colorSpace.name] as number) ?? 0
    return colorMixing.mixColorsVariants[mode as keyof typeof colorMixing.mixColorsVariants] ?? colorMixing.mixColorsLinear
}

/**
 * Map the child's luminance to a two-color ramp (dark → colorA, bright → colorB), mixed in the
 * compile-time color space. Child alpha preserved.
 */
export function duotone(slots: {colorA: PropRef; colorB: PropRef; blend: ArgSpec; colorSpace: PropRef}): PointwiseEffect {
    return pointwiseOp(colorOps.duotoneT, 'duotoneT', [slots.blend], {
        compose: (t, params) => {
            const variant = mixVariantFor(slots.colorSpace, params)
            const duotoneColor = call(variant, 'mixColors', [uniformOf(slots.colorA, params), uniformOf(slots.colorB, params), t])
            return vec4(duotoneColor.member('rgb'), params.childNode.member('a'))
        },
    })
}

/**
 * Map the child's luminance to a three-color ramp (shadows → midtones → highlights), mixed in
 * the compile-time color space. Child alpha preserved.
 */
export function tritone(slots: {colorA: PropRef; colorB: PropRef; colorC: PropRef; blendMid: ArgSpec; colorSpace: PropRef}): PointwiseEffect {
    return pointwiseOp(colorOps.tritoneFactors, 'tritoneFactors', [slots.blendMid], {
        compose: (factors, params) => {
            const shadowToMid = factors.member('x')
            const midToHighlight = factors.member('y')
            const finalT = factors.member('z')
            const variant = mixVariantFor(slots.colorSpace, params)
            const colorA = uniformOf(slots.colorA, params)
            const colorB = uniformOf(slots.colorB, params)
            const colorC = uniformOf(slots.colorC, params)
            // Shadows→midtones, then midtones→highlights, then blend the two by luminance position.
            const lowerBlend = call(variant, 'mixColors', [colorA, colorB, shadowToMid])
            const upperBlend = call(variant, 'mixColors', [colorB, colorC, midToHighlight])
            const finalColor = call(variant, 'mixColors', [lowerBlend, upperBlend, finalT])
            return vec4(finalColor.member('rgb'), params.childNode.member('a'))
        },
    })
}

// Inigo Quilez cosine palettes: color(t) = a + b*cos(2π*(c*t + d))
// https://iquilezles.org/articles/palettes/
const PALETTES: Record<number, {a: number[]; b: number[]; c: number[]; d: number[]}> = {
    0: { a: [0.5, 0.5, 0.5], b: [0.5, 0.5, 0.5], c: [1, 1, 1], d: [0.0, 0.333, 0.667] },   // Rainbow
    1: { a: [0.5, 0.5, 0.5], b: [0.5, 0.5, 0.5], c: [1, 1, 1], d: [0.0, 0.1, 0.2] },       // Sunset
    2: { a: [0.5, 0.5, 0.5], b: [0.5, 0.5, 0.5], c: [1, 1, 1], d: [0.3, 0.2, 0.2] },       // Ocean
    3: { a: [0.5, 0.5, 0.5], b: [0.5, 0.5, 0.5], c: [1, 1, 1], d: [0.8, 0.9, 0.3] },       // Fire
    4: { a: [0.5, 0.5, 0.5], b: [0.25, 0.25, 0.25], c: [1, 1, 1], d: [0.0, 0.25, 0.5] },   // Pastel
    5: { a: [0.5, 0.5, 0.5], b: [0.5, 0.5, 0.5], c: [2, 1, 1], d: [0.5, 0.2, 0.25] }       // Neon
}

// The `palette` prop is a compile-time cpu-only string; map string → index at builder level.
const PALETTE_INDEX: Record<string, number> = {rainbow: 0, sunset: 1, ocean: 2, fire: 3, pastel: 4, neon: 5, custom: 6}

/** A vec3f literal Expr (there is no `vec3` builder factory; build the raw WGSL, decimal-safe). */
const vec3E = (v: number[]): Expr => expr(`vec3f(${formatFloat(v[0])}, ${formatFloat(v[1])}, ${formatFloat(v[2])})`)

/**
 * Photoshop-style gradient map: remap the child's luminance through black/white points and
 * contrast, then color it — with a built-in cosine palette, or (palette 'custom') a cyclic
 * low→mid→high ramp mixed in the compile-time color space — scrolled over time by speed, and
 * blended back over the original by strength.
 */
export function gradientMap(slots: {
    palette: PropRef
    colorLow: PropRef
    colorMid: PropRef
    colorHigh: PropRef
    speed: PropRef
    contrast: ArgSpec
    blackPoint: ArgSpec
    whitePoint: ArgSpec
    strength: PropRef
    colorSpace: PropRef
}): PointwiseEffect {
    return pointwiseOp(colorOps.gradientMapT, 'gradientMapT', [slots.blackPoint, slots.whitePoint, slots.contrast], {
        // The luminance→t body runs first; everything after it is builder-level: a compile-time
        // palette branch, and for custom colors the per-segment mixes in the compile-time color space.
        compose: (t, params): Expr => {
            const {childNode, propValues, ctx} = params
            const strength = uniformOf(slots.strength, params)
            // Global clock; phase = speed × time — animates the gradient scroll.
            const phase = uniformOf(slots.speed, params).mul(ctx.time)

            // `palette` is a compile-time string → JS-branch the whole fragment (recomposes on change).
            const paletteIdx = PALETTE_INDEX[(propValues[slots.palette.name] as string) ?? 'rainbow'] ?? 0
            if (paletteIdx !== 6) {
                const p = PALETTES[paletteIdx] ?? PALETTES[0]
                const mapped = call(colorOps.gradientMapCosine, 'gradientMapCosine', [t, phase, vec3E(p.a), vec3E(p.b), vec3E(p.c), vec3E(p.d)])
                return call(colorOps.gradientMapCompose, 'gradientMapCompose', [childNode, mapped, strength])
            }

            // Custom 3-stop cyclic ramp (low → mid → high → low) in the compile-time color space.
            const variant = mixVariantFor(slots.colorSpace, params)
            const phaseData = call(colorOps.gradientMapCustomPhase, 'gradientMapCustomPhase', [t, phase])
            const fr = phaseData.member('x')
            const i = phaseData.member('y')
            const colorLow = uniformOf(slots.colorLow, params)
            const colorMid = uniformOf(slots.colorMid, params)
            const colorHigh = uniformOf(slots.colorHigh, params)
            const seg0 = call(variant, 'mixColors', [colorLow, colorMid, fr])
            const seg1 = call(variant, 'mixColors', [colorMid, colorHigh, fr])
            const seg2 = call(variant, 'mixColors', [colorHigh, colorLow, fr])
            const mappedColor = call(colorOps.gradientMapSelect, 'gradientMapSelect', [seg0, seg1, seg2, i])
            return call(colorOps.gradientMapCompose, 'gradientMapCompose', [childNode, mappedColor.member('rgb'), strength])
        },
    })
}

// ── 3D-LUT atlas grade ──────────────────────────────────────────────────────────────────────
//
// A measured N³ color LUT (HaldCLUT convention: sRGB-in/sRGB-out) uploaded as an rgba8unorm 2D
// atlas — blue slices side by side along x (width N·N, height N). A lookup is two hardware-
// bilinear taps (the blue slice below and above) mixed by the blue fraction: the classic
// 2D-atlas trilinear. Sample coords are half-texel inset so bilinear never bleeds across slices.

interface LutAtlasFns {
    uv: ReturnType<typeof makeLutAtlasUvFn>
    apply: ReturnType<typeof makeLutAtlasApplyFn>
}

function makeLutAtlasUvFn(size: number) {
    const LUT_MAX = size - 1
    const INV_ATLAS_W = 1 / (size * size)
    const INV_LUT_N = 1 / size
    /**
     * Atlas UV for one blue slice. The linear input is clamped to the LUT domain and
     * sRGB-encoded first; `hi` selects the slice below (0) or above (1) the encoded blue value.
     * Red/green land on hardware bilinear inside the slice.
     */
    return tgpu.fn([d.vec4f, d.f32], d.vec2f)((color, hi) => {
        'use gpu'
        const enc = tonemap.linearToSrgb(std.clamp(color.xyz, d.vec3f(0.0, 0.0, 0.0), d.vec3f(1.0, 1.0, 1.0)))
        const slice = enc.z * d.f32(LUT_MAX)
        const s = std.clamp(std.floor(slice) + hi, d.f32(0), d.f32(LUT_MAX))
        const u = (s * d.f32(size) + 0.5 + enc.x * d.f32(LUT_MAX)) * d.f32(INV_ATLAS_W)
        const v = (0.5 + enc.y * d.f32(LUT_MAX)) * d.f32(INV_LUT_N)
        return d.vec2f(u, v)
    }).$name('lutAtlasUv')
}

function makeLutAtlasApplyFn(size: number) {
    const LUT_MAX = size - 1
    /**
     * Complete the trilinear grade: mix the two slice taps by the blue fraction (in sRGB, the
     * LUT's own space), decode back to linear, and blend toward the graded color by `strength`.
     * Alpha preserved; colors above 1.0 grade at their clamped value (the LUT domain is SDR).
     */
    return tgpu.fn([d.vec4f, d.vec4f, d.vec4f, d.f32], d.vec4f)(
        (color, tapLo, tapHi, strength) => {
            'use gpu'
            const enc = tonemap.linearToSrgb(std.clamp(color.xyz, d.vec3f(0.0, 0.0, 0.0), d.vec3f(1.0, 1.0, 1.0)))
            const slice = enc.z * d.f32(LUT_MAX)
            const f = slice - std.floor(slice)
            const graded = tonemap.srgbToLinear(std.mix(tapLo.xyz, tapHi.xyz, d.vec3f(f, f, f)))
            const rgb = std.mix(color.xyz, graded, d.vec3f(strength, strength, strength))
            return d.vec4f(rgb.x, rgb.y, rgb.z, color.w)
        },
    ).$name('lutAtlasApply')
}

const lutAtlasFnCache = new Map<number, LutAtlasFns>()

/** The per-size LUT atlas fn pair (size is a compile-time constant baked into the bodies). */
export function lutAtlasFnsFor(size: number): LutAtlasFns {
    let fns = lutAtlasFnCache.get(size)
    if (!fns) {
        fns = {uv: makeLutAtlasUvFn(size), apply: makeLutAtlasApplyFn(size)}
        lutAtlasFnCache.set(size, fns)
    }
    return fns
}

/**
 * 3D-LUT atlas grade stage: color in → graded color out. `select` binds a compile-time prop
 * whose value keys `decode` (the CPU-side LUT bytes for the baked selection — decoded once per
 * composition into an rgba8unorm slice atlas); `strength` blends toward the graded color.
 */
export function lutAtlasGrade(slots: {
    select: PropRef
    strength: PropRef
    size: number
    decode: (key: string) => Uint8Array
    fallback: string
    label: string
}): (color: Expr, params: FilterParams | GpuFragmentParams) => Expr {
    const fns = lutAtlasFnsFor(slots.size)
    return (color, params) => {
        const {propValues, uniforms, createDataTexture, registerMediaTexture, onCleanup} = params as GpuFragmentParams
        const key = (propValues[slots.select.name] as string) ?? slots.fallback
        const lutTex = createDataTexture({
            width: slots.size * slots.size, height: slots.size, format: 'rgba8unorm',
            data: slots.decode(key), label: `${slots.label}-${key}`,
        })
        onCleanup(() => lutTex.destroy())
        const lutKit = registerMediaTexture(() => lutTex.texture)

        const tapLo = lutKit.sample(call(fns.uv, 'lutAtlasUvLo', [color, floatE(0)]), 'linearClamp')
        const tapHi = lutKit.sample(call(fns.uv, 'lutAtlasUvHi', [color, floatE(1)]), 'linearClamp')
        return call(fns.apply, 'lutAtlasApply', [color, tapLo, tapHi, uniforms[slots.strength.name]])
    }
}

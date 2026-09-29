/**
 * std/effects — blur, sharpen, pixelate, shadow, and retro-screen vocabulary.
 *
 * Nearly all of it rides the gather pipeline: the child renders to a texture, taps are
 * premultiplied, and the species owns the RTT boundary and the premultiplied → straight
 * unpremultiply tail (a stack that finishes in straight alpha passes
 * `{resultAlpha: 'straight'}` to `gatherStack`). Multi-part recipes are composed IN THE
 * SHADER FILE from the stage parts exported here — a `SampleStage`, ordered
 * `OverlayStage`s, and, where several stages read one geometry, a shared `GatherFrame`.
 * The GPU math lives in the kit's `motionBlur`/`warpMaps` modules; these parts only bind
 * props and context.
 */
import type {Expr, GpuComputeNode, GpuFragmentParams, KitTexture} from '../../gpu/contract'
import {call, floatE, mixExpr, vec4} from '../../gpu/composer'
import {asLocal, expr} from '../../gpu/composer'
import {formatFloat, animatedTime, createGuardedCompute, ZERO} from '../../gpu/porters'
import {blur as blurKit, motionBlur as motionBlurKit} from '../../gpu/kit/index'
import {blend, sampling, warpMaps, edges, tgpu, d, std, constants} from '../../gpu/kit/index'
import type {RttFilterParams} from '../../gpu/scaffolds/rttFilter'
import {resolveArg, uniformOf, type ArgSpec} from '../invoke'
import type {GatherEffect} from '../types'
import type {PropRef} from '../values'

// ── Gather pipeline ─────────────────────────────────────────────────────────────────────────
//
// A gather effect as a STACK: one sampling stage produces the initial color from the child RTT,
// then overlay stages transform it in order. Each stage is a named part; an effect noun reads as
// the recipe of its stages.

/** The sampling stage: produce the initial color from the child RTT. */
export type SampleStage = (params: RttFilterParams) => Expr

/** One post-sample overlay stage: color in → color out. */
export type OverlayStage = (color: Expr, params: RttFilterParams) => Expr

/** Compose a gather effect from a sampling stage and an ordered list of overlays. */
export function gatherStack(
    sample: SampleStage,
    overlays: OverlayStage[],
    opts?: {resultAlpha?: 'premultiplied' | 'straight'},
): GatherEffect {
    return {
        kind: 'gather',
        resultAlpha: opts?.resultAlpha,
        build: (params): Expr => overlays.reduce((color, stage) => stage(color, params), sample(params)),
    }
}

/**
 * A shared geometry frame: one Expr (usually an `asLocal` struct) that several stages of a
 * recipe read. The producer is memoized per composition, so every stage sees the same local.
 */
export type GatherFrame = (params: RttFilterParams) => Expr

/** Memoize a frame producer per composition params. */
function frameOf(make: (params: RttFilterParams) => Expr): GatherFrame {
    const cache = new WeakMap<RttFilterParams, Expr>()
    return (params) => {
        const hit = cache.get(params)
        if (hit) return hit
        const made = make(params)
        cache.set(params, made)
        return made
    }
}

// ── Sampling-stage parts ────────────────────────────────────────────────────────────────────

/**
 * RGB split: sample the child at ±`amount·0.002` horizontal offsets and take red from the +tap,
 * green from the centre, blue from the −tap.
 */
export function rgbSplit(opts: {amount: ArgSpec}): SampleStage {
    return (params) => {
        const amount = resolveArg(opts.amount, params)
        const redUV = call(motionBlurKit.crtSampleUV, 'crtSampleUV', [params.ctx.uv, amount, floatE(1)])
        const blueUV = call(motionBlurKit.crtSampleUV, 'crtSampleUV', [params.ctx.uv, amount, floatE(-1)])
        return call(motionBlurKit.rgbSplitCombine, 'rgbSplitCombine', [
            params.texture.sample(redUV), params.texture.sample(params.ctx.uv), params.texture.sample(blueUV),
        ])
    }
}

/** VHS tape-warp UVs: vec4(lumaUV.xy, chromaUV.zw) from the wobble/jitter tape geometry, driven by
 *  the global clock × speed. Feed into {@link chromaSmearTaps}. */
export function tapeWarp(opts: {wobble: ArgSpec; jitter: ArgSpec; speed: ArgSpec}): (params: RttFilterParams) => Expr {
    return (params) =>
        call(motionBlurKit.vhsSampleUVs, 'vhsSampleUVs', [
            params.ctx.uv, params.ctx.time, resolveArg(opts.speed, params), resolveArg(opts.wobble, params), resolveArg(opts.jitter, params),
        ])
}

/**
 * Chroma smear: one sharp luma tap + 5 chroma taps trailing by `smear·0.0075` each, recombined
 * through YIQ (sharp Y, smeared I/Q). Alpha comes from the luma tap.
 */
export function chromaSmearTaps(opts: {warp: (params: RttFilterParams) => Expr; smear: ArgSpec}): SampleStage {
    return (params) => {
        const uvs = asLocal(opts.warp(params), 'tapeUVs')
        const chromaUV = uvs.member('zw')
        const lumaSample = params.texture.sample(uvs.member('xy'))
        const smearScale = resolveArg(opts.smear, params).mul(0.0075)
        // 5 chroma taps i=1..5 (the i=0 tap has weight 0 → omitted).
        const tap = (i: number): Expr =>
            params.texture.sample(call(motionBlurKit.vhsChromaTapUV, 'vhsChromaTapUV', [chromaUV, smearScale, floatE(i)]))
        const rgb = call(motionBlurKit.yiqRecombine, 'yiqRecombine', [lumaSample, tap(1), tap(2), tap(3), tap(4), tap(5)])
        return vec4(rgb, lumaSample.member('a'))
    }
}

// ── Overlay parts ───────────────────────────────────────────────────────────────────────────

/** Brightness/contrast adjustment around mid-grey. */
export function adjust(opts: {brightness: ArgSpec; contrast: ArgSpec}): OverlayStage {
    return (color, params) =>
        call(motionBlurKit.adjustShade, 'adjustShade', [color, resolveArg(opts.contrast, params), resolveArg(opts.brightness, params)])
}

/** Sinusoidal scanlines over uv.y. */
export function scanlines(opts: {frequency: ArgSpec; intensity: ArgSpec}): OverlayStage {
    return (color, params) =>
        call(motionBlurKit.scanlineShade, 'scanlineShade', [
            color, params.ctx.uv.member('y'), resolveArg(opts.frequency, params), resolveArg(opts.intensity, params),
        ])
}

/** Subtle RGB phosphor mask; `pitch` is the phosphor cell size. */
export function phosphorMask(opts: {pitch: ArgSpec}): OverlayStage {
    return (color, params) =>
        call(motionBlurKit.phosphorShade, 'phosphorShade', [color, params.ctx.uv, resolveArg(opts.pitch, params)])
}

/** Aspect-corrected circular vignette darkening. */
export function vignetteOverlay(opts: {radius: ArgSpec; intensity: ArgSpec}): OverlayStage {
    return (color, params) =>
        call(motionBlurKit.vignetteShade, 'vignetteShade', [
            color, params.ctx.uv, params.ctx.aspect, resolveArg(opts.radius, params), resolveArg(opts.intensity, params),
        ])
}

/** Finish a vec3 color stack as an opaque vec4 (alpha 1). */
export function opaque(): OverlayStage {
    return (color) => vec4(color, floatE(1))
}

/** AC beat: a very subtle brightness pulse from the global clock, clamped to [0,1]. */
export function beatPulse(opts: {wobble: ArgSpec; speed: ArgSpec}): OverlayStage {
    return (color, params) =>
        call(motionBlurKit.beatShade, 'beatShade', [
            color,
            call(motionBlurKit.vhsAcBeat, 'vhsAcBeat', [
                params.ctx.uv, params.ctx.time, resolveArg(opts.speed, params), resolveArg(opts.wobble, params),
            ]),
        ])
}

/** A motion-blur tap trajectory: the path kind plus the prop that anchors it. */
export interface BlurPathSpec {
    readonly kind: motionBlurKit.MotionBlurPathKind
    readonly focus: PropRef
}

/** The tap trajectories {@link motionBlur} can follow. */
export const blurPath = {
    /** Straight-line smear along a fixed direction (an angle prop, in degrees). */
    linear: (angle: PropRef): BlurPathSpec => ({kind: 'linear', focus: angle}),
    /** Arc smear rotating around a center point (a position prop). */
    orbit: (center: PropRef): BlurPathSpec => ({kind: 'orbit', focus: center}),
    /** Radial smear streaking outward from a center point (a position prop). */
    zoom: (center: PropRef): BlurPathSpec => ({kind: 'zoom', focus: center}),
}

/**
 * Motion blur: 32 Gaussian-weighted samples of the child gathered along a tap trajectory —
 * `blurPath.linear` (directional), `blurPath.orbit` (rotational), or `blurPath.zoom` (radial).
 */
export function motionBlur(opts: {path: BlurPathSpec; amount: ArgSpec}): GatherEffect {
    const {path, amount} = opts
    return {
        kind: 'gather',
        build: (params): Expr =>
            motionBlurKit.motionBlurGather(
                path.kind,
                {
                    focus: resolveArg(path.focus, params),
                    amount: resolveArg(amount, params),
                    uv: params.ctx.uv,
                    aspect: params.ctx.aspect,
                    viewportSize: params.ctx.viewportSize,
                },
                (coord) => params.texture.sample(coord),
            ),
    }
}

/**
 * Scatter: displace every pixel by a per-pixel random offset of up to `amount` pixels — a
 * grain-like diffusion, not a Gaussian blur. `edges` binds a structural edge-mode prop
 * ('stretch' | 'transparent' | 'mirror' | 'wrap').
 */
export function scatter(opts: {amount: ArgSpec; edges: PropRef}): GatherEffect {
    const {amount, edges} = opts
    return {
        kind: 'gather',
        build: (params): Expr =>
            motionBlurKit.scatterGather({
                uv: params.ctx.uv,
                amount: resolveArg(amount, params),
                viewportSize: params.ctx.viewportSize,
                edgeMode: (params.propValues[edges.name] as number) ?? 0,
                sample: (coord) => params.texture.sample(coord),
            }),
    }
}

/**
 * Sharpen: a 5-tap unsharp-mask convolution — centre × (1 + 4·amount) minus the four orthogonal
 * one-pixel neighbours × amount. At `amount` 0 the kernel is the identity.
 */
export function sharpen(amount: ArgSpec): GatherEffect {
    return {
        kind: 'gather',
        build: (params): Expr =>
            motionBlurKit.sharpenGather({
                uv: params.ctx.uv,
                viewportSize: params.ctx.viewportSize,
                amount: resolveArg(amount, params),
                sample: (coord) => params.texture.sample(coord),
            }),
    }
}

/**
 * Pixelate: quantise the child to a grid of `scale` cells along the longest edge, sampling each
 * cell once, and cut each cell to a rounded rectangle via `gap` (spacing) and `roundness`
 * (0 = square, 1 = circle) — the cut is an alpha mask, not a coordinate bend.
 */
export function pixelate(opts: {scale: ArgSpec; gap: ArgSpec; roundness: ArgSpec}): GatherEffect {
    const {scale, gap, roundness} = opts
    return {
        kind: 'gather',
        resultAlpha: 'straight',
        build: (params): Expr =>
            motionBlurKit.pixelateGather({
                uv: params.ctx.uv,
                aspect: params.ctx.aspect,
                scale: resolveArg(scale, params),
                gap: resolveArg(gap, params),
                roundness: resolveArg(roundness, params),
                sampleStraight: params.sampleStraight,
            }),
    }
}

/** Sphere-bulge geometry: bulged UV + boundary coverage + surface normal. */
export function sphereFrame(opts: {center: ArgSpec; radius: ArgSpec; depth: ArgSpec}): GatherFrame {
    return frameOf((params) =>
        asLocal(call(warpMaps.sphereBulge, 'sphereBulge', [
            params.ctx.uv, params.ctx.viewportSize,
            resolveArg(opts.center, params), resolveArg(opts.radius, params), resolveArg(opts.depth, params),
        ]), 'sphere'))
}

/**
 * One Catmull-Rom tap at the frame's warped UV, unpremultiplied. Catmull-Rom, not bilinear: a
 * magnifying warp blows up its centre, and one bilinear tap turns any hard edge underneath
 * (text, a logo) into visible facets.
 */
export function crispTap(frame: GatherFrame): SampleStage {
    return (params) =>
        call(blend.unpremultiplyAlpha, 'unpremultiplyAlpha', [
            sampling.sampleCatmullRomExpr(params.texture, frame(params).member('uv')),
        ])
}

/**
 * Directional fresnel rim over the frame's surface normal — the tinted rim is ADDED and the
 * alpha gated by the frame's boundary coverage. Added color is what a uv+mask analytic fold
 * can't carry, so a recipe with this stage stays on the RTT fragment path.
 */
export function rimLit(frame: GatherFrame, opts: {
    position: ArgSpec
    intensity: ArgSpec
    softness: ArgSpec
    color: PropRef
}): OverlayStage {
    return (color, params) => {
        const rim = call(warpMaps.rimLight, 'rimLight', [
            frame(params).member('normal'), resolveArg(opts.position, params),
            resolveArg(opts.intensity, params), resolveArg(opts.softness, params),
        ])
        return call(warpMaps.rimComposite, 'rimComposite', [
            color, resolveArg(opts.color, params).member('rgb'), rim, frame(params).member('coverage'),
        ])
    }
}

/**
 * The shared glitch geometry: burst pulse, jitter bands, block shifts, and mirror flips — its
 * burst/band hashes feed the split, fills, and scanlines. Animated by the GLOBAL clock scaled
 * by `speed` (not a per-node animated time).
 */
export function glitchFrame(opts: {
    intensity: ArgSpec
    speed: ArgSpec
    blockDensity: ArgSpec
    mirrorAmount: ArgSpec
}): GatherFrame {
    return frameOf((params) =>
        asLocal(call(motionBlurKit.glitchGeom, 'glitchGeom', [
            params.ctx.uv, params.ctx.time,
            resolveArg(opts.intensity, params), resolveArg(opts.speed, params),
            resolveArg(opts.blockDensity, params), resolveArg(opts.mirrorAmount, params),
        ]), 'glitch'))
}

/**
 * RGB-split sampling around the frame's mirrored coordinate. Alpha comes from the centred green
 * tap, so a transparent child stays transparent through the glitch.
 */
export function glitchRgbSplit(frame: GatherFrame, opts: {shift: ArgSpec}): SampleStage {
    return (params) => {
        const geom = frame(params)
        const splitUVs = asLocal(call(motionBlurKit.glitchSplitUVs, 'glitchSplitUVs', [
            geom.member('mirroredUV'), resolveArg(opts.shift, params), geom.member('spread'),
        ]), 'splitUVs')
        const greenSample = params.texture.sample(geom.member('mirroredUV'))
        return vec4(
            params.texture.sample(splitUVs.member('xy')).member('r'),
            greenSample.member('g'),
            params.texture.sample(splitUVs.member('zw')).member('b'),
            greenSample.member('a'),
        )
    }
}

/** Vivid color-bar fills in the frame's active glitch blocks. */
export function colorBarFills(frame: GatherFrame, opts: {intensity: ArgSpec}): OverlayStage {
    return (color, params) => {
        const geom = frame(params)
        return call(motionBlurKit.fillBarsShade, 'fillBarsShade', [
            color, params.ctx.uv, geom.member('bandY1'), geom.member('slowFrame'),
            resolveArg(opts.intensity, params), geom.member('strength'),
            geom.member('bandGate'), geom.member('blockGate'),
        ])
    }
}

/** CRT-style scanlines confined to the frame's distorted regions. */
export function distortedScanlines(frame: GatherFrame, opts: {intensity: ArgSpec}): OverlayStage {
    return (color, params) =>
        call(motionBlurKit.distortScanShade, 'distortScanShade', [
            color, params.ctx.uv.member('y'), params.ctx.viewportSize.member('y'),
            resolveArg(opts.intensity, params), frame(params).member('distortion'),
        ])
}

// ── Compute-backed blurs ────────────────────────────────────────────────────────────────────
//
// These ride the kit's compute lifecycle wrappers (`withFixedBlurCompute` / `withVariableBlurCompute`
// / `withBloomCompute`) rather than the gather scaffold, so each noun returns BOTH halves for a
// custom-tier definition — spread it into the definition: `...gaussianBlur({intensity: p('intensity')})`.

/** The two halves a compute-backed blur contributes to a custom-tier definition. */
export interface ComputeBackedEffect {
    compute: GpuComputeNode
    gpu: {fragment: (params: GpuFragmentParams) => Expr}
}

/**
 * Gaussian blur: a 2-pass separable Gaussian at fixed compute resolution (decoupled from canvas).
 * A static/mouse/auto intensity runs the fixed-kernel path with a live per-frame radius read; an
 * intensity bound to a map fills a per-pixel radius map from the map source and runs the variable
 * Gaussian. color comes from the blurred buffer, alpha from the sharp child.
 */
export function gaussianBlur(opts: {intensity: PropRef}): ComputeBackedEffect {
    const prop = opts.intensity.name
    return {
        compute: (params) => {
            const {getCpuValue, getMapInfo} = params
            const mapInfo = getMapInfo(prop)
            if (mapInfo) {
                return blurKit.withVariableBlurCompute(params, {
                    source: mapInfo,
                    buildFill: (cw, ch) => blurKit.buildFillBlurMapGraph(cw, ch, mapInfo.channel as blurKit.BlurMapChannel),
                    // inputWidth/Height map compute px → map-source canvas px; the window is the live
                    // post-dynamic-bound remap, matching the fragment path's `_map_<prop>_*` uniforms.
                    fillValues: (dims, window) => ({
                        inputWidth: dims.width,
                        inputHeight: dims.height,
                        ...blurKit.remapWindowValues(window!),
                    }),
                })
            }
            // Static intensity: fixed-kernel Gaussian (uniform radius, live per-frame CPU read so a
            // mouse / auto driver pulls through without a recompose).
            return blurKit.withFixedBlurCompute(params, {
                radius: () => {
                    const raw = getCpuValue(prop)
                    return typeof raw === 'number' ? blurKit.intensityToRadius(raw) : 4
                },
            })
        },
        gpu: {
            fragment: (params): Expr =>
                blurKit.composeBlurredOverSharp(params, (blurred, sharp) => vec4(blurred.member('rgb'), sharp.member('a'))),
        },
    }
}

/**
 * Channel blur: one fixed Gaussian at the MAX per-channel radius, with each channel mixed between
 * the sharp source and the blurred buffer by `channelRadius / maxRadius` — the max-intensity
 * channel takes the full blur, a zero channel stays exactly sharp.
 */
export function channelBlur(opts: {red: PropRef; green: PropRef; blue: PropRef}): ComputeBackedEffect {
    const {red, green, blue} = opts
    return {
        compute: (params) =>
            blurKit.withFixedBlurCompute(params, {
                // Live per-frame per-channel radii → the fixed Gaussian runs at their max.
                radius: () => {
                    const r = ((params.getCpuValue(red.name) as number) ?? 0) * blurKit.CHANNEL_INTENSITY_TO_RADIUS
                    const g = ((params.getCpuValue(green.name) as number) ?? 0) * blurKit.CHANNEL_INTENSITY_TO_RADIUS
                    const b = ((params.getCpuValue(blue.name) as number) ?? 0) * blurKit.CHANNEL_INTENSITY_TO_RADIUS
                    return Math.max(r, g, b, 0.01)
                },
            }),
        gpu: {
            fragment: (params): Expr =>
                blurKit.composeBlurredOverSharp(params, (blurred, sharp) =>
                    call(blurKit.channelBlurCompose, 'channelBlurCompose', [
                        sharp, blurred, uniformOf(red, params), uniformOf(green, params), uniformOf(blue, params),
                    ])),
        },
    }
}

/**
 * Progressive blur: a variable-radius Gaussian whose per-pixel radius ramps directionally from
 * `center` along `angle` over `falloff`, up to the intensity's max radius. A map-driven intensity
 * samples the per-pixel max radius from the map source instead.
 */
export function progressiveBlur(opts: {
    intensity: PropRef
    angle: PropRef
    center: PropRef
    falloff: PropRef
}): ComputeBackedEffect {
    const {intensity, angle, center, falloff} = opts
    return {
        compute: (params) => {
            const {getCpuValue, getMapInfo} = params

            // Live per-frame geometry → fill-map params. `getCpuValue(center)` is the POST-transform
            // VectorFieldView (its `.y` is `1 - authoredY`; the kernel recovers y). angle is degrees.
            const readGeometry = (dims: {width: number; height: number}) => {
                const centerValue = getCpuValue(center.name) as {x?: number; y?: number} | undefined
                return {
                    angle: (getCpuValue(angle.name) as number) ?? 0,
                    centerX: typeof centerValue?.x === 'number' ? centerValue.x : 0,
                    centerY: typeof centerValue?.y === 'number' ? centerValue.y : 0.5,
                    falloff: (getCpuValue(falloff.name) as number) ?? 1,
                    aspect: dims.width / dims.height,
                }
            }

            // Intensity bound to a map → per-pixel max radius sampled from the map source.
            const mapInfo = getMapInfo(intensity.name)
            if (mapInfo) {
                return blurKit.withVariableBlurCompute(params, {
                    source: mapInfo,
                    buildFill: (cw, ch) => blurKit.buildProgressiveBlurFillMapGraph(cw, ch, mapInfo.channel as blurKit.BlurMapChannel),
                    fillValues: (dims, window) => ({
                        ...readGeometry(dims),
                        inputWidth: dims.width,
                        inputHeight: dims.height,
                        ...blurKit.remapWindowValues(window!),
                    }),
                })
            }

            // Static / mouse / auto intensity: single uniform max radius (per-frame CPU value).
            return blurKit.withVariableBlurCompute(params, {
                buildFill: blurKit.buildProgressiveBlurFillGraph,
                fillValues: (dims) => ({
                    ...readGeometry(dims),
                    maxRadius: blurKit.intensityToRadius((getCpuValue(intensity.name) as number) ?? 0),
                }),
            })
        },
        gpu: {
            // The variable blur already applied the per-pixel ramp → color from the blurred buffer,
            // alpha from the sharp child (blur can spread alpha at edges).
            fragment: (params): Expr =>
                blurKit.composeBlurredOverSharp(params, (blurred, sharp) => vec4(blurred.member('rgb'), sharp.member('a'))),
        },
    }
}

/**
 * Tilt shift: a variable-radius Gaussian whose per-pixel radius ramps with perpendicular distance
 * from a focus line (halfKernel 14 — the ~36px max radius keeps tap spacing under the banding
 * threshold with ~40% fewer taps). The fragment re-derives the same blur amount to mix the
 * canvas-res sharp source against the compute-res blurred buffer, so in-focus pixels stay crisp.
 */
export function tiltShift(opts: {
    intensity: PropRef
    width: PropRef
    falloff: PropRef
    angle: PropRef
    center: PropRef
}): ComputeBackedEffect {
    const {intensity, width, falloff, angle, center} = opts
    return {
        compute: (params) => {
            const {getCpuValue, getMapInfo} = params

            // Live per-frame geometry → fill-map params (same POST-transform center convention as
            // progressiveBlur).
            const readGeometry = (dims: {width: number; height: number}) => {
                const centerValue = getCpuValue(center.name) as {x?: number; y?: number} | undefined
                return {
                    angle: (getCpuValue(angle.name) as number) ?? 0,
                    centerX: typeof centerValue?.x === 'number' ? centerValue.x : 0.5,
                    centerY: typeof centerValue?.y === 'number' ? centerValue.y : 0.5,
                    width: (getCpuValue(width.name) as number) ?? 0.3,
                    falloff: (getCpuValue(falloff.name) as number) ?? 0.3,
                    aspect: dims.width / dims.height,
                }
            }

            const mapInfo = getMapInfo(intensity.name)
            if (mapInfo) {
                return blurKit.withVariableBlurCompute(params, {
                    halfKernel: 14,
                    source: mapInfo,
                    buildFill: (cw, ch) => blurKit.buildTiltShiftFillMapGraph(cw, ch, mapInfo.channel as blurKit.BlurMapChannel),
                    fillValues: (dims, window) => ({
                        ...readGeometry(dims),
                        inputWidth: dims.width,
                        inputHeight: dims.height,
                        ...blurKit.remapWindowValues(window!),
                    }),
                })
            }

            return blurKit.withVariableBlurCompute(params, {
                halfKernel: 14,
                buildFill: blurKit.buildTiltShiftFillGraph,
                fillValues: (dims) => ({
                    ...readGeometry(dims),
                    maxRadius: blurKit.intensityToRadius((getCpuValue(intensity.name) as number) ?? 0),
                }),
            })
        },
        gpu: {
            // Recompute the focus-line blur amount per fragment (cheap), then mix sharp ↔ blurred.
            fragment: (params): Expr =>
                blurKit.composeBlurredOverSharp(params, (blurred, sharp) => {
                    const blurAmount = call(blurKit.tiltShiftBlurAmount, 'tiltShiftBlurAmount', [
                        uniformOf(angle, params), uniformOf(center, params), uniformOf(width, params),
                        uniformOf(falloff, params), params.ctx.uv, params.ctx.aspect,
                    ])
                    return mixExpr(sharp, blurred, blurAmount)
                }),
        },
    }
}

/**
 * Bloom (Glow): bright-extract the child above `threshold` at aspect-aware compute resolution,
 * blur the extract by `size` (per-pixel from the map source when size carries a map driver), and
 * composite `original + bloom × intensity` with the halo allowed to extend past the child's alpha.
 * A scalar size of 0 skips compute entirely (pair it with `recompile: crosses(0)` on the size prop).
 */
export function bloom(opts: {intensity: PropRef; threshold: PropRef; size: PropRef}): ComputeBackedEffect {
    const {intensity, threshold, size} = opts
    return {
        compute: (params) => {
            const {getCpuValue, getMapInfo} = params
            const mapInfo = getMapInfo(size.name)
            if (!mapInfo && ((getCpuValue(size.name) as number) ?? 0) === 0) return null // size=0 bypass (scalar path only).

            return blurKit.withBloomCompute(params, {
                mapInfo,
                buildExtract: blurKit.buildGlowPrepassGraph,
                buildExtractMap: blurKit.buildGlowPrepassMapGraph,
                threshold: () => (getCpuValue(threshold.name) as number) ?? 0.5,
                radius: () => (getCpuValue(size.name) as number) ?? 25,
            })
        },
        gpu: {
            // Composite at canvas resolution: original from the canvas-res child RTT (sharp); bloom
            // from the compute-res buffer (intentionally soft).
            fragment: (params): Expr =>
                blurKit.composeBlurredOverSharp(params, (bloomSample, original) =>
                    call(blurKit.glowCompose, 'glowCompose', [original, bloomSample, uniformOf(intensity, params)])),
        },
    }
}

/** The plain centre tap of the child RTT (premultiplied). */
export function childTap(): SampleStage {
    return (params) => params.texture.sample(params.ctx.uv)
}

/** The compass-angle shadow displacement: where to read the child's silhouette from. */
export function shadowOffset(opts: {angle: ArgSpec; distance: ArgSpec}): (params: RttFilterParams) => Expr {
    return (params) =>
        call(motionBlurKit.dropShadowUV, 'dropShadowUV', [
            params.ctx.uv, params.ctx.viewportSize, resolveArg(opts.angle, params), resolveArg(opts.distance, params),
        ])
}

/** Two-pass separable Gaussian over the child's alpha silhouette, read at the `at` coordinate. */
export function silhouetteCoverage(opts: {at: (params: RttFilterParams) => Expr; blur: ArgSpec}): (params: RttFilterParams) => Expr {
    return (params) =>
        motionBlurKit.silhouetteBlur({
            at: opts.at(params),
            uv: params.ctx.uv,
            viewportSize: params.ctx.viewportSize,
            blurRadius: resolveArg(opts.blur, params),
            sample: (coord) => params.texture.sample(coord),
            convertToTexture: params.convertToTexture,
        })
}

/**
 * Tint the coverage `color` × `intensity` and composite the shadow behind the child — or, with
 * `cutout` (a structural boolean prop), show only the shadow with the child's silhouette
 * punched out.
 */
export function shadowComposite(opts: {
    coverage: (params: RttFilterParams) => Expr
    color: PropRef
    intensity: ArgSpec
    cutout: PropRef
}): OverlayStage {
    return (original, params) => {
        const shadowAlpha = opts.coverage(params).mul(resolveArg(opts.intensity, params))
        return call(motionBlurKit.dropShadowComposite, 'dropShadowComposite', [
            original, resolveArg(opts.color, params), shadowAlpha,
            floatE((params.propValues[opts.cutout.name] as number) === 1 ? 1 : 0),
        ])
    }
}

// ── Warp-shading recipes ────────────────────────────────────────────────────────────────────

/** Corner names → screenUV corner + opposite corner (y=0 top). Baked as compile-time literals. */
const PEEL_CORNERS: Record<string, {corner: [number, number]; opposite: [number, number]}> = {
    'top-left': {corner: [0, 0], opposite: [1, 1]},
    'top-right': {corner: [1, 0], opposite: [0, 1]},
    'bottom-left': {corner: [0, 1], opposite: [1, 0]},
    'bottom-right': {corner: [1, 1], opposite: [0, 0]},
}

/**
 * Fluted-glass geometry: refracted UV + chromatic offset + flute slope. `shape` binds a
 * compile-time prop baking the slope-exponent endpoints + waves flag. The flute pattern drifts
 * by the definition's animated clock (declare `animatedTime: {speed: 'speed'}`), negated so
 * positive speed drifts visually right at angle 0.
 */
export function fluteFrame(opts: {
    shape: PropRef
    angle: ArgSpec
    frequency: ArgSpec
    softness: ArgSpec
    waveAmplitude: ArgSpec
    waveFrequency: ArgSpec
    refraction: ArgSpec
    aberration: ArgSpec
}): GatherFrame {
    return frameOf((params) => {
        const shapeId = (params.propValues[opts.shape.name] as number) ?? 0
        const expHi = shapeId === 0 ? 16 : 8 // bars vs rounded/waves
        const expLo = shapeId === 0 ? 4 : 3
        const wavesFlag = shapeId === 2 ? 1 : 0
        // Speed negated so positive speed drifts +u (visually right at angle=0).
        const t = animatedTime(params).mul(-1)
        return asLocal(call(warpMaps.flutedGlassGeom, 'flutedGlassGeom', [
            params.ctx.uv, params.ctx.aspect, t,
            resolveArg(opts.angle, params), resolveArg(opts.frequency, params), resolveArg(opts.softness, params),
            resolveArg(opts.waveAmplitude, params), resolveArg(opts.waveFrequency, params),
            resolveArg(opts.refraction, params), resolveArg(opts.aberration, params),
            expr(`vec3f(${formatFloat(expHi)}, ${formatFloat(expLo)}, ${formatFloat(wavesFlag)})`),
        ]), 'flute')
    })
}

/**
 * The refracted tap(s) at the frame's UV, edge-clipped and unpremultiplied. `edges` binds the
 * compile-time sampling branch; `aberration` is a uniform whose on/off crossing is structural —
 * one tap vs a 3-tap chromatic split along the frame's offset (pair it with a crosses-0
 * recompile rule).
 */
export function refractedTaps(frame: GatherFrame, opts: {aberration: PropRef; edges: PropRef}): SampleStage {
    return (params) => {
        const edgeMode = (params.propValues[opts.edges.name] as number) ?? 2
        const aberrationEnabled = ((params.propValues[opts.aberration.name] as number) ?? 0.2) > 0
        const geom = frame(params)
        const sampleAt = (uvE: Expr): Expr =>
            warpMaps.edgeClipSample((uv) => params.texture.sample(uv), uvE, edgeMode)
        const refractedUV = geom.member('refractedUV')
        const sampled = aberrationEnabled
            ? warpMaps.rgbSplitTaps(sampleAt, refractedUV, geom.member('chrOff'))
            : sampleAt(refractedUV)
        // RTT children come back premultiplied; the shading stages run on straight color.
        return call(blend.unpremultiplyAlpha, 'unpremultiplyAlpha', [sampled])
    }
}

/**
 * Blinn specular over the frame's flute slope, tinted and added on top — weighted by alpha so
 * transparent regions pick up no phantom highlight.
 */
export function fluteHighlight(frame: GatherFrame, opts: {
    lightAngle: ArgSpec
    highlight: ArgSpec
    softness: ArgSpec
    color: PropRef
}): OverlayStage {
    return (color, params) => {
        const spec = call(warpMaps.blinnHighlight, 'blinnHighlight', [
            frame(params).member('slope'),
            resolveArg(opts.lightAngle, params), resolveArg(opts.highlight, params), resolveArg(opts.softness, params),
        ])
        const specA = spec.mul(color.member('a'))
        const litRgb = color.member('rgb').add(resolveArg(opts.color, params).member('rgb').mul(specA))
        return vec4(litRgb, color.member('a'))
    }
}

/** A scalar shading term of a peel recipe, read against the peel frame. */
export type PeelShade = (params: RttFilterParams) => Expr

/**
 * Page-peel geometry: curl UV, fold angle, crease distances + shadow reaches. `corner` binds the
 * compile-time cpu-only STRING prop (no transform — an inline transform would make the bridge
 * write the string into an f32 field); its geometry lands as literal args.
 */
export function peelFrame(opts: {corner: PropRef; amount: ArgSpec; radius: ArgSpec}): GatherFrame {
    return frameOf((params) => {
        const cornerGeom = PEEL_CORNERS[(params.propValues[opts.corner.name] as string) ?? 'bottom-right'] ?? PEEL_CORNERS['bottom-right']
        return asLocal(call(warpMaps.pagePeelGeom, 'pagePeelGeom', [
            params.ctx.uv, params.ctx.aspect,
            floatE(cornerGeom.corner[0]), floatE(cornerGeom.corner[1]),
            floatE(cornerGeom.opposite[0]), floatE(cornerGeom.opposite[1]),
            resolveArg(opts.amount, params), resolveArg(opts.radius, params),
        ]), 'peel')
    })
}

/** Crook darkening down the underside of the curl. */
export function curlShading(frame: GatherFrame, opts: {shading: ArgSpec}): PeelShade {
    return (params) =>
        call(warpMaps.curlShade, 'curlShade', [frame(params).member('theta'), resolveArg(opts.shading, params)])
}

/** The specular band running up the curl. */
export function curlSheen(frame: GatherFrame, opts: {highlight: ArgSpec; softness: ArgSpec}): PeelShade {
    return (params) =>
        call(warpMaps.curlSheen, 'curlSheen', [
            frame(params).member('theta'), resolveArg(opts.highlight, params), resolveArg(opts.softness, params),
        ])
}

/** The contact shadow the curl casts on the flat page past the crease. */
export function overhangShadow(frame: GatherFrame, opts: {amount: ArgSpec; shadow: ArgSpec}): PeelShade {
    return (params) => {
        const g = frame(params)
        return call(warpMaps.overhangShade, 'overhangShade', [
            g.member('distPastCrease'), g.member('flatReach'),
            resolveArg(opts.amount, params), resolveArg(opts.shadow, params),
        ])
    }
}

/** The shadow on the revealed backing just inside the crease. */
export function revealShadow(frame: GatherFrame, opts: {amount: ArgSpec; shadow: ArgSpec}): PeelShade {
    return (params) => {
        const g = frame(params)
        return call(warpMaps.revealShadow, 'revealShadow', [
            g.member('distIntoPeel'), g.member('peelReach'),
            resolveArg(opts.amount, params), resolveArg(opts.shadow, params),
        ])
    }
}

/**
 * The fold composite — cast shadow, shaded flat page, lit curl, back to front. Samples the child
 * at the frame's curl UV (the lip) and at the screen UV (the flat page). Premultiplied — the
 * gather species appends the unpremultiply tail.
 */
export function peelCompose(frame: GatherFrame, parts: {
    shade: PeelShade
    sheen: PeelShade
    overhang: PeelShade
    reveal: PeelShade
}): SampleStage {
    return (params) => {
        const g = frame(params)
        const curlSample = params.texture.sample(g.member('curlUV'))
        const flatSample = params.texture.sample(params.ctx.uv)
        return call(warpMaps.pagePeelCompose, 'pagePeelCompose', [
            curlSample, flatSample,
            parts.shade(params), parts.sheen(params), parts.overhang(params), parts.reveal(params),
            g.member('foldBlend'), g.member('showCurl'),
        ])
    }
}

// ── Planar reflection ───────────────────────────────────────────────────────────────────────
//
// The mirror-floor recipe: reflect the content across a horizontal line, blur the mirror image
// progressively by its depth below the line, and fade it out with distance. Three stage parts —
// the reflected coordinate, the depth-ramped variable blur (compute), and the over-composite.

/**
 * Reflected sample coordinate: mirror across the horizontal line at `lineY`. Y is down (top=0),
 * so a point at uv.y > lineY reflects to `2·lineY - uv.y`. Pure function.
 */
export const mirrorRowUV = tgpu.fn([d.vec2f, d.f32], d.vec2f)((uv, lineY) => {
    'use gpu'
    return d.vec2f(uv.x, lineY * 2.0 - uv.y)
})

/**
 * Planar-reflection composite (pre-unpremultiply). Above the line returns the original content
 * untouched; below it, the (edge-handled) reflection sample composited "over" the original via
 * premultiplied alpha, faded by `visibility` (full up to distance−fadeWidth, smoothly to 0 by
 * `distance`; `falloff` is the fade-zone width as a fraction of distance). Pure function.
 */
export const planarReflectionCompose = tgpu.fn([d.vec4f, d.vec4f, d.f32, d.f32, d.f32, d.f32], d.vec4f)(
    (original, refl, uvY, lineY, distance, falloff) => {
        'use gpu'
        const distBelow = std.max(uvY - lineY, 0.0)
        const distanceMax = std.max(distance, 0.001)
        const fadeWidth = std.max(distanceMax * falloff, 0.0001)
        const fadeStart = distanceMax - fadeWidth
        const visibility = 1.0 - std.smoothstep(fadeStart, distanceMax, distBelow)

        const effReflA = refl.w * visibility
        const oneMinusReflA = 1.0 - effReflA
        const belowRgb = refl.xyz.mul(visibility).add(original.xyz.mul(oneMinusReflA))
        const belowA = effReflA + original.w * oneMinusReflA
        const belowLine = d.vec4f(belowRgb, belowA)

        const isBelowLine = std.step(lineY, uvY)
        return std.mix(original, belowLine, d.vec4f(isBelowLine))
    },
)

/** Depth-ramp blur-map units: blur slider 1.0 → this many source pixels of Gaussian radius. */
const DEPTH_RAMP_RADIUS_PX = 12

/** Depth-ramp fill kernel params: line height + blur amount + (clamped) blur ramp distance. */
function depthRampFillParams() {
    return d.struct({height: d.f32, blurAmount: d.f32, blurDistance: d.f32})
}

/**
 * GPU-free construction of the depth-ramp blur-map fill graph. Per compute pixel, the desired
 * blur radius ramps with the row's depth below the line:
 * `blur × smoothstep(0, blurDistance, height − v) × DEPTH_RAMP_RADIUS_PX` (source pixels).
 * Row-based (depends only on cy/computeHeight); 2D dispatch, STORAGE textureStore.
 */
export function buildDepthRampFillGraph(computeWidth: number, computeHeight: number) {
    void computeWidth // row-based fill; width kept for buildFill API parity.
    const Params = depthRampFillParams()
    const layout = tgpu.bindGroupLayout({
        blurMap: {storageTexture: d.textureStorage2d(blurKit.BLUR_MAP_FORMAT, 'write-only')},
        params: {uniform: Params},
    })

    const kernel = tgpu.fn([d.u32, d.u32])((cx, cy) => {
        'use gpu'
        const p = layout.$.params
        // Y-down v — same convention as screenUV + the height prop.
        const v = (d.f32(cy) + 0.5) / computeHeight
        const distBelowAtOutput = std.max(p.height - v, 0.0)
        const ramp = std.smoothstep(0.0, p.blurDistance, distBelowAtOutput)
        const radius = p.blurAmount * ramp * DEPTH_RAMP_RADIUS_PX
        std.textureStore(layout.$.blurMap, d.vec2u(cx, cy), d.vec4f(radius, 0.0, 0.0, 1.0))
    }).$name('depthRampFillBlurMap')

    return {layout, kernel, Params}
}

/** The reflected sample coordinate stage (identity above the line; the composite gates on it). */
export function mirrorAcrossRow(line: PropRef): (params: GpuFragmentParams) => Expr {
    return ({ctx, uniforms}) => call(mirrorRowUV, 'mirrorRowUV', [ctx.uv, uniforms[line.name]])
}

/**
 * Depth-ramped variable blur (compute): RTT the child, fill a per-pixel radius map from the
 * line geometry, run kit/blur's variable Gaussian. Map-driven slots use their static scalar
 * (compute maps don't drive these per-pixel — the Blur/ProgressiveBlur precedent).
 */
export function depthRampBlur(opts: {line: PropRef; blur: PropRef; blurDistance: PropRef; halfKernel: number}): {compute: GpuComputeNode} {
    return {
        compute: (params: GpuFragmentParams) => blurKit.withVariableBlurCompute(params, {
            halfKernel: opts.halfKernel,
            buildFill: buildDepthRampFillGraph,
            fillValues: () => ({
                height: (params.getCpuValue(opts.line.name) as number) ?? 0.5,
                blurAmount: (params.getCpuValue(opts.blur.name) as number) ?? 1,
                // blurDistance is floored at 0.01 (smoothstep is undefined at edge0 == edge1).
                blurDistance: Math.max(0.01, (params.getCpuValue(opts.blurDistance.name) as number) ?? 0.5),
            }),
        }),
    }
}

/** The over-composite stage: the (edge-handled) reflection tap composited "over" the original
 *  below the line, faded to transparent over `distance`/`falloff`. */
export function planarReflection(
    opts: {line: PropRef; distance: PropRef; falloff: PropRef; edges: PropRef},
    original: Expr,
    reflection: (uv: Expr) => Expr,
    reflectedUV: (params: GpuFragmentParams) => Expr,
): (params: GpuFragmentParams) => Expr {
    return (params) => {
        const {ctx, uniforms, propValues} = params
        const edgeMode = (propValues[opts.edges.name] as number) ?? 0
        const refl = edges.applyEdgeHandlingExpr(reflectedUV(params), reflection, edgeMode)
        return call(planarReflectionCompose, 'planarReflectionCompose', [
            original, refl, ctx.uv.member('y'), uniforms[opts.line.name], uniforms[opts.distance.name], uniforms[opts.falloff.name],
        ])
    }
}

// ── Screened highlight bloom + frame sway ───────────────────────────────────────────────────

/**
 * Screen a tinted glow (blurred bright highlights) over a base color.
 * `tintStrength = (rgb tint, strength)`. The glow's coverage extends the alpha into transparent
 * areas (Glow's aura precedent) so the halo isn't clipped to the child's α.
 */
export const tintedScreenGlow = tgpu.fn([d.vec4f, d.vec4f, d.vec4f], d.vec4f)(
    (base, glow, tintStrength) => {
        'use gpu'
        const strength = tintStrength.w
        const bloomC = std.clamp(
            glow.xyz.mul(tintStrength.xyz).mul(strength),
            d.vec3f(0.0, 0.0, 0.0), d.vec3f(1.0, 1.0, 1.0),
        )
        const one = d.vec3f(1.0, 1.0, 1.0)
        const screened = one.sub(one.sub(base.xyz).mul(one.sub(bloomC)))
        const a = base.w + glow.w * strength * (1.0 - base.w)
        return d.vec4f(screened.x, screened.y, screened.z, std.clamp(a, 0.0, 1.0))
    },
).$name('tintedScreenGlow')

/**
 * Screened-bloom recipe: the compute half is the kit's highlight-bloom mechanism
 * (`withBloomCompute`: bright-extract + variable-Gaussian blur at capped compute resolution);
 * the screen half tints the blurred highlights and screens them back over the graded color
 * (no-op when the bloom buffer is absent — GPU-free resolve, or `strength` = 0, which returns
 * null from `compute`; pair the strength prop with `recompile: crosses(0)`).
 */
export function screenedBloom(slots: {
    strength: PropRef
    radius: PropRef
    tint: [number, number, number]
    threshold: number
    output: string
    extractName: string
}): {
    compute: GpuComputeNode
    screen: (base: Expr, at: Expr, params: GpuFragmentParams) => Expr
} {
    return {
        compute: (params) => {
            const {getCpuValue} = params
            if (((getCpuValue(slots.strength.name) as number) ?? 0) === 0) return null // bloom off → no compute.

            return blurKit.withBloomCompute(params, {
                outputKey: slots.output,
                buildExtract: (w, h) => blurKit.buildBloomExtractGraph(w, h, slots.extractName),
                threshold: () => slots.threshold,
                radius: () => (getCpuValue(slots.radius.name) as number) ?? 28,
            })
        },
        screen: (base, at, params) => {
            const glowTex = params.computeOutputs?.[slots.output] as KitTexture | undefined
            const glowSample = glowTex ? glowTex.sample(at) : expr('vec4f(0.0, 0.0, 0.0, 0.0)')
            const tintStrength = vec4(slots.tint[0], slots.tint[1], slots.tint[2], params.uniforms[slots.strength.name])
            return call(tintedScreenGlow, 'tintedScreenGlow', [base, glowSample, tintStrength])
        },
    }
}

/**
 * Frame sway — a tiny animated translation + rotation about the frame centre, like an unsteady
 * projector gate. `amount` scales both amplitude and (via the gated clock) presence; `t` is the
 * per-frame sway-time accumulator. Returns the UV to sample the content at (a clamp sampler
 * handles the sliver pushed off-frame). amount=0 → returns `uv` unchanged.
 */
export const swayUV = tgpu.fn([d.vec2f, d.f32, d.f32], d.vec2f)(
    (uv, amount, t) => {
        'use gpu'
        const amp = amount * 0.006
        const ox = (std.sin(t * 1.7) * 0.5 + std.sin(t * 0.9 + 1.3) * 0.5) * amp
        const oy = (std.sin(t * 1.3 + 0.7) * 0.5 + std.sin(t * 2.3) * 0.5) * amp
        const ang = std.sin(t * 0.6) * amount * 0.004
        const c = std.cos(ang)
        const s = std.sin(ang)
        const p = uv.sub(d.vec2f(0.5, 0.5))
        const rx = p.x * c - p.y * s
        const ry = p.x * s + p.y * c
        return d.vec2f(rx + 0.5 + ox, ry + 0.5 + oy)
    },
).$name('swayUV')

/**
 * The frame-sway stage: a gated clock (the `field` extraField — advances only while `amount` > 0,
 * freezes rather than snapping back at 0) drives a tiny animated translation + rotation of the
 * content sample UV. Declare the extraField `{schema: f32, initial: 0}` on the definition.
 */
export function frameSway(slots: {amount: PropRef; field: string}): (params: GpuFragmentParams) => Expr {
    return (params) => {
        let acc = 0
        params.onBeforeRender(({deltaTime}) => {
            const w = (params.getCpuValue(slots.amount.name) as number) ?? 0
            acc += deltaTime * (w > 0 ? 1 : 0)
            params.setExtraField(slots.field, acc)
        })
        return call(swayUV, 'swayUV', [params.ctx.uv, params.uniforms[slots.amount.name], params.uniforms[slots.field]])
    }
}

// ── Bokeh defocus ───────────────────────────────────────────────────────────────────────────
//
// The photographic lens-blur noun over the kit's aperture-table gather: RTT the composed child,
// run ONE scatter-as-gather pass over the CPU-precomputed aperture tap table at aspect-aware
// compute resolution, and bilinear-sample the defocused buffer at canvas resolution (true
// defocus — color AND coverage spread, so silhouettes soften like a real lens). The gather is
// in input-pixel space so discs stay round regardless of compute resolution.

const BOKEH_DEG_TO_RAD = constants.DEG_TO_RAD
const BOKEH_GATHER_FORMAT = 'rgba16float' as const

/** BakedTable part — the aperture tap table: regenerate + rewrite the uniform only when the
 *  (shape, blades) key changes. */
function apertureTable(
    opts: {shape: PropRef; blades: PropRef},
    getCpuValue: GpuFragmentParams['getCpuValue'],
    tapsUniform: {write: (value: d.v4f[]) => void},
): () => void {
    let key = ''
    return () => {
        const shape = (getCpuValue(opts.shape.name) as string) ?? 'blades'
        const count = (getCpuValue(opts.blades.name) as number) ?? 6
        const next = `${shape}|${count}`
        if (next === key) return
        key = next
        tapsUniform.write(blurKit.generateBokehTaps(shape, count, blurKit.BOKEH_TAP_COUNT).map((tap) => d.vec4f(tap.x, tap.y, tap.rim, 0)))
    }
}

/**
 * The bokeh-defocus recipe: the aperture-table gather compute (uniform-radius, or the map-driven
 * fork when `radius` binds a spatial map — mouse/auto radius stays a per-frame scalar) + the
 * defocused-buffer sampling fragment. The aperture (shape + blades) is fully RUNTIME: changing
 * it regenerates the CPU tap table and rewrites one small uniform — no recompile. Compute
 * unavailable (GPU-free resolve / no device) → sharp unpremultiplied passthrough.
 */
export function bokehDefocus(opts: {
    radius: PropRef
    gain: PropRef
    threshold: PropRef
    shape: PropRef
    blades: PropRef
    rotation: PropRef
    fringe: PropRef
}): {compute: GpuComputeNode; gpu: {fragment: (params: GpuFragmentParams) => Expr}} {
    return {
        compute: (params: GpuFragmentParams) => {
            const {childNode, gpu, convertToTexture, registerComputeTexture, getCpuValue, getMapInfo, onCleanup, onResize, dimensions} = params
            if (!childNode) return null
            const root = gpu?.root
            if (!root) return null // GPU-free resolve/tests: fragment falls back to sharp passthrough.

            const childTexture = convertToTexture(childNode)
            // `dimensions` is already the device-pixel backing size (the renderer owns DPR).
            let curWidth = Math.max(1, Math.round(dimensions.width))
            let curHeight = Math.max(1, Math.round(dimensions.height))

            // Aspect-aware compute resolution: cap the LONGER edge at the default long edge and derive the
            // other from the canvas aspect (a fixed 1024×640 would squash the discs on non-16:10 canvases).
            const LONG_EDGE = Math.max(blurKit.DEFAULT_COMPUTE_WIDTH, blurKit.DEFAULT_COMPUTE_HEIGHT)
            const aspect = curHeight > 0 ? curWidth / curHeight : 1
            const computeWidth = Math.max(8, aspect >= 1 ? LONG_EDGE : Math.round(LONG_EDGE * aspect))
            const computeHeight = Math.max(8, aspect >= 1 ? Math.round(LONG_EDGE / aspect) : LONG_EDGE)

            // Gathered-output buffer at compute res: written by the gather (storage), sampled by the fragment.
            const outputTex = root.createTexture({size: [computeWidth, computeHeight], format: BOKEH_GATHER_FORMAT}).$usage('storage', 'sampled')
            onCleanup(() => outputTex.destroy())
            const blurredTexture = registerComputeTexture(outputTex)

            onResize(({width, height}) => {
                curWidth = Math.max(1, Math.round(width))
                curHeight = Math.max(1, Math.round(height))
            })

            // Look controls (per-frame CPU values; rotation degrees → a cos/sin pair for the kernel).
            const readLook = () => {
                const rot = ((getCpuValue(opts.rotation.name) as number) ?? 0) * BOKEH_DEG_TO_RAD
                return {
                    highlightGain: (getCpuValue(opts.gain.name) as number) ?? 4,
                    highlightThreshold: (getCpuValue(opts.threshold.name) as number) ?? 0.6,
                    rotCos: Math.cos(rot),
                    rotSin: Math.sin(rot),
                    chromaticFringe: (getCpuValue(opts.fringe.name) as number) ?? 0.2,
                }
            }

            // radius bound to a map → per-pixel radius sampled from the map source.
            const mapInfo = getMapInfo(opts.radius.name)
            if (mapInfo) {
                const graph = blurKit.buildBokehMapGraph(computeWidth, computeHeight, blurKit.BOKEH_TAP_COUNT, mapInfo.channel as blurKit.BokehMapChannel)
                const kernelParams = root.createUniform(graph.Params)
                const tapsUniform = root.createUniform(graph.TapArray)
                const ensureTaps = apertureTable({shape: opts.shape, blades: opts.blades}, getCpuValue, tapsUniform)
                let bokehPass = createGuardedCompute(root, (cx: number, cy: number) => {
                    'use gpu'
                    graph.kernel(cx, cy)
                }, {size: [computeWidth, computeHeight]})

                return {
                    outputs: {blurredTexture},
                    bindInputs: (resolve) => {
                        const src = resolve(childTexture.key)
                        const mapSrc = resolve(mapInfo.sourceTexture.key)
                        if (!src || !mapSrc) return
                        const bindGroup = root.createBindGroup(graph.layout, {
                            input: src.texture as never,
                            source: mapSrc.texture as never,
                            output: outputTex,
                            params: kernelParams.buffer,
                            taps: tapsUniform.buffer,
                        })
                        bokehPass = bokehPass.with(bindGroup)
                    },
                    getComputeNodes: () => {
                        ensureTaps()
                        const w = mapInfo.window()
                        kernelParams.write({
                            ...readLook(),
                            inputWidth: curWidth,
                            inputHeight: curHeight,
                            inputMin: w.inputMin,
                            inputMax: w.inputMax,
                            outputMin: w.outputMin,
                            outputMax: w.outputMax,
                            curve: w.curve,
                        })
                        return [bokehPass]
                    },
                }
            }

            // ── Static / mouse / auto radius: single uniform gather radius (per-frame CPU value). ──
            const graph = blurKit.buildBokehGraph(computeWidth, computeHeight, blurKit.BOKEH_TAP_COUNT)
            const kernelParams = root.createUniform(graph.Params)
            const tapsUniform = root.createUniform(graph.TapArray)
            const ensureTaps = apertureTable({shape: opts.shape, blades: opts.blades}, getCpuValue, tapsUniform)
            let bokehPass = createGuardedCompute(root, (cx: number, cy: number) => {
                'use gpu'
                graph.kernel(cx, cy)
            }, {size: [computeWidth, computeHeight]})

            return {
                outputs: {blurredTexture},
                // The child RTT is allocated after composition → build the gather bind group once it exists.
                bindInputs: (resolve) => {
                    const src = resolve(childTexture.key)
                    if (!src) return
                    const bindGroup = root.createBindGroup(graph.layout, {
                        input: src.texture as never,
                        output: outputTex,
                        params: kernelParams.buffer,
                        taps: tapsUniform.buffer,
                    })
                    bokehPass = bokehPass.with(bindGroup)
                },
                getComputeNodes: () => {
                    ensureTaps()
                    const radius = (getCpuValue(opts.radius.name) as number) ?? 50
                    kernelParams.write({
                        radius: blurKit.bokehRadiusToPixels(radius),
                        ...readLook(),
                        inputWidth: curWidth,
                        inputHeight: curHeight,
                    })
                    return [bokehPass]
                },
            }
        },

        gpu: {
            fragment: ({childNode, computeOutputs, ctx, convertToTexture}: GpuFragmentParams): Expr => {
                if (!childNode) return ZERO

                const blurred = computeOutputs?.blurredTexture as KitTexture | undefined
                if (!blurred) {
                    // Compute unavailable (GPU-free resolve / no device): sharp passthrough, unpremultiplied.
                    const tex = convertToTexture(childNode)
                    return call(blend.unpremultiplyAlpha, 'unpremultiplyAlpha', [tex.sample(ctx.uv)])
                }

                // The gather produced a full defocus in premultiplied space (color + coverage). Sample it at
                // canvas resolution and unpremultiply back into the straight-alpha blend pipeline.
                return call(blend.unpremultiplyAlpha, 'unpremultiplyAlpha', [blurred.sample(ctx.uv)])
            },
        },
    }
}

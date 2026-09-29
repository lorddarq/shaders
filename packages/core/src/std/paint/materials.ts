/**
 * std/paint/materials — the SHAPE-EFFECT SPINE.
 *
 * Every SDF shape-effect material (Glass, Chrome, LiquidMetal, Plastic, Crystal, Frost, Water, …)
 * used to repeat the same ~60–100 lines of plumbing: the `VOLUMETRIC_FIELD_EXTRA_FIELDS` +
 * `ANALYTIC_SDF_EXTRA_FIELDS` spreads, the `createVolumetricFieldComputeNode` compute hook, the
 * `resolveShapeFieldSampler` routing (compute-marched volumetric field → flat SVG → analytic),
 * the `sdfSpaceUV` placement, and the neighbour-tap stencil. {@link shapedSurface} is that spine
 * as ONE spread noun: the definition spreads its `{extraFields, compute, gpu}` halves
 * (`...shapedSurface({...})` — the gaussianBlur/ComputeBackedEffect precedent) and supplies the
 * MATERIAL as the `surface:` slot, which receives the resolved {@link SurfaceFrame} (placement UV,
 * field taps, samplers, flags, child textures) and returns the shaded color Expr.
 *
 * Config axes are declared, not copy-pasted:
 *  - `pattern` / `chord`      — forwarded to the volumetric pre-march AND the sampler routing,
 *                               so the two paths can never disagree.
 *  - `stencil`                — the field-tap frame the material reads: `'none'` (the material
 *                               samples for itself — Glass/ThinFilm/Neon/Emboss), `'centre'`
 *                               (placement + centre tap only — Heatmap/LightEdge), a forward
 *                               2-neighbour stencil at a per-material eps (the metals), or the
 *                               5-tap central stencil (Hologram).
 *  - `placeUV`                — optional pre-placement displacement of the screen UV (Hologram's
 *                               beam wobble); the default is `ctx.uv`.
 *  - `child: 'required'`      — resolve the child RTT (compute-created when present, else
 *                               `convertToTexture`) and return transparent without a child.
 *
 * Material-specific prepasses (Glass's frosted blur) are NOT axes here: a consumer wraps the
 * returned `compute` half and merges its own compute node's outputs — the fragment picks the
 * compute-created `childTexture` up automatically, and anything else rides `params.computeOutputs`.
 */
import type {EmitContext, GpuComputeNode, GpuFragmentParams, KitTexture} from '../../gpu/contract'
import {Expr} from '../../gpu/contract'
import {call, floatE, ZERO} from '../../gpu/composer'
import {effects, lighting, materialParts, noise, sdf, sdf3d, tonemap} from '../../gpu/kit/index'
import {
    add, clamp as clampE, div, dot as dotE, exp as expE, exp2, float, fract as fractE, length as lengthE, local, max as maxE,
    mix as mixE, mul, neg, normalize as normalizeE, pow as powE, sin as sinE, smoothstep as smoothstepE, splat3,
    sub, vec2 as vec2E, vec3 as vec3E,
} from '../math'
import type {DirectionFrame} from '../frames'

const {sdfSpaceUV, offsetUV} = effects.glass
const {ANALYTIC_SDF_EXTRA_FIELDS} = sdf
const {createVolumetricFieldComputeNode, resolveShapeFieldSampler, VOLUMETRIC_FIELD_EXTRA_FIELDS} = sdf3d

// ─── Conditional region ───────────────────────────────────────────────────────────────────────
//
// Expr algebra is pure expressions — it cannot express the `if (outside) return transparent`
// early-exit the material bodies rely on for their perf (a shape usually covers a fraction of the
// canvas). `guarded` restores that as vocabulary: a REAL WGSL branch whose interior statements
// (including every `local()` hoist emitted while serialising `inner`) land inside the branch, so
// pixels failing `cond` pay only the condition. NOTE: promote-to-math candidate — this is
// universal Expr flow control, not a materials concern.

let guardCounter = 0

/**
 * Evaluate `inner` only where `cond` holds; elsewhere the (cheap) `fallback`. Emits
 * `var x = fallback; if (cond) { …inner hoists…; x = inner; }`. Values hoisted BEFORE the guard
 * (emitted by `cond`/`fallback`, or by earlier expressions) stay in the enclosing scope and remain
 * visible inside; values first used inside are declared inside and must not be re-read outside.
 * A `local()` shared between the guarded interior and any later expression MUST be listed in
 * `deps` — that emits it in the enclosing scope first (a hoist whose first use is inside the
 * branch would otherwise be block-scoped and unreachable afterwards).
 */
export function guarded(cond: Expr, inner: Expr, fallback: Expr, hint = 'guarded', deps: Expr[] = []): Expr {
    const id = guardCounter++
    return new Expr((ctx) =>
        ctx.memo(`guard:${id}`, () => {
            const name = ctx.freshLocal(hint)
            const condText = cond._emit(ctx)
            const fallbackText = fallback._emit(ctx)
            for (const dep of deps) dep._emit(ctx)
            const stmts: string[] = []
            const scoped: EmitContext = {
                external: (v, h) => ctx.external(v, h),
                statement: (w) => stmts.push(w),
                freshLocal: (h) => ctx.freshLocal(h),
                memo: (k, f) => ctx.memo(k, f),
            }
            const innerText = inner._emit(scoped)
            ctx.statement(`var ${name} = ${fallbackText};`)
            ctx.statement(`if (${condText}) {\n  ${stmts.join('\n  ')}\n  ${name} = ${innerText};\n}`)
            return name
        }),
    )
}

// reflect/refract/exp2 PROMOTED to std/math (2026-09-01) — re-exported here so existing
// material recipes keep their import surface.
export {reflect, refract, exp2} from '../math'

/** The material-body early-exit condition as vocabulary: NOT outside the shape by > 2 device
 *  pixels (the exact complement of the kit's `outsideShape` early-exit test). */
export function insideShape(sdf: Expr, pxH: Expr): Expr {
    const outside = call(lighting.outsideShape, 'outsideShape', [sdf, pxH])
    return new Expr((ctx) => `(!(${outside._emit(ctx)}))`)
}

/** The resolved per-fragment frame the `surface:` slot shades. */
export interface SurfaceFrame {
    /** Placement-space UV (`sdfSpaceUV` over center/scale/rotation) — undefined for `stencil: 'none'`. */
    sdfUV?: Expr
    /** Centre field tap (`.x` sdf/−chord, `.y/.z` pattern coords or baked gradients, `.w` depth). */
    surf0?: Expr
    /** Forward/central stencil neighbour taps (per the declared `stencil`). */
    surfX?: Expr
    surfY?: Expr
    /** Central-stencil extras (−eps taps); only with `stencil: {kind: 'central'}`. */
    surfL?: Expr
    surfD?: Expr
    /** Centre-quality field sampler (materials with `stencil: 'none'` take their own taps). */
    sampler: (uv: Expr) => Expr
    /** Neighbour-tap sampler (cheap bilinear on the volumetric path unless `gradSampler: 'same'`). */
    gradSampler: (uv: Expr) => Expr
    /** Unfiltered nearest-texel tap — for discrete payload a pre-march packs into the field
     *  (the voxel G-buffer's face/cell channels). Aliases `sampler` on the flat paths. */
    texelSampler: (uv: Expr) => Expr
    /** Build-time: the active shape resolved to the compute-marched volumetric path. */
    volumetric: boolean
    /** `volumetric` as the runtime 0/1 f32 the composite bodies `std.select` on. */
    volFlag: Expr
    /** Build-time: the sampler carries analytic/baked gradients in `.g/.b` (skip FD taps). */
    bakedGradients: boolean
    /** Child RTT (with `child: 'required'`) — compute-created when a consumer prepass made one. */
    childTexture?: KitTexture
}

export interface ShapedSurfaceSpec {
    /** Surface-pattern coords the volumetric pre-march bakes into `.g/.b` (default 'none'). */
    pattern?: 'none' | 'raw' | 'triplanar'
    /** Chord measure along the march ray (default 'span'; 'firstLobe' = front-wall thickness). */
    chord?: 'span' | 'firstLobe'
    /** Neighbour-tap sampler quality (default 'fast'; 'same' = centre sampler everywhere). */
    gradSampler?: 'fast' | 'same'
    /** Baked-gradient routing (default 'none'); see `ShapeFieldSamplerOptions`. */
    bakedGradients?: 'none' | 'volumetric' | 'all'
    /** The field-tap frame to resolve for the material (default forward stencil at eps 0.01). */
    stencil?: 'none' | 'centre' | {kind: 'forward' | 'central'; eps: number}
    /**
     * Which sampler takes the CENTRE tap (default 'quality' — bicubic on the volumetric path).
     * 'fast' takes the cheap bilinear neighbour sampler instead: for a material whose open-canvas
     * pixels only gate on the centre value (coverage, an outside branch) and that takes its own
     * quality tap inside the branches that need one — the metals' bicubic-per-pixel is 16 texel
     * loads at every canvas pixel on a 3D shape.
     */
    centreTap?: 'quality' | 'fast'
    /** Displace the screen UV before placement (returns the UV fed to `sdfSpaceUV`). */
    placeUV?: (params: GpuFragmentParams) => Expr
    /** Resolve the child RTT into the frame; without a child the fragment returns transparent. */
    child?: 'required'
    /** THE MATERIAL: shade the resolved frame. */
    surface: (frame: SurfaceFrame, params: GpuFragmentParams) => Expr
}

/** The three definition halves the spine contributes — spread into the definition. */
export interface ShapedSurfaceEffect {
    extraFields: Record<string, {schema: import('typegpu/data').AnyWgslData; initial: number | number[]}>
    compute: GpuComputeNode
    gpu: {fragment: (params: GpuFragmentParams) => Expr}
}

/** The shape-effect spine noun. See the module header. */
export function shapedSurface(spec: ShapedSurfaceSpec): ShapedSurfaceEffect {
    const pattern = spec.pattern ?? 'none'
    const chord = spec.chord ?? 'span'
    const stencil = spec.stencil ?? {kind: 'forward' as const, eps: 0.01}

    // Volumetric SDF pre-march (3D / SVG-extrude shapes) — dirty-keyed; flat shapes and
    // GPU-free resolve return null (the fragment falls back to the flat-SVG/analytic sampler).
    const compute: GpuComputeNode = (params) =>
        createVolumetricFieldComputeNode(params, () => params.getCpuValue('shape'), pattern, chord)

    const fragment = (params: GpuFragmentParams): Expr => {
        const {childNode, computeOutputs, convertToTexture, uniforms, ctx} = params

        let childTexture: KitTexture | undefined
        if (spec.child === 'required') {
            if (!childNode) return ZERO
            childTexture = (computeOutputs?.childTexture as KitTexture | undefined) ?? convertToTexture(childNode)
        }

        // Sampler routing — the SAME pattern mode as the pre-march, by construction.
        const {sampler, gradSampler, texelSampler, volumetric, bakedGradients} = resolveShapeFieldSampler(params, {
            patternMode: pattern,
            gradSampler: spec.gradSampler ?? 'fast',
            bakedGradients: spec.bakedGradients ?? 'none',
        })

        const frame: SurfaceFrame = {
            sampler, gradSampler, texelSampler, volumetric, bakedGradients,
            volFlag: floatE(volumetric ? 1 : 0),
            childTexture,
        }

        if (stencil !== 'none') {
            const screenUV = spec.placeUV ? spec.placeUV(params) : ctx.uv
            frame.sdfUV = local(call(sdfSpaceUV, 'sdfSpaceUV', [uniforms.center, uniforms.scale, uniforms.rotation, screenUV, ctx.aspect]), 'sdfUV')
            frame.surf0 = (spec.centreTap === 'fast' ? gradSampler : sampler)(frame.sdfUV)
            if (stencil !== 'centre') {
                const {kind, eps} = stencil
                frame.surfX = gradSampler(call(offsetUV, 'offsetUV', [frame.sdfUV, floatE(eps), floatE(0)]))
                if (kind === 'central') {
                    frame.surfL = gradSampler(call(offsetUV, 'offsetUV', [frame.sdfUV, floatE(-eps), floatE(0)]))
                }
                frame.surfY = gradSampler(call(offsetUV, 'offsetUV', [frame.sdfUV, floatE(0), floatE(eps)]))
                if (kind === 'central') {
                    frame.surfD = gradSampler(call(offsetUV, 'offsetUV', [frame.sdfUV, floatE(0), floatE(-eps)]))
                }
            }
        }

        return spec.surface(frame, params)
    }

    return {
        extraFields: {...VOLUMETRIC_FIELD_EXTRA_FIELDS, ...ANALYTIC_SDF_EXTRA_FIELDS},
        compute,
        gpu: {fragment},
    }
}

// ─── Material vocabulary ────────────────────────────────────────────────────────────────────────
//
// The shared anatomy of a shaded shape-effect material, as words. A material file should read as
// a recipe over these — field basics → geometric normal → (relief → tilt) → view → lighting →
// tint — with only its own constants and composition at the declaration site.

export interface SurfaceField {
    /** Centre / forward-x / forward-y field taps, locals-bound. */
    s0: Expr
    sX?: Expr
    sY?: Expr
    /** Signed distance (centre tap `.x`, divided by `scale` when given). */
    sdf: Expr
    /** One pixel in UV height units. */
    pxH: Expr
    /** Viewport aspect (width / height). */
    aspect: Expr
}

/** The opening block of every material: bind the field taps + sdf + pixel size to locals. */
export function surfaceField(
    frame: SurfaceFrame,
    params: GpuFragmentParams,
    opts: {scale?: Expr} = {},
): SurfaceField {
    const {ctx} = params
    const s0 = local(frame.surf0!, 's0')
    const pxH = local(div(1, ctx.viewportSize.member('y')), 'pxH')
    const sdf = local(opts.scale ? div(s0.member('x'), opts.scale) : s0.member('x'), 'sdf')
    const aspect = div(ctx.viewportSize.member('x'), ctx.viewportSize.member('y'))
    return {
        s0,
        sX: frame.surfX ? local(frame.surfX, 'sX') : undefined,
        sY: frame.surfY ? local(frame.surfY, 'sY') : undefined,
        sdf,
        pxH,
        aspect,
    }
}

/**
 * The geometric surface normal, resolved at build time: marched volumetric shapes take the
 * field-tap normal; flat shapes take the gradient normal tilted by the bevel profile.
 */
export function geometricNormal(
    frame: SurfaceFrame,
    field: SurfaceField,
    opts: {bevelWidth: Expr; bevelShape: Expr; sharpness?: number; eps?: number; hint?: string},
): Expr {
    const eps = float(opts.eps ?? 0.01)
    const hint = opts.hint ?? 'nGeo'
    if (frame.volumetric) {
        return local(
            call(lighting.volumetricNormal, 'volumetricNormal', [field.s0, field.sX!, field.sY!, eps, float(opts.sharpness ?? 5)]),
            hint,
        )
    }
    const rimT = clampE(div(neg(field.sdf), maxE(opts.bevelWidth, 0.003)), 0, 1)
    return local(
        call(lighting.bevelledFlatNormal, 'bevelledFlatNormal', [
            call(lighting.fieldGradient, 'fieldGradient', [field.s0, field.sX!, field.sY!, eps]),
            call(effects.bevel.bevelSin, 'bevelSin', [rimT, opts.bevelShape]),
        ]),
        hint,
    )
}

/** Surface-pattern coordinates (the coords a pattern rides the geometry in), locals-bound. */
export function surfacePattern(frame: SurfaceFrame, field: SurfaceField, opts: {uv?: Expr} = {}): Expr {
    return local(call(lighting.patternCoords, 'patternCoords', [opts.uv ?? frame.sdfUV!, field.s0, frame.volFlag]), 'pat')
}

/** The in-plane field gradient (the outward direction a flat shape's shading rides). */
export function fieldSlope(field: SurfaceField, eps = 0.01): Expr {
    return local(call(lighting.fieldGradient, 'fieldGradient', [field.s0, field.sX!, field.sY!, float(eps)]), 'grad')
}

/** The perspective view ray for a pseudo-3D surface, locals-bound. */
export function viewRay(params: GpuFragmentParams, field: SurfaceField, fov = 0.6): Expr {
    return local(call(lighting.perspectiveViewRay, 'perspectiveViewRay', [params.ctx.uv, field.aspect, float(fov)]), 'viewI')
}

/** How edge-on the surface is to the view (0 face-on → 1 grazing), locals-bound. */
export function grazingOf(n: Expr, view: Expr): Expr {
    return local(sub(1, clampE(neg(dotE(n, view)), 0, 1)), 'grazing')
}

/** Tilt a normal within a direction frame: `along`/`across` are slope amounts per axis. */
export function tiltNormal(n: Expr, frame: DirectionFrame, slopes: {along?: Expr; across?: Expr}, hint = 'n'): Expr {
    const terms = (axis: 'x' | 'y'): Expr[] => {
        const out: Expr[] = []
        if (slopes.across) out.push(mul(frame.perp.member(axis), slopes.across))
        if (slopes.along) out.push(mul(frame.tangent.member(axis), slopes.along))
        return out
    }
    const sum = (base: Expr, extra: Expr[]): Expr => extra.reduce((acc, e) => add(acc, e), base)
    return local(
        normalizeE(vec3E(sum(n.member('x'), terms('x')), sum(n.member('y'), terms('y')), n.member('z'))),
        hint,
    )
}

/** The shared 5-softbox studio bank, looked up at a reflected direction. */
export function studioSoftboxes(x: Expr, y: Expr, opts: {
    /** The studio rotation as a `direction()` vector. */
    rotation: Expr
    keyRadius: Expr | number
    drift: Expr | number
    strength: Expr | number
    skyGain: number
}): Expr {
    const toE = (v: Expr | number): Expr => (typeof v === 'number' ? float(v) : v)
    return call(materialParts.studio5Softbox, 'studio5Softbox', [
        x, y, opts.rotation.member('x'), opts.rotation.member('y'),
        toE(opts.keyRadius), toE(opts.drift), toE(opts.strength), float(opts.skyGain),
    ])
}

/**
 * Grain-space noise: one octave of value noise over a direction frame's coordinates, with a
 * frequency per axis. Octave sums, gains, and drift terms are algebra at the call site — they
 * are the look; this is the sampling.
 */
export function grainNoise(
    coords: {along: Expr; across: Expr},
    opts: {freq: [Expr | number, Expr | number]; offset?: [number, number]; drift?: Expr},
): Expr {
    let x: Expr = mul(coords.along, opts.freq[0])
    let y: Expr = mul(coords.across, opts.freq[1])
    if (opts.offset?.[0]) x = add(x, opts.offset[0])
    if (opts.offset?.[1]) y = add(y, opts.offset[1])
    if (opts.drift) x = add(x, opts.drift)
    return call(noise.mxNoiseFloat2, 'mxNoiseFloat2', [vec2E(x, y)])
}

/**
 * A gaussian-weighted directional gather: sample `taps` points spread along `axis` around
 * `center`, summed with baked normalized weights. The anisotropic smear of any 2D lookup —
 * a reflected environment along a brush grain, a streaked highlight, a motion smear.
 */
export function smearAlong(
    opts: {center: Expr; axis: Expr; spread: Expr; taps?: number; sigma?: number; hint?: string},
    sample: (x: Expr, y: Expr) => Expr,
): Expr {
    const taps = opts.taps ?? 9
    const sigma = opts.sigma ?? 2.1
    const half = (taps - 1) / 2
    const raw = Array.from({length: taps}, (_, i) => Math.exp(-((i - half) ** 2) / (2 * sigma * sigma)))
    const sum = raw.reduce((a, b) => a + b, 0)
    const weights = raw.map((w) => w / sum)

    const step = local(div(opts.spread, half), 'step')
    let acc: Expr = float(0)
    for (let i = 0; i < taps; i++) {
        const off = local(mul(step, float(i - half)), 'off')
        acc = add(acc, mul(
            sample(
                add(opts.center.member('x'), mul(opts.axis.member('x'), off)),
                add(opts.center.member('y'), mul(opts.axis.member('y'), off)),
            ),
            float(weights[i]),
        ))
    }
    return local(acc, opts.hint ?? 'smear')
}

/** A 3D light direction from a `direction()` vector and an eye-tuned elevation z. */
export function lightVec3(dir: Expr, z: number, hint = 'L'): Expr {
    return local(normalizeE(vec3E(dir.member('x'), dir.member('y'), float(z))), hint)
}

/** The house roughness/anisotropy → Ward lobe-width mapping (stretch along, squeeze across). */
export function wardAlphas(roughness: Expr, anisotropy: Expr): {along: Expr; across: Expr} {
    const baseA = local(mixE(0.05, 0.45, roughness), 'baseA')
    return {
        along: mul(baseA, add(1, mul(anisotropy, 7))),
        across: maxE(mul(baseA, sub(1, mul(anisotropy, 0.55))), 0.02),
    }
}

/**
 * Anisotropic (Ward) specular lobe: stretched along `tangent`, widths per axis from `alphas`
 * (see {@link wardAlphas} for the designer-terms mapping).
 */
export function anisoSpecular(opts: {
    normal: Expr
    /** The grain/fibre tangent (2D — in-plane). */
    tangent: Expr
    /** Normalized 3D light direction (see {@link lightVec3}). */
    light: Expr
    view: Expr
    alphas: {along: Expr; across: Expr}
    gain?: Expr | number
}): Expr {
    const spec = call(lighting.wardAnisotropicSpecular, 'wardAnisotropicSpecular', [
        call(lighting.WardSpecularInput, 'WardSpecularInput', [
            opts.normal,
            vec3E(opts.tangent.member('x'), opts.tangent.member('y'), 0),
            opts.light,
            opts.view,
            opts.alphas.along,
            opts.alphas.across,
        ]),
    ])
    return opts.gain === undefined ? spec : mul(spec, opts.gain)
}

/** Tilt a normal along one in-plane axis by a slope amount. */
export function tiltAlong(n: Expr, axis: Expr, amount: Expr, hint = 'n'): Expr {
    return local(normalizeE(vec3E(
        add(n.member('x'), mul(axis.member('x'), amount)),
        add(n.member('y'), mul(axis.member('y'), amount)),
        n.member('z'),
    )), hint)
}

/** Fresnel edge lift: `1 + grazing^power · amount`. */
export function fresnelBoost(grazing: Expr, opts: {power?: number; amount: Expr | number}): Expr {
    return add(1, mul(powE(grazing, float(opts.power ?? 3)), opts.amount))
}

/** A two-tone tint ramp: dark → light by a lighting term (Duotone's ramp as material tint). */
export function tintRamp(t: Expr, dark: Expr, light: Expr): Expr {
    return mixE(dark.member('rgb'), light.member('rgb'), splat3(t))
}

/** The soft silhouette alpha at the shape boundary. */
export function silhouette(field: SurfaceField, edgeSoftness: Expr): Expr {
    return call(materialParts.silhouetteAlpha, 'silhouetteAlpha', [field.sdf, edgeSoftness, field.pxH])
}

/** The neutral filmic tonemap shoulder (exposure grading's final step). */
export function neutralTone(rgb: Expr): Expr {
    return call(tonemap.neutral, 'tonemapNeutral', [rgb])
}

/** Nudge a normal by screen-space slope deltas (the isotropic cousin of `tiltNormal`). */
export function nudgeNormal(n: Expr, dx: Expr, dy: Expr, hint = 'n'): Expr {
    return local(call(materialParts.nudgeNormal, 'nudgeNormal', [n, dx, dy]), hint)
}

/** The volumetric field-tap normal (marched shapes). Flat shapes shape their own — see
 *  {@link geometricNormal} for the standard bevelled path. */
export function marchedNormal(field: SurfaceField, sharpness = 5, hint = 'nGeo'): Expr {
    return local(call(lighting.volumetricNormal, 'volumetricNormal', [field.s0, field.sX!, field.sY!, float(0.01), float(sharpness)]), hint)
}

/** The key-light frame from a `direction()` vector: `.L` light dir, `.H` half vector.
 *  `elevation` is the material's eye-tuned z (the fleet uses −0.6 … −0.9). */
export function keyLightAt(dir: Expr, elevation: number, hint = 'kl'): Expr {
    return keyLightXY(dir.member('x'), dir.member('y'), elevation, hint)
}

/** {@link keyLightAt} from explicit in-plane components (shimmered / cross lights). */
export function keyLightXY(x: Expr, y: Expr, elevation: number, hint = 'kl'): Expr {
    return local(call(materialParts.keyLight, 'keyLight', [x, y, float(elevation)]), hint)
}

/** Grazing factor for the fixed −z view (the fresnel input when no perspective ray exists). */
export function grazingFlat(n: Expr): Expr {
    return local(call(materialParts.grazingView, 'grazingView', [n]), 'grazing')
}

/** Sharp key-light glint lobe. */
export function sharpGlint(ndh: Expr, sharpness: Expr, cfg: Expr): Expr {
    return call(materialParts.sharpGlint, 'sharpGlint', [ndh, sharpness, cfg])
}

/** Clamped perlin slope — the pressed/molten waviness gradient two metals nudge normals with. */
export function perlinSlope(q: Expr, gain: Expr | number): Expr {
    return call(materialParts.clampedPerlinGrad, 'clampedPerlinGrad', [q, typeof gain === 'number' ? float(gain) : gain])
}

/** Optical body thickness of the shape field (chord on marched shapes, rim proxy on flat). */
export function opticalThickness(field: SurfaceField, frame: SurfaceFrame): Expr {
    return call(materialParts.opticalThickness, 'opticalThickness', [field.sdf, frame.volFlag])
}

/** Beer–Lambert transmission: `exp(−thickness · absorb)` per channel. */
export function beerLambert(thickness: Expr, absorb: Expr): Expr {
    return call(materialParts.beerLambert, 'beerLambert', [thickness, absorb])
}

/** The shared incommensurate-frequency flow-warp offset (molten folds / wind-driven waves). */
export function flowWarp(warpInput: Expr, flowT: Expr): Expr {
    return call(materialParts.flowWarpOffset, 'flowWarpOffset', [warpInput, flowT])
}

/** Signed value noise sampled at a surface point (the material grain/crumple basis). */
export function surfaceNoise(x: Expr | number, y: Expr | number): Expr {
    return call(noise.mxNoiseFloat2, 'mxNoiseFloat2', [vec2E(x, y)])
}

/** {@link surfaceNoise} at a pre-built vec2 point. */
export function surfaceNoiseAt(p: Expr): Expr {
    return call(noise.mxNoiseFloat2, 'mxNoiseFloat2', [p])
}

/** Signed noise sampled at an interior volume point (vec3) — the 3D cousin of
 *  {@link surfaceNoiseAt}: the basis for anything sampled THROUGH a body rather than
 *  across its surface (gas, smoke, inclusions, subsurface density). */
export function volumeNoiseAt(p: Expr): Expr {
    return call(noise.mxNoiseFloat3, 'mxNoiseFloat3', [p])
}

/** IQ cosine palette: a smooth rainbow ramp with per-channel phase offsets. */
export function cosineRainbow(t: Expr, phases: [number, number, number]): Expr {
    return call(lighting.cosinePalette, 'cosinePalette', [t, float(phases[0]), float(phases[1]), float(phases[2])])
}

/** Worley (cellular) noise at a surface point — the facet/crack cell basis. */
export function cellNoiseAt(p: Expr): Expr {
    return call(noise.mxWorleyNoiseFloat2Pub, 'mxWorleyNoiseFloat2Pub', [p, float(1)])
}

/**
 * Finite-difference slope of any scalar surface field: the field's value at `p` plus its
 * x/y derivatives at step `eps` — the relief-to-normal bridge every bump recipe rides.
 */
export function fdSlope(
    sampleAt: (p: Expr) => Expr,
    p: Expr,
    eps: number,
    hint = 'fd',
): {value: Expr; dx: Expr; dy: Expr} {
    const value = local(sampleAt(p), `${hint}0`)
    return {
        value,
        dx: local(div(sub(sampleAt(add(p, vec2E(eps, 0))), value), eps), `${hint}X`),
        dy: local(div(sub(sampleAt(add(p, vec2E(0, eps))), value), eps), `${hint}Y`),
    }
}

/** Unsigned white-noise hash of a 2D point (film grain, flicker, per-pixel jitter). */
export function hashNoise(x: Expr | number, y: Expr | number): Expr {
    return call(noise.hash12, 'hash12', [vec2E(x, y)])
}

/**
 * Interleaved gradient noise over device pixels: a [0,1) pattern whose neighbouring pixels are
 * maximally different (Jimenez 2014) — the low-discrepancy cousin of {@link hashNoise} for
 * jittering samples, where white noise clumps into visible grain. `clock` advances the pattern
 * by the golden ratio per unit so successive frames decorrelate.
 */
export function interleavedNoise(pixel: Expr, clock: Expr | number = 0): Expr {
    const cell = add(mul(pixel.member('x'), 0.06711056), mul(pixel.member('y'), 0.00583715))
    return fractE(add(mul(52.9829189, fractE(cell)), mul(clock, 0.61803398875)))
}

/** Unsigned smooth value noise at a surface point (the [0,1] cousin of {@link surfaceNoise}). */
export function valueNoise(x: Expr | number, y: Expr | number): Expr {
    return call(noise.value12, 'value12', [vec2E(x, y)])
}

/**
 * How rotated the 3D shape is, as an in-plane pan vector: one field tap at the placement
 * centre reads the surface-locked pattern coords of the centre hit (zero at rest on every
 * shape, growing with 3D rotation). Spatially constant per frame, so it pans content
 * (star planes, interior backdrops) without shearing it; flat shapes gate to zero.
 * Requires `pattern: 'raw'`.
 */
export function rotationSensor(frame: SurfaceFrame): {x: Expr; y: Expr} {
    const s0C = local(frame.gradSampler(vec2E(0.5, 0.5)), 's0C')
    return {
        x: local(mul(neg(s0C.member('y')), frame.volFlag), 'rotSenseX'),
        y: local(mul(neg(s0C.member('z')), frame.volFlag), 'rotSenseY'),
    }
}

/**
 * Interior coordinates bent at the shell: centered placement coords displaced along the
 * surface normal's in-plane tilt. `n.xy` is ~0 on a flat face and large at bevels and
 * silhouettes, so the bend concentrates where real glass would bend the interior — the
 * cheap single-interface cousin of a full refraction trace. Also returns the unbent
 * centered coords for anything anchored to the container rather than the interior.
 */
export function shellRefract(frame: SurfaceFrame, n: Expr, amount: Expr): {
    x: Expr
    y: Expr
    centered: {x: Expr; y: Expr}
} {
    const cx = local(sub(frame.sdfUV!.member('x'), 0.5), 'cx')
    const cy = local(sub(frame.sdfUV!.member('y'), 0.5), 'cy')
    return {
        x: local(add(cx, mul(n.member('x'), amount)), 'bentX'),
        y: local(add(cy, mul(n.member('y'), amount)), 'bentY'),
        centered: {x: cx, y: cy},
    }
}

/** Schlick fresnel reflectance: `r0 + grazing^power · gain`. Physical glass is
 *  `{r0: 0.04, gain: 0.96}`; lower the gain to keep grazing faces from curtaining
 *  an interior behind full mirror. */
export function schlickFresnel(grazing: Expr, opts: {r0?: number; gain: Expr | number; power?: number}): Expr {
    return add(opts.r0 ?? 0.04, mul(powE(grazing, float(opts.power ?? 5)), opts.gain))
}

/** A per-element shimmer factor around 1: `1 − amount + sin(t·rate + phase·spread)·amount`.
 *  `phase` decorrelates elements (a per-star/per-flake variation field). */
export function twinkle(t: Expr, phase: Expr, opts: {amount: Expr; rate: number; spread: number}): Expr {
    return add(sub(1, opts.amount), mul(sinE(add(mul(t, opts.rate), mul(phase, opts.spread))), opts.amount))
}

/**
 * A plane of point glints — stars, glitter, bokeh dust: worley feature points lit only
 * within a tiny radius, gated to a sparse subset and sized/brightened per point by a slow
 * variation field. Returns the glint intensity and the variation field (for tinting).
 * The worley metric is squared distance in cell units, so radii are in those units.
 */
export function pointStars(p: Expr, opts: {
    /** Frequency of the per-point variation field relative to `p` (slow: ~0.1). */
    variationFreq: number
    /** Point radius: base + variation · jitter (squared-distance units). */
    radius: [number, number]
    /** smoothstep band of the variation field that keeps a point at all (sparsity). */
    keep: [number, number]
    /** Brightness: base + variation² · gain. */
    brightness: [number, number]
    /** Per-point shimmer factor, built from the variation field (see {@link twinkle}). */
    shimmer?: (variation: Expr) => Expr
    /** Overall gain. */
    gain: Expr
    /** color the glints: variation mixes `from` → `to`, scaled by the glint × `gain`. */
    tint?: {from: [number, number, number]; to: [number, number, number]; gain: number}
    hint?: string
}): {glint: Expr; variation: Expr; rgb?: Expr} {
    const hint = opts.hint ?? 'stars'
    const variation = local(add(mul(surfaceNoiseAt(mul(p, opts.variationFreq)), 0.5), 0.5), `${hint}V`)
    const shimmer = opts.shimmer ? opts.shimmer(variation) : float(1)
    const glint = local(mul(mul(mul(
        smoothstepE(add(opts.radius[0], mul(variation, opts.radius[1])), 0.0005, cellNoiseAt(p)),
        smoothstepE(opts.keep[0], opts.keep[1], variation)),
        mul(add(opts.brightness[0], mul(mul(variation, variation), opts.brightness[1])), shimmer),
    ), opts.gain), hint)
    const rgb = opts.tint
        ? mul(
            mixE(vec3E(...opts.tint.from), vec3E(...opts.tint.to), splat3(variation)),
            splat3(mul(glint, opts.tint.gain)),
        )
        : undefined
    return {glint, variation, rgb}
}

// ─── Placement-space points, the field beyond its texture, and positional light ──────────────
//
// The words a shape effect needs to light its surroundings from a POINT (a cursor, an anchor)
// rather than a direction: bring a position prop into the field's own coordinates, keep the
// distance field honest far from the shape, find the boundary a pixel's light comes from, and
// evaluate a point light against a normal.

/**
 * A `transformPosition` prop mapped into the shape's placement (sdfUV) space — the same
 * aspect/rotation/scale placement the field is sampled in, so a light, anchor or pointer position
 * can be compared against field coordinates directly. Placement-space distances are the field's
 * units (the shape's own scale), not screen units.
 */
export function placementPoint(params: GpuFragmentParams, position: Expr, hint = 'placed'): Expr {
    const {uniforms, ctx} = params
    // transformPosition stores `(x, 1 − y)`; sdfSpaceUV takes a screen UV.
    const screenUV = vec2E(position.member('x'), sub(1, position.member('y')))
    return local(call(sdfSpaceUV, 'sdfSpaceUV', [uniforms.center, uniforms.scale, uniforms.rotation, screenUV, ctx.aspect]), hint)
}

/**
 * The field continued beyond its texture. The flat SVG sampler clamps to its 0–1 square, so a
 * far-reaching effect (a glow, a cast shadow) would see a plateau past it. The continuation is a
 * sphere-trace-safe LOWER bound — `max(beyond, border − beyond)`, the shape being inside the
 * square — not `border + beyond`, which is an upper bound that a march overshoots through
 * (analytic fields need nothing). Returns the continued distance at the pixel, the outward push vector (zero within the square) to add to any
 * gradient taken from the clamped taps, and `at(uv)` — the continued distance anywhere (fast
 * neighbour-quality taps, flow-safe: the sampler for ray marches and shadow rays). Expects an
 * unscaled {@link surfaceField}.
 */
export function continuedField(
    frame: SurfaceFrame,
    field: SurfaceField,
    params: GpuFragmentParams,
): {sdf: Expr; outward: Expr; at: (uv: Expr) => Expr} {
    const flatSvg = !frame.volumetric && Boolean(params.propValues.shapeSdfUrl)
    if (!flatSvg) return {sdf: field.sdf, outward: vec2E(0, 0), at: (uv) => frame.gradSampler(uv).member('x')}
    const uv = frame.sdfUV!
    const beyond = (p: Expr): Expr => sub(p, clampE(p, vec2E(0, 0), vec2E(1, 1)))
    const outward = local(beyond(uv), 'outward')
    const continued = (border: Expr, dist: Expr): Expr => maxE(border, maxE(dist, sub(border, dist)))
    return {
        sdf: local(continued(field.sdf, lengthE(outward)), 'sdfFar'),
        outward,
        at: (p) => {
            const dist = lengthE(beyond(p))
            return continued(frame.gradSampler(p).member('x'), dist)
        },
    }
}

/**
 * The nearest point of the shape boundary to this pixel and the outward unit normal there — where
 * a rim glow, contact shadow or edge light originates for pixels outside the shape. `slope` is
 * the in-plane field gradient (see {@link fieldSlope}); its length is guarded.
 */
export function nearestEdge(sdfUV: Expr, sdf: Expr, slope: Expr, hint = 'edge'): {point: Expr; normal: Expr} {
    const normal = local(div(slope, maxE(lengthE(slope), 0.0001)), `${hint}N`)
    const point = local(sub(sdfUV, mul(normal, maxE(sdf, 0))), `${hint}P`)
    return {point, normal}
}

/**
 * A point light in placement space as seen from `at` (a 2D placement point on the shape plane):
 * the unit 3D light vector in the material convention (z negative = toward the viewer, matching
 * {@link lightVec3}), the in-plane unit direction toward the light, and the in-plane distance.
 */
export function pointLightFrom(
    light: {position: Expr; height: Expr | number},
    at: Expr,
    hint = 'pl',
): {L: Expr; toLight: Expr; distance: Expr} {
    const delta = local(sub(light.position, at), `${hint}D`)
    const distance = local(lengthE(delta), `${hint}Dist`)
    const toLight = local(div(delta, maxE(distance, 0.0001)), `${hint}Dir`)
    const L = local(normalizeE(vec3E(delta.member('x'), delta.member('y'), neg(light.height))), `${hint}L`)
    return {L, toLight, distance}
}

/**
 * Diffuse (Lambert) response of a normal to a unit light vector (2D or 3D pairs alike), with
 * optional wrap lighting: `wrap` 0 = strict hemisphere, 1 = the light reaches all the way round.
 */
export function lambert(normal: Expr, light: Expr, opts: {wrap?: Expr | number} = {}): Expr {
    const ndl = dotE(normal, light)
    if (opts.wrap === undefined) return maxE(ndl, 0)
    return clampE(div(add(ndl, opts.wrap), add(1, opts.wrap)), 0, 1)
}

/** Inverse-square attenuation of a point light: 1 at the source, half strength at `range`. */
export function inverseSquare(distance: Expr, range: Expr | number): Expr {
    const r = div(distance, range)
    return div(1, add(1, mul(r, r)))
}

/**
 * Photographic exposure response `1 − e^(−rgb)`: light piles up toward white the way a bright
 * emitter burns out on film, instead of clipping per channel (the HDR → display step of any
 * emissive look; {@link neutralTone} is the graded-image cousin).
 */
export function exposureTone(rgb: Expr): Expr {
    return sub(1, expE(neg(rgb)))
}

/**
 * Multiplicative sensor grain `rgb · (1 + (hash − 0.5) · amount)`, keyed per device pixel and
 * re-rolled by `clock` — the photon-noise floor that makes rendered light read as photographed.
 */
export function sensorGrain(rgb: Expr, opts: {amount: Expr | number; pixel: Expr; clock: Expr | number}): Expr {
    const n = hashNoise(add(opts.pixel.member('x'), mul(opts.clock, 37.1)), add(opts.pixel.member('y'), mul(opts.clock, 17.3)))
    return mul(rgb, add(1, mul(sub(n, 0.5), opts.amount)))
}

/**
 * Two-lobe specular glint — a tight hot core plus a wide dim halo. `softness` widens or
 * tightens both lobes around the authored exponents (0.5 is neutral: the exponents as
 * given; 0 is ~4× sharper, 1 is ~4× softer).
 */
export function dualLobeGlint(ndh: Expr, opts: {
    /** [exponent, gain] of the hot core lobe. */
    core: [number, number]
    /** [exponent, gain] of the wide halo lobe. */
    halo: [number, number]
    softness?: Expr
    gain: Expr
}): Expr {
    const widen = opts.softness ? local(exp2(mixE(2, -2, opts.softness)), 'lobeWiden') : undefined
    const exponent = (e: number): Expr => (widen ? mul(e, widen) : float(e))
    return mul(add(
        mul(powE(ndh, exponent(opts.core[0])), opts.core[1]),
        mul(powE(ndh, exponent(opts.halo[0])), opts.halo[1]),
    ), opts.gain)
}

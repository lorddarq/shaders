/**
 * std/warps — noun producers for the warp role.
 *
 * Each producer takes a slot object (prop bindings via `p()`, or plain numbers) and returns
 * a `UvMapSource`: the single coordinate map (screen UV → source UV, plus coverage where the
 * warp clips) that the warp role turns into both engine paths. The GPU bodies live in the
 * kit (`warpMaps`); these functions only wire slots into them.
 */
import type {Expr, KitTexture} from '../gpu/contract'
import {call, expr, floatE, animatedTime} from '../gpu/porters'
import {warpMaps, edges} from '../gpu/kit/index'
import {lerpToIdentity, selectMap} from '../gpu/scaffolds/uvRemapShader'
import type {UvMapSource, UvRemapHookParams} from '../gpu/scaffolds/uvRemapShader'
import type {ArgSpec} from './invoke'
import {resolveScalar, uniformOf} from './invoke'
import {Scalar} from './values'
import type {GatherEffect} from './types'
import type {PropRef} from './values'

/**
 * Resolve a warp slot to an Expr: prop bindings, number literals, and Scalar value-graphs
 * (including time signals — which ride the node's animated clock here, since warp hooks
 * carry no `ctx`; declare `animatedTime: {speed: '…'}` on the definition).
 */
function arg(spec: ArgSpec, params: UvRemapHookParams): Expr {
    if (typeof spec === 'number') return floatE(spec)
    if (spec instanceof Scalar) return resolveScalar(spec, params)
    if (typeof spec === 'object' && spec !== null && 'kind' in spec) {
        if (spec.kind === 'prop') return uniformOf(spec as PropRef, params)
        return resolveScalar(spec as never, params)
    }
    throw new Error('std/warps: warp slots accept prop bindings, numbers, and Scalar signals')
}

/**
 * Reflect content across the line through `center` at `angle` (degrees).
 *
 * The reflection body returns a hard 0/1 side selector alongside the reflected coordinate,
 * so `selectMap` picks per pixel between the original coordinate (near side) and the
 * reflected one (far side). Selecting COORDINATES and sampling once is pixel-identical to
 * mixing two sampled colors by the same hard selector, with one texture fetch instead of
 * two, and both halves go through the same reconstruction filter so the filtering never
 * changes across the mirror line.
 */
export function mirrorLine(slots: {center: ArgSpec; angle: ArgSpec}): UvMapSource {
    return (params) => {
        // `Expr` has no CSE — `member()` re-emits its whole subtree at each use site — so calling
        // this twice emits `mirrorReflect(...)` twice in the WGSL. Same text, no extra cost.
        const reflect = (uv: Expr, aspect: Expr): Expr =>
            call(warpMaps.mirrorReflect, 'mirrorReflect', [arg(slots.center, params), arg(slots.angle, params), uv, aspect])
        return selectMap(
            (uv) => uv,
            (uv, aspect) => reflect(uv, aspect).member('xy'),
            (uv, aspect) => reflect(uv, aspect).member('z'),
        )
    }
}

/**
 * Mirror content across the canvas axes. `flipX`/`flipY` bind boolean props stored as ±1
 * uniforms (transformBoolean). The map sends [0,1] onto [0,1], so pair it with
 * `edges: 'none'` — no lookup can land outside the source.
 */
export function flip(slots: {flipX: ArgSpec; flipY: ArgSpec}): UvMapSource {
    return (params) => ({
        map: (uv) => call(warpMaps.flipUV, 'flipUV', [uv, arg(slots.flipX, params), arg(slots.flipY, params)]),
    })
}

/**
 * Bend the ends of the frame toward (or away from) the viewer like a curved display —
 * the inverse bent-sheet projection, under real perspective. `angle` sets the bend axis in
 * degrees; `falloff` concentrates the bend toward the ends.
 */
export function bend(slots: {strength: ArgSpec; falloff: ArgSpec; angle: ArgSpec}): UvMapSource {
    return (params) => ({
        map: (uv, aspect) => call(warpMaps.bendRemap, 'bendRemap', [
            arg(slots.strength, params), arg(slots.falloff, params), arg(slots.angle, params), uv, aspect,
        ]),
    })
}

/**
 * Magnify or pinch content around `center` — positive strength bulges out, negative
 * pinches in, with `radius`/`falloff` shaping the affected disc.
 */
export function bulge(slots: {center: ArgSpec; strength: ArgSpec; radius: ArgSpec; falloff: ArgSpec}): UvMapSource {
    return (params) => ({
        map: (uv, aspect) => call(warpMaps.bulgeUV, 'bulgeUV', [
            arg(slots.center, params), arg(slots.strength, params), arg(slots.radius, params), arg(slots.falloff, params), uv, aspect,
        ]),
    })
}

/** Rotate and twist content around `center`, with the twist angle growing with distance. */
export function twirl(slots: {center: ArgSpec; intensity: ArgSpec}): UvMapSource {
    return (params) => ({
        map: (uv, aspect) => call(warpMaps.twirlUV, 'twirlUV', [
            arg(slots.center, params), arg(slots.intensity, params), uv, aspect,
        ]),
    })
}

/** Fold content into radial mirrored segments around `center` (`angle` in degrees). */
export function kaleidoscope(slots: {center: ArgSpec; segments: ArgSpec; angle: ArgSpec}): UvMapSource {
    return (params) => ({
        map: (uv, aspect) => call(warpMaps.kaleidoscopeUV, 'kaleidoscopeUV', [
            arg(slots.center, params), arg(slots.segments, params), arg(slots.angle, params), uv, aspect,
        ]),
    })
}

/**
 * Stretch content along a direction (`angle` in degrees) away from `center`, with
 * `falloff` controlling how sharply the stretch ramps in past the center.
 */
export function stretch(slots: {center: ArgSpec; strength: ArgSpec; angle: ArgSpec; falloff: ArgSpec}): UvMapSource {
    return (params) => ({
        map: (uv, aspect) => call(warpMaps.stretchUV, 'stretchUV', [
            arg(slots.center, params), arg(slots.strength, params), arg(slots.angle, params), arg(slots.falloff, params), uv, aspect,
        ]),
    })
}

/** Waveform names → mode numbers, matching the `waveType` prop transform. */
const WAVE_MODES: Record<string, number> = {sine: 0, triangle: 1, square: 2, sawtooth: 3, bounce: 4}

/** Normalise a `waveType` prop value (mapped number, or a raw string for robustness). */
function waveTypeMode(raw: unknown): number {
    if (typeof raw === 'number') return raw
    return WAVE_MODES[raw as string] ?? 0
}

/** Pick the active waveform body from the compile-time waveType (0=sine … 4=bounce). */
function waveBodyFor(waveType: number): {fn: unknown; hint: string} {
    switch (waveType) {
        case 1: return {fn: warpMaps.waveDistortTriangle, hint: 'waveDistortTriangle'}
        case 2: return {fn: warpMaps.waveDistortSquare, hint: 'waveDistortSquare'}
        case 3: return {fn: warpMaps.waveDistortSawtooth, hint: 'waveDistortSawtooth'}
        case 4: return {fn: warpMaps.waveDistortBounce, hint: 'waveDistortBounce'}
        default: return {fn: warpMaps.waveDistortSine, hint: 'waveDistortSine'}
    }
}

/**
 * Displace content along a direction by an animated wave. `waveType` binds a compile-time
 * prop (sine/triangle/square/sawtooth/bounce): the producer branches once per composition
 * and only the active waveform body is emitted — no runtime branch. The phase advances by
 * the definition's animated clock (declare `animatedTime: {speed: 'speed'}`), at half rate.
 */
export function wave(slots: {strength: ArgSpec; frequency: ArgSpec; angle: ArgSpec; waveType: PropRef}): UvMapSource {
    return (params) => {
        const {fn, hint} = waveBodyFor(waveTypeMode(params.propValues[slots.waveType.name]))
        const t = animatedTime(params).mul(0.5)
        return {
            map: (uv, aspect) => call(fn, hint, [
                uv, aspect, arg(slots.angle, params), arg(slots.frequency, params), arg(slots.strength, params), t,
            ]),
        }
    }
}

/**
 * Convert rectangular coordinates to polar space around `center` (`wrap` scales the
 * angular range, `radius` the radial one), lerped toward the identity by `intensity` —
 * 0 leaves the content alone, 1 is the full coordinate-system change. `lerpToIdentity`
 * blends COORDINATES, not colors, so intermediate values are a genuine partial warp
 * rather than a cross-fade.
 */
export function toPolar(slots: {center: ArgSpec; wrap: ArgSpec; radius: ArgSpec; intensity: ArgSpec}): UvMapSource {
    return (params) => lerpToIdentity(
        {
            map: (uv, aspect) => call(warpMaps.polarCoordsUV, 'polarCoordsUV', [
                arg(slots.center, params), arg(slots.wrap, params), arg(slots.radius, params), uv, aspect,
            ]),
        },
        () => arg(slots.intensity, params),
    )
}

/**
 * Convert polar coordinates back to rectangular space around `center` (the mirror of
 * {@link toPolar}), lerped toward the identity by `intensity`.
 */
export function fromPolar(slots: {center: ArgSpec; scale: ArgSpec; intensity: ArgSpec}): UvMapSource {
    return (params) => lerpToIdentity(
        {
            map: (uv, aspect) => call(warpMaps.rectCoordsUV, 'rectCoordsUV', [
                arg(slots.center, params), arg(slots.scale, params), uv, aspect,
            ]),
        },
        () => arg(slots.intensity, params),
    )
}

/**
 * Pin each corner of the content to an arbitrary position — a true square→quad corner-pin
 * homography, inverted analytically. The body returns a packed `vec3f` (projected
 * coordinate in `.xy`, front-facing gate in `.z`), so the map returns the COVERAGE form:
 * the warp role applies coverage to sampled alpha on the fragment path and to the coverage
 * mask on the analytic path, clipping the folded-back region either way. `amount` blends
 * each corner toward its neutral rectangle position.
 */
export function cornerPin(slots: {
    topLeft: ArgSpec
    topRight: ArgSpec
    bottomRight: ArgSpec
    bottomLeft: ArgSpec
    amount: ArgSpec
}): UvMapSource {
    return (params) => ({
        map: (uv) => {
            const packed = call(warpMaps.cornerPinSample, 'cornerPinSample', [
                uv, arg(slots.amount, params),
                arg(slots.topLeft, params), arg(slots.topRight, params), arg(slots.bottomRight, params), arg(slots.bottomLeft, params),
            ])
            return {uv: packed.member('xy'), coverage: packed.member('z')}
        },
    })
}

/**
 * Rotate the plane in 3D space with pan and tilt (degrees), `fov` controlling perspective
 * intensity and `zoom` scaling the sampled area. The map ignores `aspect` — it samples in
 * raw UV space, unlike the other warps here.
 */
export function perspective(slots: {
    center: ArgSpec
    pan: ArgSpec
    tilt: ArgSpec
    fov: ArgSpec
    zoom: ArgSpec
    offset: ArgSpec
}): UvMapSource {
    return (params) => ({
        map: (uv) => call(warpMaps.perspectiveUV, 'perspectiveUV', [
            arg(slots.center, params), arg(slots.pan, params), arg(slots.tilt, params),
            arg(slots.fov, params), arg(slots.zoom, params), arg(slots.offset, params), uv,
        ]),
    })
}

/**
 * Concentric rings that each rotate the content by different amounts — per-ring hashed
 * static + animated rotation, blended between adjacent rings along the shortest angular
 * path. The ring rotation advances by the definition's animated clock (declare
 * `animatedTime: {speed: 'speed'}`).
 */
export function concentricRings(slots: {
    center: ArgSpec
    intensity: ArgSpec
    rings: ArgSpec
    smoothness: ArgSpec
    seed: ArgSpec
    speedRandomness: ArgSpec
}): UvMapSource {
    return (params) => {
        const t = animatedTime(params)
        return {
            map: (uv, aspect) => call(warpMaps.concentricSpinUV, 'concentricSpinUV', [
                arg(slots.center, params), arg(slots.intensity, params), arg(slots.rings, params),
                arg(slots.smoothness, params), arg(slots.seed, params), arg(slots.speedRandomness, params),
                t, uv, aspect,
            ]),
        }
    }
}

/**
 * Fluid-like distortion with constant smooth motion: three noise layers drifting through a
 * 3D field, combined and normalised so the flow speed is constant and `strength` is the
 * only magnitude control. Reads TWO animated clocks — the primary flow drift and the
 * `evolution` extra clock (declare `animatedTime: {speed: ...}` plus
 * `extraAnimatedTimes: {evolution: ...}`), each at 1/10 rate. `seed` binds a prop added to
 * both clocks on the GPU, shifting the noise domain along the drift/evolution axes — a
 * different static pattern per seed value, even at speed 0.
 */
export function flowNoise(slots: {strength: ArgSpec; detail: ArgSpec; seed: PropRef}): UvMapSource {
    return (params) => {
        const time = animatedTime(params, slots.seed.name).mul(0.1)
        const evolutionTime = animatedTime(params, slots.seed.name, '_animTime_evolution').mul(0.1)
        return {
            map: (uv, aspect) => call(warpMaps.flowFieldUV, 'flowFieldUV', [
                uv, aspect, arg(slots.detail, params), arg(slots.strength, params), time, evolutionTime,
            ]),
        }
    }
}

/**
 * Slice the content into strips that slide away in alternating directions — the "shredder"
 * transition. At progress 1 every lookup is out of bounds, so pair it with a fixed
 * TRANSPARENT edge mode (`edges: 1`): clipping the vacated space is what makes it a wipe.
 */
export function slicedSlide(slots: {angle: ArgSpec; count: ArgSpec; progress: ArgSpec}): UvMapSource {
    return (params) => ({
        map: (uv, aspect) => call(warpMaps.sliceWipeUV, 'sliceWipeUV', [
            uv, aspect, arg(slots.angle, params), arg(slots.count, params), arg(slots.progress, params),
        ]),
    })
}

/**
 * Slice content into parallel bars, each offset independently by a hash for a fractured or
 * glitch-like look. Each bar drifts at its own hashed rate by the definition's animated
 * clock (declare `animatedTime: {speed: 'speed'}`).
 */
export function barOffset(slots: {count: ArgSpec; angle: ArgSpec; intensity: ArgSpec; seed: ArgSpec}): UvMapSource {
    return (params) => {
        const t = animatedTime(params)
        return {
            map: (uv, aspect) => call(warpMaps.barShiftUV, 'barShiftUV', [
                arg(slots.count, params), arg(slots.angle, params), arg(slots.intensity, params),
                arg(slots.seed, params), t, uv, aspect,
            ]),
        }
    }
}

/**
 * Displace by this node's own compute-generated displacement field, sampled per grid cell:
 * snap the incoming UV to the centre of its `gridSize` cell, read the displacement texture
 * there, offset (±0.1 clamp), then edge-handle.
 *
 * When the compute texture is absent (GPU-free composition, or before the compute hook has
 * run) the map substitutes a zero displacement and keeps its structure identical — pair the
 * definition with `uvRemapIdentityWhen` so the analytic path returns the incoming remap
 * verbatim in that case instead.
 */
export function gridCellDisplace(slots: {gridSize: ArgSpec; output: string}): UvMapSource {
    return (params) => ({
        map: (uv, aspect) => {
            const dispTex = params.computeOutputs?.[slots.output] as KitTexture | undefined
            const gridCellUV = call(warpMaps.gridCellSnap, 'gridCellSnap', [uv, arg(slots.gridSize, params), aspect])
            const disp = dispTex ? dispTex.sample(gridCellUV, 'linearClamp').member('xy') : expr('vec2f(0.0, 0.0)')
            return call(warpMaps.gridDistortOffsetUV, 'gridDistortOffsetUV', [uv, disp])
        },
    })
}

/**
 * Displace by this node's own compute-generated displacement field, sampled continuously at
 * the incoming UV and scaled by `intensity` (±0.15 clamp) — the liquid/cloth look. Same
 * absent-texture handling as {@link gridCellDisplace}: zero displacement here, with
 * `uvRemapIdentityWhen` covering the analytic path on the definition.
 */
export function liquidDisplace(slots: {intensity: ArgSpec; output: string}): UvMapSource {
    return (params) => ({
        map: (uv) => {
            const dispTex = params.computeOutputs?.[slots.output] as KitTexture | undefined
            const disp = dispTex ? dispTex.sample(uv, 'linearClamp').member('xy') : expr('vec2f(0.0, 0.0)')
            return call(warpMaps.liquifyOffsetUV, 'liquifyOffsetUV', [uv, disp, arg(slots.intensity, params)])
        },
    })
}

/** Channel-mode names → mode numbers, matching the `channelMode` prop transform. */
const DISPLACE_CHANNEL_MODES: Record<string, number> = {twoAxis: 0, directional: 1}

/** Normalise a `channelMode` prop value (mapped number, or a raw string for robustness). */
function displaceChannelMode(raw: unknown): number {
    if (typeof raw === 'number') return raw
    return DISPLACE_CHANNEL_MODES[raw as string] ?? 0
}

/**
 * Distort the child using ANOTHER LAYER's pixels as a displacement map — the classic
 * Photoshop/AE displacement, driven cross-layer rather than by a procedural field, hence a
 * gather effect (the source arrives via `getLayerTexture`, the same shared RTT boundary
 * prop maps use, so the layer renders once even when it's also visible on canvas). An
 * animated source layer displaces live.
 *
 * `source` binds the compile-time layer prop; `channels` picks the applier at composition
 * (red/green two-axis vs luminance along `angle`); `edges` binds the compile-time edge
 * mode. No source selected (or an unresolvable id) → a sharp passthrough of the child,
 * exactly as sampled at the incoming coordinate.
 */
export function displaceByLayer(slots: {
    source: PropRef
    amount: ArgSpec
    channels: PropRef
    angle: ArgSpec
    edges: PropRef
}): GatherEffect {
    return {
        kind: 'gather',
        build: (params) => {
            const source = params.getLayerTexture(slots.source.name)
            if (!source) return params.texture.sample(params.ctx.uv)
            const edgeMode = (params.propValues[slots.edges.name] as number) ?? 0
            const src = source.sample(params.ctx.uv)
            const distorted = displaceChannelMode(params.propValues[slots.channels.name]) === 1
                ? call(warpMaps.dmDisplaceLuminance, 'dmDisplaceLuminance', [params.ctx.uv, src, params.ctx.aspect, arg(slots.amount, params), arg(slots.angle, params)])
                : call(warpMaps.dmDisplaceRG, 'dmDisplaceRG', [params.ctx.uv, src, params.ctx.aspect, arg(slots.amount, params)])
            return edges.sampleRemappedExpr(params.texture, distorted, edgeMode)
        },
    }
}

/**
 * std/effects — lens vocabulary: chromatic split sampling.
 *
 * The rgbSplit family (see `blurs.ts`) reads the child RTT at per-channel offset UVs and
 * recombines one channel from each tap. This module carries the directional per-channel
 * variant: a fan of three taps along one angle, each channel with its own signed offset
 * multiplier. Taps are premultiplied — the gather species owns the unpremultiply tail.
 */
import type {GpuFragmentParams, EmitContext} from '../../gpu/contract'
import {Expr, formatFloat} from '../../gpu/contract'
import {call, vec4, expr, ZERO} from '../../gpu/composer'
import {tgpu, d, std, blend} from '../../gpu/kit/index'
import * as lensParts from '../../gpu/kit/lensParts'
import {resolveArg, type ArgSpec} from '../invoke'
import type {PropRef} from '../values'
import type {SampleStage} from './blurs'

// ── Kernel ──────────────────────────────────────────────────────────────────────────────────
//
// The child sample UV for one channel. `angle` is the TRANSFORMED prop value (degrees,
// transformAngle normalises to 0–360), so `std.radians(angle)` recovers the direction. Aspect-
// correct X on the direction so the angle reads visually correct on any canvas; strength scales
// 0–1 to a 0–0.1 UV offset. Called three times (one channelOffset per channel).
export const chromaticOffsetUV = tgpu.fn([d.vec2f, d.f32, d.f32, d.f32, d.f32], d.vec2f)(
    (uv, angle, strength, aspect, channelOffset) => {
        'use gpu'
        const angleRad = std.radians(angle)
        const direction = d.vec2f(std.cos(angleRad) / aspect, std.sin(angleRad))
        const scaledStrength = strength * 0.1
        const offsetBase = direction.mul(scaledStrength)
        return uv.add(offsetBase.mul(channelOffset))
    })

/**
 * Chromatic fan: sample the child at three per-channel offset UVs along `angle` (each channel's
 * signed multiplier × `strength`) and recombine R/G/B — alpha from the green tap, so a centred
 * green (offset 0) keeps the silhouette put. Not a uvRemap candidate: three DIFFERENT sample UVs
 * can't fold into one coordinate.
 */
export function chromaticFan(opts: {
    strength: ArgSpec
    angle: ArgSpec
    red: ArgSpec
    green: ArgSpec
    blue: ArgSpec
}): SampleStage {
    return (params) => {
        const tapUV = (offset: ArgSpec): Expr =>
            call(chromaticOffsetUV, 'chromaticOffsetUV', [
                params.ctx.uv, resolveArg(opts.angle, params), resolveArg(opts.strength, params),
                params.ctx.aspect, resolveArg(offset, params),
            ])
        const redSample = params.texture.sample(tapUV(opts.red))
        const greenSample = params.texture.sample(tapUV(opts.green))
        const blueSample = params.texture.sample(tapUV(opts.blue))
        return vec4(redSample.member('r'), greenSample.member('g'), blueSample.member('b'), greenSample.member('a'))
    }
}

// ── Spectral lens ───────────────────────────────────────────────────────────────────────────

/** The prop roles the spectral-lens recipe binds. */
export interface SpectralLensSlots {
    center: PropRef
    spread: PropRef
    bias: PropRef
    angle: PropRef
    perspective: PropRef
    count: PropRef
    dispersion: PropRef
    dispersionShift: PropRef
    dispersionColor: PropRef
    focusCenter: PropRef
    focusEdges: PropRef
    swirl: PropRef
    noise: PropRef
    noiseFrequency: PropRef
    noiseOffset: PropRef
    lensBulge: PropRef
    lensCircle: PropRef
    grainMixer: PropRef
    grainOverlay: PropRef
}

/**
 * The spectral-lens recipe: split the child into shifting chromatic layers (a runtime-count
 * gather fan) with barrel/pincushion lens warp, optional circular crop, swirl, noise scatter and
 * film grain. Composed from the kit's lens statement parts — lensGeometry → spreadAxis →
 * spectralFan → grainOverlay, all appending to one statement stream (they share locals via the
 * fresh prefix). The fan loop is raw WGSL because per-iteration texture sampling can't fold
 * into the Expr graph; the accumulated result is premultiplied → straight alpha on the way out
 * like every RTT filter.
 */
export function spectralLens(slots: SpectralLensSlots): (params: GpuFragmentParams) => Expr {
    return (params) => {
        const {uniforms, childNode, ctx, convertToTexture} = params
        if (!childNode) return ZERO

        const childTex = convertToTexture(childNode)
        const texKey = childTex.key

        return new Expr((ec: EmitContext) => {
            const em: lensParts.LensEmitFrame = {
                p: ec.freshLocal('lens'),
                stmts: [],
                U: (name: string): string => uniforms[slots[name as keyof SpectralLensSlots].name]._emit(ec),
                L: (n: number): string => formatFloat(n),
                noiseIdx: {value: 0},
            }

            lensParts.lensGeometryStmts(em, ctx.uv._emit(ec), ctx.aspect._emit(ec))
            lensParts.spreadAxisStmts(em)
            lensParts.spectralFanStmts(em, texKey)
            lensParts.grainOverlayStmts(em)

            for (const s of em.stmts) ec.statement(s)

            return call(blend.unpremultiplyAlpha, 'unpremultiplyAlpha', [expr(`${em.p}_out`)])._emit(ec)
        })
    }
}

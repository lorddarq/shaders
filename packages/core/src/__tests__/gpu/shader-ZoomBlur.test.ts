import {describe, it, expect} from 'vitest'
import {tgpu, d} from '@coreroot/gpu/kit'
import {composeNodeTree, call} from '@coreroot/gpu/composer'
import type {GpuShaderDefinition, GpuFragmentParams, Expr} from '@coreroot/gpu/contract'
import ZoomBlur from '@coreroot/shaders/ZoomBlur/index'
import {zoomBlurTapCoord, MOTION_BLUR_WEIGHTS, MOTION_BLUR_TAP_COUNT} from '@coreroot/gpu/kit/motionBlur'
import {buildRegistry, RootContainer} from './_patternHarness'

/**
 * ZoomBlur port gate (W7-D) — an RTT multi-tap FILTER (requiresRTT/requiresChild, no uvRemap), the
 * AngularBlur twin. It RTTs the child and accumulates 32 weighted Gaussian samples scaled radially
 * from `center`, then unpremultiplies (RTT premultiplied — Twirl trap #2). v1's JS `for` loop unrolls
 * at build; this builder unrolls 32 `texture.sample()` calls, each tap's scaled coord from a shared
 * body fn (`1 + radius·(tap/31)`, radius = intensity·0.01). CPU-goldened.
 */
const ZB = ZoomBlur as GpuShaderDefinition

const genBody = tgpu.fn([d.vec2f], d.vec4f)((uv) => {
    'use gpu'
    return d.vec4f(uv.x, uv.y, 0.5, 1.0)
})
const Generator: GpuShaderDefinition = {
    name: 'Generator',
    props: {} as never,
    fragment: ({ctx}: GpuFragmentParams): Expr => call(genBody, 'genBody', [ctx.uv]),
}

describe('ZoomBlur (a) RTT multi-tap filter path', () => {
    it('RTTs the child, unrolls 32 taps of zoomBlurTapCoord, unpremultiplies', () => {
        const {registry} = buildRegistry([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'zb', def: ZB, parentId: 'root', metadata: {renderOrder: 0}},
            {id: 'gen', def: Generator, parentId: 'zb', metadata: {renderOrder: 0}},
        ])
        const ir = composeNodeTree(registry)
        expect(ir.rttPasses.length).toBe(1)
        const finalWgsl = tgpu.resolve([ir.finalPass.entry], {names: 'strict'})
        expect(finalWgsl).toMatch(/zoomBlurTapCoord/)
        expect(finalWgsl).toMatch(/unpremultiplyAlpha/)
        expect(finalWgsl).toMatch(/textureSample\(rtt_/)
        expect((finalWgsl.match(/textureSample\(/g) ?? []).length).toBe(MOTION_BLUR_TAP_COUNT)
        expect(finalWgsl).toMatchSnapshot('final-pass')
    })
})

describe('ZoomBlur (b) weights', () => {
    it('normalized Gaussian weights sum to 1 over 32 taps', () => {
        expect(MOTION_BLUR_WEIGHTS.length).toBe(32)
        expect(MOTION_BLUR_WEIGHTS.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10)
    })
})

describe('ZoomBlur (c) CPU golden — scaled tap coordinate', () => {
    it('scales the (aspect-corrected) offset from center by 1 + intensity·0.01·(tap/31), un-corrects', () => {
        // center transformed = (cx, 1 - cy_authored); the body flips y back via 1 - center.y.
        const cases = [
            {center: [0.5, 0.5] as const, intensity: 30, uv: [0.7, 0.6] as const, aspect: 1.5, tap: 0},
            {center: [0.5, 0.5] as const, intensity: 100, uv: [0.2, 0.3] as const, aspect: 1.0, tap: 31},
            {center: [0.3, 0.4] as const, intensity: 50, uv: [0.6, 0.9] as const, aspect: 2.0, tap: 16},
        ]
        for (const {center, intensity, uv, aspect, tap} of cases) {
            const cpX = center[0]
            const cpY = 1 - center[1]
            const radius = intensity * 0.01
            const scale = 1 + radius * (tap / 31)
            const acdX = (uv[0] - cpX) * aspect
            const acdY = uv[1] - cpY
            const ex = acdX / scale / aspect + cpX
            const ey = acdY / scale + cpY
            const out = zoomBlurTapCoord(
                d.vec2f(center[0], center[1]), intensity, d.vec2f(uv[0], uv[1]), aspect, tap,
            ) as unknown as {x: number; y: number}
            expect(out.x).toBeCloseTo(ex, 5)
            expect(out.y).toBeCloseTo(ey, 5)
        }
        // tap 0 → scale 1 → the identity sample (aspect multiply/divide cancel) → returns uv unchanged.
        const id = zoomBlurTapCoord(d.vec2f(0.5, 0.5), 30, d.vec2f(0.7, 0.6), 1.5, 0) as unknown as {x: number; y: number}
        expect(id.x).toBeCloseTo(0.7, 6)
        expect(id.y).toBeCloseTo(0.6, 6)
    })
})

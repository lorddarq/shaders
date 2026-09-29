import {describe, it, expect} from 'vitest'
import {tgpu} from '@coreroot/gpu/kit'
import {composeNodeTree, collectStructuralHashInputs} from '@coreroot/gpu/composer'
import type {GpuShaderDefinition} from '@coreroot/gpu/contract'
import Hologram from '@coreroot/shaders/Hologram/index'
import {buildRegistry, RootContainer} from './_patternHarness'

/**
 * Hologram gate (W7-A → std sweep). A GENERATOR whose volumetric-hologram material is std algebra
 * in the definition file (optical thickness + fresnel rim + depth scan-lines + CRT
 * scanlines/sweep/flicker/grain + chromatic fringe, inside the `guarded` shape region; the beam
 * wobble is algebra in the spine's placeUV slot). GPU-free → flat analytic path. Validates the
 * composition, the wobble-displaced sampling, and the animated `_animTime` read.
 */
const H = Hologram as GpuShaderDefinition

describe('Hologram (a) default analytic projection generator', () => {
    it('emits the analytic circle sampler + the guarded projection algebra + hash grain, no RTT pass', () => {
        const {registry} = buildRegistry([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'h', def: H, parentId: 'root', metadata: {renderOrder: 0}},
        ])
        const ir = composeNodeTree(registry)
        expect(ir.rttPasses.length).toBe(0) // generator
        const finalWgsl = tgpu.resolve([ir.finalPass.entry], {names: 'strict'})
        expect(finalWgsl).toMatch(/analyticSdf_circleSDF/)
        expect(finalWgsl).toMatch(/sdfSpaceUV/)
        expect(finalWgsl).toMatch(/outsideShape/)
        expect(finalWgsl).toMatch(/hash12/)
        expect(finalWgsl).toMatch(/mxNoiseFloat2/)
        expect(finalWgsl).toMatch(/_animTime/)
        expect(finalWgsl).toMatch(/_saRadius/)
        expect(finalWgsl).toMatchSnapshot('final-pass')
    })

    it('registers the animatedTime clock (speed prop) in the structural hash surface', () => {
        expect(H.animatedTime).toEqual({speed: 'speed'})
        const {registry} = buildRegistry([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'h', def: H, parentId: 'root', metadata: {renderOrder: 0}},
        ])
        expect(collectStructuralHashInputs(registry).join('\n')).toContain('Hologram')
    })
})

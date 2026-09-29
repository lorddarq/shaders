import {describe, it, expect} from 'vitest'
import {tgpu} from '@coreroot/gpu/kit'
import {composeNodeTree, collectStructuralHashInputs} from '@coreroot/gpu/composer'
import type {GpuShaderDefinition} from '@coreroot/gpu/contract'
import Chrome from '@coreroot/shaders/Chrome/index'
import {buildRegistry, RootContainer} from './_patternHarness'

/**
 * Chrome gate (std sweep). A GENERATOR whose studio-chrome material is std algebra in the
 * definition file over shared kit parts (controllable multi-facet bevel via the shared kit
 * bevelSin + sphere/pillow face dome + the shared clamped-Perlin-gradient waviness + Chrome's own
 * inline studio, tapped 3× spectrally-offset on desktop + PBR-neutral shoulder), inside the
 * `guarded` shape region. GPU-free → flat analytic path (default roundedRectSDF); the compute
 * volumetric field is skipped. Validates: the analytic sampler + the shared material parts +
 * `bevelSin` + the perlin relief + the animated `_animTime` read all reach the emitted WGSL.
 */
const C = Chrome as GpuShaderDefinition

describe('Chrome (a) default analytic studio-chrome generator', () => {
    it('emits the analytic roundedRect sampler + the shared material parts + bevelSin, no RTT pass', () => {
        const {registry} = buildRegistry([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'c', def: C, parentId: 'root', metadata: {renderOrder: 0}},
        ])
        const ir = composeNodeTree(registry)
        expect(ir.rttPasses.length).toBe(0) // generator
        const finalWgsl = tgpu.resolve([ir.finalPass.entry], {names: 'strict'})
        expect(finalWgsl).toMatch(/analyticSdf_roundedRectSDF/)
        expect(finalWgsl).toMatch(/sdfSpaceUV/)
        expect(finalWgsl).toMatch(/silhouetteAlpha/)
        expect(finalWgsl).toMatch(/clampedPerlinGrad/)
        expect(finalWgsl).toMatch(/nudgeNormal/)
        expect(finalWgsl).toMatch(/tonemapNeutral/)
        expect(finalWgsl).toMatch(/bevelSin/)
        expect(finalWgsl).toMatch(/perlin12d/)
        expect(finalWgsl).toMatch(/_animTime/)
        expect(finalWgsl).toMatch(/_saRadius/)
        expect(finalWgsl).toMatchSnapshot('final-pass')
    })

    it('registers the animatedTime clock (speed prop) in the structural hash surface', () => {
        expect(C.animatedTime).toEqual({speed: 'speed'})
        const {registry} = buildRegistry([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'c', def: C, parentId: 'root', metadata: {renderOrder: 0}},
        ])
        expect(collectStructuralHashInputs(registry).join('\n')).toContain('Chrome')
    })
})

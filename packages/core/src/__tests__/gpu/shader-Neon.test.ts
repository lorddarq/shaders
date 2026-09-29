import {describe, it, expect} from 'vitest'
import {tgpu} from '@coreroot/gpu/kit'
import {composeNodeTree} from '@coreroot/gpu/composer'
import type {GpuShaderDefinition} from '@coreroot/gpu/contract'
import Neon from '@coreroot/shaders/Neon/index'
import {buildRegistry, RootContainer} from './_patternHarness'

/**
 * Neon port gate (W6-C). Neon is a GENERATOR (no RTT child) — it renders the neon tube on a
 * transparent cutout. GPU-free: the compute volumetric pre-march is skipped (no device) so the
 * fragment takes the FLAT analytic path (default circleSDF). The tube-shading / glow / flicker math
 * is golden-tested in kit-effects-lighting.test.ts; this checks the shader's routing + composition.
 */
const N = Neon as GpuShaderDefinition

describe('Neon (a) default analytic circle generator', () => {
    it('emits the analytic circle sampler + the neon composite, no RTT pass', () => {
        const {registry} = buildRegistry([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'n', def: N, parentId: 'root', metadata: {renderOrder: 0}},
        ])
        const ir = composeNodeTree(registry)
        expect(ir.rttPasses.length).toBe(0) // generator — no child RTT
        const finalWgsl = tgpu.resolve([ir.finalPass.entry], {names: 'strict'})
        expect(finalWgsl).toMatch(/analyticSdf_circleSDF/)
        expect(finalWgsl).toMatch(/neonComposite/)
        expect(finalWgsl).toMatch(/sdfSpaceUV/)
        // color uniforms are vec4 rgba — the builder swizzles `.rgb` for the vec3-taking composite.
        expect(finalWgsl).toMatch(/_saRadius/)
        expect(finalWgsl).toMatchSnapshot('final-pass')
    })
})

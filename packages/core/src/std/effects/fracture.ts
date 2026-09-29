/**
 * std/effects/fracture — Voronoi-fracture vocabulary: a nearest/second-nearest region field
 * precomputed on the GPU, per-cell data-lane addressing, and the crack render parts (crack
 * geometry from the two nearest sites, per-channel refracted taps, tilted-shard composite).
 */
import type {GpuComputeNode, GpuFragmentParams} from '../../gpu/contract'
import {createGuardedCompute, createStateBuffer} from '../../gpu/porters'
import {tgpu, d, std} from '../../gpu/kit/index'

const VORONOI_FORMAT = 'rgba16float' as const

interface VoronoiGraph {
    layout: ReturnType<typeof makeVoronoiGraph>['layout']
    kernel: ReturnType<typeof makeVoronoiGraph>['kernel']
}

function makeVoronoiGraph(count: number, size: number) {
    const layout = tgpu.bindGroupLayout({
        cellPos: {storage: d.arrayOf(d.f32, count * 2), access: 'readonly'},
        voronoiTex: {storageTexture: d.textureStorage2d(VORONOI_FORMAT, 'write-only')},
    })
    /** Nearest + second-nearest cell per pixel. The running (d1, d2, idx1, idx2) selection is one
     *  serial scan whose intermediates every branch shares — an atomic core. */
    const kernel = tgpu.fn([d.u32, d.u32])((cx, cy) => {
        'use gpu'
        const u = (d.f32(cx) + 0.5) / size
        const v = (d.f32(cy) + 0.5) / size
        let d1 = d.f32(99999.0)
        let d2 = d.f32(99999.0)
        let idx1 = d.f32(0.0)
        let idx2 = d.f32(0.0)
        for (let i = 0; i < count; i++) {
            const cellX = layout.$.cellPos[i * 2]
            const cellY = layout.$.cellPos[i * 2 + 1]
            const ddx = u - cellX
            const ddy = v - cellY
            const distSq = ddx * ddx + ddy * ddy
            if (distSq < d1) {
                d2 = d1
                idx2 = idx1
                d1 = distSq
                idx1 = d.f32(i)
            } else {
                if (distSq < d2) {
                    d2 = distSq
                    idx2 = d.f32(i)
                }
            }
        }
        std.textureStore(layout.$.voronoiTex, d.vec2u(cx, cy), d.vec4f(idx1, idx2, 0.0, 0.0))
    }).$name('voronoiNearest2')
    return {layout, kernel}
}

const voronoiGraphCache = new Map<string, VoronoiGraph>()

/** The per-(count, size) Voronoi nearest-2 graph (both baked as compile-time constants). */
export function voronoiNearest2Graph(count: number, size: number): VoronoiGraph {
    const key = `${count}|${size}`
    let g = voronoiGraphCache.get(key)
    if (!g) {
        g = makeVoronoiGraph(count, size)
        voronoiGraphCache.set(key, g)
    }
    return g
}

/** UV of cell `idx`'s `field` slot in a 1-tall data texture of `laneWidth` texels (4 per cell). */
const cellLaneUVCache = new Map<number, ReturnType<typeof makeCellLaneUV>>()
function makeCellLaneUV(laneWidth: number) {
    return tgpu.fn([d.f32, d.f32], d.vec2f)((idx, field) => {
        'use gpu'
        return d.vec2f((idx * 4.0 + field + 0.5) / laneWidth, 0.5)
    })
}
export function cellLaneUVFor(laneWidth: number) {
    let fn = cellLaneUVCache.get(laneWidth)
    if (!fn) {
        fn = makeCellLaneUV(laneWidth)
        cellLaneUVCache.set(laneWidth, fn)
    }
    return fn
}

// Crack geometry for one pixel (nearest/second cell positions + the nearest cell's displacement).
export const CrackGeom = d.struct({
    crackIntensity: d.f32,
    edgeNormal: d.vec2f,
    displacedUV: d.vec2f,
    disp: d.vec2f,
})

export const crackGeom = tgpu.fn(
    [d.vec2f, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32],
    CrackGeom,
)((uv, nPosX, nPosY, sPosX, sPosY, dispX, dispY, crackWidth) => {
    'use gpu'
    const nearestPos = d.vec2f(nPosX, nPosY)
    const secondPos = d.vec2f(sPosX, sPosY)
    const disp = d.vec2f(dispX, dispY)
    const fromNearest = uv.sub(nearestPos)
    const minDist1 = std.length(fromNearest)
    const minDist2 = std.length(uv.sub(secondPos))
    const displacementMag = std.length(disp)
    const edgeDiff = minDist2 - minDist1
    const crackThreshold = crackWidth * 0.005
    const baseCrackIntensity = 1.0 - std.smoothstep(0.0, crackThreshold, edgeDiff)
    const displacementFactor = std.smoothstep(0.0, 0.01, displacementMag)
    const crackIntensity = baseCrackIntensity * displacementFactor
    const toNearest = fromNearest.div(std.max(minDist1, 1e-5))
    const edgeNormal = d.vec2f(toNearest.y * -1.0, toNearest.x)
    const displacedUV = uv.sub(disp)
    return CrackGeom({crackIntensity, edgeNormal, displacedUV, disp})
})

/** Per-channel refracted sample UV (channel −1/0/+1 → chromatic split). */
export const crackRefractUV = tgpu.fn([d.vec2f, d.vec2f, d.f32, d.f32, d.f32, d.f32], d.vec2f)(
    (displacedUV, edgeNormal, crackIntensity, refractionStrength, chromaticSplit, channel) => {
        'use gpu'
        const refractionOffset = edgeNormal.mul(refractionStrength * 0.01)
        const chromaticOffset = chromaticSplit * 0.005
        const chromaShift = edgeNormal.mul(channel * chromaticOffset)
        const offset = refractionOffset.add(chromaShift)
        return displacedUV.add(offset.mul(crackIntensity))
    },
)

// normalize(vec2(0.3, 0.6)), pre-folded so the shard light direction is a pair of literals.
const SHARD_LIGHT_DIR_X = 0.3 / Math.hypot(0.3, 0.6)
const SHARD_LIGHT_DIR_Y = 0.6 / Math.hypot(0.3, 0.6)

/** Final composite: crack refraction blend + tilted-shard lighting (pre-unpremultiply). */
export const crackShardCompose = tgpu.fn([d.vec4f, d.vec4f, d.vec4f, d.vec4f, d.f32, d.vec2f, d.f32], d.vec4f)(
    (normalColor, rFinal, gFinal, bFinal, crackIntensity, disp, shardLighting) => {
        'use gpu'
        const refractedRGB = d.vec3f(rFinal.x, gFinal.y, bFinal.z)
        const shadedRGB = normalColor.xyz.mul(1.0 - crackIntensity).add(refractedRGB.mul(crackIntensity))
        const displacementLength = std.length(disp)
        const tiltX = disp.x / (displacementLength + 0.001)
        const tiltY = disp.y / (displacementLength + 0.001)
        const normalDot = tiltX * SHARD_LIGHT_DIR_X + tiltY * SHARD_LIGHT_DIR_Y
        const lightingFactor = 1.0 + normalDot * shardLighting
        const lightingIntensity = std.smoothstep(0.0, 0.02, displacementLength)
        const finalLighting = 1.0 + (lightingFactor - 1.0) * lightingIntensity
        const finalRGB = shadedRGB.mul(finalLighting)
        return d.vec4f(finalRGB, normalColor.w)
    },
)

/**
 * The Voronoi region field as a compute part: nearest + second-nearest cell ID per pixel,
 * precomputed on the GPU and re-dispatched only when the seed prop changes (avoiding a
 * many-megaop CPU stall). Cell positions live in a small storage buffer regenerated CPU-side by
 * `sites(seed)` on seed change; between changes the field is static and the frame program
 * dispatches nothing. Published under `output`.
 */
export function voronoiRegionField(opts: {
    count: number
    size: number
    sites: (seed: number) => Float32Array
    seedProp: string
    seedDefault: number
    output: string
}): GpuComputeNode {
    return (params: GpuFragmentParams) => {
        const {gpu, registerComputeTexture, getCpuValue, onCleanup} = params
        const root = gpu?.root
        if (!root) return null // GPU-free (no device): fragment falls back to passthrough.

        const {layout, kernel} = voronoiNearest2Graph(opts.count, opts.size)
        const cellPos = createStateBuffer(root, d.f32, opts.count * 2)
        const voronoiTex = root.createTexture({size: [opts.size, opts.size], format: VORONOI_FORMAT}).$usage('storage', 'sampled')
        onCleanup(() => voronoiTex.destroy())
        const field = registerComputeTexture(voronoiTex)

        const bindGroup = root.createBindGroup(layout, {cellPos, voronoiTex})
        const pipeline = createGuardedCompute(root, (cx: number, cy: number) => {
            'use gpu'
            kernel(cx, cy)
        }, {size: [opts.size, opts.size], bindGroup})

        const generateCells = (seed: number): void => {
            cellPos.write(opts.sites(seed) as never)
        }

        let currentSeed = (getCpuValue(opts.seedProp) as number) ?? opts.seedDefault
        generateCells(currentSeed)
        let needsDispatch = true

        return {
            outputs: {[opts.output]: field},
            getComputeNodes: () => {
                const newSeed = (getCpuValue(opts.seedProp) as number) ?? opts.seedDefault
                if (newSeed !== currentSeed) {
                    currentSeed = newSeed
                    generateCells(currentSeed)
                    needsDispatch = true
                }
                if (needsDispatch) {
                    needsDispatch = false
                    return [pipeline]
                }
                return null // the field is static between seed changes.
            },
        }
    }
}

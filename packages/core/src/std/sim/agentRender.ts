/**
 * std/sim — `renderAgents`: the splat/resolve subsystem of the agent simulations.
 *
 * Every agent shader renders the same way: per-agent additive splat of an oriented/point shape
 * profile into fixed-point atomic accumulators, then one full-target resolve that reads-and-
 * zeroes them and writes the rgba16f output texture the fragment bilinear-upsamples. This
 * module owns that subsystem as kernel factories over the shader's layout, in four splat
 * variants (matching the state families) and three resolve variants:
 *
 *   splat: 'oriented-world' (Boids / MagneticFilings / ParticleFlow — heading-space shapes in
 *          world units), 'point-world' (FloatingParticles — live softness, placement/brightness
 *          slots), 'volume' (Particles — z-perspective + speed ramp accumulation), 'relief'
 *          (ParticleField — camera rows + depth-weighted RGBW color).
 *   resolve: 'ramp' (rest→excited color in a baked color space, optional trail canvas),
 *          'tint' (single color alpha), 'weighted-color' (RGBW average — ParticleField).
 *
 * Splat windows, gains and the alpha curve stay on the harness (`scaffolds/agentSystem`); the
 * shape menu and profile parts on `kit/agents`. Every resolve CLEARS the accumulators as it
 * reads them (one thread owns each cell, nothing else is in flight — read-then-zero needs no
 * exchange, and WebGPU zero-inits storage buffers so frame 0 is correct with no clear pass).
 */
import {tgpu, d, std, agents, colorMixing, noise} from '../../gpu/kit/index'
import {
    worldSplatWindow, texelSplatWindow, energyCoverageAlpha, FIXED_POINT_GAINS,
} from '../../gpu/scaffolds/agentSystem'
import type {Vec4StateArray, AgentHome2} from './agentForces'

type AtomicU32 = d.atomicU32
type StorageTex = d.textureStorage2d<'rgba16float', 'write-only'>

// ── Layout views ──────────────────────────────────────────────────────────────────────────
// Public factory signatures take the CORE shape every consumer of that variant has; variant-
// gated entries (`agit`, `trailBuf`, `speedNorm`, `exposure`) live on private full views the
// factory casts to once internally — a layout missing an entry its variant needs fails at
// resolve time, which every consumer's resolve gate exercises (the fluids-view precedent).

/** The world-splat core: per-agent vec4 state + the energy/excitement accumulators. */
export interface WorldSplatLayout {
    readonly $: {
        readonly agents: Vec4StateArray
        readonly accumE: AtomicU32[]
        readonly accumS: AtomicU32[]
        readonly params: {
            readonly aspect: number
            readonly bodyR: number
        }
    }
}

interface FullWorldSplatView {
    readonly $: WorldSplatLayout['$'] & {
        readonly agit: number[]
        readonly params: WorldSplatLayout['$']['params'] & {readonly speedNorm: number}
    }
}

/** The ramp-resolve core: the two accumulators, the output texture, the color endpoints. */
export interface RampResolveLayout {
    readonly $: {
        readonly accumE: AtomicU32[]
        readonly accumS: AtomicU32[]
        readonly outTex: StorageTex
        readonly params: {
            readonly colA: d.v4f
            readonly colB: d.v4f
        }
    }
}

interface FullRampResolveView {
    readonly $: RampResolveLayout['$'] & {
        readonly trailBuf: Vec4StateArray
        readonly params: RampResolveLayout['$']['params'] & {
            readonly trails: number
            readonly exposure: number
        }
    }
}

/** One agent's rendered pose: position, unit heading, speed (0 for angle-headed agents). */
const AgentPose = d.struct({pos: d.vec2f, dir: d.vec2f, spd: d.f32}).$name('AgentPose')

// ── Oriented world splat (Boids / MagneticFilings / ParticleFlow) ─────────────────────────

export interface OrientedWorldSplatConfig {
    shape: string
    /** Square accumulator resolution. */
    res: number
    /** Sanity cap on the body half-width, in texels (the window itself is dynamic). */
    splatRCap: number
    /** Where the agent's heading comes from: its velocity (xy = vel) or its angle (z = θ). */
    heading: 'velocity' | 'angle'
    /** Excitement source: the per-agent envelope buffer, or local speed / params.speedNorm. */
    agitation: 'buffer' | 'speed'
    /** Stretch the round glow into a comet along the heading (velocity consumers). */
    comet: boolean
    /** Derive the position from a home part + stored offset (MagneticFilings). */
    home?: AgentHome2
    name: string
}

/**
 * Rasterize one oriented agent into the accumulators by evaluating the baked shape's coverage
 * per texel in heading space (t along the heading, n perpendicular), in WORLD units so the
 * shape reads undistorted on any aspect. Hard shapes get a single-texel AA edge and over-drive
 * the fixed-point gain (interiors resolve fully opaque); the glow is a Gaussian falloff, comet-
 * stretched for the velocity consumers. The window is sized per agent from its ACTUAL accept
 * radius per axis and clipped once (the shared harness helper), and additive fixed-point
 * atomics are order-independent — no sort. Symmetric shapes (a plain dot, an un-stretched
 * glow) skip the heading rotation entirely — it cannot change a radial profile.
 */
function splatOrientedWorld(layout: WorldSplatLayout, cfg: OrientedWorldSplatConfig): (i: number) => void {
    const L = layout as FullWorldSplatView
    const profile = agents.orientedShapeProfile(cfg.shape, {comet: cfg.comet})
    const RES = cfg.res
    const INV_RES = 1.0 / RES
    const AA_W = 0.75 * INV_RES // hard-edge anti-alias half-width (¾ of a render texel, world units)
    const SPLAT_R = cfg.splatRCap
    const EXT = profile.ext
    const GAIN = profile.soft ? FIXED_POINT_GAINS.SOFT : FIXED_POINT_GAINS.HARD

    // Pose: where the agent is and which way it points, per the declared state family.
    const home = cfg.home
    const pose = cfg.heading === 'angle'
        ? tgpu.fn([d.u32], AgentPose)((i) => {
            'use gpu'
            const a = L.$.agents[i]
            const homeP = home!(d.f32(i))
            // Heading is the agent's own angle (compass-needle orientation), NOT a velocity.
            return AgentPose({
                pos: d.vec2f(homeP.x + a.x, homeP.y + a.y),
                dir: d.vec2f(std.cos(a.z), std.sin(a.z)),
                spd: d.f32(0),
            })
        }).$name(`${cfg.name}Pose`)
        : tgpu.fn([d.u32], AgentPose)((i) => {
            'use gpu'
            const a = L.$.agents[i]
            const vel = d.vec2f(a.z, a.w)
            const spd = std.length(vel)
            return AgentPose({
                pos: d.vec2f(a.x, a.y),
                dir: std.select(d.vec2f(1.0, 0.0), vel.div(std.max(spd, 1e-5)), spd > 1e-5),
                spd,
            })
        }).$name(`${cfg.name}Pose`)

    // Excitement for the rest→excited ramp: the per-agent envelope, or normalized local speed.
    const excitement = cfg.agitation === 'buffer'
        ? tgpu.fn([d.u32, d.f32], d.f32)((i, _spd) => {
            'use gpu'
            return std.clamp(L.$.agit[i], 0.0, 1.0)
        }).$name(`${cfg.name}Excitement`)
        : tgpu.fn([d.u32, d.f32], d.f32)((_i, spd) => {
            'use gpu'
            return std.clamp(spd / std.max(L.$.params.speedNorm, 1e-5), 0.0, 1.0)
        }).$name(`${cfg.name}Excitement`)

    // Per-texel coverage: the comet-stretched heading-space weight, the plain heading-space
    // weight, or — for symmetric shapes — the radial weight with the rotation baked out (it
    // cannot change a radial profile, so it was pure cost).
    const cometWeight = agents.makeOrientedCometWeightFn(cfg.shape)
    const plainWeight = agents.makeAgentTexelWeightFn(profile.name)
    const useComet = cfg.comet && !profile.symmetric
    const texelWeight = useComet
        ? tgpu.fn([d.f32, d.f32, d.f32, d.vec2f, d.f32, d.f32, d.f32], d.f32)((_dSq, offX, offY, dir, bodyR, _bodyRSq, aaW) => {
            'use gpu'
            const t = offX * dir.x + offY * dir.y
            const n = offX * -dir.y + offY * dir.x
            return cometWeight(t, n, bodyR, aaW)
        }).$name(`${cfg.name}Weight`)
        : profile.symmetric
            ? tgpu.fn([d.f32, d.f32, d.f32, d.vec2f, d.f32, d.f32, d.f32], d.f32)((dSq, offX, offY, _dir, bodyR, bodyRSq, aaW) => {
                'use gpu'
                return plainWeight(dSq / bodyRSq, offX, offY, bodyR, aaW)
            }).$name(`${cfg.name}Weight`)
            : tgpu.fn([d.f32, d.f32, d.f32, d.vec2f, d.f32, d.f32, d.f32], d.f32)((dSq, offX, offY, dir, bodyR, bodyRSq, aaW) => {
                'use gpu'
                const t = offX * dir.x + offY * dir.y
                const n = offX * -dir.y + offY * dir.x
                return plainWeight(dSq / bodyRSq, t, n, bodyR, aaW)
            }).$name(`${cfg.name}Weight`)

    return tgpu.fn([d.u32])((i) => {
        'use gpu'
        const prm = L.$.params
        const p = pose(i)
        const agitI = excitement(i, p.spd)
        const agitGain = agitI * d.f32(GAIN)

        const aspect = std.max(prm.aspect, 1e-5)
        const bodyR = std.clamp(prm.bodyR, 1e-6, d.f32(SPLAT_R) * INV_RES)
        const bodyRSq = bodyR * bodyR
        // Accept radius in WORLD units: the shape's radial extent plus the AA margin.
        const radW = bodyR * EXT + AA_W * 2.0
        const extSq = radW * radW

        // Per-axis window sized to the ACTUAL accept radius and clipped once (the shared harness
        // helper — the aspect asymmetry and the "no bounds test in the inner loop" trick live
        // there). An off-screen agent clips to an empty range.
        const win = worldSplatWindow(p.pos, radW, aspect, d.f32(RES))
        for (let py = win.y0; py <= win.y1; py++) {
            const offY = (d.f32(py) + 0.5) * INV_RES - p.pos.y
            const offYSq = offY * offY
            for (let px = win.x0; px <= win.x1; px++) {
                const offX = (d.f32(px) + 0.5) * INV_RES * aspect - p.pos.x
                const dSq = offX * offX + offYSq
                if (dSq < extSq) {
                    const w = texelWeight(dSq, offX, offY, p.dir, bodyR, bodyRSq, AA_W)
                    const e = d.u32(w * d.f32(GAIN))
                    if (e > d.u32(0)) {
                        const idx = d.u32(py * d.i32(RES) + px)
                        std.atomicAdd(L.$.accumE[idx], e)
                        // A calm agent's agitI is exactly 0 (the common case), and a zero add is
                        // still a full atomic round-trip — so skip its traffic entirely.
                        if (agitI > 0.0) {
                            std.atomicAdd(L.$.accumS[idx], d.u32(w * agitGain))
                        }
                    }
                }
            }
        }
    }).$name(cfg.name)
}

// ── Point world splat (FloatingParticles) ─────────────────────────────────────────────────

// The generic parts that feed the point splat's slots. Each is a factory over the consumer's
// layout (REQUIRED param names documented per part); the per-agent hash phases keep the field
// from ever phase-locking, and the presence cascade (kit `agents.presenceVariation`) couples
// size and brightness through ONE hash so near-reading agents are bigger AND brighter together.
export const pointSlot = {
    /** Placement: the stored position plus a circular orbital wander whose radius and rate
     *  scale with `params.randomness`, phase + rate hashed per agent. Needs
     *  `params.{time, randomness}`. */
    orbitalPlace(
        layout: {readonly $: {readonly params: {readonly time: number; readonly randomness: number}}},
        cfg: {radius: number; rate: number},
    ): PointWorldSplatConfig['place'] {
        const RATE = cfg.rate
        const RAD = cfg.radius
        const TAU = 6.283185307179586
        return tgpu.fn([d.f32, d.vec2f], d.vec2f)((fi, stored) => {
            'use gpu'
            const prm = layout.$.params
            const h = noise.hash22(d.vec2f(fi * 2.41 + 0.37, fi * 0.577 + 8.19))
            const rate = (0.5 + h.x) * RATE * prm.randomness
            const ph = h.y * TAU
            const rad = RAD * prm.randomness * (0.5 + h.x * 0.5)
            const t = prm.time * rate + ph
            return stored.add(d.vec2f(std.cos(t), std.sin(t)).mul(rad))
        }).$name('pointSlotOrbitalPlace') as PointWorldSplatConfig['place']
    },

    /** Brightness: a hash-phased sinusoidal twinkle (blend set by `params.twinkle`) × the
     *  presence cascade. Needs `params.{time, twinkle}`. */
    twinkleBrightness(
        layout: {readonly $: {readonly params: {readonly time: number; readonly twinkle: number}}},
        cfg: {freq: number},
    ): PointWorldSplatConfig['brightness'] {
        const FREQ = cfg.freq
        const TAU = 6.283185307179586
        return tgpu.fn([d.f32], d.f32)((fi) => {
            'use gpu'
            const prm = layout.$.params
            const ph = noise.hash11(fi * 3.7 + 0.93) * TAU
            const twk = std.sin(prm.time * FREQ + ph) * 0.5 + 0.5
            return std.mix(d.f32(1), twk, prm.twinkle) * (0.4 + 0.6 * agents.presenceVariation(fi))
        }).$name('pointSlotTwinkleBrightness') as PointWorldSplatConfig['brightness']
    },

    /** Body radius: the shared `params.bodyR` scaled by the presence cascade, floored. */
    presenceRadius(
        layout: {readonly $: {readonly params: {readonly bodyR: number}}},
    ): PointWorldSplatConfig['bodyRadius'] {
        return tgpu.fn([d.f32], d.f32)((fi) => {
            'use gpu'
            return std.max(layout.$.params.bodyR * (0.55 + 0.45 * agents.presenceVariation(fi)), 1e-6)
        }).$name('pointSlotPresenceRadius') as PointWorldSplatConfig['bodyRadius']
    },
} as const

export interface PointWorldSplatConfig {
    shape: string
    res: number
    /** Placement part: stored position → rendered position (the orbital wander). */
    place: (fi: number, stored: d.v2f) => d.v2f
    /** Brightness part: the twinkle × presence energy modulation. */
    brightness: (fi: number) => number
    /** Body-radius part: per-agent size (presence cascade), already floored. */
    bodyRadius: (fi: number) => number
    name: string
}

/**
 * Rasterize one un-oriented mote with a LIVE softness uniform (`params.softness` feathers the
 * baked hard shapes toward the glow skirt at runtime; the glow shape is always fully soft —
 * its mix folds away and the fast texel fn emits the Gaussian skirt only). Placement,
 * brightness and per-agent size arrive as slot parts — they are the consumer's look.
 */
/** The point-splat core: agent state + the energy accumulator + the live softness. */
export interface PointWorldSplatLayout {
    readonly $: {
        readonly agents: Vec4StateArray
        readonly accumE: AtomicU32[]
        readonly params: {
            readonly aspect: number
            readonly softness: number
        }
    }
}

function splatPointWorld(layout: PointWorldSplatLayout, cfg: PointWorldSplatConfig): (i: number) => void {
    const s = agents.AGENT_SHAPES[agents.resolveAgentShape(cfg.shape, 'dot')]
    const name = agents.resolveAgentShape(cfg.shape, 'dot')
    const FORCE_SOFT = s.soft
    const EXT = s.ext
    const RES = cfg.res
    const INV_RES = 1.0 / RES
    const AA_W = 0.75 * INV_RES
    const HARD_GAIN = FIXED_POINT_GAINS.HARD
    const SOFT_GAIN = FIXED_POINT_GAINS.SOFT
    const {place, brightness, bodyRadius} = cfg
    // `softness` is a live uniform for the hard shapes, so those keep the general weight fn that
    // evaluates and mixes both profiles. For `glow` the registry pins soft = 1, so the mix — and
    // the general SDF call — fold away: the fast texel fn emits the Gaussian skirt only.
    const weightFn = s.soft === 1
        ? (() => {
            const fast = agents.makeAgentTexelWeightFn(name)
            return tgpu.fn([d.f32, d.f32, d.f32, d.f32, d.f32, d.f32], d.f32)((q, t, n, bodyR, aaW, _soft) => {
                'use gpu'
                return fast(q, t, n, bodyR, aaW)
            }).$name('agentPointWeight')
        })()
        : (() => {
            const general = agents.makeAgentWeightFn(name)
            return tgpu.fn([d.f32, d.f32, d.f32, d.f32, d.f32, d.f32], d.f32)((_q, t, n, bodyR, aaW, soft) => {
                'use gpu'
                return general(t, n, bodyR, aaW, soft)
            }).$name('agentPointWeight')
        })()
    return tgpu.fn([d.u32])((i) => {
        'use gpu'
        const prm = layout.$.params
        const s4 = layout.$.agents[i]
        const fi = d.f32(i)
        const pos = place(fi, d.vec2f(s4.x, s4.y))
        const bright = brightness(fi)

        const aspect = std.max(prm.aspect, 1e-5)
        const softV = std.max(prm.softness, d.f32(FORCE_SOFT))
        // Runtime softness widens the footprint toward the glow skirt → widen the reject with it.
        const extK = std.mix(d.f32(EXT), std.max(d.f32(EXT), 2.4), softV)
        const gain = std.mix(d.f32(HARD_GAIN), d.f32(SOFT_GAIN), softV)
        const bodyR = bodyRadius(fi)
        const invBodyR = 1.0 / bodyR
        // World-space accept radius of THIS mote — most motes iterate a handful of texels.
        const reach = bodyR * extK + AA_W * 2.0
        const reachQ = (reach * invBodyR) * (reach * invBodyR) // reject in normalized q, not world d²

        const win = worldSplatWindow(pos, reach, aspect, d.f32(RES))
        for (let py = win.y0; py <= win.y1; py++) {
            const offY = (d.f32(py) + 0.5) * INV_RES - pos.y
            for (let px = win.x0; px <= win.x1; px++) {
                const offX = (d.f32(px) + 0.5) * INV_RES * aspect - pos.x
                const q = (offX * offX + offY * offY) * invBodyR * invBodyR
                if (q < reachQ) {
                    const w = weightFn(q, offX, offY, bodyR, AA_W, softV) * bright
                    const e = d.u32(w * gain)
                    if (e > d.u32(0)) {
                        const idx = d.u32(py * RES + px)
                        std.atomicAdd(layout.$.accumE[idx], e)
                    }
                }
            }
        }
    }).$name(cfg.name)
}

// ── Volume splat (Particles) ──────────────────────────────────────────────────────────────

interface VolumeRenderLayout {
    readonly $: {
        readonly pos: Vec4StateArray
        readonly vel: Vec4StateArray
        readonly accumE: AtomicU32[]
        readonly accumS: AtomicU32[]
        readonly outTex: StorageTex
        readonly params: {
            readonly colA: d.v4f
            readonly colB: d.v4f
            readonly centerX: number
            readonly centerYv: number
            readonly scale: number
            readonly rotC: number
            readonly rotS: number
            readonly aspect: number
            readonly size: number
            readonly exposure: number
            readonly softness: number
            readonly speedColorK: number
        }
    }
}

/** The Gaussian puff — the whole profile for the `glow` shape (its forced soft pins the blend
 *  at 1, so the hard half was dead code), and the soft end of the blend for the crisp shapes. */
function makeSoftenedProfile(shape: agents.AgentShapeName, names: {glow: string; blend: string}) {
    const glowProfile = tgpu.fn([d.f32, d.f32, d.f32, d.f32, d.f32], d.f32)((q, _ddx, _ddy, _size, _soft) => {
        'use gpu'
        return std.exp(q * -2.5)
    }).$name(names.glow)
    if (agents.AGENT_SHAPES[shape].soft === 1) return glowProfile
    const weightFn = agents.makeAgentTexelWeightFn(shape)
    return tgpu.fn([d.f32, d.f32, d.f32, d.f32, d.f32], d.f32)((q, ddx, ddy, size, soft) => {
        'use gpu'
        const wHard = weightFn(q, ddx, ddy, size, 0.9)
        const wSoft = std.exp(q * -2.5)
        return std.mix(wHard, wSoft, soft)
    }).$name(names.blend)
}

export interface VolumeSplatConfig {
    shape: string
    outRes: number
    maxSplatSize: number
    /** Radial reject: a texel contributes while q = dist²/size² < extQ. */
    extQ: number
    names: {splat: string; glow: string; profile: string}
}

/**
 * Project a 3D swarm particle to screen (2D placement rotation + a soft perspective from its
 * z), then accumulate the particle's softness-blended profile into the energy buffer and a
 * speed-weighted copy into the second buffer (the resolve derives per-texel average speed for
 * the rest→excited ramp). Additive fixed-point atomics are order-independent — no z-sort.
 */
function splatVolume(layout: VolumeRenderLayout, cfg: VolumeSplatConfig): (i: number) => void {
    const name = agents.resolveAgentShape(cfg.shape, 'dot')
    const profileFn = makeSoftenedProfile(name, {glow: cfg.names.glow, blend: cfg.names.profile})
    const OUT_RES = cfg.outRes
    const MAX_SPLAT_SIZE = cfg.maxSplatSize
    const EXTQ = cfg.extQ
    const EXT = Math.sqrt(cfg.extQ) // pre-folded — never Math.* in a 'use gpu' body
    return tgpu.fn([d.u32])((i) => {
        'use gpu'
        const prm = layout.$.params
        const pos = layout.$.pos[i]
        const vel = layout.$.vel[i]

        // Soft perspective: z toward the viewer enlarges and brightens.
        const persp = 1.0 / std.clamp(1.0 - pos.z * 0.35, 0.45, 2.0)
        const xs = pos.x * persp
        const ys = pos.y * persp * -1.0 // shape-local y-up → screen y-down
        const rx = xs * prm.rotC - ys * prm.rotS
        const ry = ys * prm.rotC + xs * prm.rotS
        const u = prm.centerX + rx * prm.scale / prm.aspect
        const v = prm.centerYv + ry * prm.scale

        const outF = d.f32(OUT_RES)
        const tx = u * outF
        const ty = v * outF
        const size = std.clamp(prm.size * (0.65 + 0.7 * pos.w) * persp * prm.scale, 0.6, d.f32(MAX_SPLAT_SIZE))
        const bright = (0.7 + 0.6 * pos.w) * persp * persp
        const speedNorm = std.clamp(vel.w * prm.speedColorK, 0.0, 1.0)
        // The speed accumulator's quantity is w · bright · speedNorm · 256 with w ≤ 1, so once
        // the w-free part can't reach one fixed-point step the atomic could only ever add zero.
        const doSpeed = bright * speedNorm * 256.0 >= 1.0

        // Iterate only the texels the profile can actually reach, clipped to the target.
        const rad = d.i32(std.ceil(size * d.f32(EXT)))
        const radF = d.f32(rad)
        const onTarget = tx > radF * -1.0 && tx < outF + radF && ty > radF * -1.0 && ty < outF + radF
        if (onTarget) {
            // Isotropic window (square render texels), clipped to the target ONCE by the shared
            // harness helper — so the bounds test drops out of the inner loop entirely.
            const win = texelSplatWindow(d.vec2f(tx, ty), rad, d.vec2i(d.i32(OUT_RES - 1), d.i32(OUT_RES - 1)))
            const sizeSq = std.max(size * size, 0.01)
            for (let py = win.y0; py <= win.y1; py++) {
                const ddy = d.f32(py) + 0.5 - ty
                const ddySq = ddy * ddy
                for (let px = win.x0; px <= win.x1; px++) {
                    const ddx = d.f32(px) + 0.5 - tx
                    const q = (ddx * ddx + ddySq) / sizeSq
                    if (q < EXTQ) {
                        // Softness blends the particle profile: the crisp baked shape with a
                        // thin AA rim at 0, the classic Gaussian glow puff at 1.
                        const w = profileFn(q, ddx, ddy, size, prm.softness)
                        const e = d.u32(w * bright * 256.0)
                        if (e > d.u32(0)) {
                            const idx = d.u32(py * d.i32(OUT_RES) + px)
                            std.atomicAdd(layout.$.accumE[idx], e)
                            if (doSpeed) std.atomicAdd(layout.$.accumS[idx], d.u32(w * bright * speedNorm * 256.0))
                        }
                    }
                }
            }
        }
    }).$name(cfg.names.splat)
}

// ── Relief splat (ParticleField) ──────────────────────────────────────────────────────────

interface ReliefRenderLayout {
    readonly $: {
        readonly pos: Vec4StateArray
        readonly col: Vec4StateArray
        readonly accumR: AtomicU32[]
        readonly accumG: AtomicU32[]
        readonly accumB: AtomicU32[]
        readonly accumW: AtomicU32[]
        readonly outTex: StorageTex
        readonly params: {
            readonly rowX: d.v4f
            readonly rowY: d.v4f
            readonly rowZ: d.v4f
            readonly time: number
            readonly gridW: number
            readonly gridH: number
            readonly outW: number
            readonly outH: number
            readonly depth: number
            readonly wobbleAmp: number
            readonly depthShading: number
            readonly spacing: number
            readonly particleSize: number
            readonly aspect: number
            readonly zoom: number
            readonly transX: number
            readonly transY: number
        }
    }
}

export interface ReliefSplatConfig {
    shape: string
    /** Cap on a particle's body radius in texels (the window itself is dynamic). */
    splatRCap: number
    /** Hard-edge anti-alias half-width, in render texels. */
    aaW: number
    /** Child alpha below which a particle is skipped entirely. */
    minAlpha: number
    /** Fixed-point scale for the atomic accumulators. */
    fp: number
    /** Idle-wobble base frequency (× params.time). */
    wobbleFreq: number
    /** Projection strength. */
    perspK: number
    name: string
}

/**
 * The camera + colorFrom-layer splat variant: rotate the particle's 3D world position by the
 * camera rows, project through a perspective from screen center (near ones enlarge, brighten
 * and spread outward), then accumulate the shape's coverage as DEPTH-WEIGHTED color into the
 * four fixed-point buffers. The weight (persp²·shade) makes near particles dominate the
 * per-texel average → stylized pseudo-occlusion, no z-sort.
 */
function splatRelief(layout: ReliefRenderLayout, cfg: ReliefSplatConfig): (x: number, y: number) => void {
    const name = agents.resolveAgentShape(cfg.shape, 'dot')
    const s = agents.AGENT_SHAPES[name]
    const weightFn = agents.makeAgentTexelWeightFn(name)
    // The registry's soft `ext` is sized for a gain of ~255; here the per-texel factor is
    // FP × depthWeight (peaking ≈ 8.2), so a near particle deposits nonzero quanta out to
    // √q ≈ 2.34 — widen the soft reject to cover that.
    const EXT_SHAPE = s.soft === 1 ? Math.max(s.ext, 2.4) : s.ext
    const SPLAT_R = cfg.splatRCap
    const AA_W = cfg.aaW
    const MIN_ALPHA = cfg.minAlpha
    const FP = cfg.fp
    const MIN_WEIGHT = 1.0 / cfg.fp
    const WOBBLE_FREQ = cfg.wobbleFreq
    const PERSP_K = cfg.perspK
    const TAU = 6.283185307179586
    return tgpu.fn([d.u32, d.u32])((gx, gy) => {
        'use gpu'
        const P = layout.$.params
        const i = gy * d.u32(P.gridW) + gx
        const p4 = layout.$.pos[i]
        const col = layout.$.col[i]
        const childA = col.w

        // Early-out 1: the particle sits over transparent child content, so every texel it
        // would touch scales to zero.
        if (childA > MIN_ALPHA) {
            const homeU = (d.f32(gx) + 0.5) / P.gridW
            const homeV = (d.f32(gy) + 0.5) / P.gridH

            // Idle wobble — a hash-phased per-particle breath in xy + z, keeps the field alive.
            const ph = p4.w * TAU
            const wob = P.wobbleAmp
            const wx = std.sin(P.time * WOBBLE_FREQ + ph) * wob * 0.012
            const wy = std.cos(P.time * WOBBLE_FREQ * 1.13 + ph * 1.7) * wob * 0.012
            const wz = std.sin(P.time * WOBBLE_FREQ * 0.87 + ph * 2.3) * wob * 0.06

            // World position → camera rotation (rows built on the CPU) → perspective projection.
            const pw = d.vec3f((homeU - 0.5) * P.aspect + p4.x + wx, homeV - 0.5 + p4.y + wy, p4.z + wz)
            const rv = agents.applyCameraRows(pw, P.rowX, P.rowY, P.rowZ)

            // Zoom scales the projected offset from center and the particle size (a camera
            // dolly), but NOT the render weight — persp0 keeps brightness/occlusion zoom-invariant.
            const persp0 = 1.0 / std.clamp(1.0 - rv.z * PERSP_K, 0.35, 2.6)
            const persp = persp0 * P.zoom
            const su = 0.5 + rv.x * persp / P.aspect + P.transX
            const sv = 0.5 + rv.y * persp + P.transY

            const tx = su * P.outW
            const ty = sv * P.outH
            const dotR = std.clamp(P.particleSize * P.spacing * 0.5 * persp, 0.5, d.f32(SPLAT_R))

            // Depth shading: near → full brightness, far → dimmer. The render weight leans on
            // near particles so they win the weighted-average color.
            const dr = std.max(P.depth, 1e-3)
            const near = std.smoothstep(dr * -1.0, dr, rv.z)
            const shade = std.mix(d.f32(1), 0.35 + 0.65 * near, P.depthShading)
            const depthWeight = persp0 * persp0 * shade
            // Everything in the per-texel weight that does NOT vary across the window.
            const cw = childA * depthWeight

            // The accept radius in TEXELS is `dotR · EXT_SHAPE + AA_W` — the shape extent scales
            // with the body, the anti-alias skirt does not. Divided back into dotR units it gives
            // the reject radius `extR`, so the window and the per-texel radial test agree.
            const extR = d.f32(EXT_SHAPE) + d.f32(AA_W) / dotR
            const rad = d.i32(std.ceil(dotR * extR))
            const radF = d.f32(rad)

            // Early-out 2: the projected particle can't reach the target at all.
            const onTarget = tx > radF * -1.0 && tx < P.outW + radF && ty > radF * -1.0 && ty < P.outH + radF
            if (cw > MIN_WEIGHT && onTarget) {
                const owU = d.u32(P.outW)
                const win = texelSplatWindow(d.vec2f(tx, ty), rad, d.vec2i(d.i32(P.outW) - 1, d.i32(P.outH) - 1))
                for (let py = win.y0; py <= win.y1; py++) {
                    const ddy = d.f32(py) + 0.5 - ty
                    for (let px = win.x0; px <= win.x1; px++) {
                        const ddx = d.f32(px) + 0.5 - tx
                        const q = (ddx * ddx + ddy * ddy) / std.max(dotR * dotR, 0.01)
                        if (q < extR * extR) {
                            const wc = weightFn(q, ddx, ddy, dotR, d.f32(AA_W)) * cw
                            if (wc > 0.0) {
                                const idx = d.u32(py) * owU + d.u32(px)
                                std.atomicAdd(layout.$.accumR[idx], d.u32(col.x * wc * FP))
                                std.atomicAdd(layout.$.accumG[idx], d.u32(col.y * wc * FP))
                                std.atomicAdd(layout.$.accumB[idx], d.u32(col.z * wc * FP))
                                std.atomicAdd(layout.$.accumW[idx], d.u32(wc * FP))
                            }
                        }
                    }
                }
            }
        }
    }).$name(cfg.name)
}

// ── Resolves ──────────────────────────────────────────────────────────────────────────────

export interface RampResolveConfig {
    /** Baked color-space mode for the rest→excited mix (the canonical mixColorsVariants path). */
    colorSpace: number
    res: number
    /** Fixed-point inverse gain (1/255 for the world splats, 1/256 for the volume splat). */
    invGain: number
    /** Scale the energy by `params.exposure` (the volume consumer's brightness knob). */
    exposure?: boolean
    /** Trail canvas: 'on' (always composited), {baked: boolean} (read baked out at trails = 0,
     *  write kept so scrubbing up never ghosts), or 'none' (no trail buffer in the layout). */
    trails: 'on' | 'none' | {baked: boolean}
    name: string
}

/**
 * Accumulators → color: the rest→excited ramp mixes in the baked color space, driven by the
 * per-texel average excitement (`accumS / accumE`). Most texels get no splat energy at all and
 * the endpoints are uniforms, so zero-energy texels skip the mix — RGB must still be colA (not
 * black) or the fragment's bilinear upsample would drag a dark fringe into every edge texel.
 */
function resolveRamp(layout: RampResolveLayout, cfg: RampResolveConfig): (x: number, y: number) => void {
    const L = layout as FullRampResolveView
    const mixFn = colorMixing.mixColorsVariants[cfg.colorSpace as keyof typeof colorMixing.mixColorsVariants] ?? colorMixing.mixColorsLinear
    const RES = cfg.res
    const INV_GAIN = cfg.invGain
    // Fixed-point energy, optionally scaled by the exposure knob (baked — the field only
    // exists on the consumers that declare it).
    const energyOf = cfg.exposure === true
        ? tgpu.fn([d.f32], d.f32)((eRaw) => {
            'use gpu'
            return eRaw * INV_GAIN * L.$.params.exposure
        }).$name(`${cfg.name}Energy`)
        : tgpu.fn([d.f32], d.f32)((eRaw) => {
            'use gpu'
            return eRaw * INV_GAIN
        }).$name(`${cfg.name}Energy`)

    if (cfg.trails === 'none') {
        // No trail canvas: write the ramped color straight to the output texture.
        return tgpu.fn([d.u32, d.u32])((x, y) => {
            'use gpu'
            const prm = L.$.params
            const idx = y * d.u32(RES) + x
            const eRaw = d.f32(std.atomicLoad(L.$.accumE[idx]))
            std.atomicStore(L.$.accumE[idx], d.u32(0))
            const sRaw = d.f32(std.atomicLoad(L.$.accumS[idx]))
            std.atomicStore(L.$.accumS[idx], d.u32(0))
            if (eRaw > 0.0) {
                const energy = energyOf(eRaw)
                const agitAvg = sRaw / std.max(eRaw, 1.0)
                const col = mixFn(prm.colA, prm.colB, std.clamp(agitAvg, 0.0, 1.0))
                const a = energyCoverageAlpha(energy, 1.6) * col.w
                std.textureStore(L.$.outTex, d.vec2u(x, y), d.vec4f(col.x, col.y, col.z, a))
            } else {
                std.textureStore(L.$.outTex, d.vec2u(x, y), d.vec4f(prm.colA.x, prm.colA.y, prm.colA.z, d.f32(0)))
            }
        }).$name(cfg.name)
    }

    // Trail-canvas variant: composite into the persistent decayed canvas (max-with-decay), or —
    // with the read baked out at trails = 0 — write the bare splat through the same path.
    const trailsOn = cfg.trails === 'on' || cfg.trails.baked
    const composite = trailsOn
        ? tgpu.fn([d.vec4f, d.f32, d.u32], d.vec4f)((col, curA, idx) => {
            'use gpu'
            return agents.trailMaxDecay(col, curA, L.$.trailBuf[idx], L.$.params.trails)
        }).$name(`${cfg.name}TrailComposite`)
        : tgpu.fn([d.vec4f, d.f32, d.u32], d.vec4f)((col, curA, _idx) => {
            'use gpu'
            return d.vec4f(col.x, col.y, col.z, curA)
        }).$name(`${cfg.name}TrailComposite`)
    return tgpu.fn([d.u32, d.u32])((x, y) => {
        'use gpu'
        const prm = L.$.params
        const idx = y * d.u32(RES) + x
        const eRaw = d.f32(std.atomicLoad(L.$.accumE[idx]))
        std.atomicStore(L.$.accumE[idx], d.u32(0))
        const sRaw = d.f32(std.atomicLoad(L.$.accumS[idx]))
        std.atomicStore(L.$.accumS[idx], d.u32(0))
        const energy = energyOf(eRaw)
        // The rest→excited mix is the expensive part (a full color-space round trip). At zero
        // energy it resolves to exactly colA and `curA` is 0 regardless, so empty texels skip it.
        let col = d.vec4f(prm.colA)
        if (eRaw > 0.0) {
            const agitAvg = sRaw / std.max(eRaw, 1.0)
            col = mixFn(prm.colA, prm.colB, std.clamp(agitAvg, 0.0, 1.0))
        }
        const curA = energyCoverageAlpha(energy, 1.6) * col.w

        const out = composite(col, curA, idx)
        // The trail WRITE stays even when the read is baked out — dropping it would leave stale
        // content that ghosts for a frame when the user scrubs trails back up off zero.
        L.$.trailBuf[idx] = d.vec4f(out)
        std.textureStore(L.$.outTex, d.vec2u(x, y), out)
    }).$name(cfg.name)
}

/** Accumulated energy → a single tint's alpha (`params.color`) — transparent between agents. */
function resolveTint(layout: {
    readonly $: {
        readonly accumE: AtomicU32[]
        readonly outTex: StorageTex
        readonly params: {readonly color: d.v4f}
    }
}, cfg: {res: number; name: string}): (x: number, y: number) => void {
    const RES = cfg.res
    return tgpu.fn([d.u32, d.u32])((x, y) => {
        'use gpu'
        const prm = layout.$.params
        const idx = y * d.u32(RES) + x
        const energy = d.f32(std.atomicLoad(layout.$.accumE[idx])) * (1.0 / 255.0)
        std.atomicStore(layout.$.accumE[idx], d.u32(0))
        const a = energyCoverageAlpha(energy, 1.6) * prm.color.w
        std.textureStore(layout.$.outTex, d.vec2u(x, y), d.vec4f(prm.color.x, prm.color.y, prm.color.z, a))
    }).$name(cfg.name)
}

/**
 * Weighted-average color resolve (RGB/W — the fixed-point scale cancels) + a coverage alpha
 * from W (the relief consumer).
 */
function resolveWeightedColor(layout: ReliefRenderLayout, cfg: {fp: number; alphaK: number; name: string}): (x: number, y: number) => void {
    const FP = cfg.fp
    const ALPHA_K = cfg.alphaK
    return tgpu.fn([d.u32, d.u32])((x, y) => {
        'use gpu'
        const P = layout.$.params
        const idx = y * d.u32(P.outW) + x
        const wRaw = d.f32(std.atomicLoad(layout.$.accumW[idx]))
        const rRaw = d.f32(std.atomicLoad(layout.$.accumR[idx]))
        const gRaw = d.f32(std.atomicLoad(layout.$.accumG[idx]))
        const bRaw = d.f32(std.atomicLoad(layout.$.accumB[idx]))
        std.atomicStore(layout.$.accumR[idx], d.u32(0))
        std.atomicStore(layout.$.accumG[idx], d.u32(0))
        std.atomicStore(layout.$.accumB[idx], d.u32(0))
        std.atomicStore(layout.$.accumW[idx], d.u32(0))
        const denom = std.max(wRaw, d.f32(1))
        const r = rRaw / denom
        const g = gRaw / denom
        const b = bRaw / denom
        const alpha = energyCoverageAlpha(wRaw * (1.0 / FP), ALPHA_K)
        std.textureStore(layout.$.outTex, d.vec2u(x, y), d.vec4f(r, g, b, alpha))
    }).$name(cfg.name)
}

// ── The renderAgents surface ──────────────────────────────────────────────────────────────

export const splat = {
    orientedWorld: splatOrientedWorld,
    pointWorld: splatPointWorld,
    volume: splatVolume,
    relief: splatRelief,
} as const

export const resolve = {
    ramp: resolveRamp,
    tint: resolveTint,
    weightedColor: resolveWeightedColor,
} as const

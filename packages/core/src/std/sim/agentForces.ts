/**
 * std/sim — the agent-simulation FORCE/TORQUE vocabulary and its integrator factories.
 *
 * The six agent shaders (Boids, MagneticFilings, FloatingParticles, ParticleField, Particles,
 * ParticleFlow) share one architecture (state buffers → integrate → splat → resolve, run by
 * `createAgentSystem`); what differed was the physics. This module names that physics as parts:
 *
 *   - FORCE parts — `tgpu.fn` factories over a shader's bind-group layout. Each returns a
 *     force/impulse contribution with a family-standard signature, so an integrator factory can
 *     compose a declared force list into ONE kernel, preserving the declared force-sum order.
 *   - TORQUE parts — the orientation dynamics (MagneticFilings' nematic directors).
 *   - INTEGRATOR factories — one per state family (3D volume swarm, 2D Reynolds steering,
 *     orientation directors, uniform drift, image relief, field advection). The family is the
 *     irreducible serial skeleton (state read → forces → integrate → write, with the family's
 *     envelope/clamp/wrap policy); everything look-bearing arrives as parts or constants.
 *
 * Layout contracts follow the Stable-Fluids precedent (`scaffolds/fluids.ts`): factories take
 * the shader's module-scope layout and reference entries by REQUIRED NAMES (documented per
 * factory), so the WGSL identifiers keep coming from the shader's own layout literal.
 */
import {tgpu, d, std, agents, noise} from '../../gpu/kit/index'
import {integrateSemiImplicitEuler, R2_ALPHA} from '../../gpu/scaffolds/agentSystem'

// ── Shared structural view pieces ─────────────────────────────────────────────────────────

type AtomicU32 = d.atomicU32

/** A mutable vec4f state array entry (`agents`, `pos`, `vel`, …). */
export type Vec4StateArray = d.v4f[]

// ── 3D volume-swarm family (Particles) ────────────────────────────────────────────────────
//
// State: pos = (xyz, seed), vel = (xyz, |v|). Forces have the family signature
// `(pos, vel, seed) → vec3f`; the integrator folds the declared list in order and runs the
// shared semi-implicit Euler with drag + speed clamp.

/** A 3D force contribution: `(pos, vel, seed) → vec3f`. */
export type AgentForce3 = (pos: d.v3f, vel: d.v3f, seed: number) => d.v3f

/** A baked shape-local signed-distance field: `(pos) → f32`. */
export type ShapeField3 = (pos: d.v3f) => number

interface VolumeParamsView {
    readonly dt: number
    readonly time: number
    readonly spread: number
    readonly agitation: number
    readonly dragMul: number
    readonly gravX: number
    readonly gravY: number
    readonly cursorX: number
    readonly cursorY: number
    readonly cursorForce: number
    readonly cursorRadSq: number
    readonly omegaX: number
    readonly omegaY: number
    readonly omegaZ: number
    readonly entrain: number
    readonly gridOffX: number
    readonly gridOffY: number
    readonly gridOffZ: number
}

/** Layout view for the volume family: vec4 pos/vel state + the density grid + params. */
export interface VolumeSwarmLayout {
    readonly $: {
        readonly pos: Vec4StateArray
        readonly vel: Vec4StateArray
        readonly dens: AtomicU32[]
        readonly params: VolumeParamsView
    }
}

/** What `force.pressure` returns: the force part plus the density-grid machinery it owns. */
export interface PressureForce {
    force: AgentForce3
    /** Full-grid clear, dispatched `fixed` over `cells`. */
    clearKernel: (i: number) => void
    /** Per-agent trilinear density splat. */
    splatKernel: (i: number) => void
    /** The density grid's cell count (the clear dispatch size). */
    cells: number
}

/**
 * Gas-pressure force over an auxiliary 3D density grid — even filling as an emergent property.
 * The part owns the grid ENTIRELY: the per-frame clear + trilinear splat kernels, the trilinear
 * sampler the force differentiates (piecewise-constant per-cell gradients herd agents onto the
 * cell lattice), and the R3 sub-cell dither contract (`params.gridOffX/Y/Z`, written per frame
 * from `agentFrame.r3SubCellOffset`) that keeps the lattice from standing in moiré with a shape
 * boundary. Layout needs: `pos`, `dens` (atomic u32, `gridDim³`), `params.{spread, gridOff*}`.
 */
function pressure(layout: VolumeSwarmLayout, cfg: {
    gridDim: number
    /** The grid covers shape-local [−domain, domain]³. */
    domain: number
    names: {clear: string; splat: string}
}): PressureForce {
    const DGRID = cfg.gridDim
    const DOMAIN = cfg.domain

    /** Density-cell read (agent count in FIX_D=64 fixed point) at a clamped 3D cell offset. */
    const densAt = tgpu.fn([d.i32, d.i32, d.i32], d.f32)((cx, cy, cz) => {
        'use gpu'
        const g = d.i32(DGRID)
        const x = std.clamp(cx, 0, g - 1)
        const y = std.clamp(cy, 0, g - 1)
        const z = std.clamp(cz, 0, g - 1)
        const idx = d.u32((z * g + y) * g + x)
        return d.f32(std.atomicLoad(layout.$.dens[idx])) * (1.0 / 64.0)
    })

    /** Trilinearly interpolated density AND its analytic gradient at a shape-local position
     *  (dithered by the frame's sub-cell grid offset, shared with the splat within a frame).
     *  Both are closed-form combinations of the SAME 8 corner values, so one 8-load pass feeds
     *  the whole pressure term. Returns (∂x, ∂y, ∂z, value); the gradient is with respect to
     *  the FRACTIONAL cell coordinate — numerically identical to the old ±half-cell central
     *  difference of the trilinear field on locally linear data. */
    const densTrilinearGrad = tgpu.fn([d.vec3f], d.vec4f)((pos) => {
        'use gpu'
        const prm = layout.$.params
        const g = d.f32(DGRID)
        const c = d.vec3f(
            (pos.x + prm.gridOffX + DOMAIN) / (2.0 * DOMAIN) * g - 0.5,
            (pos.y + prm.gridOffY + DOMAIN) / (2.0 * DOMAIN) * g - 0.5,
            (pos.z + prm.gridOffZ + DOMAIN) / (2.0 * DOMAIN) * g - 0.5,
        )
        const base = std.clamp(std.floor(c), d.vec3f(0.0, 0.0, 0.0), d.vec3f(g - 2.0, g - 2.0, g - 2.0))
        const f = std.clamp(c.sub(base), d.vec3f(0.0, 0.0, 0.0), d.vec3f(1.0, 1.0, 1.0))
        const bi = d.i32(base.x)
        const bj = d.i32(base.y)
        const bk = d.i32(base.z)
        let acc = d.f32(0)
        let gx = d.f32(0)
        let gy = d.f32(0)
        let gz = d.f32(0)
        for (let dz = 0; dz < 2; dz++) {
            for (let dy = 0; dy < 2; dy++) {
                for (let dx = 0; dx < 2; dx++) {
                    const wx = std.mix(1.0 - f.x, f.x, d.f32(dx))
                    const wy = std.mix(1.0 - f.y, f.y, d.f32(dy))
                    const wz = std.mix(1.0 - f.z, f.z, d.f32(dz))
                    // d/df of mix(1−f, f, corner): −1 at the low corner, +1 at the high corner.
                    const sx = d.f32(dx) * 2.0 - 1.0
                    const sy = d.f32(dy) * 2.0 - 1.0
                    const sz = d.f32(dz) * 2.0 - 1.0
                    const cv = densAt(bi + dx, bj + dy, bk + dz)
                    acc = acc + cv * (wx * wy * wz)
                    gx = gx + cv * (sx * wy * wz)
                    gy = gy + cv * (wx * sy * wz)
                    gz = gz + cv * (wx * wy * sz)
                }
            }
        }
        return d.vec4f(gx, gy, gz, acc)
    })

    const clearKernel = tgpu.fn([d.u32])((i) => {
        'use gpu'
        std.atomicStore(layout.$.dens[i], d.u32(0))
    }).$name(cfg.names.clear)

    /** Trilinear density splat (8 corner cells, 6-bit fixed point) — same per-frame grid offset
     *  as the sampler, so splat and read agree within a frame. */
    const splatKernel = tgpu.fn([d.u32])((i) => {
        'use gpu'
        const prm = layout.$.params
        const pos = layout.$.pos[i]
        const g = d.f32(DGRID)
        const c = d.vec3f(
            (pos.x + prm.gridOffX + DOMAIN) / (2.0 * DOMAIN) * g - 0.5,
            (pos.y + prm.gridOffY + DOMAIN) / (2.0 * DOMAIN) * g - 0.5,
            (pos.z + prm.gridOffZ + DOMAIN) / (2.0 * DOMAIN) * g - 0.5,
        )
        const base = std.clamp(std.floor(c), d.vec3f(0.0, 0.0, 0.0), d.vec3f(g - 2.0, g - 2.0, g - 2.0))
        const f = std.clamp(c.sub(base), d.vec3f(0.0, 0.0, 0.0), d.vec3f(1.0, 1.0, 1.0))
        const bi = d.i32(base.x)
        const bj = d.i32(base.y)
        const bk = d.i32(base.z)
        const gi = d.i32(DGRID)
        for (let dz = 0; dz < 2; dz++) {
            for (let dy = 0; dy < 2; dy++) {
                for (let dx = 0; dx < 2; dx++) {
                    const wx = std.mix(1.0 - f.x, f.x, d.f32(dx))
                    const wy = std.mix(1.0 - f.y, f.y, d.f32(dy))
                    const wz = std.mix(1.0 - f.z, f.z, d.f32(dz))
                    const w = wx * wy * wz
                    const idx = d.u32(((bk + dz) * gi + (bj + dy)) * gi + (bi + dx))
                    std.atomicAdd(layout.$.dens[idx], d.u32(w * 64.0 + 0.5))
                }
            }
        }
    }).$name(cfg.names.splat)

    /** Pressure: descend the TRILINEAR density gradient (analytic, from the same 8 corners as
     *  the value — one grid pass instead of seven), relative to local density → self-normalizing.
     *  The fractional-coordinate gradient carries the same implicit 2·(half-cell) scale the old
     *  central differences had, so `pressNorm` is unchanged. */
    const force = tgpu.fn([d.vec3f, d.vec3f, d.f32], d.vec3f)((pos, _vel, _seed) => {
        'use gpu'
        const prm = layout.$.params
        const gv = densTrilinearGrad(pos)
        const pressNorm = prm.spread * 4.4 / std.max(gv.w * 2.0 + 6.0, 6.0)
        return d.vec3f(gv.x * -1.0 * pressNorm, gv.y * -1.0 * pressNorm, gv.z * -1.0 * pressNorm)
    }).$name('forcePressure')

    return {force: force as AgentForce3, clearKernel, splatKernel, cells: DGRID * DGRID * DGRID}
}

/**
 * The containment BUNDLE — the four forces that share one SDF evaluation (a signed distance,
 * its central-difference gradient, and the radial distance): the surface spring wall, the
 * far-field recall, the gradient-free homing, and the rotation entrainment. Bundled so the
 * shared intermediates are computed once, exactly as the hand-written integrator did.
 */
function containment(layout: VolumeSwarmLayout, cfg: {
    field: ShapeField3
    /** Central-difference epsilon for the field gradient. */
    gradEps: number
    /** Spring through the surface: feathers in over [featherFrom, featherTo], plus a linear
     *  outside term — strong outside, diffuse inside (a sharp feather packs a coherent shell). */
    wall: {featherFrom: number; featherTo: number; base: number; springK: number}
    /** Smooth pull home once an agent leaves the domain (a hard clamp piles strays into lines). */
    recall: {from: number; to: number; k: number}
    /** SDF-VALUE-driven pull toward the origin — direction-safe where the gradient degenerates. */
    homing: {from: number; to: number; k: number}
    /** Drag toward the rotating container's feature velocity ω×r, feathered to inside. */
    entrainment: {featherHalf: number}
}): AgentForce3 {
    const field = cfg.field
    const E = cfg.gradEps
    const {wall, recall, homing, entrainment} = cfg
    return tgpu.fn([d.vec3f, d.vec3f, d.f32], d.vec3f)((pos, vel, _seed) => {
        'use gpu'
        const prm = layout.$.params
        // Signed distance + gradient (central differences).
        const sd = field(pos)
        const e = d.f32(E)
        const gx = field(d.vec3f(pos.x + e, pos.y, pos.z)) - field(d.vec3f(pos.x - e, pos.y, pos.z))
        const gy = field(d.vec3f(pos.x, pos.y + e, pos.z)) - field(d.vec3f(pos.x, pos.y - e, pos.z))
        const gz = field(d.vec3f(pos.x, pos.y, pos.z + e)) - field(d.vec3f(pos.x, pos.y, pos.z - e))
        const gv = d.vec3f(gx, gy, gz)
        const gn = gv.div(std.max(std.length(gv), 1e-4))

        let F = d.vec3f(0.0, 0.0, 0.0)
        // Wall: a spring through the surface — strong outside, feathering to zero over a wide
        // inner band.
        const out = std.max(sd, 0.0)
        const wallF = std.smoothstep(wall.featherFrom, wall.featherTo, sd) * wall.base + out * wall.springK
        F = F.sub(gn.mul(wallF))

        // Far-field recall.
        const rl = std.length(pos)
        const recallF = std.smoothstep(recall.from, recall.to, rl) * recall.k
        F = F.sub(pos.mul(recallF / std.max(rl, 1e-4)))

        // Gradient-free homing — fades out near the surface where the real containment takes over.
        const homingF = std.smoothstep(homing.from, homing.to, sd) * homing.k
        F = F.sub(pos.mul(homingF / std.max(rl, 1e-4)))

        // Rotation entrainment: drag toward ω×r inside the shape. `entrain` is gated to 0 on the
        // CPU when nothing rotates.
        const featureVel = std.cross(d.vec3f(prm.omegaX, prm.omegaY, prm.omegaZ), pos)
        const insideM = 1.0 - std.smoothstep(-entrainment.featherHalf, entrainment.featherHalf, sd)
        F = F.add(featureVel.sub(vel).mul(prm.entrain * insideM))
        return F
    }).$name('forceContainment') as AgentForce3
}

/** The cursor magnet as an xy cylinder around the pointer ray (positive pushes away) — the
 *  shared kit force field. Layout needs `params.{cursorX, cursorY, cursorForce, cursorRadSq}`. */
function cursorXY(layout: VolumeSwarmLayout): AgentForce3 {
    return tgpu.fn([d.vec3f, d.vec3f, d.f32], d.vec3f)((pos, _vel, _seed) => {
        'use gpu'
        const prm = layout.$.params
        const mxy = agents.cursorMagnet(d.vec2f(pos.x - prm.cursorX, pos.y - prm.cursorY), prm.cursorRadSq, prm.cursorForce)
        return d.vec3f(mxy.x, mxy.y, 0.0)
    }).$name('forceCursorXY') as AgentForce3
}

/** Constant gravity (pre-rotated into shape space on the CPU). Needs `params.{gravX, gravY}`. */
function gravity(layout: VolumeSwarmLayout): AgentForce3 {
    return tgpu.fn([d.vec3f, d.vec3f, d.f32], d.vec3f)((_pos, _vel, _seed) => {
        'use gpu'
        const prm = layout.$.params
        return d.vec3f(prm.gravX, prm.gravY, 0.0)
    }).$name('forceGravity') as AgentForce3
}

/**
 * Hash-turbulence kick, scaled by `params.agitation`. The large multipliers are load-bearing:
 * the fract-based hash needs its input to wrap many times to be white — at small ranges it
 * degrades into a smooth, near-symmetric function of position and the whole swarm gets herded
 * into coherent (mirror-symmetric) drift cells.
 */
function turbulence(layout: VolumeSwarmLayout, cfg: {posScale: number; timeX: number; timeY: number; seedScale: number; gain: number}): AgentForce3 {
    const {posScale, timeX, timeY, seedScale, gain} = cfg
    return tgpu.fn([d.vec3f, d.vec3f, d.f32], d.vec3f)((pos, _vel, seed) => {
        'use gpu'
        const prm = layout.$.params
        const rnd = noise.hash33(pos.mul(posScale).add(d.vec3f(prm.time * timeX, prm.time * timeY, seed * seedScale))).sub(d.vec3f(0.5, 0.5, 0.5))
        return rnd.mul(prm.agitation * gain)
    }).$name('forceTurbulence') as AgentForce3
}

/** Fold a declared 3D force list into one part, preserving the declared sum order. */
export function composeForces3(parts: AgentForce3[]): AgentForce3 {
    return parts.reduce((acc, part, index) =>
        tgpu.fn([d.vec3f, d.vec3f, d.f32], d.vec3f)((pos, vel, seed) => {
            'use gpu'
            return acc(pos, vel, seed).add(part(pos, vel, seed))
        }).$name(`forceSum${index}`) as AgentForce3)
}

/**
 * The volume-family integrator: fold the declared forces, run the shared semi-implicit Euler
 * (drag + speed clamp), advance, and hard-clamp position as a distant safety net. State written
 * back as pos = (xyz, seed), vel = (xyz, |v|) — the resolve's speed ramp reads vel.w.
 */
function forces3d(layout: VolumeSwarmLayout, cfg: {
    forces: AgentForce3[]
    maxSpeed: number
    posClamp: number
    name: string
}): (i: number) => void {
    const total = composeForces3(cfg.forces)
    const MAX_SPEED = cfg.maxSpeed
    const C = cfg.posClamp
    return tgpu.fn([d.u32])((i) => {
        'use gpu'
        const prm = layout.$.params
        const p4 = layout.$.pos[i]
        const v4 = layout.$.vel[i]
        let pos = d.vec3f(p4.x, p4.y, p4.z)
        const vPrev = d.vec3f(v4.x, v4.y, v4.z)
        const dt = prm.dt

        const F = total(pos, vPrev, p4.w)

        // Integrate (the shared semi-implicit Euler + drag + speed clamp). The position clamp is
        // a distant last-resort safety net — the recall force turns agents around long before it.
        const v3 = integrateSemiImplicitEuler(vPrev, F, dt, prm.dragMul, d.f32(MAX_SPEED))
        pos = std.clamp(pos.add(v3.mul(dt)), d.vec3f(-C, -C, -C), d.vec3f(C, C, C))

        layout.$.pos[i] = d.vec4f(pos.x, pos.y, pos.z, p4.w)
        layout.$.vel[i] = d.vec4f(v3.x, v3.y, v3.z, std.length(v3))
    }).$name(cfg.name)
}

// ── 2D Reynolds-steering family (Boids) ───────────────────────────────────────────────────
//
// State: agents = (posX, posY, velX, velY) in screen-proportional world space, updated in
// place (Gauss-Seidel). Steering parts have the signature `(pos, vel, maxSpeedI) → vec2f`.

/** A steering-force contribution: `(pos, vel, maxSpeedI) → vec2f`, clamped by the part. */
export type AgentSteer2 = (pos: d.v2f, vel: d.v2f, maxSpeedI: number) => d.v2f

interface SteeringParamsView {
    readonly count: number
    readonly dt: number
    readonly domainX: number
    readonly maxSpeed: number
    readonly maxForce: number
    readonly perceptionSq: number
    readonly sepRadiusSq: number
    readonly sepW: number
    readonly aliW: number
    readonly cohW: number
    readonly cursorX: number
    readonly cursorY: number
    readonly cursorMode: number
    readonly cursorRadius: number
    readonly cursorRadiusSq: number
    readonly cursorForce: number
    readonly margin: number
    readonly turnForce: number
}

/** Layout view for the steering family. */
export interface SteeringLayout {
    readonly $: {
        readonly agents: Vec4StateArray
        readonly agit: number[]
        readonly params: SteeringParamsView
    }
}

/**
 * The Reynolds triple — separation / alignment / cohesion over ONE brute-force neighbour scan
 * (the static-MAX loop bound with the runtime-count early break is engine policy inside the
 * part: the count slider never recompiles). Weights and radii are runtime uniforms
 * (`sepW/aliW/cohW`, `perceptionSq/sepRadiusSq`); each steer is maxForce-clamped.
 */
function flocking(layout: SteeringLayout, cfg: {maxAgents: number}): AgentSteer2 {
    const MAX_AGENTS = cfg.maxAgents
    return tgpu.fn([d.vec2f, d.vec2f, d.f32], d.vec2f)((pos, vel, maxSpeedI) => {
        'use gpu'
        const prm = layout.$.params
        let sep = d.vec2f(0.0, 0.0)
        let ali = d.vec2f(0.0, 0.0)
        let coh = d.vec2f(0.0, 0.0)
        let nCount = d.f32(0)
        let sCount = d.f32(0)
        // Brute-force neighbour scan. The self term (dsq ≈ 0) is excluded by the epsilon gate,
        // so no index comparison is needed.
        for (let j = 0; j < MAX_AGENTS; j++) {
            if (d.f32(j) >= prm.count) { break }
            const other = layout.$.agents[d.u32(j)]
            const opos = d.vec2f(other.x, other.y)
            const off = pos.sub(opos)
            const dsq = std.dot(off, off)
            if (dsq > 1e-7 && dsq < prm.perceptionSq) {
                ali = ali.add(d.vec2f(other.z, other.w))
                coh = coh.add(opos)
                nCount = nCount + 1.0
                if (dsq < prm.sepRadiusSq) {
                    sep = sep.add(off.div(dsq)) // inverse-distance: closer neighbours push harder
                    sCount = sCount + 1.0
                }
            }
        }
        let accel = d.vec2f(0.0, 0.0)
        if (nCount > 0.5) {
            const aliSteer = agents.reynoldsSteer(ali.div(nCount), vel, maxSpeedI, prm.maxForce)
            const cohSteer = agents.reynoldsSteer(coh.div(nCount).sub(pos), vel, maxSpeedI, prm.maxForce)
            accel = accel.add(aliSteer.mul(prm.aliW)).add(cohSteer.mul(prm.cohW))
        }
        if (sCount > 0.5) {
            accel = accel.add(agents.reynoldsSteer(sep, vel, maxSpeedI, prm.maxForce).mul(prm.sepW))
        }
        return accel
    }).$name('forceFlocking') as AgentSteer2
}

/** Cursor field: attract (mode 1) or repel/predator (mode 2), fading to zero at the radius. */
function cursorSteer(layout: SteeringLayout): AgentSteer2 {
    return tgpu.fn([d.vec2f, d.vec2f, d.f32], d.vec2f)((pos, _vel, _maxSpeedI) => {
        'use gpu'
        const prm = layout.$.params
        let accel = d.vec2f(0.0, 0.0)
        if (prm.cursorMode > 0.5) {
            const cd = d.vec2f(prm.cursorX, prm.cursorY).sub(pos)
            const cdsq = std.dot(cd, cd)
            if (cdsq > 1e-7 && cdsq < prm.cursorRadiusSq) {
                const cdist = std.sqrt(cdsq)
                const falloff = 1.0 - cdist / std.max(prm.cursorRadius, 1e-5)
                const sign = std.select(d.f32(1), d.f32(-1), prm.cursorMode > 1.5)
                accel = cd.div(cdist).mul(sign * falloff * prm.cursorForce)
            }
        }
        return accel
    }).$name('forceCursorSteer') as AgentSteer2
}

/** Soft edge turn: steer back before reaching a wall so the flock curves away in an arc.
 *  (At most one x and one y term is nonzero, so the per-axis sums are exact.) */
function wallTurn(layout: SteeringLayout): AgentSteer2 {
    return tgpu.fn([d.vec2f, d.vec2f, d.f32], d.vec2f)((pos, _vel, _maxSpeedI) => {
        'use gpu'
        const prm = layout.$.params
        const m = prm.margin
        const tf = prm.turnForce
        const ax = std.select(d.f32(0), tf, pos.x < m) + std.select(d.f32(0), -tf, pos.x > prm.domainX - m)
        const ay = std.select(d.f32(0), tf, pos.y < m) + std.select(d.f32(0), -tf, pos.y > 1.0 - m)
        return d.vec2f(ax, ay)
    }).$name('forceWallTurn') as AgentSteer2
}

/** Fold a declared steering list into one part, preserving the declared sum order. */
export function composeSteer2(parts: AgentSteer2[]): AgentSteer2 {
    return parts.reduce((acc, part, index) =>
        tgpu.fn([d.vec2f, d.vec2f, d.f32], d.vec2f)((pos, vel, maxSpeedI) => {
            'use gpu'
            return acc(pos, vel, maxSpeedI).add(part(pos, vel, maxSpeedI))
        }).$name(`steerSum${index}`) as AgentSteer2)
}

/**
 * The steering-family integrator: per-agent cruise variation, the folded steering forces, the
 * agitation envelope (steering effort above the cruising baseline charges instantly, cools
 * exponentially — the resolve's rest→excited ramp reads it), then Euler with a speed clamp
 * that keeps a minimum cruise, and a reflective hard safety net at the domain walls.
 */
function steering2d(layout: SteeringLayout, cfg: {
    forces: AgentSteer2[]
    agitation: {rest: number; gain: number; cool: number}
    /** Minimum cruise as a ratio of the per-agent max speed. */
    cruiseFloor: number
    /** Velocity retained (negated) on wall contact. */
    wallRestitution: number
    name: string
}): (i: number) => void {
    const total = composeSteer2(cfg.forces)
    const {rest: AGIT_REST, gain: AGIT_GAIN, cool: AGIT_COOL} = cfg.agitation
    const FLOOR = cfg.cruiseFloor
    const REST = cfg.wallRestitution
    return tgpu.fn([d.u32])((i) => {
        'use gpu'
        const prm = layout.$.params
        const self = layout.$.agents[i]
        let pos = d.vec2f(self.x, self.y)
        let vel = d.vec2f(self.z, self.w)
        const dt = prm.dt

        // Per-agent cruise variation (hash of index) so the flock isn't robotic.
        const maxSpeedI = prm.maxSpeed * agents.cruiseVariation(d.f32(i))

        const accel = total(pos, vel, maxSpeedI)

        // Agitation envelope: steering effort above the cruising baseline charges instantly,
        // cools exponentially — the splat/resolve turn it into the rest→excited color ramp.
        const effort = std.length(accel) / std.max(prm.maxForce, 1e-5)
        const raw = std.clamp((effort - AGIT_REST) * AGIT_GAIN, 0.0, 1.0)
        const cooled = layout.$.agit[i] * std.exp(dt * -AGIT_COOL)
        layout.$.agit[i] = std.max(raw, cooled)

        // Integrate + clamp speed into [floor·maxSpeedI, maxSpeedI] (keeps the swarm alive).
        vel = vel.add(accel.mul(dt))
        const spd = std.length(vel)
        const clamped = std.clamp(spd, maxSpeedI * FLOOR, maxSpeedI)
        vel = vel.div(std.max(spd, 1e-5)).mul(clamped)
        pos = pos.add(vel.mul(dt))

        // Hard safety net at the domain walls (the soft turn does the real work); reflect.
        const hitX = pos.x < 0.0 || pos.x > prm.domainX
        const hitY = pos.y < 0.0 || pos.y > 1.0
        const nvx = std.select(vel.x, vel.x * -REST, hitX)
        const nvy = std.select(vel.y, vel.y * -REST, hitY)
        const cx = std.clamp(pos.x, 0.0, prm.domainX)
        const cy = std.clamp(pos.y, 0.0, 1.0)
        layout.$.agents[i] = d.vec4f(cx, cy, nvx, nvy)
    }).$name(cfg.name)
}

// ── Orientation-director family (MagneticFilings) ─────────────────────────────────────────
//
// State: agents = (offX, offY, θ, ω) — anchored near a derived home, the physics lives in the
// rotation. Torque parts have the signature `(field, θ, fi) → f32` over a shared field sample.

/** The field sampled at one agent: line direction (φ), strength, and the radial unit. */
export const FieldSample = d.struct({rhat: d.vec2f, strength: d.f32, phi: d.f32}).$name('AgentFieldSample')
export type FieldSampleValue = d.Infer<typeof FieldSample>

/** A torque contribution: `(field, θ, fi) → f32`. */
export type AgentTorque = (fld: FieldSampleValue, theta: number, fi: number) => number

/** A field sampler part: `(pos) → FieldSample`. */
export type AgentFieldAt = (pos: d.v2f) => FieldSampleValue

/** A derived home-position part: `(fi) → vec2f`. */
export type AgentHome2 = (fi: number) => d.v2f

interface OrientationParamsView {
    readonly dt: number
    readonly cursorX: number
    readonly cursorY: number
    readonly axisX: number
    readonly axisY: number
    readonly fieldType: number
    readonly restMode: number
    readonly strength: number
    readonly reachSq: number
    readonly alignK: number
    readonly damping: number
    readonly restK: number
    readonly pullK: number
    readonly homeK: number
    readonly omegaRef: number
    readonly agitCool: number
    readonly gridCols: number
    readonly cellW: number
    readonly cellH: number
    readonly jitter: number
}

/** Layout view for the orientation family. */
export interface OrientationLayout {
    readonly $: {
        readonly agents: Vec4StateArray
        readonly agit: number[]
        readonly params: OrientationParamsView
    }
}

/**
 * The cursor field sample every torque/pull shares: r̂ from the pointer, the dipole-or-radial
 * line direction (`params.fieldType` selects at runtime), and a Gaussian strength falloff by
 * distance (`params.{strength, reachSq}`). The dipole axis is the CPU-smoothed cursor motion
 * (`params.{axisX, axisY}` — see `agentFrame.createMotionAxis`).
 */
function dipoleOrRadial(layout: OrientationLayout): AgentFieldAt {
    return tgpu.fn([d.vec2f], FieldSample)((pos) => {
        'use gpu'
        const prm = layout.$.params
        const cursor = d.vec2f(prm.cursorX, prm.cursorY)
        const delta = pos.sub(cursor)
        const r = std.max(std.length(delta), 1e-4)
        const rhat = delta.div(r)
        const axis = d.vec2f(prm.axisX, prm.axisY)
        const B = agents.dipoleOrRadialField(rhat, axis, prm.fieldType)
        // Gaussian falloff by distance (influence ≈ 2·reach). Beyond it the field is ~0 and only
        // the rest torque acts, so far agents relax slowly to their rest orientation.
        const fieldStr = prm.strength * std.exp(r * r / std.max(prm.reachSq, 1e-6) * -1.0)
        return FieldSample({rhat, strength: fieldStr, phi: std.atan2(B.y, B.x)})
    }).$name('agentFieldDipoleOrRadial') as AgentFieldAt
}

/** Each agent's rest orientation (radians): random per agent (mode 0), horizontal (1),
 *  vertical (2) — `restMode` is a runtime uniform. */
export const restOrientationAngle = tgpu.fn([d.f32, d.f32], d.f32)((fi, restMode) => {
    'use gpu'
    const randAng = noise.hash11(fi * 2.3987 + 0.517) * 3.141592653589793
    const a = std.select(randAng, d.f32(0), restMode > 0.5) // random | horizontal
    return std.select(a, d.f32(1.5707963267948966), restMode > 1.5) // → vertical
}).$name('agentRestAngle')

/** Swing onto the local field line: nematic torque `alignK · strength · sin(2Δ)`. */
function alignToField(layout: OrientationLayout): AgentTorque {
    return tgpu.fn([FieldSample, d.f32, d.f32], d.f32)((fld, theta, _fi) => {
        'use gpu'
        return agents.nematicTorque(layout.$.params.alignK * fld.strength, fld.phi, theta)
    }).$name('torqueAlignToField') as AgentTorque
}

/** Weak nematic pull to the rest orientation — dominant only where the field is faint. */
function restTorque(layout: OrientationLayout): AgentTorque {
    return tgpu.fn([FieldSample, d.f32, d.f32], d.f32)((_fld, theta, fi) => {
        'use gpu'
        const prm = layout.$.params
        return agents.nematicTorque(prm.restK, restOrientationAngle(fi, prm.restMode), theta)
    }).$name('torqueRest') as AgentTorque
}

/** Fold a declared torque list into one part, preserving the declared sum order. */
export function composeTorques(parts: AgentTorque[]): AgentTorque {
    return parts.reduce((acc, part, index) =>
        tgpu.fn([FieldSample, d.f32, d.f32], d.f32)((fld, theta, fi) => {
            'use gpu'
            return acc(fld, theta, fi) + part(fld, theta, fi)
        }).$name(`torqueSum${index}`) as AgentTorque)
}

/**
 * Rest (home) position on a jittered grid (`params.{gridCols, cellW, cellH, jitter}`), laid out
 * over an `overscan` border beyond the viewport so pulled-in edges backfill from off-screen.
 */
function jitteredGridHome(layout: OrientationLayout, cfg: {overscan: number}): AgentHome2 {
    const OVERSCAN = cfg.overscan
    return tgpu.fn([d.f32], d.vec2f)((fi) => {
        'use gpu'
        const prm = layout.$.params
        const cols = prm.gridCols
        const row = std.floor(fi / std.max(cols, 1.0))
        const col = fi - row * std.max(cols, 1.0)
        const h = noise.hash22(d.vec2f(fi * 0.7548 + 0.11, fi * 1.3236 + 5.31))
        const jx = (h.x - 0.5) * prm.jitter
        const jy = (h.y - 0.5) * prm.jitter
        return d.vec2f((col + 0.5 + jx) * prm.cellW - OVERSCAN, (row + 0.5 + jy) * prm.cellH - OVERSCAN)
    }).$name('agentJitteredHome') as AgentHome2
}

/**
 * The orientation-family integrator: derive home + sample the field once, integrate the folded
 * torques with angular drag (underdamped → visible settling), charge the angular-activity
 * envelope from |ω|, then the gentle position dynamics — the field pull (−r̂·pullK·strength),
 * the exponential spring home, and the hard wander clamp. Orientation carries the effect.
 */
function orientation2d(layout: OrientationLayout, cfg: {
    home: AgentHome2
    fieldAt: AgentFieldAt
    torques: AgentTorque[]
    maxOffset: number
    name: string
}): (i: number) => void {
    const {home, fieldAt} = cfg
    const total = composeTorques(cfg.torques)
    const MAX_OFFSET = cfg.maxOffset
    return tgpu.fn([d.u32])((i) => {
        'use gpu'
        const prm = layout.$.params
        const st = layout.$.agents[i]
        let off = d.vec2f(st.x, st.y)
        let theta = st.z
        let omega = st.w
        const dt = prm.dt
        const fi = d.f32(i)

        const homeP = home(fi)
        const pos = d.vec2f(homeP.x + off.x, homeP.y + off.y)
        const fld = fieldAt(pos)

        omega = omega + total(fld, theta, fi) * dt
        omega = omega * std.exp(prm.damping * dt * -1.0) // angular drag (underdamped → settling wobble)
        theta = theta + omega * dt

        // Angular-activity envelope: fast spin charges instantly, cools exponentially.
        const raw = std.clamp(std.abs(omega) / std.max(prm.omegaRef, 1e-5), 0.0, 1.0)
        const cooled = layout.$.agit[i] * std.exp(dt * -prm.agitCool)
        layout.$.agit[i] = std.max(raw, cooled)

        // Position: subtle pull toward the pole (−r̂, scaled by field strength) + spring back
        // home. Kept gentle and clamped so agents barely crowd — the rotation carries the effect.
        const pull = fld.rhat.mul(prm.pullK * fld.strength * -1.0)
        off = off.add(pull.mul(dt))
        off = off.mul(std.exp(prm.homeK * dt * -1.0))
        const offLen = std.length(off)
        off = std.select(off, off.div(std.max(offLen, 1e-5)).mul(MAX_OFFSET), offLen > MAX_OFFSET)

        layout.$.agents[i] = d.vec4f(off.x, off.y, theta, omega)
    }).$name(cfg.name)
}

// ── Uniform-drift family (FloatingParticles) ──────────────────────────────────────────────
//
// State: agents = (posX, posY, gustVX, gustVY) — the base drift is deterministic from the
// index, only the decaying cursor gust is stored. Motion is pure drift, so a toroidal wrap
// preserves even coverage exactly.

interface DriftParamsView {
    readonly dt: number
    readonly domainX: number
    readonly dragMul: number
    readonly cursorX: number
    readonly cursorY: number
    readonly cursorRadSq: number
    readonly cursorForce: number
}

/** Layout view for the drift family. */
export interface DriftLayout {
    readonly $: {
        readonly agents: Vec4StateArray
        readonly params: DriftParamsView
    }
}

/** A per-agent deterministic drift velocity part: `(fi) → vec2f`. */
export type AgentDrift = (fi: number) => d.v2f

/** Params view for the varied-heading drift part. */
interface HeadingDriftParamsView {
    readonly driftBase: number
    readonly angleRad: number
    readonly speedVar: number
    readonly angleVarRad: number
}

/**
 * Shared-heading drift with per-agent variance: the shared heading varied per agent by
 * `params.angleVarRad` (± the half-range) and the shared speed by `params.speedVar` (±½),
 * both hashed from the index — a coherent stream that never reads robotic. Layout needs
 * `params.{driftBase, angleRad, speedVar, angleVarRad}`.
 */
function variedHeading(layout: {readonly $: {readonly params: HeadingDriftParamsView}}): AgentDrift {
    return tgpu.fn([d.f32], d.vec2f)((fi) => {
        'use gpu'
        const prm = layout.$.params
        const h = noise.hash22(d.vec2f(fi * 0.7548 + 0.13, fi * 1.3236 + 5.71))
        const speedMul = 1.0 + (h.x - 0.5) * prm.speedVar
        const ang = prm.angleRad + (h.y - 0.5) * 2.0 * prm.angleVarRad
        return d.vec2f(std.cos(ang), std.sin(ang)).mul(prm.driftBase * speedMul)
    }).$name('driftVariedHeading') as AgentDrift
}

/** Cursor gust: a kit magnet impulse on the stored velocity (already ×dt), zero while the
 *  force slider sits at 0 (a uniform-valued branch — fully coherent). */
function cursorGust(layout: DriftLayout): (pos: d.v2f) => d.v2f {
    return tgpu.fn([d.vec2f], d.vec2f)((pos) => {
        'use gpu'
        const prm = layout.$.params
        let impulse = d.vec2f(0.0, 0.0)
        if (prm.cursorForce > 1e-5) {
            const delta = pos.sub(d.vec2f(prm.cursorX, prm.cursorY))
            impulse = agents.cursorMagnet(delta, prm.cursorRadSq, prm.cursorForce).mul(prm.dt)
        }
        return impulse
    }).$name('forceCursorGust') as (pos: d.v2f) => d.v2f
}

/**
 * The drift-family integrator: gust impulse → exponential gust decay → advance by the
 * deterministic drift + gust → toroidal wrap over [0, domainX] × [0, 1] (floor-subtract
 * handles any overshoot in one step).
 */
function drift2d(layout: DriftLayout, cfg: {
    drift: AgentDrift
    gust: (pos: d.v2f) => d.v2f
    name: string
}): (i: number) => void {
    const {drift, gust} = cfg
    return tgpu.fn([d.u32])((i) => {
        'use gpu'
        const prm = layout.$.params
        const s = layout.$.agents[i]
        const fi = d.f32(i)
        const dt = prm.dt
        let pos = d.vec2f(s.x, s.y)
        let gustV = d.vec2f(s.z, s.w)

        gustV = gustV.add(gust(pos))
        gustV = gustV.mul(prm.dragMul)

        pos = pos.add(drift(fi).add(gustV).mul(dt))

        const wx = pos.x - std.floor(pos.x / std.max(prm.domainX, 1e-4)) * prm.domainX
        const wy = pos.y - std.floor(pos.y)
        layout.$.agents[i] = d.vec4f(wx, wy, gustV.x, gustV.y)
    }).$name(cfg.name)
}

// ── Image-relief family (ParticleField) ───────────────────────────────────────────────────
//
// State: pos = (offX, offY, curZ, seed), vel = (xyz, seen-latch), col = the child color.
// A 2D grid dispatch (clean u32 grid coordinates — a 1D index reconstructed via `idx / gw` is
// unsafe because TGSL transpiles `/` to FLOAT division). Each agent has a fixed home on the
// grid; a channel of the live child drives its depth target.

interface ReliefParamsView {
    readonly rowX: d.v4f
    readonly rowY: d.v4f
    readonly rowZ: d.v4f
    readonly dt: number
    readonly snap: number
    readonly gridW: number
    readonly gridH: number
    readonly depth: number
    readonly dragMul: number
    readonly aspect: number
    readonly zoom: number
    readonly transX: number
    readonly transY: number
    readonly pointerX: number
    readonly pointerY: number
    readonly cursorForce: number
    readonly cursorZForce: number
    readonly cursorRadSq: number
}

/** Layout view for the relief family (`src` is the late-bound child RTT). */
export interface ReliefLayout {
    readonly $: {
        readonly pos: Vec4StateArray
        readonly vel: Vec4StateArray
        readonly col: Vec4StateArray
        readonly src: d.Infer<d.WgslTexture2d<d.F32>>
        readonly params: ReliefParamsView
    }
}

/** A depth-channel extractor: `(rgb, a) → f32` over the UNPREMULTIPLIED child color. */
export type AgentChannel = (rgb: d.v3f, a: number) => number

// The channel-extractor menu: one scalar in [0, 1] out of an unpremultiplied color, for any
// consumer that maps a channel of an image to a physical quantity (depth, mass, heat, …).
const channelLuminance = tgpu.fn([d.vec3f, d.f32], d.f32)((rgb, _a) => {
    'use gpu'
    return std.dot(rgb, d.vec3f(0.299, 0.587, 0.114))
})
const channelLuminanceInv = tgpu.fn([d.vec3f, d.f32], d.f32)((rgb, _a) => {
    'use gpu'
    return 1.0 - std.dot(rgb, d.vec3f(0.299, 0.587, 0.114))
})
const channelRed = tgpu.fn([d.vec3f, d.f32], d.f32)((rgb, _a) => {'use gpu'; return rgb.x})
const channelGreen = tgpu.fn([d.vec3f, d.f32], d.f32)((rgb, _a) => {'use gpu'; return rgb.y})
const channelBlue = tgpu.fn([d.vec3f, d.f32], d.f32)((rgb, _a) => {'use gpu'; return rgb.z})
const channelSaturation = tgpu.fn([d.vec3f, d.f32], d.f32)((rgb, _a) => {
    'use gpu'
    const mx = std.max(std.max(rgb.x, rgb.y), rgb.z)
    const mn = std.min(std.min(rgb.x, rgb.y), rgb.z)
    return mx - mn
})
const channelAlpha = tgpu.fn([d.vec3f, d.f32], d.f32)((_rgb, a) => {'use gpu'; return a})

/** The extractor menu + its canonical mode numbering (0 luminance … 6 alpha). */
export const channel = {
    luminance: channelLuminance as AgentChannel,
    luminanceInverted: channelLuminanceInv as AgentChannel,
    red: channelRed as AgentChannel,
    green: channelGreen as AgentChannel,
    blue: channelBlue as AgentChannel,
    saturation: channelSaturation as AgentChannel,
    alpha: channelAlpha as AgentChannel,
    byMode: {
        0: channelLuminance, 1: channelLuminanceInv, 2: channelRed,
        3: channelGreen, 4: channelBlue, 5: channelSaturation, 6: channelAlpha,
    } as Record<number, AgentChannel>,
} as const

/**
 * The view-space cursor magnet: project the agent's CURRENT position through the same camera
 * path as the splat (rotation → perspective·zoom → screen offset) so the cursor interacts with
 * what is actually under the pointer at any camera angle, apply the kit magnet in view space
 * (xy shove + z bulge), and rotate the force back into field space by Rᵀ.
 */
function cursorInView(layout: ReliefLayout, cfg: {perspK: number}): (homeW: d.v2f, off: d.v2f, z: number) => d.v3f {
    const PERSP_K = cfg.perspK
    return tgpu.fn([d.vec2f, d.vec2f, d.f32], d.vec3f)((homeW, off, z) => {
        'use gpu'
        const P = layout.$.params
        const pw = d.vec3f(homeW.x + off.x, homeW.y + off.y, z)
        const rv = agents.applyCameraRows(pw, P.rowX, P.rowY, P.rowZ)
        const persp = P.zoom / std.clamp(1.0 - rv.z * PERSP_K, 0.35, 2.6)
        const su = 0.5 + rv.x * persp / P.aspect + P.transX
        const sv = 0.5 + rv.y * persp + P.transY
        // Screen-proportional offset from the pointer (isotropic — x lifted by aspect).
        const deltaS = d.vec2f((su - P.pointerX) * P.aspect, sv - P.pointerY)
        const mxy = agents.cursorMagnet(deltaS, P.cursorRadSq, P.cursorForce)
        const mz = agents.cursorFalloff(deltaS, P.cursorRadSq) * P.cursorZForce
        return agents.applyCameraRowsTransposed(d.vec3f(mxy.x, mxy.y, mz), P.rowX, P.rowY, P.rowZ)
    }).$name('forceCursorInView') as (homeW: d.v2f, off: d.v2f, z: number) => d.v3f
}

/**
 * The relief-family integrator: bilinear-sample the live child at the home UV, unpremultiply
 * and store the color, derive the depth target from the declared channel, then the spring
 * toward home (xy) / target (z) plus the view-space cursor magnet, through the shared
 * semi-implicit Euler. A particle SNAPS to its target until it has actually seen child content
 * (the vel.w latch) — no start lurch, no fly-in on load.
 */
function relief(layout: ReliefLayout, cfg: {
    channel: AgentChannel
    cursor: (homeW: d.v2f, off: d.v2f, z: number) => d.v3f
    stiffness: number
    maxSpeed: number
    /** Child alpha below which a particle is never latched (matches the splat's early-out). */
    minAlpha: number
    name: string
}): (x: number, y: number) => void {
    const {channel, cursor} = cfg
    const STIFF = cfg.stiffness
    const MAX_SPEED = cfg.maxSpeed
    const MIN_ALPHA = cfg.minAlpha
    return tgpu.fn([d.u32, d.u32])((gx, gy) => {
        'use gpu'
        const P = layout.$.params
        const i = gy * d.u32(P.gridW) + gx
        const homeU = (d.f32(gx) + 0.5) / P.gridW
        const homeV = (d.f32(gy) + 0.5) / P.gridH

        // Bilinear tap from the child RTT (premultiplied), size queried from the texture itself.
        const dims = std.textureDimensions(layout.$.src)
        const bf = agents.bilinearTexelFrame(d.vec2f(homeU, homeV), d.vec2f(d.f32(dims.x), d.f32(dims.y)))
        const c00 = std.textureLoad(layout.$.src, d.vec2u(bf.x0, bf.y0), 0)
        const c10 = std.textureLoad(layout.$.src, d.vec2u(bf.x1, bf.y0), 0)
        const c01 = std.textureLoad(layout.$.src, d.vec2u(bf.x0, bf.y1), 0)
        const c11 = std.textureLoad(layout.$.src, d.vec2u(bf.x1, bf.y1), 0)
        const prem = std.mix(std.mix(c00, c10, bf.fx), std.mix(c01, c11, bf.fx), bf.fy)
        const a = prem.w
        const rgb = prem.xyz.div(std.max(a, d.f32(1e-4)))
        layout.$.col[i] = d.vec4f(rgb.x, rgb.y, rgb.z, a)

        // Target Z from the child channel, centered so mid-gray floats at z = 0.
        const ch = channel(rgb, a)
        const targetZ = (ch - 0.5) * P.depth

        const p4 = layout.$.pos[i]
        const v4 = layout.$.vel[i]
        const dt = P.dt
        // Compute passes run BEFORE the RTT passes, so on frame 0 the child texture is still
        // zero-filled and `targetZ` reads as fully-receded — snapping to that would launch a
        // spring settle from a bogus depth the moment real content lands. The snap is therefore
        // CONTENT-driven: vel.w is a sticky "has ever seen valid content" latch.
        const needsSnap = P.snap > 0.5 || v4.w < 0.5

        const homeW = d.vec2f((homeU - 0.5) * P.aspect, homeV - 0.5)
        const off = d.vec2f(p4.x, p4.y)
        const cf = cursor(homeW, off, p4.z)

        // Forces: spring home (xy) / toward the depth target (z) + the cursor field.
        const F = d.vec3f(off.x * -STIFF, off.y * -STIFF, (targetZ - p4.z) * STIFF).add(cf)

        // Integrate (the shared semi-implicit Euler + drag + speed clamp).
        let v3 = integrateSemiImplicitEuler(d.vec3f(v4.x, v4.y, v4.z), F, dt, P.dragMul, d.f32(MAX_SPEED))
        let offN = off.add(d.vec2f(v3.x, v3.y).mul(dt))
        let curZ = p4.z + v3.z * dt

        // Before the particle has ever seen content: settle exactly at the target, no lurch.
        offN = std.select(offN, d.vec2f(0.0, 0.0), needsSnap)
        curZ = std.select(curZ, targetZ, needsSnap)
        v3 = std.select(v3, d.vec3f(0.0, 0.0, 0.0), needsSnap)

        const seed = noise.hash12(d.vec2f(d.f32(gx) + 0.5, d.f32(gy) + 0.5))
        // Latch on the SAME alpha threshold the splat's transparency early-out uses, so "has
        // seen content" and "is rendered" agree by construction.
        const seenN = std.select(v4.w, d.f32(1), a > MIN_ALPHA)
        layout.$.pos[i] = d.vec4f(offN.x, offN.y, curZ, seed)
        layout.$.vel[i] = d.vec4f(v3.x, v3.y, v3.z, seenN)
    }).$name(cfg.name)
}

// ── Field-advection family (ParticleFlow) ─────────────────────────────────────────────────
//
// State: agents = (posX, posY, velX, velY). Each agent bilinear-samples a solved fluid
// velocity texture, eases toward it (inertia), gets a weak spring to its R2 home, and
// integrates in the clamped domain.

interface AdvectParamsView {
    readonly dt: number
    readonly aspect: number
    readonly advect: number
}

/** Layout view for the advection family (`velTex` is the solved fluid velocity handoff). */
export interface AdvectLayout {
    readonly $: {
        readonly agents: Vec4StateArray
        readonly velTex: d.Infer<d.WgslTexture2d<d.F32>>
        readonly params: AdvectParamsView
    }
}

/**
 * The advection-family integrator: sample the fluid grid bilinearly at the agent's screen-uv,
 * convert grid cells/sec to world units/sec, ease velocity toward the field (framerate-
 * independent inertia), add the weak R2 home spring (re-evens coverage once the flow dies —
 * the fluid box does not wrap), integrate, and clamp to the domain.
 */
function advect2d(layout: AdvectLayout, cfg: {
    gridN: number
    inertiaRate: number
    homeRate: number
    name: string
}): (i: number) => void {
    const N = cfg.gridN
    const INERTIA_RATE = cfg.inertiaRate
    const HOME_RATE = cfg.homeRate
    const R2A = R2_ALPHA[0]
    const R2B = R2_ALPHA[1]
    return tgpu.fn([d.u32])((i) => {
        'use gpu'
        const prm = layout.$.params
        const a = layout.$.agents[i]
        const pos = d.vec2f(a.x, a.y)
        const dt = prm.dt
        const aspect = std.max(prm.aspect, 1e-5)

        // Agent world pos → screen-uv → fluid-grid texel; bilinear-sample the velocity field.
        const NF = d.f32(N)
        const bf = agents.bilinearTexelFrame(d.vec2f(pos.x / aspect, pos.y), d.vec2f(NF, NF))
        const v00 = std.textureLoad(layout.$.velTex, d.vec2u(bf.x0, bf.y0), 0)
        const v10 = std.textureLoad(layout.$.velTex, d.vec2u(bf.x1, bf.y0), 0)
        const v01 = std.textureLoad(layout.$.velTex, d.vec2u(bf.x0, bf.y1), 0)
        const v11 = std.textureLoad(layout.$.velTex, d.vec2u(bf.x1, bf.y1), 0)
        const gvx = std.mix(std.mix(v00.x, v10.x, bf.fx), std.mix(v01.x, v11.x, bf.fx), bf.fy)
        const gvy = std.mix(std.mix(v00.y, v10.y, bf.fx), std.mix(v01.y, v11.y, bf.fx), bf.fy)

        // Grid cells/sec → world units/sec: uv/sec = grid/N, then worldX = uvX·aspect.
        const fieldVx = gvx / NF * aspect * prm.advect
        const fieldVy = gvy / NF * prm.advect

        // Inertia: ease velocity toward the field over ~1/rate seconds (framerate-independent).
        const blend = 1.0 - std.exp(dt * INERTIA_RATE * -1.0)
        // Weak spring toward this agent's R2 home — negligible against an active flow, but it
        // re-evens the distribution over ~1–2s once the flow dies (no wrap, no edge pile-up).
        const fi = d.f32(i)
        const homeX = std.fract(0.5 + fi * R2A) * aspect
        const homeY = std.fract(0.5 + fi * R2B)
        const velX = std.mix(a.z, fieldVx, blend) + (homeX - pos.x) * (HOME_RATE * dt)
        const velY = std.mix(a.w, fieldVy, blend) + (homeY - pos.y) * (HOME_RATE * dt)
        const nposX = std.clamp(pos.x + velX * dt, 0.0, aspect)
        const nposY = std.clamp(pos.y + velY * dt, 0.0, 1.0)
        layout.$.agents[i] = d.vec4f(nposX, nposY, velX, velY)
    }).$name(cfg.name)
}

// ── Namespaced surface ────────────────────────────────────────────────────────────────────

export const force = {
    pressure,
    containment,
    cursorXY,
    gravity,
    turbulence,
    flocking,
    cursorSteer,
    wallTurn,
    cursorGust,
    cursorInView,
} as const

export const torque = {
    alignToField,
    rest: restTorque,
} as const

export const field = {
    dipoleOrRadial,
    jitteredGridHome,
} as const

export const drift = {
    variedHeading,
} as const

export const integrator = {
    forces3d,
    steering2d,
    orientation2d,
    drift2d,
    relief,
    advect2d,
} as const

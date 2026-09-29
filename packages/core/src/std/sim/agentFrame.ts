/**
 * std/sim — CPU frame-program recipes for the agent simulations.
 *
 * The per-frame CPU math the agent shaders share, as named, reusable functions: placement
 * inversion (pointer → shape-local), rotation-delta → entrainment omega conjugation, grid
 * fitting, camera rows, low-discrepancy dithers, and exponential decays. Everything here is
 * pure CPU value derivation — GPU parts live in `agentForces`/`agentRender`, dispatch in the
 * `createAgentSystem` harness.
 */
import {R3_ALPHA} from '../../gpu/scaffolds/agentSystem'
import {rotateVecCpu} from '../../gpu/kit/sdf3d'

// ── Exponential rates ───────────────────────────────────────────────────────────────────

/** Frame-rate-independent exponential decay multiplier: `exp(−rate·dt)`. */
export function expDecay(rate: number, dt: number): number {
    return Math.exp(-rate * dt)
}

// ── Placement inversion ─────────────────────────────────────────────────────────────────

/**
 * Invert a center/scale/rotation placement: screen-UV pointer → shape-local xy (y up).
 * The forward transform is `screen = center + R(rot)·(local·scale)/aspectLift`; this is its
 * exact inverse, with x lifted by aspect so the local frame is isotropic.
 */
export function pointerToShapeLocal(opts: {
    pointerX: number
    pointerY: number
    centerX: number
    centerYv: number
    scale: number
    rotC: number
    rotS: number
    aspect: number
}): {x: number; y: number} {
    const dxAc = (opts.pointerX - opts.centerX) * opts.aspect
    const dyv = opts.pointerY - opts.centerYv
    return {
        x: (dxAc * opts.rotC + dyv * opts.rotS) / opts.scale,
        y: -((dyv * opts.rotC - dxAc * opts.rotS) / opts.scale),
    }
}

// ── Entrainment omega (rotation-delta conjugation) ─────────────────────────────────────

/**
 * Body-frame angular velocity from consecutive euler angles (radians), for a field sampled at
 * `R·p`: features move at `−(Rᵀω)×r`, so the euler rates are conjugated by Rᵀ and negated.
 * Returns null on the first frame (no previous sample).
 */
export function omegaFromRotationDeltas(
    prev: {x: number; y: number; z: number} | null,
    next: {x: number; y: number; z: number},
    dt: number,
): {x: number; y: number; z: number} | null {
    if (!prev) return null
    const wx = (next.x - prev.x) / dt
    const wy = (next.y - prev.y) / dt
    const wz = (next.z - prev.z) / dt
    const s = {
        cx: Math.cos(next.x), sx: Math.sin(next.x),
        cy: Math.cos(next.y), sy: Math.sin(next.y),
        cz: Math.cos(next.z), sz: Math.sin(next.z),
    }
    const ex = rotateVecCpu(1, 0, 0, s)
    const ey = rotateVecCpu(0, 1, 0, s)
    const ez = rotateVecCpu(0, 0, 1, s)
    return {
        x: -(ex.x * wx + ex.y * wy + ex.z * wz),
        y: -(ey.x * wx + ey.y * wy + ey.z * wz),
        z: -(ez.x * wx + ez.y * wy + ez.z * wz),
    }
}

/** Cap an omega vector's magnitude and derive the entrainment gain from it. */
export function entrainmentFromOmega(om: {x: number; y: number; z: number}, opts: {cap: number; gainRate: number; gainMax: number}) {
    const omLen = Math.hypot(om.x, om.y, om.z)
    const omCap = omLen > opts.cap ? opts.cap / omLen : 1
    return {
        omegaX: om.x * omCap,
        omegaY: om.y * omCap,
        omegaZ: om.z * omCap,
        entrain: Math.min(1, omLen * opts.gainRate) * opts.gainMax,
    }
}

// ── Pointer-motion axis (the magnet's dipole heading) ───────────────────────────────────

/**
 * A stateful smoothed-motion-direction tracker: the axis eases toward the pointer's travel
 * direction with a teleport guard — a jump wider than `teleport` world units in one frame is
 * the pointer re-entering the canvas, not a stroke, so the axis holds its previous heading.
 */
export function createMotionAxis(opts?: {smoothing?: number; teleport?: number}) {
    const smoothing = opts?.smoothing ?? 0.1
    const teleport = opts?.teleport ?? 0.25
    let lastX = 0.5
    let lastY = 0.5
    let axisX = 1
    let axisY = 0
    return {
        update(x: number, y: number): {x: number; y: number} {
            const mvx = x - lastX
            const mvy = y - lastY
            const mvLen = Math.hypot(mvx, mvy)
            if (mvLen > 1e-3 && mvLen < teleport) {
                const nx = mvx / mvLen
                const ny = mvy / mvLen
                axisX += (nx - axisX) * smoothing
                axisY += (ny - axisY) * smoothing
                const al = Math.hypot(axisX, axisY) || 1
                axisX /= al
                axisY /= al
            }
            lastX = x
            lastY = y
            return {x: axisX, y: axisY}
        },
    }
}

// ── Grid fitting ────────────────────────────────────────────────────────────────────────

/**
 * Fit ~`count` roughly-square cells over an overscanned `[−ov, domainX+ov] × [−ov, 1+ov]`
 * domain (the jittered home grid — off-screen agents backfill pulled-in edges).
 */
export function fitJitteredGrid(count: number, domainX: number, overscan: number) {
    const spanX = domainX + 2 * overscan
    const spanY = 1 + 2 * overscan
    const cols = Math.max(1, Math.round(Math.sqrt(count * (spanX / spanY))))
    const rows = Math.max(1, Math.ceil(count / cols))
    return {cols, cellW: spanX / cols, cellH: spanY / rows}
}

/** Fit an isotropic W×H agent grid to a count and aspect, each axis clamped to [min, max]. */
export function fitIsoGrid(count: number, aspect: number, min: number, max: number) {
    let w = Math.round(Math.sqrt(count * aspect))
    w = Math.min(Math.max(w, min), max)
    let h = Math.round(count / w)
    h = Math.min(Math.max(h, min), max)
    return {w, h}
}

// ── Camera rows ─────────────────────────────────────────────────────────────────────────

/**
 * Camera rotation matrix rows (Rz·Ry·Rx), conjugated for screen space (y down): the y-up
 * matrix with the off-diagonal y terms sign-flipped, so positive angles read as the natural
 * orbit directions in the editor. Angles in radians.
 */
export function cameraRowsYDown(rx: number, ry: number, rz: number): {
    rowX: [number, number, number]
    rowY: [number, number, number]
    rowZ: [number, number, number]
} {
    const cx = Math.cos(rx), sx = Math.sin(rx)
    const cy = Math.cos(ry), sy = Math.sin(ry)
    const cz = Math.cos(rz), sz = Math.sin(rz)
    const m00 = cz * cy, m01 = cz * sy * sx - sz * cx, m02 = cz * sy * cx + sz * sx
    const m10 = sz * cy, m11 = sz * sy * sx + cz * cx, m12 = sz * sy * cx - cz * sx
    const m20 = -sy, m21 = cy * sx, m22 = cy * cx
    return {
        rowX: [m00, -m01, m02],
        rowY: [-m10, m11, -m12],
        rowZ: [m20, -m21, m22],
    }
}

// ── Low-discrepancy dither ──────────────────────────────────────────────────────────────

/**
 * The R3 sub-cell dither for a voxel lattice: frame `i`'s offset, scaled by one cell — evenly
 * covers the sub-cell cube over frames, so a grid can never stand in coherent moiré with a
 * boundary. (See `R3_ALPHA` on the harness.)
 */
export function r3SubCellOffset(frameIdx: number, cell: number): {x: number; y: number; z: number} {
    return {
        x: ((frameIdx * R3_ALPHA[0]) % 1) * cell,
        y: ((frameIdx * R3_ALPHA[1]) % 1) * cell,
        z: ((frameIdx * R3_ALPHA[2]) % 1) * cell,
    }
}

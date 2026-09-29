/**
 * std — signal nouns. The first slice: the pointer signals a cursor-driven
 * simulation needs. Pointer signals carry POLICY as data — the teleport guard is a
 * per-shader aesthetic decision (ON for ripple/RD-class sims, OFF for smoke/trail-class),
 * and the speed clamp names the gating curve.
 */

/** The pointer position signal. `teleportGuard: 'on'` zeroes velocity on a jump frame. */
export interface PointerSignal {
    readonly kind: 'pointer'
    readonly teleportGuard: 'on' | 'off'
}

export function pointer(opts?: {teleportGuard?: 'on' | 'off'}): PointerSignal {
    return {kind: 'pointer', teleportGuard: opts?.teleportGuard ?? 'on'}
}

/** The pointer's per-second speed, clamped to `max` (the splat-amount gating curve). */
export interface PointerSpeedSignal {
    readonly kind: 'pointerSpeed'
    readonly max: number
}

export function pointerSpeed(opts?: {max?: number}): PointerSpeedSignal {
    return {kind: 'pointerSpeed', max: opts?.max ?? 2}
}

/**
 * Drive three extraFields with the CPU-normalized unit vector of three scalar props, refreshed
 * each frame — the standard way a direction prop reaches the GPU pre-normalized (saves a
 * per-fragment `normalize()`). Declare the fields on the definition's `extraFields`; a
 * zero-length input writes the zero vector.
 */
export function driveUnitDirection(
    params: {
        onBeforeRender(cb: () => void): void
        getCpuValue(name: string): unknown
        setExtraField(name: string, value: number): void
    },
    props: {x: string; y: string; z: string},
    fields: {x: string; y: string; z: string},
): void {
    params.onBeforeRender(() => {
        const x = params.getCpuValue(props.x) as number
        const y = params.getCpuValue(props.y) as number
        const z = params.getCpuValue(props.z) as number
        const len = Math.sqrt(x * x + y * y + z * z)
        if (len > 0) {
            params.setExtraField(fields.x, x / len)
            params.setExtraField(fields.y, y / len)
            params.setExtraField(fields.z, z / len)
        } else {
            params.setExtraField(fields.x, 0)
            params.setExtraField(fields.y, 0)
            params.setExtraField(fields.z, 0)
        }
    })
}

/**
 * CPU mirror of the classic in-shader `hash01` (`fract(sin(x·12.9898)·43758.5453)`) — for
 * pixel-invariant per-seed constants computed off the GPU (JS Math.sin at these argument
 * magnitudes is exact where GPU sin is driver-approximate).
 */
export function cpuHash01(x: number): number {
    const v = Math.sin(x * 12.9898) * 43758.5453
    return v - Math.floor(v)
}

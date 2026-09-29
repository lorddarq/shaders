/**
 * std/effects/reveal — the wipe/transition vocabulary.
 *
 * Every pointwise wipe is the same effect with an interchangeable *coverage coordinate*: a
 * per-pixel scalar in [0,1] saying "when does this pixel go away". {@link reveal} is the noun —
 * it takes a coverage and the three shared timing slots (progress / feather / invert) and lowers
 * to the kit's reveal tail (`revealMask` → `applyReveal`), scaling the STRAIGHT-alpha child's
 * alpha with RGB preserved (no render-to-texture).
 *
 * Coverages come from the {@link coverage} namespace — one producer per coordinate family
 * (directional projection, radial distance, angular sweep, cell grid, organic noise) — and
 * compose through modifiers (`.folded()`, `.tiled(n)`, `.bands(n)`, `.shuffled()`), so a wipe
 * reads as its recipe:
 *
 *     effect: reveal({
 *         coverage: coverage.directional(p('angle')).bands(p('barCount')).shuffled(),
 *         progress: p('progress'), feather: p('softness'), invert: p('invert'),
 *     })
 */
import type {Expr} from '../../gpu/contract'
import {call, floatE} from '../../gpu/porters'
import {reveal as kit} from '../../gpu/kit/index'
import type {FilterParams} from '../../gpu/scaffolds/pointwiseFilter'
import type {PointwiseEffect} from '../types'
import {resolveArg, type ArgSpec} from '../invoke'
import type {PropRef} from '../values'

/**
 * A coverage coordinate: a per-pixel scalar in [0,1] ordering when each pixel is wiped
 * (0 = goes first, 1 = goes last). Produced by the {@link coverage} namespace; refined by the
 * modifier methods, each of which returns a new coverage.
 */
export class Coverage {
    constructor(readonly build: (params: FilterParams) => Expr) {}

    /**
     * Fold the coordinate about its midpoint: 0 on the center line, 1 at both ends — one sweeping
     * front becomes two fronts opening outward (barn doors).
     */
    folded(): Coverage {
        return new Coverage((params) => call(kit.foldAboutCenter, 'foldAboutCenter', [this.build(params)]))
    }

    /**
     * Tile the coordinate into `count` repeats, each tile crossing the same front in lockstep
     * (venetian blinds).
     */
    tiled(count: ArgSpec): Coverage {
        return new Coverage((params) =>
            call(kit.tilePhase, 'tilePhase', [this.build(params), resolveArg(count, params)]))
    }

    /**
     * Quantize the coordinate into `count` bands that vanish whole, one after another in
     * coordinate order (ring pops over a radial coverage). Chain `.shuffled()` to randomize the
     * band order instead.
     */
    bands(count: ArgSpec): BandedCoverage {
        return new BandedCoverage(this, count)
    }
}

/** A banded coverage — bands vanish in coordinate order, or in stable random order via {@link BandedCoverage.shuffled}. */
export class BandedCoverage extends Coverage {
    constructor(base: Coverage, count: ArgSpec) {
        super((params) =>
            call(kit.bandMidpointCoord, 'bandMidpointCoord', [base.build(params), resolveArg(count, params)]))
        this.base = base
        this.count = count
    }

    private readonly base: Coverage
    private readonly count: ArgSpec

    /**
     * Give each band a stable random coverage (a hash of its index, no time dependence) so the
     * bands vanish in a frozen shuffled order (random bars).
     */
    shuffled(): Coverage {
        return new Coverage((params) =>
            call(kit.bandShuffleCoord, 'bandShuffleCoord', [this.base.build(params), resolveArg(this.count, params)]))
    }
}

/**
 * The cell-grid coverage family: the frame diced into square-ish cells `size` wide (as a fraction
 * of frame width, height aspect-corrected). Not a coverage by itself — pick how the cells order
 * their pixels: {@link diamond}, {@link checker}, or {@link shuffled}.
 */
export class CellCoverage {
    constructor(private readonly size: ArgSpec) {}

    /** Each pixel's coverage is its diamond distance from the cell center — a lattice of growing diamonds. */
    diamond(): Coverage {
        return new Coverage((params) =>
            call(kit.cellDiamondCoord, 'cellDiamondCoord', [params.ctx.uv, params.ctx.aspect, resolveArg(this.size, params)]))
    }

    /** Cells vanish in the staggered checkerboard order: even-parity cells sweep corner-to-corner first, odd-parity second. */
    checker(): Coverage {
        return new Coverage((params) =>
            call(kit.cellCheckerCoord, 'cellCheckerCoord', [params.ctx.uv, params.ctx.aspect, resolveArg(this.size, params)]))
    }

    /** Each cell gets a stable random coverage (a hash of its index, no time dependence) — blocks dissolve in a frozen random order. */
    shuffled(): Coverage {
        return new Coverage((params) =>
            call(kit.cellShuffleCoord, 'cellShuffleCoord', [params.ctx.uv, params.ctx.aspect, resolveArg(this.size, params)]))
    }
}

/** The sweep direction of an angular coverage — a structural choice, baked into the compiled shader. */
export type AngularDirection = 'cw' | 'ccw' | 'both'

const ANGULAR_MODES: Record<AngularDirection, number> = {cw: 0, ccw: 1, both: 2}

/** The coverage coordinate producers — one per family. All read the frame's own UV and aspect. */
export const coverage = {
    /**
     * 0→1 along `angleDeg` (degrees, 0 = left to right), aspect corrected so a diagonal reads as
     * a true angle and both extremes of the frame are reached exactly.
     */
    directional(angleDeg: ArgSpec): Coverage {
        return new Coverage((params) =>
            call(kit.directionalCoord, 'directionalCoord', [params.ctx.uv, params.ctx.aspect, resolveArg(angleDeg, params)]))
    },

    /**
     * Distance from `center` (a position prop), normalized by the distance to the FARTHEST corner
     * so coverage 1 is reached at every pixel however off-center the origin is.
     */
    radial(center: ArgSpec): Coverage {
        return new Coverage((params) =>
            call(kit.radialCornerNormCoord, 'radialCornerNormCoord', [params.ctx.uv, params.ctx.aspect, resolveArg(center, params)]))
    },

    /**
     * Clock-sweep fraction around `center`, measured from `startDeg`. `direction` is STRUCTURAL —
     * a compile-time string prop (bind with `p()`) or a literal `'cw' | 'ccw' | 'both'`, baked
     * into the compiled shader as a mode literal ('both' opens two wedges meeting on the far side).
     */
    angular(center: ArgSpec, startDeg: ArgSpec, direction: PropRef | AngularDirection): Coverage {
        return new Coverage((params) => {
            const value = typeof direction === 'string'
                ? direction
                : ((params.propValues[direction.name] as AngularDirection | undefined) ?? 'cw')
            const mode = ANGULAR_MODES[value] ?? 0
            return call(kit.angularCoord, 'angularCoord', [
                params.ctx.uv, params.ctx.aspect,
                resolveArg(center, params), resolveArg(startDeg, params), floatE(mode),
            ])
        })
    },

    /**
     * The frame diced into square-ish cells `size` wide (as a fraction of frame width). Pick the
     * per-cell ordering: `.diamond()`, `.checker()`, or `.shuffled()`.
     */
    cells(size: ArgSpec): CellCoverage {
        return new CellCoverage(size)
    },

    /**
     * An organic noise field (a stable 3-octave fbm, no time dependence) — the classic film
     * dissolve. `scale` sets the blob frequency; `seed` shifts the pattern.
     */
    noise(scale: ArgSpec, seed: ArgSpec): Coverage {
        return new Coverage((params) =>
            call(kit.fbmCoverageCoord, 'fbmCoverageCoord', [params.ctx.uv, params.ctx.aspect, resolveArg(scale, params), resolveArg(seed, params)]))
    },
}

/** The reveal noun's slots: what orders the pixels, and the shared wipe timing. */
export interface RevealSlots {
    /** The coverage coordinate — which pixels go first. */
    coverage: Coverage
    /** How far the wipe has travelled (0 = fully visible, 1 = fully wiped away). */
    progress: ArgSpec
    /** Softness of the wipe front (the progress window a pixel crosses). */
    feather: ArgSpec
    /** Reverse the coverage ordering (a boolean prop; flips the coordinate, not the progress). */
    invert: ArgSpec
}

/**
 * Wipe the child away along a coverage coordinate: pixels vanish in coverage order as `progress`
 * grows, crossing a ±`feather` soft front remapped so both progress extremes clear completely.
 * Scales the straight-alpha child's alpha with RGB preserved — pointwise, no render-to-texture.
 */
export function reveal(slots: RevealSlots): PointwiseEffect {
    return {
        kind: 'pointwise',
        body: {fn: kit.applyReveal, hint: 'applyReveal'},
        args: (params) => [
            call(kit.revealMask, 'revealMask', [
                slots.coverage.build(params),
                resolveArg(slots.progress, params),
                resolveArg(slots.feather, params),
                resolveArg(slots.invert, params),
            ]),
        ],
    }
}

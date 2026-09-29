/**
 * std — mask nouns. Scalar coverage fields usable inside pointwise effects
 * without forcing RTT (Vignette is the proof case). GPU bodies live in the kit
 * (`kit/fields.ts`); these constructors capture data only.
 *
 * The Expr-level coverage words (`softBand`, `softDisc`) live here too — concept-level
 * masks any paint can shape with.
 */
import type {Expr} from '../gpu/contract'
import {abs, add, mul, neg, pow, smoothstep, sub} from './math'
import type {PropRef} from './values'
import {Scalar} from './values'

/**
 * An asymmetric soft band around a line/curve: a sharp fade-in on one side, a long fade-out
 * on the other. `distance` is the signed distance from the line (negative = below);
 * `below` fades coverage in across `[from, to]`, `above` fades it out across
 * `[from ?? 0, to]`. The band concept under curtains, horizons, beam falloffs, underlines.
 */
export function softBand(opts: {
    distance: Expr
    below: {from: Expr | number; to: Expr | number}
    above: {from?: Expr | number; to: Expr | number}
}): Expr {
    return mul(
        smoothstep(opts.below.from, opts.below.to, opts.distance),
        sub(1, smoothstep(opts.above.from ?? 0, opts.above.to, opts.distance)),
    )
}

/**
 * Disc coverage with a soft, gamma-curved edge: a smoothstep band of half-width
 * `softness.width` around `radius`, raised through `softness.curve`. The soft cousin of the
 * kit's crisp `aa.discCoverage` — this one is a LOOK dial (blobs, orbs, glow dots), not an
 * anti-aliasing footprint.
 */
export function softDisc(opts: {
    dist: Expr
    radius: Expr | number
    softness: {width: Expr | number; curve: Expr | number}
}): Expr {
    const {width, curve} = opts.softness
    return pow(sub(1, smoothstep(sub(opts.radius, width), add(opts.radius, width), opts.dist)), curve)
}

/**
 * Crisp fill of a signed-distance field: 1 inside (`distance < 0`), 0 outside, feathered
 * across one `footprint` (a pixel in field units for anti-aliasing; wider for a deliberate
 * soft edge). The concept-level "fill this shape" word for any paint that draws SDFs
 * outside the shape scaffold (bars, blocks, curve envelopes).
 */
export function crispFill(opts: {distance: Expr; footprint: Expr | number}): Expr {
    const half = mul(opts.footprint, 0.5)
    return sub(1, smoothstep(neg(half), half, opts.distance))
}

/**
 * Crisp stroke of a signed-distance field: a band of `width` centred on the zero contour,
 * feathered across `footprint`. Sugar over {@link crispFill} on `|distance| − width/2` — the
 * "outline this shape / draw this curve" word.
 */
export function crispStroke(opts: {distance: Expr; width: Expr | number; footprint: Expr | number}): Expr {
    return crispFill({distance: sub(abs(opts.distance), mul(opts.width, 0.5)), footprint: opts.footprint})
}

/**
 * Aspect-corrected radial falloff: 0 inside `radius`, rising to 1 at `radius + falloff`.
 * `center` binds a position prop (transformPosition-stored). Body: `fields.radialFalloffMask`.
 */
export function radialMask(slots: {center: PropRef; radius: PropRef; falloff: PropRef}): Scalar {
    return new Scalar({kind: 'radialMask', center: slots.center, radius: slots.radius, falloff: slots.falloff})
}

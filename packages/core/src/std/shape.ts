/**
 * std — the 2D geometry vocabulary.
 *
 * Signed-distance fields as expressions, the language's `clip-path` shapes: each function
 * takes the shape-local coordinates plus its dimensions (expressions or numbers) and returns
 * the signed distance. Shape shaders declare `distance: (local, u) => circle(local.x, local.y,
 * u('radius'))` — anything shape-specific (Circle's radius→edge halving, Arc's degrees→radians)
 * is plain algebra at the declaration site.
 */
import type {Expr} from '../gpu/contract'
import {call, floatE} from '../gpu/composer'
import {sdf} from '../gpu/kit/index'

type Val = Expr | number

const toE = (v: Val): Expr => (typeof v === 'number' ? floatE(v) : v)

const field =
    (fn: unknown, hint: string) =>
    (...args: Val[]): Expr =>
        call(fn, hint, args.map(toE))

export const circle = field(sdf.circleSdf, 'circleSdf') as (x: Val, y: Val, radius: Val) => Expr
export const ellipse = field(sdf.ellipseSdf, 'ellipseSdf') as (x: Val, y: Val, radiusX: Val, radiusY: Val) => Expr
export const roundedRect = field(sdf.roundedRectSdf, 'roundedRectSdf') as (x: Val, y: Val, width: Val, height: Val, rounding: Val) => Expr
export const polygon = field(sdf.polygonSdf, 'polygonSdf') as (x: Val, y: Val, radius: Val, sides: Val) => Expr
export const star = field(sdf.starSdf, 'starSdf') as (x: Val, y: Val, outerRadius: Val, sides: Val, innerRatio: Val) => Expr
export const flower = field(sdf.flowerSdf, 'flowerSdf') as (x: Val, y: Val, outerRadius: Val, sides: Val, innerRatio: Val) => Expr
export const heart = field(sdf.heartSdf, 'heartSdf') as (x: Val, y: Val, radius: Val) => Expr
export const cross = field(sdf.crossSdf, 'crossSdf') as (x: Val, y: Val, size: Val, thickness: Val, rounding: Val) => Expr
export const ring = field(sdf.ringSdf, 'ringSdf') as (x: Val, y: Val, radius: Val, thickness: Val) => Expr
export const vesica = field(sdf.vesicaSdf, 'vesicaSdf') as (x: Val, y: Val, radius: Val, spread: Val) => Expr
export const crescent = field(sdf.crescentSdf, 'crescentSdf') as (x: Val, y: Val, outerRadius: Val, innerRatio: Val, offset: Val) => Expr
export const trapezoid = field(sdf.trapezoidSdf, 'trapezoidSdf') as (x: Val, y: Val, topWidth: Val, bottomWidth: Val, height: Val) => Expr
export const teardrop = field(sdf.teardropSdf, 'teardropSdf') as (x: Val, y: Val, radius: Val, height: Val) => Expr
export const parallelogram = field(sdf.parallelogramSdf, 'parallelogramSdf') as (x: Val, y: Val, width: Val, height: Val, skew: Val) => Expr
export const arc = field(sdf.arcSdf, 'arcSdf') as (x: Val, y: Val, radius: Val, halfAngle: Val) => Expr

/** Degrees of aperture → the half-angle in radians `arc` takes (π/360). */
export const APERTURE_TO_HALF_ANGLE = sdf.APERTURE_TO_HALFANGLE

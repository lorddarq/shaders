/**
 * std/paint — media vocabulary: sources whose pixels come from the host (an image file, a
 * video element, the user's camera) rather than from math.
 *
 * Every media noun is the same two halves:
 *   - a HOST LIFECYCLE (kit/host/mediaLifecycle): a swappable media texture or a
 *     readiness-gated external-texture source, created once at composition and torn down in
 *     its onCleanup;
 *   - the shared GPU FIT SURFACE (kit/media's `mediaSurface`): object-fit UV scale → fitted
 *     UV → sample → decode, with the media's pixel size read on the GPU via
 *     `textureDimensions` (no CPU aspect uniforms).
 *
 * The nouns return `{gpu: {fragment}}` for the definition to spread (the compute-backed-blur
 * precedent, minus the compute half). Declarative lifecycle flags (`naturalSizeKey`,
 * `acceptsUVContext`, resize-fit bounds) stay declared on the definition — a noun never owns
 * an engine flag. Error TAXONOMY stays per-shader by design (only ImageTexture knows how to
 * phrase a bad image URL; only WebcamTexture knows a camera permission denial), so each noun
 * takes the consumer's `onError` as a slot.
 */
import type {Expr, GpuFragmentParams} from '../../gpu/contract'
import {call} from '../../gpu/composer'
import {media} from '../../gpu/kit/index'
import {
    createSwappableMediaTexture,
    createUrlSourceLoader,
    createVideoElementSource,
    decodeImageSource,
} from '../../gpu/kit/host/mediaLifecycle'
import {registerNaturalSize} from '../../utilities/naturalSize'
import type {PropRef} from '../values'

/** The GPU half a media noun contributes to a custom-tier definition — spread it. */
export interface MediaSurfaceHalf {
    gpu: {fragment: (params: GpuFragmentParams) => Expr}
}

/** The fit-surface slots every media noun shares. */
interface MediaFitSlots {
    /** The structural object-fit prop (`compileTime` — only the selected mode's math is emitted). */
    fit: PropRef
    /**
     * How the sample is decoded. Required rather than defaulted so each consumer states its
     * choice (kit/media's D-1 rule).
     */
    decode: 'srgb-linear-alpha-cut'
}

/** Assemble the shared fit surface against the composition params. */
function fitSurface(
    params: GpuFragmentParams,
    slots: MediaFitSlots,
    texture: media.MediaSurfaceOptions['texture'],
    extra?: Pick<media.MediaSurfaceOptions, 'allowNone' | 'uvPostProcess'>,
): Expr {
    const {ctx, propValues, uvContext, effectiveViewportSize} = params
    return media.mediaSurface({
        texture,
        uv: uvContext ?? ctx.uv,
        viewport: effectiveViewportSize ?? ctx.viewportSize,
        fit: propValues[slots.fit.name],
        decode: slots.decode,
        ...extra,
    })
}

/**
 * A static image, fetched → decoded → written into a media texture sized to the image's
 * NATIVE pixels. A 1×1 transparent placeholder shows until the image lands; the real texture
 * is swapped in when it loads (the pass manager rebuilds the sampling bind group on the swap —
 * no recompose), which is what makes `textureDimensions` report the real aspect. sRGB samples
 * are decoded to linear in-shader (the media analog of the external-texture decode).
 */
export function imageMedia(opts: MediaFitSlots & {
    /** The URL prop, read live via the URL-loader state machine. */
    src: PropRef
    /** Debug label for the media texture. */
    label: string
    /** Load-failure reporting — phrasing stays with the consumer. */
    onError?: (error: unknown, url: string) => void
}): MediaSurfaceHalf {
    return {gpu: {fragment: (params: GpuFragmentParams): Expr => {
        const tex = createSwappableMediaTexture(params, {label: opts.label})

        createUrlSourceLoader(params, {
            prop: opts.src.name,
            async load(url, loadCtx) {
                const decoded = await decodeImageSource(url)
                if (loadCtx.isDisposed()) {
                    decoded.close()
                    return
                }
                loadCtx.commit()
                // Layout measures the source's INTRINSIC size (a 32px SVG logo slots at 32px), not
                // the raster the texture is baked at — for a bitmap the two are the same.
                registerNaturalSize(url, decoded.naturalWidth, decoded.naturalHeight)
                tex.ensureSize(decoded.width, decoded.height)
                tex.write(decoded.source)
                decoded.close()
            },
            onError: opts.onError,
        })

        return fitSurface(params, opts, tex.kit)
    }}}
}

/**
 * A URL video: the shared video-element lifecycle (created once at composition, torn down in
 * its onCleanup) feeding a per-frame zero-copy `importExternalTexture` sample. The registered
 * getter returns the element only when it has a decodable frame; the pass manager re-imports
 * it each frame and skips the pass while it is null (the WGSL is fixed, so readiness never
 * recompiles). Firefox's CanvasTexture copy-path fallback is not implemented — its single
 * site is in the host module.
 */
export function videoMedia(opts: MediaFitSlots & {
    /** The URL prop. */
    src: PropRef
    /** The loop prop, synced to `video.loop` each frame. */
    loop: PropRef
    /** Milliseconds to wait for `loadedmetadata` before giving up. */
    metadataTimeoutMs: number
    onError?: (error: unknown, info: {url: string | null; stage: 'acquire' | 'autoplay'}) => void
}): MediaSurfaceHalf {
    return {gpu: {fragment: (params: GpuFragmentParams): Expr => {
        const source = createVideoElementSource(params, {
            source: {kind: 'url', prop: opts.src.name},
            loop: {prop: opts.loop.name},
            metadataTimeoutMs: opts.metadataTimeoutMs,
            onError: opts.onError,
        })
        return fitSurface(params, opts, params.registerExternalTexture(source.getSource))
    }}}
}

/**
 * The user's camera: the exact `videoMedia` path, only the source is `getUserMedia` instead
 * of a URL (which is why both share the host's `createVideoElementSource`). Autoplay is
 * required — a webcam element that would not play is not a usable feed, unlike a URL video
 * where a blocked autoplay still yields a first frame. `mirror` is a runtime uniform (selfie
 * flip must not recompose) applied to the fitted UV BEFORE the letterbox alpha cut, so the
 * cut sees the same UV that was sampled.
 */
export function webcamMedia(opts: MediaFitSlots & {
    /** The selfie-mirror boolean prop (runtime uniform). */
    mirror: PropRef
    /** `getUserMedia` constraints — behaviour data, stated by the consumer. */
    constraints: MediaStreamConstraints
    /** The fixed natural-size registry key (a webcam has no URL). */
    naturalSizeKey: string
    onError?: (error: unknown, info: {url: string | null; stage: 'acquire' | 'autoplay'}) => void
}): MediaSurfaceHalf {
    return {gpu: {fragment: (params: GpuFragmentParams): Expr => {
        const source = createVideoElementSource(params, {
            source: {kind: 'webcam', constraints: opts.constraints},
            naturalSizeKey: opts.naturalSizeKey,
            autoplayRequired: true,
            onError: opts.onError,
        })
        return fitSurface(params, opts, params.registerExternalTexture(source.getSource), {
            allowNone: true,
            uvPostProcess: (uv) => call(media.applyMirror, 'applyMirror', [uv, params.uniforms[opts.mirror.name]]),
        })
    }}}
}

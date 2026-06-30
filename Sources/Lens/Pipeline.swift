// Lens text-to-image pipeline — Swift/MLX. Mirrors lens_mlx/pipeline_mlx.py.
// This file carries the pipeline MATH (CFG, denoise loop, latent packing) and the
// chat-template constants (T5). The GPT-OSS encoder wrapper lands in TextEncoder.swift
// (P4); until then the pipeline is drivable from precomputed features (the goldens).

import Foundation
import MLX
import MLXRandom
import Flux2VAE

// Chat template constants used by the Lens text encoder (verbatim from upstream).
public enum LensChatTemplate {
    public static let system =
        "Describe the image by detailing the color, shape, size, texture, "
        + "quantity, text, spatial relationships of the objects and background."
    public static let assistantThinking = "Need to generate one image according to the description."
    public static let defaultTxtOffset = 97
}

public enum LensPipeline {
    public static let vaeScaleFactor = 16
    public static let latentChannels = 128

    /// Norm-rescaled classifier-free guidance (NOT vanilla CFG — pitfall #11).
    /// comb = uncond + g*(cond-uncond); rescale per-token by ||cond||/||comb||.
    public static func lensCFG(_ noise: MLXArray, guidanceScale: Float) -> MLXArray {
        let parts = split(noise, parts: 2, axis: 0)
        let (cond, uncond) = (parts[0], parts[1])
        let comb = uncond + guidanceScale * (cond - uncond)
        let condNorm = sqrt(cond.square().sum(axis: -1, keepDims: true))
        let combNorm = sqrt(comb.square().sum(axis: -1, keepDims: true))
        let scale = MLX.where(
            combNorm .> 0,
            condNorm / maximum(combNorm, MLXArray(Float(1e-12))),
            MLXArray.ones(like: combNorm))
        return comb * scale
    }

    /// Lens flow-match denoising loop with CFG batching + norm-rescaled guidance.
    /// `latents` [1, S, C]; `encoderFeatures`/`encoderMask` already CFG-batched [cond; uncond].
    public static func denoise(
        transformer: LensTransformer2DModel,
        latents: MLXArray,
        encoderFeatures: [MLXArray],
        encoderMask: MLXArray,
        imgShape: (Int, Int, Int),
        numInferenceSteps: Int,
        guidanceScale: Float = 4.0
    ) -> MLXArray {
        var latents = latents
        let seqLen = latents.dim(1)
        let mu = computeEmpiricalMu(imageSeqLen: seqLen, numSteps: numInferenceSteps)
        let N = numInferenceSteps
        let sigmas: [Double] = N > 1
            ? (0..<N).map { 1.0 - Double($0) * (1.0 - 1.0 / Double(N)) / Double(N - 1) }
            : [1.0]
        let scheduler = FlowMatchEulerDiscreteScheduler()
        scheduler.setTimesteps(sigmas: sigmas, mu: Double(mu))

        for t in scheduler.timesteps {
            let hiddenStates = concatenated([latents, latents], axis: 0)
            let timestep = MLXArray(
                [Float](repeating: Float(t / 1000.0), count: hiddenStates.dim(0)))
            let noise = transformer(
                hiddenStates: hiddenStates, encoderHiddenStates: encoderFeatures,
                encoderHiddenStatesMask: encoderMask, timestep: timestep, imgShape: imgShape)
            let noisePred = lensCFG(noise, guidanceScale: guidanceScale)
            latents = scheduler.step(modelOutput: noisePred, sample: latents)
            eval(latents)
        }
        return latents
    }

    /// [b, h*w, c*4] -> packed [b, c*4, h, w] (Lens _decode: rearrange then patchify).
    public static func packLatentsForDecode(
        _ latents: MLXArray, latentH: Int, latentW: Int
    ) -> MLXArray {
        let b = latents.dim(0)
        let c = latents.dim(2) / 4
        var x = latents.reshaped(b, latentH, latentW, c, 2, 2).transposed(0, 3, 1, 4, 2, 5)
        x = x.reshaped(b, c, latentH * 2, latentW * 2)
        let H = latentH * 2
        let W = latentW * 2
        x = x.reshaped(b, c, H / 2, 2, W / 2, 2).transposed(0, 1, 3, 5, 2, 4)
        return x.reshaped(b, c * 4, H / 2, W / 2)
    }
}

// MARK: - Full pipeline (assembled after P1–P5 locked)

import Tokenizers

/// Assembled Lens text-to-image pipeline: tokenizer + GPT-OSS encoder + DiT + VAE.
///
/// **Per-stage residency (efficiency contract 1.14.0).** The GPT-OSS-20B text encoder
/// (~40 GB bf16 — the bulk of the old flat 62 GB footprint) is used ONCE per request to
/// produce the multi-layer text features, then sits idle through the entire DiT denoise
/// loop and the FLUX.2 VAE decode — the heaviest, longest phase. So the generator does NOT
/// hold the encoder resident: it owns an async `encoderProvider` (the wrapper's loader),
/// loads the encoder on demand, encodes, then **evicts it (`nil` + `Memory.clearCache()`)
/// before the denoise peak**, reclaiming the ~40 GB. Only the 3.8B DiT (the resident floor
/// + the activation peak) and the small VAE stay resident. Both wrappers (base + Turbo)
/// share this core, so both inherit the eviction. Tradeoff: the encoder re-loads per
/// request (cheap encode vs. expensive denoise) — a `keepEncoderResident` flag covers
/// big-RAM tiers (and the parity tests, which can't reload it).
public final class LensGenerator {
    public let transformer: LensTransformer2DModel
    public let vae: Flux2VAE
    /// Lazy loader for the GPT-OSS-20B encoder. Invoked per request, then evicted before
    /// the denoise peak (unless `keepEncoderResident`).
    public let encoderProvider: () async throws -> LensGptOssEncoder
    public let tokenizer: any Tokenizers.Tokenizer
    /// Keep the encoder resident across requests (skip per-request evict+reload). Default
    /// `false` = evict-between-stages, the memory-citizen default; `true` on big-RAM tiers.
    public let keepEncoderResident: Bool

    /// Hot encoder when `keepEncoderResident` is set (avoids the reload each request).
    private var residentEncoder: LensGptOssEncoder?

    /// Staged init: the encoder is loaded on demand via `encoderProvider`, not held resident.
    public init(
        transformer: LensTransformer2DModel, vae: Flux2VAE,
        encoderProvider: @escaping () async throws -> LensGptOssEncoder,
        tokenizer: any Tokenizers.Tokenizer,
        keepEncoderResident: Bool = false
    ) {
        self.transformer = transformer
        self.vae = vae
        self.encoderProvider = encoderProvider
        self.tokenizer = tokenizer
        self.keepEncoderResident = keepEncoderResident
    }

    /// Back-compat init from an already-loaded encoder. The encoder is kept resident (the
    /// pre-staged behavior) since the caller has no way to reload it. Prefer the
    /// `encoderProvider` init to get per-stage eviction.
    public convenience init(
        transformer: LensTransformer2DModel, vae: Flux2VAE,
        encoder: LensGptOssEncoder, tokenizer: any Tokenizers.Tokenizer
    ) {
        self.init(
            transformer: transformer, vae: vae, encoderProvider: { encoder },
            tokenizer: tokenizer, keepEncoderResident: true)
    }

    /// Obtain the text encoder for this request. Reuses the hot encoder when
    /// `keepEncoderResident`, otherwise loads a fresh one via `encoderProvider`.
    private func loadEncoder(isolation: isolated (any Actor)? = #isolation) async throws
        -> LensGptOssEncoder
    {
        if keepEncoderResident, let residentEncoder { return residentEncoder }
        let encoder = try await encoderProvider()
        if keepEncoderResident { residentEncoder = encoder }
        return encoder
    }

    /// Drop the encoder's weights before the denoise peak. A no-op when keeping it resident;
    /// otherwise nils the caller's last strong reference and clears the buffer cache,
    /// reclaiming the ~40 GB before the DiT denoise loop.
    private func evictEncoder(_ encoder: inout LensGptOssEncoder?) {
        guard !keepEncoderResident else { return }
        encoder = nil           // release the encoder's MLXArrays (last strong ref)
        Memory.clearCache()     // return the freed buffers to the OS before denoise
    }

    /// Chat-template encode + offset slice -> (CFG-batched features, mask). Uses the
    /// supplied (per-request) encoder so the caller can evict it before the denoise loop.
    func encode(prompt: String, encoder: LensGptOssEncoder) -> ([MLXArray], MLXArray) {
        let rendered = LensChatTemplate.render(prompt: prompt)
        let ids = tokenizer.encode(text: rendered, addSpecialTokens: false)
        let idArray = MLXArray(ids.map { Int32($0) }).expandedDimensions(axis: 0)
        let layers = encoder(idArray)
        let off = LensChatTemplate.defaultTxtOffset
        let feats = layers.map { layer -> MLXArray in
            let pos = layer[0..., off..., 0...]
            return concatenated([pos, MLXArray.zeros(like: pos)], axis: 0)
        }
        let seqLen = ids.count - off
        let posMask = MLXArray.ones([1, seqLen], dtype: .int32)
        let mask = concatenated([posMask, MLXArray.zeros(like: posMask)], axis: 0)
        return (feats, mask)
    }

    /// Returns interleaved RGB uint8 pixels (height × width × 3) in row-major order.
    public func generate(
        prompt: String, height: Int = 1024, width: Int = 1024,
        numInferenceSteps: Int = 20, guidanceScale: Float = 4.0, seed: UInt64 = 0,
        isolation: isolated (any Actor)? = #isolation
    ) async throws -> (pixels: [UInt8], height: Int, width: Int) {
        guard height % LensPipeline.vaeScaleFactor == 0,
              width % LensPipeline.vaeScaleFactor == 0
        else { throw LensError.generation("height/width must be divisible by 16") }
        let latentH = height / LensPipeline.vaeScaleFactor
        let latentW = width / LensPipeline.vaeScaleFactor

        // PER-STAGE EVICTION: load the GPT-OSS-20B encoder, encode, force-materialize the
        // features (`eval`), then drop the encoder + clear the cache BEFORE the denoise loop
        // so the ~40 GB encoder is not co-resident with the DiT activation peak.
        var encoderRef: LensGptOssEncoder? = try await loadEncoder()
        let (enc, mask) = encode(prompt: prompt, encoder: encoderRef!)
        eval(enc + [mask])         // materialize off the encoder graph
        evictEncoder(&encoderRef)  // reclaim the ~40 GB before the DiT denoise peak

        MLXRandom.seed(seed)
        var latents = MLXRandom.normal(
            [1, latentH * latentW, LensPipeline.latentChannels])
        latents = latents.asType(transformer.imgIn.weight.dtype)

        latents = LensPipeline.denoise(
            transformer: transformer, latents: latents,
            encoderFeatures: enc.map { $0.asType(latents.dtype) },
            encoderMask: mask,
            imgShape: (1, latentH, latentW),
            numInferenceSteps: numInferenceSteps, guidanceScale: guidanceScale)

        let packed = LensPipeline.packLatentsForDecode(
            latents.asType(.float32), latentH: latentH, latentW: latentW)
        var img = vae.decodePackedLatents(packed)  // [1, 3, H, W] in [-1, 1]
        img = clip(img, min: -1, max: 1)
        img = (img + 1) * 127.5
        img = img[0].transposed(1, 2, 0)  // [H, W, 3]
        eval(img)
        let pixels = img.asType(.uint8).asArray(UInt8.self)
        return (pixels, height, width)
    }
}

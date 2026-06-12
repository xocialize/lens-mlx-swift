// Lens text-to-image pipeline — Swift/MLX. Mirrors lens_mlx/pipeline_mlx.py.
// This file carries the pipeline MATH (CFG, denoise loop, latent packing) and the
// chat-template constants (T5). The GPT-OSS encoder wrapper lands in TextEncoder.swift
// (P4); until then the pipeline is drivable from precomputed features (the goldens).

import Foundation
import MLX
import MLXRandom

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
public final class LensGenerator {
    public let transformer: LensTransformer2DModel
    public let vae: Flux2VAE
    public let encoder: LensGptOssEncoder
    public let tokenizer: any Tokenizers.Tokenizer

    public init(
        transformer: LensTransformer2DModel, vae: Flux2VAE,
        encoder: LensGptOssEncoder, tokenizer: any Tokenizers.Tokenizer
    ) {
        self.transformer = transformer
        self.vae = vae
        self.encoder = encoder
        self.tokenizer = tokenizer
    }

    /// Chat-template encode + offset slice -> (CFG-batched features, mask).
    func encode(prompt: String) -> ([MLXArray], MLXArray) {
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
        numInferenceSteps: Int = 20, guidanceScale: Float = 4.0, seed: UInt64 = 0
    ) throws -> (pixels: [UInt8], height: Int, width: Int) {
        guard height % LensPipeline.vaeScaleFactor == 0,
              width % LensPipeline.vaeScaleFactor == 0
        else { throw LensError.generation("height/width must be divisible by 16") }
        let latentH = height / LensPipeline.vaeScaleFactor
        let latentW = width / LensPipeline.vaeScaleFactor

        let (enc, mask) = encode(prompt: prompt)

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

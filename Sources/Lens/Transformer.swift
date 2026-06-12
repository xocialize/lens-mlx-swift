// Lens denoising transformer (DiT) — Swift/MLX port.
//
// Isomorphic to lens_mlx/model/transformer.py (the parity-locked Python MLX port,
// itself isomorphic to refs/Lens/lens/transformer.py). Class / method names and the
// forward call order match; only Python MLX ops are swapped for mlx-swift.
//
// Epsilons (confirmed from the reference): QK-norm 1e-5, block norms 1e-6,
// txt_norm 1e-5, norm_out 1e-6.

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Embeddings & RoPE

/// Sinusoidal timestep embeddings (DDPM-style). Mirrors the reference helper.
func getTimestepEmbedding(
    timesteps: MLXArray,
    embeddingDim: Int,
    flipSinToCos: Bool = false,
    downscaleFreqShift: Float = 1.0,
    scale: Float = 1.0,
    maxPeriod: Int = 10000
) -> MLXArray {
    precondition(timesteps.ndim == 1, "Timesteps should be 1-D")
    let halfDim = embeddingDim / 2
    var exponent = -log(Float(maxPeriod)) * MLXArray(0..<halfDim).asType(.float32)
    exponent = exponent / (Float(halfDim) - downscaleFreqShift)
    var emb = exp(exponent)
    emb = timesteps[0..., .newAxis].asType(.float32) * emb[.newAxis, 0...]
    emb = scale * emb
    var out = concatenated([sin(emb), cos(emb)], axis: -1)
    if flipSinToCos {
        out = concatenated([out[0..., halfDim...], out[0..., ..<halfDim]], axis: -1)
    }
    // embeddingDim is even for Lens (256); odd-pad branch omitted.
    return out
}

/// Apply complex-valued RoPE (Lens variant) as a real interleaved rotation.
///
/// x: [B, S, H, D]; cos/sin: [S, D/2] (= real/imag of upstream `freqs_cis`).
/// Upstream pairs (x[...,2i], x[...,2i+1]) as (real, imag): out_r = x_r*cos - x_i*sin;
/// out_i = x_r*sin + x_i*cos. Adjacent pairs == reshape (..., D/2, 2).
func applyRotaryEmbLens(_ x: MLXArray, cos cosT: MLXArray, sin sinT: MLXArray) -> MLXArray {
    let shape = x.shape
    let pairs = x.reshaped(shape[0], shape[1], shape[2], shape[3] / 2, 2)
    let xR = pairs[.ellipsis, 0]
    let xI = pairs[.ellipsis, 1]
    let c = cosT[.newAxis, 0..., .newAxis, 0...]
    let s = sinT[.newAxis, 0..., .newAxis, 0...]
    let outR = xR * c - xI * s
    let outI = xR * s + xI * c
    return stacked([outR, outI], axis: -1).reshaped(shape).asType(x.dtype)
}

/// SwiGLU MLP used by the transformer blocks.
public final class GateMLP: Module, UnaryLayer {
    @ModuleInfo(key: "w1") var w1: Linear
    @ModuleInfo(key: "w2") var w2: Linear
    @ModuleInfo(key: "w3") var w3: Linear

    public init(dim: Int, hiddenDim: Int) {
        self._w1.wrappedValue = Linear(dim, hiddenDim, bias: false)
        self._w2.wrappedValue = Linear(hiddenDim, dim, bias: false)
        self._w3.wrappedValue = Linear(dim, hiddenDim, bias: false)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)) * w3(x))
    }
}

public final class TimestepEmbedding: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    public init(inChannels: Int, timeEmbedDim: Int) {
        self._linear1.wrappedValue = Linear(inChannels, timeEmbedDim)
        self._linear2.wrappedValue = Linear(timeEmbedDim, timeEmbedDim)
        super.init()
    }

    public func callAsFunction(_ sample: MLXArray) -> MLXArray {
        linear2(silu(linear1(sample)))
    }
}

public final class LensTimestepProjEmbeddings: Module {
    @ModuleInfo(key: "timestep_embedder") var timestepEmbedder: TimestepEmbedding

    public init(embeddingDim: Int) {
        // time_proj = Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)
        self._timestepEmbedder.wrappedValue = TimestepEmbedding(
            inChannels: 256, timeEmbedDim: embeddingDim)
        super.init()
    }

    public func callAsFunction(_ timestep: MLXArray, _ hiddenStates: MLXArray) -> MLXArray {
        let proj = getTimestepEmbedding(
            timesteps: timestep, embeddingDim: 256, flipSinToCos: true,
            downscaleFreqShift: 0, scale: 1000)
        return timestepEmbedder(proj.asType(hiddenStates.dtype))
    }
}

/// Frame/H/W axial RoPE shared between image and text streams (emits cos/sin).
///
/// Plain class (NOT a Module): only computed rope tables, no learnable params —
/// as a Module the tables would be collected as parameters and expected in the
/// checkpoint (upstream stores them as non-persistent buffers).
public final class LensEmbedRope {
    let theta: Int
    let axesDim: [Int]
    let scaleRope: Bool
    let posFreqs: MLXArray  // [4096, sum(axesDim)/2] angles
    let negFreqs: MLXArray

    public init(theta: Int, axesDim: [Int], scaleRope: Bool = false) {
        self.theta = theta
        self.axesDim = axesDim
        self.scaleRope = scaleRope
        let posIndex = MLXArray(0..<4096)
        // arange(4096)[::-1] * -1 - 1 == arange(4096) - 4096  (= [-4096 … -1])
        let negIndex = MLXArray(0..<4096) - 4096
        self.posFreqs = concatenated(
            axesDim.map { Self.ropeParams(index: posIndex, dim: $0, theta: theta) }, axis: 1)
        self.negFreqs = concatenated(
            axesDim.map { Self.ropeParams(index: negIndex, dim: $0, theta: theta) }, axis: 1)
    }

    static func ropeParams(index: MLXArray, dim: Int, theta: Int = 10000) -> MLXArray {
        precondition(dim % 2 == 0)
        let invFreq = 1.0 / pow(
            MLXArray(Float(theta)),
            MLXArray(stride(from: 0, to: dim, by: 2).map { Float($0) }) / Float(dim))
        return outer(index.asType(.float32), invFreq)  // angle; cos/sin at apply time
    }

    /// Returns ((imgCos, imgSin), (txtCos, txtSin)).
    public func callAsFunction(
        videoFHW: (frame: Int, height: Int, width: Int), txtSeqLen: Int
    ) -> ((MLXArray, MLXArray), (MLXArray, MLXArray)) {
        let (frame, height, width) = videoFHW
        let videoFreq = computeVideoFreqs(frame: frame, height: height, width: width)
        let maxVidIndex = scaleRope ? max(height / 2, width / 2) : max(height, width)
        let txtFreqs = posFreqs[maxVidIndex ..< (maxVidIndex + txtSeqLen), 0...]
        return ((cos(videoFreq), sin(videoFreq)), (cos(txtFreqs), sin(txtFreqs)))
    }

    func computeVideoFreqs(frame: Int, height: Int, width: Int, idx: Int = 0) -> MLXArray {
        let seqLens = frame * height * width
        let splits = axesDim.map { $0 / 2 }
        var bounds: [Int] = [0]
        for s in splits { bounds.append(bounds.last! + s) }
        let fp = (0..<splits.count).map { posFreqs[0..., bounds[$0] ..< bounds[$0 + 1]] }
        let fn = (0..<splits.count).map { negFreqs[0..., bounds[$0] ..< bounds[$0 + 1]] }

        let freqsFrame = broadcast(
            fp[0][idx ..< (idx + frame)].reshaped(frame, 1, 1, -1),
            to: [frame, height, width, splits[0]])

        let freqsHeight: MLXArray
        let freqsWidth: MLXArray
        if scaleRope {
            freqsHeight = broadcast(
                concatenated([fn[1][(4096 - (height - height / 2))...], fp[1][..<(height / 2)]], axis: 0)
                    .reshaped(1, height, 1, -1),
                to: [frame, height, width, splits[1]])
            freqsWidth = broadcast(
                concatenated([fn[2][(4096 - (width - width / 2))...], fp[2][..<(width / 2)]], axis: 0)
                    .reshaped(1, 1, width, -1),
                to: [frame, height, width, splits[2]])
        } else {
            freqsHeight = broadcast(
                fp[1][..<height].reshaped(1, height, 1, -1),
                to: [frame, height, width, splits[1]])
            freqsWidth = broadcast(
                fp[2][..<width].reshaped(1, 1, width, -1),
                to: [frame, height, width, splits[2]])
        }
        return concatenated([freqsFrame, freqsHeight, freqsWidth], axis: -1)
            .reshaped(seqLens, -1)
    }
}

// MARK: - Attention (joint image + text, plain SDPA)

/// Joint image+text attention with fused QKV and SDPA backend.
public final class LensJointAttention: Module {
    let innerDim: Int
    let heads: Int
    let dimHead: Int
    let outDim: Int

    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm
    @ModuleInfo(key: "norm_added_q") var normAddedQ: RMSNorm
    @ModuleInfo(key: "norm_added_k") var normAddedK: RMSNorm

    @ModuleInfo(key: "img_qkv") var imgQKV: Linear
    @ModuleInfo(key: "txt_qkv") var txtQKV: Linear

    // upstream: to_out = ModuleList([Linear, Identity]); index 0 is the Linear.
    @ModuleInfo(key: "to_out") var toOut: [Linear]
    @ModuleInfo(key: "to_add_out") var toAddOut: Linear

    public init(
        queryDim: Int, addedKVProjDim: Int, dimHead: Int = 64, heads: Int = 8,
        outDim: Int? = nil, eps: Float = 1e-5
    ) {
        self.innerDim = outDim ?? (dimHead * heads)
        self.heads = innerDim / dimHead
        self.dimHead = dimHead
        self.outDim = outDim ?? queryDim

        self._normQ.wrappedValue = RMSNorm(dimensions: dimHead, eps: eps)
        self._normK.wrappedValue = RMSNorm(dimensions: dimHead, eps: eps)
        self._normAddedQ.wrappedValue = RMSNorm(dimensions: dimHead, eps: eps)
        self._normAddedK.wrappedValue = RMSNorm(dimensions: dimHead, eps: eps)

        self._imgQKV.wrappedValue = Linear(queryDim, 3 * innerDim, bias: true)
        self._txtQKV.wrappedValue = Linear(addedKVProjDim, 3 * innerDim, bias: true)

        self._toOut.wrappedValue = [Linear(innerDim, self.outDim, bias: true)]
        self._toAddOut.wrappedValue = Linear(innerDim, queryDim, bias: true)
        super.init()
    }

    public func callAsFunction(
        hiddenStates: MLXArray,
        encoderHiddenStates: MLXArray,
        imageRotaryEmb: ((MLXArray, MLXArray), (MLXArray, MLXArray)),
        attentionMask: MLXArray?
    ) -> (MLXArray, MLXArray) {
        let bsz = hiddenStates.dim(0)
        let seqImg = hiddenStates.dim(1)
        let seqTxt = encoderHiddenStates.dim(1)
        let (H, Dh) = (heads, dimHead)

        // Fused QKV: view(B,S,3,H,Dh) — STACKED layout (3 its own axis before heads).
        let imgChunks = imgQKV(hiddenStates).reshaped(bsz, seqImg, 3, H, Dh)
        let txtChunks = txtQKV(encoderHiddenStates).reshaped(bsz, seqTxt, 3, H, Dh)
        var imgQ = imgChunks[0..., 0..., 0]
        var imgK = imgChunks[0..., 0..., 1]
        let imgV = imgChunks[0..., 0..., 2]
        var txtQ = txtChunks[0..., 0..., 0]
        var txtK = txtChunks[0..., 0..., 1]
        let txtV = txtChunks[0..., 0..., 2]

        imgQ = normQ(imgQ)
        imgK = normK(imgK)
        txtQ = normAddedQ(txtQ)
        txtK = normAddedK(txtK)

        let ((imgCos, imgSin), (txtCos, txtSin)) = imageRotaryEmb
        imgQ = applyRotaryEmbLens(imgQ, cos: imgCos[..<seqImg], sin: imgSin[..<seqImg])
        imgK = applyRotaryEmbLens(imgK, cos: imgCos[..<seqImg], sin: imgSin[..<seqImg])
        if seqTxt > 0 {
            txtQ = applyRotaryEmbLens(txtQ, cos: txtCos[..<seqTxt], sin: txtSin[..<seqTxt])
            txtK = applyRotaryEmbLens(txtK, cos: txtCos[..<seqTxt], sin: txtSin[..<seqTxt])
        }

        // Joint sequence, [B, H, S, D] for SDPA.
        let q = concatenated([imgQ, txtQ], axis: 1).transposed(0, 2, 1, 3)
        let k = concatenated([imgK, txtK], axis: 1).transposed(0, 2, 1, 3)
        let v = concatenated([imgV, txtV], axis: 1).transposed(0, 2, 1, 3)

        let scale = 1.0 / sqrt(Float(Dh))
        let maskMode: MLXFast.ScaledDotProductAttentionMaskMode =
            attentionMask.map { .array($0) } ?? .none
        var out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: maskMode)
        out = out.transposed(0, 2, 1, 3).reshaped(bsz, seqImg + seqTxt, -1)

        let imgOut = toOut[0](out[0..., ..<seqImg, 0...])
        let txtOut = toAddOut(out[0..., seqImg..., 0...])
        return (imgOut, txtOut)
    }
}

// MARK: - Transformer block

public final class LensTransformerBlock: Module {
    @ModuleInfo(key: "attn") var attn: LensJointAttention

    @ModuleInfo(key: "img_mod") var imgMod: Linear  // upstream Sequential(SiLU, Linear) -> .1
    @ModuleInfo(key: "img_norm1") var imgNorm1: RMSNorm
    @ModuleInfo(key: "img_norm2") var imgNorm2: RMSNorm
    @ModuleInfo(key: "img_mlp") var imgMLP: GateMLP

    @ModuleInfo(key: "txt_mod") var txtMod: Linear
    @ModuleInfo(key: "txt_norm1") var txtNorm1: RMSNorm
    @ModuleInfo(key: "txt_norm2") var txtNorm2: RMSNorm
    @ModuleInfo(key: "txt_mlp") var txtMLP: GateMLP

    public init(
        dim: Int, numAttentionHeads: Int, attentionHeadDim: Int, eps: Float = 1e-6
    ) {
        // rms_norm=true for Lens: RMSNorm with learnable weight, then AdaLN modulation.
        self._attn.wrappedValue = LensJointAttention(
            queryDim: dim, addedKVProjDim: dim, dimHead: attentionHeadDim,
            heads: numAttentionHeads, outDim: dim, eps: eps)

        let hidden = dim / 3 * 8
        self._imgMod.wrappedValue = Linear(dim, 6 * dim, bias: true)
        self._imgNorm1.wrappedValue = RMSNorm(dimensions: dim, eps: eps)
        self._imgNorm2.wrappedValue = RMSNorm(dimensions: dim, eps: eps)
        self._imgMLP.wrappedValue = GateMLP(dim: dim, hiddenDim: hidden)

        self._txtMod.wrappedValue = Linear(dim, 6 * dim, bias: true)
        self._txtNorm1.wrappedValue = RMSNorm(dimensions: dim, eps: eps)
        self._txtNorm2.wrappedValue = RMSNorm(dimensions: dim, eps: eps)
        self._txtMLP.wrappedValue = GateMLP(dim: dim, hiddenDim: hidden)
        super.init()
    }

    /// AdaLN: x*(1+scale)+shift; gate multiplies the residual branch (T3).
    static func modulate(_ x: MLXArray, _ modParams: MLXArray) -> (MLXArray, MLXArray) {
        let parts = split(modParams, parts: 3, axis: -1)
        let (shift, scale, gate) = (parts[0], parts[1], parts[2])
        return (
            x * (1 + scale[0..., .newAxis, 0...]) + shift[0..., .newAxis, 0...],
            gate[0..., .newAxis, 0...]
        )
    }

    public func callAsFunction(
        hiddenStates: MLXArray,
        encoderHiddenStates: MLXArray,
        temb: MLXArray,
        imageRotaryEmb: ((MLXArray, MLXArray), (MLXArray, MLXArray)),
        attentionMask: MLXArray?
    ) -> (MLXArray, MLXArray) {
        var hiddenStates = hiddenStates
        var encoderHiddenStates = encoderHiddenStates

        let imgMods = split(imgMod(silu(temb)), parts: 2, axis: -1)
        let txtMods = split(txtMod(silu(temb)), parts: 2, axis: -1)

        let (imgModulated, imgGate1) = Self.modulate(imgNorm1(hiddenStates), imgMods[0])
        let (txtModulated, txtGate1) = Self.modulate(txtNorm1(encoderHiddenStates), txtMods[0])

        let (imgAttn, txtAttn) = attn(
            hiddenStates: imgModulated, encoderHiddenStates: txtModulated,
            imageRotaryEmb: imageRotaryEmb, attentionMask: attentionMask)

        hiddenStates = hiddenStates + imgGate1 * imgAttn
        encoderHiddenStates = encoderHiddenStates + txtGate1 * txtAttn

        let (imgModulated2, imgGate2) = Self.modulate(imgNorm2(hiddenStates), imgMods[1])
        hiddenStates = hiddenStates + imgGate2 * imgMLP(imgModulated2)

        let (txtModulated2, txtGate2) = Self.modulate(txtNorm2(encoderHiddenStates), txtMods[1])
        encoderHiddenStates = encoderHiddenStates + txtGate2 * txtMLP(txtModulated2)

        return (encoderHiddenStates, hiddenStates)
    }
}

// MARK: - Top-level model

/// norm_out: SiLU -> Linear(dim, 2*dim) -> affine-less LayerNorm modulation.
public final class AdaLayerNormContinuous: Module {
    @ModuleInfo(key: "linear") var linear: Linear
    @ModuleInfo(key: "norm") var norm: LayerNorm

    public init(dim: Int, condDim: Int, eps: Float = 1e-6) {
        self._linear.wrappedValue = Linear(condDim, 2 * dim, bias: true)
        self._norm.wrappedValue = LayerNorm(dimensions: dim, eps: eps, affine: false)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, conditioning: MLXArray) -> MLXArray {
        let emb = linear(silu(conditioning))
        let parts = split(emb, parts: 2, axis: -1)
        let (scale, shift) = (parts[0], parts[1])
        return norm(x) * (1 + scale[0..., .newAxis, 0...]) + shift[0..., .newAxis, 0...]
    }
}

/// The Lens text-to-image DiT.
public final class LensTransformer2DModel: Module {
    public let inChannels: Int
    public let outChannels: Int
    public let innerDim: Int
    public let patchSize: Int
    public let multiLayerEncoderFeature: Bool
    public let selectedLayerIndex: [Int]

    public let posEmbed: LensEmbedRope  // plain class — no parameters

    @ModuleInfo(key: "time_text_embed") var timeTextEmbed: LensTimestepProjEmbeddings
    @ModuleInfo(key: "txt_norm") var txtNorm: [RMSNorm]
    @ModuleInfo(key: "txt_in") var txtIn: Linear
    @ModuleInfo(key: "img_in") var imgIn: Linear
    @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [LensTransformerBlock]
    @ModuleInfo(key: "norm_out") var normOut: AdaLayerNormContinuous
    @ModuleInfo(key: "proj_out") var projOut: Linear

    public init(
        patchSize: Int = 2,
        inChannels: Int = 128,
        outChannels: Int? = 32,
        numLayers: Int = 48,
        attentionHeadDim: Int = 64,
        numAttentionHeads: Int = 24,
        encHiddenDim: Int = 2880,
        axesDimsRope: [Int] = [8, 28, 28],
        multiLayerEncoderFeature: Bool = true,
        selectedLayerIndex: [Int] = [5, 11, 17, 23]
    ) {
        let inner = numAttentionHeads * attentionHeadDim
        self.inChannels = inChannels
        self.outChannels = outChannels ?? inChannels
        self.innerDim = inner
        self.patchSize = patchSize
        self.multiLayerEncoderFeature = multiLayerEncoderFeature
        self.selectedLayerIndex = selectedLayerIndex

        self.posEmbed = LensEmbedRope(theta: 10000, axesDim: axesDimsRope, scaleRope: true)
        self._timeTextEmbed.wrappedValue = LensTimestepProjEmbeddings(embeddingDim: inner)

        precondition(multiLayerEncoderFeature, "single-layer encoder path not used by Lens")
        self._txtNorm.wrappedValue = selectedLayerIndex.map { _ in
            RMSNorm(dimensions: encHiddenDim, eps: 1e-5)
        }
        self._txtIn.wrappedValue = Linear(encHiddenDim * selectedLayerIndex.count, inner)
        self._imgIn.wrappedValue = Linear(inChannels, inner)
        self._transformerBlocks.wrappedValue = (0..<numLayers).map { _ in
            LensTransformerBlock(
                dim: inner, numAttentionHeads: numAttentionHeads,
                attentionHeadDim: attentionHeadDim)
        }
        self._normOut.wrappedValue = AdaLayerNormContinuous(dim: inner, condDim: inner)
        self._projOut.wrappedValue = Linear(
            inner, patchSize * patchSize * (outChannels ?? inChannels), bias: true)
        super.init()
    }

    /// - Parameters:
    ///   - hiddenStates: [B, imgLen, inChannels] packed latents.
    ///   - encoderHiddenStates: per-selected-layer features, each [B, S, encHiddenDim].
    ///   - encoderHiddenStatesMask: [B, S] 1 = real token.
    ///   - timestep: [B] in 0…1 (t/1000).
    ///   - imgShape: (frame, latentH/2, latentW/2) patch grid.
    public func callAsFunction(
        hiddenStates: MLXArray,
        encoderHiddenStates: [MLXArray],
        encoderHiddenStatesMask: MLXArray,
        timestep: MLXArray,
        imgShape: (Int, Int, Int)
    ) -> MLXArray {
        let imgLen = hiddenStates.dim(1)
        let textSeqLen = encoderHiddenStates[0].dim(1)
        let normed = (0..<txtNorm.count).map { txtNorm[$0](encoderHiddenStates[$0]) }
        var encoder = concatenated(normed, axis: -1)

        var hidden = imgIn(hiddenStates)
        // SDPA requires the additive mask to promote to the stream dtype (bf16 runs).
        let attentionMask = Self.buildJointAttentionMask(
            textMask: encoderHiddenStatesMask, imgLen: imgLen
        ).asType(hidden.dtype)
        encoder = txtIn(encoder)
        let temb = timeTextEmbed(timestep.asType(hidden.dtype), hidden)
        let imageRotaryEmb = posEmbed(videoFHW: imgShape, txtSeqLen: textSeqLen)

        for block in transformerBlocks {
            (encoder, hidden) = block(
                hiddenStates: hidden, encoderHiddenStates: encoder, temb: temb,
                imageRotaryEmb: imageRotaryEmb, attentionMask: attentionMask)
        }

        hidden = normOut(hidden, conditioning: temb)
        return projOut(hidden)
    }

    /// Additive joint mask [B, 1, 1, imgLen + S_txt]; -inf on padded text positions.
    static func buildJointAttentionMask(textMask: MLXArray, imgLen: Int) -> MLXArray {
        let bsz = textMask.dim(0)
        let imgOnes = MLXArray.ones([bsz, imgLen], dtype: .bool)
        let joint = concatenated([imgOnes, textMask.asType(.bool)], axis: 1)
        let additive = MLX.where(joint, MLXArray(Float(0)), MLXArray(-Float.infinity))
            .asType(.float32)
        return additive[0..., .newAxis, .newAxis, 0...]
    }
}

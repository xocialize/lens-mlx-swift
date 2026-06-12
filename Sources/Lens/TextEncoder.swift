// GPT-OSS text-feature encoder for Lens — Swift port of lens_mlx/model/text_encoder.py.
//
// Wraps the Adapted GPTOSS model: capture hidden states at `selectedLayers`
// (default [5,11,17,23], T0 — 0-indexed), early-exit after the last selected
// layer (skip final RMSNorm + LM head), alternating sliding(128)/full causal
// attention (T8). Tokenization + the T5 chat template live in Pipeline.swift.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public let defaultSelectedLayers = [5, 11, 17, 23]

/// GPT-OSS-20B uses YaRN rope — an HF class default the checkpoint config does NOT
/// serialize. The adapted GPTOSS falls back to YarnRoPE(scalingFactor: 1.0) == plain
/// rope without it (F9: diverges at every layer/position, cosine ~0.94). Inject when
/// the on-disk config omits rope_scaling.
let gptOssYarnRope: [String: StringOrNumber] = [
    "rope_type": .string("yarn"),
    "factor": .float(32.0),
    "beta_fast": .float(32.0),
    "beta_slow": .float(1.0),
    "original_max_position_embeddings": .int(4096),
]

/// Adapted-GPTOSS wrapper that exposes selected hidden states.
public final class LensGptOssEncoder {
    public let model: GPTOSSModel
    public let selectedLayers: [Int]

    public init(model: GPTOSSModel, selectedLayers: [Int] = defaultSelectedLayers) {
        self.model = model
        self.selectedLayers = selectedLayers
    }

    /// Load a gpt-oss checkpoint dir (HF bf16 dense or MXFP4 — the adapted sanitize
    /// handles both layouts). Injects YaRN when the config omits rope_scaling (F9).
    public static func fromPretrained(
        directory: URL, selectedLayers: [Int] = defaultSelectedLayers
    ) throws -> LensGptOssEncoder {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        var config = try JSONDecoder().decode(GPTOSSConfiguration.self, from: data)
        if config.ropeScaling == nil {
            config.ropeScaling = gptOssYarnRope
        }

        let model = GPTOSSModel(config)
        var weights = try LensWeights.loadAllArrays(directory: directory)
        weights = model.sanitize(weights: weights)

        // Strict two-way check AFTER sanitize (same discipline as the other loaders).
        let moduleKeys = Set(model.parameters().flattened().map(\.0))
        let fileKeys = Set(weights.keys)
        let missing = moduleKeys.subtracting(fileKeys).sorted()
        guard missing.isEmpty else {
            throw LensError.loading(
                "encoder: missing \(missing.count) keys, e.g. "
                + missing.prefix(4).joined(separator: ", "))
        }
        let consumed = weights.filter { moduleKeys.contains($0.key) }
        model.update(parameters: ModuleParameters.unflattened(consumed))
        eval(model)
        return LensGptOssEncoder(model: model, selectedLayers: selectedLayers)
    }

    /// Per-layer hidden states at the selected layers for [B, S] token ids.
    public func callAsFunction(_ inputIds: MLXArray) -> [MLXArray] {
        model.model.captureHiddenStates(inputIds, selectedLayers: selectedLayers)
    }
}

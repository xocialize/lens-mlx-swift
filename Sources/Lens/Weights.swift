// Weight loading for the Lens DiT + FLUX.2 VAE — Swift port of lens_mlx/utils/weights.py.
//
// DiT: pure Linear + RMSNorm — PT<->MLX layouts identical, no transpose. Key renames:
// upstream wraps AdaLN modulation in Sequential(SiLU, Linear) (`img_mod.1.*`) which both
// ports flatten to a single Linear (`img_mod.*`). Converted mlx-community repos
// (Lens-3.8B-bf16/-4bit/-8bit) are already sanitized + optionally quantized.
//
// VAE: diffusers-identical keys except `to_out.0.` -> `to_out.`, drop
// `num_batches_tracked`, and 4D conv weights transpose PT (O,I,kH,kW) -> MLX (O,kH,kW,I).
// Decoder-only here: encoder.* / quant_conv.* keys are skipped.

import Foundation
import MLX
import MLXNN

public enum LensWeights {

    static func loadAllArrays(directory: URL) throws -> [String: MLXArray] {
        let files = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil
        ).filter { $0.pathExtension == "safetensors" }.sorted {
            $0.lastPathComponent < $1.lastPathComponent
        }
        guard !files.isEmpty else {
            throw LensError.loading("no .safetensors under \(directory.path)")
        }
        var merged: [String: MLXArray] = [:]
        for f in files {
            merged.merge(try MLX.loadArrays(url: f)) { a, _ in a }
        }
        return merged
    }

    static func sanitizeDiTKey(_ k: String) -> String {
        k.replacingOccurrences(of: ".img_mod.1.", with: ".img_mod.")
            .replacingOccurrences(of: ".txt_mod.1.", with: ".txt_mod.")
    }

    /// Load a CONVERTED mlx DiT repo (bf16 / int4 / int8) produced by the Python
    /// recipes/convert_lens.py. Reads config.json; for a quantized repo, rebuilds the
    /// quantized module structure (same group_size/bits/keep_hi_precision) BEFORE
    /// load. Tensors load as-saved (no dtype cast — packed quant weights).
    public static func loadDiTRepo(directory: URL) throws -> LensTransformer2DModel {
        let configData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        guard let cfg = try JSONSerialization.jsonObject(with: configData) as? [String: Any]
        else { throw LensError.loading("unreadable DiT config.json") }

        let model = LensTransformer2DModel(
            patchSize: cfg["patch_size"] as? Int ?? 2,
            inChannels: cfg["in_channels"] as? Int ?? 128,
            outChannels: cfg["out_channels"] as? Int ?? 32,
            numLayers: cfg["num_layers"] as? Int ?? 48,
            attentionHeadDim: cfg["attention_head_dim"] as? Int ?? 64,
            numAttentionHeads: cfg["num_attention_heads"] as? Int ?? 24,
            encHiddenDim: cfg["enc_hidden_dim"] as? Int ?? 2880,
            axesDimsRope: cfg["axes_dims_rope"] as? [Int] ?? [8, 28, 28],
            selectedLayerIndex: cfg["selected_layer_index"] as? [Int] ?? [5, 11, 17, 23]
        )

        var weights = try loadAllArrays(directory: directory)
        weights = Dictionary(uniqueKeysWithValues: weights.map { (sanitizeDiTKey($0.key), $0.value) })

        if let q = cfg["quantization"] as? [String: Any],
           let groupSize = q["group_size"] as? Int, let bits = q["bits"] as? Int
        {
            let keepHi = (q["keep_hi_precision"] as? [String]) ?? []
            quantize(model: model, groupSize: groupSize, bits: bits) { path, module in
                guard module is Linear else { return false }
                guard weights["\(path).scales"] != nil else { return false }
                return !keepHi.contains { path.contains($0) }
            }
        }

        try verifyAndLoad(model: model, weights: weights, label: "DiT")
        return model
    }

    /// Load the DiT from the ORIGINAL PT `transformer/` safetensors (diagnostic /
    /// parity path — mirrors load_dit_weights). Pure Linear+RMSNorm: no transposes,
    /// only the img_mod/txt_mod Sequential flattening, plus a dtype cast.
    public static func loadDiTFromPT(directory: URL, dtype: DType = .float32) throws
        -> LensTransformer2DModel
    {
        let model = LensTransformer2DModel()
        var weights: [String: MLXArray] = [:]
        for (k, v) in try loadAllArrays(directory: directory) {
            weights[sanitizeDiTKey(k)] = v.asType(dtype)
        }
        try verifyAndLoad(model: model, weights: weights, label: "DiT(PT)")
        return model
    }

    /// Load the FLUX.2 VAE (decoder-only) from a diffusers `vae/` snapshot.
    public static func loadVAE(directory: URL, dtype: DType = .float32) throws -> Flux2VAE {
        let vae = Flux2VAE()
        var state: [String: MLXArray] = [:]
        for (rawKey, rawValue) in try loadAllArrays(directory: directory) {
            if rawKey.hasSuffix("num_batches_tracked") { continue }
            // Decoder-only: skip the encoder tower + its quant projection.
            if rawKey.hasPrefix("encoder.") || rawKey.hasPrefix("quant_conv.") { continue }
            var k = rawKey.replacingOccurrences(of: ".to_out.0.", with: ".to_out.")
            var v = rawValue
            if v.ndim == 4 {  // conv weight: PT (O,I,kH,kW) -> MLX (O,kH,kW,I)
                v = v.transposed(0, 2, 3, 1)
            }
            // bn stats stay fp32 regardless of requested dtype (de-norm precision).
            if k.hasPrefix("bn.") {
                v = v.asType(.float32)
            } else {
                v = v.asType(dtype)
            }
            state[k] = v
        }
        try verifyAndLoad(model: vae, weights: state, label: "VAE")
        return vae
    }

    /// Two-way strict load (workspace discipline): all module keys must be filled and
    /// every provided key consumed — a partial load emits garbage with no other symptom.
    static func verifyAndLoad(model: Module, weights: [String: MLXArray], label: String) throws {
        let moduleKeys = Set(model.parameters().flattened().map(\.0))
        let fileKeys = Set(weights.keys)
        let missing = moduleKeys.subtracting(fileKeys).sorted()
        guard missing.isEmpty else {
            throw LensError.loading(
                "\(label): checkpoint missing \(missing.count) module keys, e.g. "
                + missing.prefix(4).joined(separator: ", "))
        }
        let unused = fileKeys.subtracting(moduleKeys).sorted()
        guard unused.isEmpty else {
            throw LensError.loading(
                "\(label): \(unused.count) unconsumed checkpoint keys, e.g. "
                + unused.prefix(4).joined(separator: ", "))
        }
        model.update(parameters: ModuleParameters.unflattened(weights))
        eval(model)
    }
}

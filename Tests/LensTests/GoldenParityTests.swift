import XCTest
import MLX
@testable import Lens

/// P2/P3 golden gates (gated LENS_PARITY=1; CPU stream; needs the Cmlx metallib in
/// .build/debug for any GPU fallback ops, plus the goldens + weights below).
/// Goldens: fp32/CPU capture from the PT reference (512×512, 4 steps, seed 42) —
/// the same oracle the Python port locked against.
final class GoldenParityTests: XCTestCase {
    static let goldens = URL(fileURLWithPath:
        "/Volumes/DEV_VOL1/VideoResearch/lens-mlx-models/goldens/lens_goldens.safetensors")
    static let ptTransformer = URL(fileURLWithPath:
        "/Volumes/DEV_ARCHIVE/lens-mlx/weights/Lens/transformer")
    static let ptVAE = URL(fileURLWithPath:
        "/Volumes/DEV_ARCHIVE/lens-mlx/weights/Lens/vae")

    func cosine(_ a: MLXArray, _ b: MLXArray) -> Float {
        let x = a.asType(.float32).flattened()
        let y = b.asType(.float32).flattened()
        return ((x * y).sum() / (sqrt(x.square().sum()) * sqrt(y.square().sum()))).item(Float.self)
    }

    /// P2: full DiT forward vs dit_out_noise. Gate: max_abs < 1e-2 (fp32, CPU).
    func testFullDiTParity() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["LENS_PARITY"] == "1", "LENS_PARITY=1")
        Device.setDefault(device: Device.cpu)

        let g = try MLX.loadArrays(url: Self.goldens)
        let hidden = g["dit_in_hidden"]!.asType(.float32)            // [2,1024,128]
        let timestep = g["dit_in_timestep"]!.asType(.float32)        // [2]
        // Reconstruct CFG-batched encoder inputs: positive goldens + zero negative.
        var enc: [MLXArray] = []
        for i in 0..<4 {
            let pos = g["text_feat_\(i)"]!.asType(.float32)          // [1,35,2880]
            enc.append(concatenated([pos, MLXArray.zeros(like: pos)], axis: 0))
        }
        let posMask = g["text_mask"]!.asType(.int32)                 // [1,35]
        let mask = concatenated([posMask, MLXArray.zeros(like: posMask)], axis: 0)

        let model = try LensWeights.loadDiTFromPT(directory: Self.ptTransformer)

        let out = model(
            hiddenStates: hidden, encoderHiddenStates: enc,
            encoderHiddenStatesMask: mask, timestep: timestep, imgShape: (1, 32, 32))
        eval(out)

        let ref = g["dit_out_noise"]!.asType(.float32)
        let maxAbs = abs(out - ref).max().item(Float.self)
        let cosv = cosine(out, ref)
        print("[P2 DiT] max_abs=\(maxAbs) cosine=\(cosv)")
        XCTAssertLessThan(maxAbs, 1e-2, "DiT diverges")
        XCTAssertGreaterThan(cosv, 0.99999)
    }

    /// P3: final_latent → bn de-norm → unpatchify → VAE decode vs decoded_image.
    /// Gate: PSNR ≥ 55 dB (Python locked 57.65).
    func testVAEDecodeParity() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["LENS_PARITY"] == "1", "LENS_PARITY=1")
        Device.setDefault(device: Device.cpu)

        let g = try MLX.loadArrays(url: Self.goldens)
        let latents = g["final_latent"]!.asType(.float32)            // [1,1024,128]
        let ref = g["decoded_image"]!.asType(.float32)               // [1,3,512,512]

        let vae = try LensWeights.loadVAE(directory: Self.ptVAE)
        let packed = LensPipeline.packLatentsForDecode(latents, latentH: 32, latentW: 32)
        let img = vae.decodePackedLatents(packed)
        eval(img)

        let mse = (img - ref).square().mean().item(Float.self)
        let psnr = 10 * log10(4.0 / mse)  // range [-1,1] → peak-to-peak 2, peak² = 4
        print("[P3 VAE] mse=\(mse) psnr=\(psnr) dB")
        XCTAssertGreaterThan(psnr, 55.0, "VAE decode diverges")
    }
}

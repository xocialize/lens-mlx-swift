import XCTest
import MLX
@testable import Lens

/// P5: full-pipeline golden gate. Initial latents = golden dit_in_hidden[0:1] (the
/// captured seed-42 noise), golden encoder features, 4-step denoise + CFG 4.0, bn
/// de-norm + unpatchify + VAE decode — compared against the PT golden decoded_image.
/// (No RNG involved: every input is the captured tensor.)
final class EndToEndTests: XCTestCase {
    func testFullPipelineGolden() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["LENS_E2E"] == "1", "LENS_E2E=1")

        let g = try MLX.loadArrays(url: GoldenParityTests.goldens)
        let latents0 = g["dit_in_hidden"]!.asType(.float32)[0..<1]   // [1,1024,128]
        var enc: [MLXArray] = []
        for i in 0..<4 {
            let pos = g["text_feat_\(i)"]!.asType(.float32)
            enc.append(concatenated([pos, MLXArray.zeros(like: pos)], axis: 0))
        }
        let posMask = g["text_mask"]!.asType(.int32)
        let mask = concatenated([posMask, MLXArray.zeros(like: posMask)], axis: 0)

        let transformer = try LensWeights.loadDiTFromPT(
            directory: GoldenParityTests.ptTransformer)
        let vae = try LensWeights.loadVAE(directory: GoldenParityTests.ptVAE)

        let t0 = Date()
        let final = LensPipeline.denoise(
            transformer: transformer, latents: latents0, encoderFeatures: enc,
            encoderMask: mask, imgShape: (1, 32, 32), numInferenceSteps: 4,
            guidanceScale: 4.0)
        print("[P5] denoise: \(String(format: "%.1f", -t0.timeIntervalSinceNow))s")

        // Intermediate gate: final latent vs golden.
        let latRef = g["final_latent"]!.asType(.float32)
        let latMax = abs(final - latRef).max().item(Float.self)
        print("[P5] final_latent max_abs=\(latMax)")

        let packed = LensPipeline.packLatentsForDecode(final, latentH: 32, latentW: 32)
        let img = vae.decodePackedLatents(packed)
        eval(img)

        let ref = g["decoded_image"]!.asType(.float32)
        let mse = (img - ref).square().mean().item(Float.self)
        let psnr = 10 * log10(4.0 / mse)
        print("[P5] e2e image PSNR=\(psnr) dB (Python port locked 45.26 vs the same golden)")
        XCTAssertGreaterThan(psnr, 35.0, "e2e image diverges from the PT golden")
    }
}

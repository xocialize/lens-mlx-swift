import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import XCTest
import MLX
import Tokenizers
@testable import Lens

/// Production-scale demo render for the eye check (gated LENS_DEMO=1): bf16 DiT +
/// bf16 dense encoder + fp32 VAE, 1024×1024, 20 steps. Writes a PNG to ~/Desktop.
final class GenerateDemoTests: XCTestCase {
    func testGenerate1024() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["LENS_DEMO"] == "1", "LENS_DEMO=1")

        let t0 = Date()
        let transformer = try LensWeights.loadDiTFromPT(
            directory: GoldenParityTests.ptTransformer, dtype: .bfloat16)
        let vae = try LensWeights.loadVAE(directory: GoldenParityTests.ptVAE)
        let encoder = try LensGptOssEncoder.fromPretrained(
            directory: EncoderParityTests.bf16Encoder)
        let tokenizer = try await AutoTokenizer.from(
            modelFolder: EncoderParityTests.tokenizerDir)
        print("[DEMO] loaded in \(String(format: "%.1f", -t0.timeIntervalSinceNow))s")

        let gen = LensGenerator(
            transformer: transformer, vae: vae, encoder: encoder, tokenizer: tokenizer)
        let t1 = Date()
        let (pixels, h, w) = try gen.generate(
            prompt: "A serene lake below snow-capped mountains, golden hour.",
            height: 1024, width: 1024, numInferenceSteps: 20, seed: 42)
        print("[DEMO] generated in \(String(format: "%.1f", -t1.timeIntervalSinceNow))s "
            + "· peak GPU \(String(format: "%.1f", Double(GPU.peakMemory) / 1_073_741_824)) GB")

        // Sanity: not degenerate.
        let mean = pixels.reduce(0.0) { $0 + Double($1) } / Double(pixels.count)
        var variance = 0.0
        for p in pixels { variance += (Double(p) - mean) * (Double(p) - mean) }
        let std = (variance / Double(pixels.count)).squareRoot()
        print("[DEMO] pixel mean \(String(format: "%.1f", mean)) std \(String(format: "%.1f", std))")
        XCTAssertGreaterThan(std, 20, "image looks degenerate (flat)")

        // PNG to Desktop.
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        let ctx = CGContext(
            data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
            space: cs, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        let buf = ctx.data!.bindMemory(to: UInt8.self, capacity: w * h * 4)
        for i in 0..<(w * h) {
            buf[i * 4] = pixels[i * 3]
            buf[i * 4 + 1] = pixels[i * 3 + 1]
            buf[i * 4 + 2] = pixels[i * 3 + 2]
            buf[i * 4 + 3] = 255
        }
        let image = ctx.makeImage()!
        let url = URL(fileURLWithPath: NSHomeDirectory() + "/Desktop/lens-swift-demo.png")
        let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(dest, image, nil)
        CGImageDestinationFinalize(dest)
        print("[DEMO] saved → \(url.path)")
    }
}

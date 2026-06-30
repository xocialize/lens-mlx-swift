// Cross-model PE validation: a prompt enhanced by the ERNIE Prompt Enhancer
// (ernie-pe-swift, llm capability) rendered by LENS — proving enhancement is
// t2i-agnostic. The enhanced text below is the verbatim ernie-pe-3b output for
// {"prompt": "a lighthouse on a stormy coast", 1024x1024} (PE smoke, 2026-06-12).
//
// Run: LENS_PE_DEMO=1 swift test --filter PECrossDemoTests

import CoreGraphics
import Foundation
import ImageIO
import Tokenizers
import UniformTypeIdentifiers
import XCTest

@testable import Lens

final class PECrossDemoTests: XCTestCase {
    static let enhanced = """
        A dramatic coastal scene captured during a severe storm. The focal point is a \
        rugged, weathered lighthouse perched on a jagged, dark granite cliff that juts \
        out into the raging sea. The lighthouse itself is a tall, cylindrical stone \
        structure, its surface covered in thick, green seaweed and patches of white \
        barnacle growth, showing signs of long-term exposure to the elements. It has a \
        small, dark entrance at the base and a narrow, dark window near the top, all of \
        which are obscured by the heavy, swirling clouds. The lighthouse is positioned \
        on the left side of the frame, its silhouette stark against the darkening sky. \
        The sea is a chaotic, stormy expanse, with massive, white-capped waves crashing \
        against the cliff base, sending up thick, dramatic billows of water and spray. \
        The water is a deep, churning blue-gray, with white foam dominating the lower \
        part of the image. The sky is dominated by a massive, swirling storm cloud, \
        which forms a dramatic, dark arch over the lighthouse, casting a deep, ominous \
        shadow. The horizon line is low in the distance, blending into the storm. There \
        is a single, intense beam of light piercing through the storm cloud, just above \
        the lighthouse, casting a sharp, white light on the tower's upper windows and \
        the surrounding cliff. The overall atmosphere is one of desolation, power, and \
        a sense of being isolated in the face of nature's fury.
        """

    func testEnhancedPromptThroughLens() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["LENS_PE_DEMO"] == "1", "LENS_PE_DEMO=1")
        // Mirror GenerateDemoTests' loading; render at 1024², 20 steps.
        let snapshot = URL(fileURLWithPath: "/Volumes/DEV_ARCHIVE/lens-mlx/weights/Lens")
        let transformer = try LensWeights.loadDiTFromPT(
            directory: snapshot.appendingPathComponent("transformer"), dtype: .bfloat16)
        let vae = try LensWeights.loadVAE(directory: snapshot.appendingPathComponent("vae"))
        let encoder = try LensGptOssEncoder.fromPretrained(
            directory: URL(fileURLWithPath: "/Volumes/DEV_ARCHIVE/lens-mlx/weights/Lens-encoder-mlx-bf16"))
        let tokenizer = try await AutoTokenizer.from(
            modelFolder: snapshot.appendingPathComponent("tokenizer"))
        let generator = LensGenerator(
            transformer: transformer, vae: vae, encoder: encoder, tokenizer: tokenizer)

        let start = Date()
        let (pixels, h, w) = try await generator.generate(
            prompt: Self.enhanced, height: 1024, width: 1024,
            numInferenceSteps: 20, guidanceScale: 4.0, seed: 7)
        print("lens render \(w)x\(h) in \(Date().timeIntervalSince(start))s")
        let out = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Desktop/lens-pe-cross-demo.png")
        try PNGWriter.write(pixels: pixels, width: w, height: h, to: out)
        print("saved \(out.path)")
    }
}

enum PNGWriter {
    static func write(pixels: [UInt8], width: Int, height: Int, to url: URL) throws {
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        let ctx = CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8,
            bytesPerRow: width * 4, space: cs,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        let buf = ctx.data!.bindMemory(to: UInt8.self, capacity: width * height * 4)
        for i in 0..<(width * height) {
            buf[i * 4] = pixels[i * 3]
            buf[i * 4 + 1] = pixels[i * 3 + 1]
            buf[i * 4 + 2] = pixels[i * 3 + 2]
            buf[i * 4 + 3] = 255
        }
        let dest = CGImageDestinationCreateWithURL(
            url as CFURL, UTType.png.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(dest, ctx.makeImage()!, nil)
        guard CGImageDestinationFinalize(dest) else {
            fatalError("PNG write failed")
        }
    }
}

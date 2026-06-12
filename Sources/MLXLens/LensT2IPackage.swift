// MLXEngine `textToImage` package over the Lens core — the engine's first T2I surface.
//
// Lens (microsoft/Lens, MIT): GPT-OSS-20B multi-layer text features → 3.8B 48-layer
// double-stream flow-matching DiT → FLUX.2 VAE decode. The Swift core is parity-locked
// against the PT goldens (DiT cosine 1.0000 · VAE 120 dB · e2e 51.9 dB); this wrapper
// is a thin conformance layer: all model logic lives in the `Lens` target.

import CoreGraphics
import Foundation
import ImageIO
import Lens
import MLX
import MLXToolKit
import Tokenizers
import UniformTypeIdentifiers

/// Init-time configuration (C9): where the component weights live and the generation
/// defaults. Per-request prompt/size/steps/seed ride the canonical `T2IRequest`.
public struct LensConfiguration: PackageConfiguration, ModelStorable {
    /// Base Lens snapshot dir providing `transformer/`, `vae/`, `tokenizer/`
    /// (a local `microsoft/Lens` layout; HF auto-download is a tracked follow-up).
    public var lensSnapshotPath: String
    /// GPT-OSS encoder dir (dense bf16 or MXFP4 HF layout — sanitize handles both).
    public var encoderPath: String
    /// DiT precision when loading from the PT snapshot (`bfloat16` production default).
    public var ditDTypeBF16: Bool
    public var defaultSteps: Int
    public var defaultGuidanceScale: Float
    public var modelsRootDirectory: URL?

    public init(
        lensSnapshotPath: String = "/Volumes/DEV_ARCHIVE/lens-mlx/weights/Lens",
        encoderPath: String = "/Volumes/DEV_ARCHIVE/lens-mlx/weights/Lens-encoder-mlx-bf16",
        ditDTypeBF16: Bool = true,
        defaultSteps: Int = 20,
        defaultGuidanceScale: Float = 4.0,
        modelsRootDirectory: URL? = nil
    ) {
        self.lensSnapshotPath = lensSnapshotPath
        self.encoderPath = encoderPath
        self.ditDTypeBF16 = ditDTypeBF16
        self.defaultSteps = defaultSteps
        self.defaultGuidanceScale = defaultGuidanceScale
        self.modelsRootDirectory = modelsRootDirectory
    }

    private enum CodingKeys: String, CodingKey {
        case lensSnapshotPath, encoderPath, ditDTypeBF16, defaultSteps, defaultGuidanceScale
    }
}

public enum LensT2IError: Error, LocalizedError {
    case unreadableSnapshot(String)
    case pngEncode

    public var errorDescription: String? {
        switch self {
        case .unreadableSnapshot(let p): return "Lens snapshot not readable at \(p)."
        case .pngEncode: return "PNG encoding failed."
        }
    }
}

@InferenceActor
public final class LensT2IPackage: ModelPackage {
    public typealias Configuration = LensConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // Lens weights + code: MIT. GPT-OSS-20B encoder: Apache-2.0. FLUX.2 VAE
            // weights load from the user's snapshot (not redistributed — license note
            // in the core PORTING-SPEC).
            license: LicenseDeclaration(weightLicense: .mit, portCodeLicense: .mit),
            provenance: Provenance(
                sourceRepo: "mlx-community/Lens-3.8B-bf16", revision: "main", tier: 1),
            requirements: RequirementsManifest(
                // Measured (1024², 20 steps, bf16 DiT + dense bf16 encoder + fp32 VAE):
                // peak ~61 GB. The dense encoder dominates (~40 GB); the MXFP4 encoder
                // path reduces it materially and is the tracked follow-up.
                footprints: [QuantFootprint(quant: .bf16, residentBytes: 62_000_000_000)],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: .max
            ),
            specialties: [],
            surfaces: [
                T2IContract.descriptor(
                    name: "lens-t2i",
                    summary: "Lens 3.8B text-to-image (GPT-OSS-conditioned flow-matching "
                        + "DiT + FLUX.2 VAE): photoreal 1024–1440px generation, 20-step.",
                    modes: []
                )
            ]
        )
    }

    private let configuration: Configuration
    private var generator: LensGenerator?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    public func load() async throws {
        guard generator == nil else { return }
        let snapshot = URL(fileURLWithPath: configuration.lensSnapshotPath)
        guard FileManager.default.fileExists(
            atPath: snapshot.appendingPathComponent("transformer").path)
        else { throw LensT2IError.unreadableSnapshot(snapshot.path) }

        let transformer = try LensWeights.loadDiTFromPT(
            directory: snapshot.appendingPathComponent("transformer"),
            dtype: configuration.ditDTypeBF16 ? .bfloat16 : .float32)
        let vae = try LensWeights.loadVAE(directory: snapshot.appendingPathComponent("vae"))
        let encoder = try LensGptOssEncoder.fromPretrained(
            directory: URL(fileURLWithPath: configuration.encoderPath))
        let tokenizer = try await AutoTokenizer.from(
            modelFolder: snapshot.appendingPathComponent("tokenizer"))
        generator = LensGenerator(
            transformer: transformer, vae: vae, encoder: encoder, tokenizer: tokenizer)
    }

    public func unload() async {
        generator = nil
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        guard let generator else { throw PackageError.notLoaded }
        guard request.capability == .textToImage, let t2i = request as? T2IRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }
        try Task.checkCancellation()

        let (pixels, h, w) = try generator.generate(
            prompt: t2i.prompt,
            height: t2i.height ?? 1024,
            width: t2i.width ?? 1024,
            numInferenceSteps: t2i.steps ?? configuration.defaultSteps,
            guidanceScale: t2i.guidanceScale.map(Float.init)
                ?? configuration.defaultGuidanceScale,
            seed: t2i.seed ?? 0)

        try Task.checkCancellation()
        let png = try Self.encodePNG(pixels: pixels, width: w, height: h)
        return T2IResponse(image: Image(format: .png, data: png, width: w, height: h))
    }

    /// Interleaved RGB8 → PNG (canonical serialized artifact form, C3).
    nonisolated static func encodePNG(pixels: [UInt8], width: Int, height: Int) throws -> Data {
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        guard let ctx = CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8,
            bytesPerRow: width * 4, space: cs,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { throw LensT2IError.pngEncode }
        let buf = ctx.data!.bindMemory(to: UInt8.self, capacity: width * height * 4)
        for i in 0..<(width * height) {
            buf[i * 4] = pixels[i * 3]
            buf[i * 4 + 1] = pixels[i * 3 + 1]
            buf[i * 4 + 2] = pixels[i * 3 + 2]
            buf[i * 4 + 3] = 255
        }
        guard let image = ctx.makeImage() else { throw LensT2IError.pngEncode }
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(
            out, UTType.png.identifier as CFString, 1, nil)
        else { throw LensT2IError.pngEncode }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else { throw LensT2IError.pngEncode }
        return out as Data
    }
}

extension LensT2IPackage {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration {
        .of(LensT2IPackage.self)
    }
}

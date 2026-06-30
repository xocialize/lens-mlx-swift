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
public struct LensConfiguration: PackageConfiguration, ModelStorable, QuantConfigured {
    /// Base Lens snapshot dir providing `transformer/`, `vae/`, `tokenizer/`
    /// (a local `microsoft/Lens` layout; HF auto-download is a tracked follow-up).
    public var lensSnapshotPath: String
    /// GPT-OSS encoder dir (dense bf16 or MXFP4 HF layout — sanitize handles both).
    public var encoderPath: String
    /// Optional CONVERTED mlx DiT repo (e.g. a downloaded `mlx-community/Lens-3.8B-4bit`
    /// or `Lens-Turbo-3.8B-bf16`). When set, the DiT loads via `loadDiTRepo` (already
    /// sanitized + optionally quantized) instead of the PT `transformer/` snapshot, and
    /// `ditDTypeBF16` is ignored. tokenizer/VAE still come from `lensSnapshotPath`.
    public var ditRepoPath: String?
    /// DiT precision when loading from the PT snapshot (`bfloat16` production default).
    public var ditDTypeBF16: Bool
    public var defaultSteps: Int
    public var defaultGuidanceScale: Float
    public var modelsRootDirectory: URL?

    /// The declared footprint tier. Both wrappers vend a single bf16 `QuantFootprint`
    /// (the DiT-resident floor + the shared transient encoder/activation), so the governor
    /// charges that figure rather than a largest-that-fits guess. A 4-bit DiT via
    /// `ditRepoPath` only shrinks the DiT term within the same envelope — it is not a
    /// separate declared variant here.
    public var quant: Quant { .bf16 }

    public init(
        lensSnapshotPath: String = "/Volumes/DEV_ARCHIVE/lens-mlx/weights/Lens",
        encoderPath: String = "/Volumes/DEV_ARCHIVE/lens-mlx/weights/Lens-encoder-mlx-bf16",
        ditRepoPath: String? = nil,
        ditDTypeBF16: Bool = true,
        defaultSteps: Int = 20,
        defaultGuidanceScale: Float = 4.0,
        modelsRootDirectory: URL? = nil
    ) {
        self.lensSnapshotPath = lensSnapshotPath
        self.encoderPath = encoderPath
        self.ditRepoPath = ditRepoPath
        self.ditDTypeBF16 = ditDTypeBF16
        self.defaultSteps = defaultSteps
        self.defaultGuidanceScale = defaultGuidanceScale
        self.modelsRootDirectory = modelsRootDirectory
    }

    /// Lens-Turbo defaults: a converted Turbo DiT repo + 4-step / guidance-1.0 sampling.
    /// Reuses the base snapshot for tokenizer + VAE and the same encoder.
    public static func turbo(
        lensSnapshotPath: String = "/Volumes/DEV_ARCHIVE/lens-mlx/weights/Lens",
        encoderPath: String = "/Volumes/DEV_ARCHIVE/lens-mlx/weights/Lens-encoder-mlx-bf16",
        ditRepoPath: String
    ) -> LensConfiguration {
        LensConfiguration(
            lensSnapshotPath: lensSnapshotPath, encoderPath: encoderPath,
            ditRepoPath: ditRepoPath, defaultSteps: 4, defaultGuidanceScale: 1.0)
    }

    private enum CodingKeys: String, CodingKey {
        case lensSnapshotPath, encoderPath, ditRepoPath, ditDTypeBF16, defaultSteps, defaultGuidanceScale
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
                // Split footprint (efficiency contract 1.14.0). Per-stage eviction (P2): the
                // GPT-OSS-20B encoder (~39 GB on disk) loads per request and is evicted
                // (`nil` + Memory.clearCache()) before the denoise peak — a TRANSIENT, not a
                // resident. Only the 3.8B DiT + fp32 VAE stay resident.
                //   Resident floor = bf16 DiT ~7.6 GB (15 GB fp32 on disk → bf16) + fp32 VAE
                //     ~0.3 GB ≈ 8 GB. The old flat 62 GB folded the ~40 GB encoder into the
                //     resident floor.
                //   activation ≈ 54 GB = worst peak (~62 GB, during ENCODE while the ~39 GB
                //     encoder is loaded over the resident DiT + encode scratch) − 8 GB floor.
                //     The encoder-load transient dominates; after evict, the denoise peak is
                //     far lower (DiT activation only). The MXFP4 encoder path shrinks this
                //     transient materially and is the tracked follow-up.
                // [residentBytes = measured on-disk weight floor (solid). peakActivationBytes
                //  is a smoke/derived estimate (old flat peak − floor); the smoke MLX-peak
                //  under-reads process phys_footprint ~2.7× (BiRefNet lesson) — FLAGGED for a
                //  clean in-app phys re-baseline once Lens is registered in the MLXEngineImage
                //  app (IMAGE_AUTORUN).]
                footprints: [
                    QuantFootprint(
                        quant: .bf16, residentBytes: 8_000_000_000,
                        peakActivationBytes: 54_000_000_000)
                ],
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
            atPath: snapshot.appendingPathComponent("vae").path)
        else { throw LensT2IError.unreadableSnapshot(snapshot.path) }

        // DiT: a converted mlx repo (bf16/int4/int8, incl. Lens-Turbo) when configured,
        // else the PT transformer/ snapshot. Both paths materialize weights on load.
        let transformer: LensTransformer2DModel
        if let ditRepoPath = configuration.ditRepoPath {
            transformer = try LensWeights.loadDiTRepo(directory: URL(fileURLWithPath: ditRepoPath))
        } else {
            transformer = try LensWeights.loadDiTFromPT(
                directory: snapshot.appendingPathComponent("transformer"),
                dtype: configuration.ditDTypeBF16 ? .bfloat16 : .float32)
        }
        let vae = try LensWeights.loadVAE(directory: snapshot.appendingPathComponent("vae"))
        // Per-stage residency (efficiency contract 1.14.0): the DiT + VAE stay resident; the
        // GPT-OSS-20B encoder (~40 GB) is loaded per request and evicted before the denoise
        // peak (see LensGenerator). Captured as a closure so it is never co-resident with the
        // DiT denoise activation peak.
        let encoderPath = URL(fileURLWithPath: configuration.encoderPath)
        let encoderProvider: () async throws -> LensGptOssEncoder = {
            try LensGptOssEncoder.fromPretrained(directory: encoderPath)
        }
        let tokenizer = try await AutoTokenizer.from(
            modelFolder: snapshot.appendingPathComponent("tokenizer"))
        generator = LensGenerator(
            transformer: transformer, vae: vae,
            encoderProvider: encoderProvider, tokenizer: tokenizer)
    }

    public func unload() async {
        generator = nil
        MLX.Memory.clearCache()  // release the retained MLX pool so eviction frees RSS (not just drop refs)
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        guard let generator else { throw PackageError.notLoaded }
        guard request.capability == .textToImage, let t2i = request as? T2IRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }
        try Task.checkCancellation()

        let (pixels, h, w) = try await generator.generate(
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

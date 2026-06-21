// MLXEngine `textToImage` package for **Lens-Turbo** — the distilled 4-step sibling of Lens.
//
// Lens-Turbo is architecture-identical to base Lens (same 3.8B DiT, GPT-OSS encoder, FLUX.2
// VAE); only the DiT weights differ (distilled for 4-step / guidance-1.0 sampling). So this is
// a thin variant: a distinct PackageID + manifest (Turbo provenance, 4-step summary) that
// delegates all lifecycle/inference to an inner `LensT2IPackage`. The engine supplies a Turbo
// `LensConfiguration` at admission (use `LensConfiguration.turbo(ditRepoPath:)`).

import Foundation
import MLXToolKit

@InferenceActor
public final class LensTurboT2IPackage: ModelPackage {
    public typealias Configuration = LensConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // Same licensing as base Lens: DiT MIT; GPT-OSS encoder Apache-2.0; FLUX.2 VAE
            // loads from the user's snapshot (not redistributed).
            license: LicenseDeclaration(weightLicense: .mit, portCodeLicense: .mit),
            provenance: Provenance(
                sourceRepo: "mlx-community/Lens-Turbo-3.8B-bf16", revision: "main", tier: 1),
            requirements: RequirementsManifest(
                // Architecture-identical to base Lens → same measured footprint envelope.
                footprints: [QuantFootprint(quant: .bf16, residentBytes: 62_000_000_000)],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: .max
            ),
            specialties: [],
            surfaces: [
                T2IContract.descriptor(
                    name: "lens-turbo-t2i",
                    summary: "Lens-Turbo 3.8B text-to-image — distilled 4-step (guidance 1.0) "
                        + "GPT-OSS-conditioned flow-matching DiT + FLUX.2 VAE; fast 1024–1440px.",
                    modes: []
                )
            ]
        )
    }

    private let inner: LensT2IPackage

    public nonisolated init(configuration: Configuration) {
        self.inner = LensT2IPackage(configuration: configuration)
    }

    public func load() async throws { try await inner.load() }
    public func unload() async { await inner.unload() }
    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        try await inner.run(request)
    }
}

extension LensTurboT2IPackage {
    /// The author one-liner the engine registers (distinct PackageID from base Lens).
    public nonisolated static var registration: PackageRegistration {
        .of(LensTurboT2IPackage.self)
    }
}

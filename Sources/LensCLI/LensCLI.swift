// GPU validation CLI for the Lens / Lens-Turbo MLXEngine package.
//
//   swift run lens-cli [ditRepoPath] [steps] [size] [outPath]
//
// Drives the real `LensT2IPackage` surface (load → run via a canonical T2IRequest) so the
// validation exercises the engine contract, not just the core generator. Defaults to the
// converted Lens-Turbo bf16 repo at 4 steps / cfg 1.0 (Turbo's distilled sampling).

import Foundation
import MLXLens
import MLXToolKit

@main
struct LensCLI {
    static func main() async throws {
        let a = CommandLine.arguments

        // `swift run lens-cli manifest` — print both package manifests (no weights/GPU).
        if a.count > 1, a[1] == "manifest" {
            for (label, m) in [("Lens", LensT2IPackage.manifest),
                               ("Lens-Turbo", LensTurboT2IPackage.manifest)] {
                print("[\(label)] sourceRepo=\(m.provenance.sourceRepo) "
                    + "surfaces=\(m.surfaces.map(\.name)) "
                    + "license=\(m.license.weightLicense)/\(m.license.portCodeLicense)")
            }
            return
        }

        let ditRepo = a.count > 1 ? a[1] : "/Volumes/DEV_ARCHIVE/lens-mlx/build/Lens-Turbo-3.8B-bf16"
        let steps = a.count > 2 ? Int(a[2])! : 4
        let size = a.count > 3 ? Int(a[3])! : 512
        let outPath = a.count > 4 ? a[4] : "/Volumes/DEV_ARCHIVE/lens-mlx/outputs/gen_turbo_swift.png"
        let prompt = "A serene lake below snow-capped mountains, golden hour."

        let cfg = LensConfiguration.turbo(ditRepoPath: ditRepo)
        let pkg = LensT2IPackage(configuration: cfg)

        let t0 = Date()
        try await pkg.load()
        print("[lens-cli] loaded (\(String(format: "%.1f", -t0.timeIntervalSinceNow))s)")

        let req = T2IRequest(
            prompt: prompt, width: size, height: size, steps: steps, guidanceScale: 1.0, seed: 42)
        let t1 = Date()
        let resp = try await pkg.run(req)
        guard let t2i = resp as? T2IResponse else { fatalError("[lens-cli] not a T2IResponse") }
        print("[lens-cli] \(size)x\(size) \(steps)-step gen "
            + "(\(String(format: "%.1f", -t1.timeIntervalSinceNow))s), "
            + "\(t2i.image.data.count) bytes PNG")

        let out = URL(fileURLWithPath: outPath)
        try FileManager.default.createDirectory(
            at: out.deletingLastPathComponent(), withIntermediateDirectories: true)
        try t2i.image.data.write(to: out)
        print("[lens-cli] saved \(out.path)")
    }
}

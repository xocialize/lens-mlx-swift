import XCTest
import MLXToolKit
@testable import MLXLens

final class LensT2IPackageTests: XCTestCase {
    /// CPU-pure: manifest + descriptor shape (C11) sanity.
    func testManifest() {
        let m = LensT2IPackage.manifest
        XCTAssertEqual(m.surfaces.count, 1)
        XCTAssertEqual(m.surfaces[0].capability, .textToImage)
        XCTAssertEqual(m.surfaces[0].name, "lens-t2i")
        XCTAssertTrue(m.requirements.footprints.contains { $0.residentBytes > 50_000_000_000 })
    }

    func testConfigurationDefaults() {
        let c = LensConfiguration()
        XCTAssertEqual(c.defaultSteps, 20)
        XCTAssertEqual(c.defaultGuidanceScale, 4.0)
        XCTAssertTrue(c.ditDTypeBF16)
    }

    /// Gated full package run (LENS_PKG=1): load → T2IRequest → PNG response.
    func testPackageRun() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["LENS_PKG"] == "1", "LENS_PKG=1")

        let pkg = LensT2IPackage(configuration: LensConfiguration())
        let t0 = Date()
        try await pkg.load()
        print("[T2I-PKG] load: \(String(format: "%.1f", -t0.timeIntervalSinceNow))s")

        let t1 = Date()
        let resp = try await pkg.run(T2IRequest(
            prompt: "A red fox in tall summer grass at golden hour, photoreal.",
            width: 512, height: 512, steps: 8, seed: 7)) as! T2IResponse
        print("[T2I-PKG] run: \(String(format: "%.1f", -t1.timeIntervalSinceNow))s · "
            + "png \(resp.image.data.count / 1024) KB · \(resp.image.width ?? 0)×\(resp.image.height ?? 0)")

        XCTAssertEqual(resp.image.format, .png)
        XCTAssertEqual(resp.image.width, 512)
        XCTAssertGreaterThan(resp.image.data.count, 50_000, "implausibly small PNG")
        try resp.image.data.write(
            to: URL(fileURLWithPath: NSHomeDirectory() + "/Desktop/lens-t2i-package.png"))
        print("[T2I-PKG] saved → ~/Desktop/lens-t2i-package.png")

        await pkg.unload()
        try await pkg.load()  // reload sanity
        await pkg.unload()
    }
}

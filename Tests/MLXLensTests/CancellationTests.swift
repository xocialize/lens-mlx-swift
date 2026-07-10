// CancellationTests.swift — Lens (base + Turbo tiers) through the engine's CAN gate
// (offline, no MLX kernels, no weights). CAN-1/2 drive the real run() pre-cancelled: the
// entry checkpoint (`try Task.checkCancellation()` as the FIRST act of run(), before
// notLoaded validation) fires before weights are touched, so a stub configuration suffices.
// CAN-3 is the document of record for the checkpoint cadence:
//   - post-encode seam — `try Task.checkCancellation()` in LensGenerator.generate right
//     after the GPT-OSS encoder is evicted (Sources/Lens/Pipeline.swift), before denoise;
//   - denoise/step — `if Task.isCancelled { break }` at the top of the denoise loop in
//     LensPipeline.denoise (non-throwing static core — sanctioned break shape);
//   - pre-decode seam — `try Task.checkCancellation()` before the monolithic FLUX.2 VAE
//     decode (ONE MLX eval, no chunk loop, so no per-chunk decode cadence is claimed).
// The Turbo tier delegates run() to an inner LensT2IPackage, so one set of checkpoints
// covers both PackageIDs; each still passes the gate independently below. No catch blocks
// exist on the run() path — nothing to launder (CAN-2).

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest

@testable import MLXLens

final class CancellationTests: XCTestCase {

    // MARK: - CAN-1 / CAN-2 — pre-cancelled run() propagation + classification

    func testCANGatePreCancelledRunBase() async {
        // Stub config; construction is cheap (C13) and the entry checkpoint throws before
        // validation or weights are touched, so this is offline-safe.
        let package = LensT2IPackage(configuration: LensConfiguration())
        let report = await CancellationConformance.checkRun(
            package: package,
            request: T2IRequest(prompt: "probe"))
        XCTAssertTrue(report.passed, report.summary)
    }

    func testCANGatePreCancelledRunTurbo() async {
        // Stub paths — never touched: the pre-cancelled run() throws at the entry checkpoint.
        let package = LensTurboT2IPackage(
            configuration: LensConfiguration.turbo(ditRepoPath: "/nonexistent/turbo-dit"))
        let report = await CancellationConformance.checkRun(
            package: package,
            request: T2IRequest(prompt: "probe"))
        XCTAssertTrue(report.passed, report.summary)
    }

    // MARK: - CAN-3 — checkpoint-cadence declaration (the document of record)

    /// Both tiers share LensGenerator/LensPipeline: post-encode seam, per-denoise-step
    /// Task.isCancelled break (LensPipeline.denoise), pre-decode seam. Only the real
    /// per-step denoise cadence is declared; encode and decode are single forwards
    /// (seams, not recurring units).
    private var posture: CancellationConformance.CheckpointPosture {
        .cadence([
            .init(phase: .denoise, unit: .step)
        ])
    }

    func testCANCadenceDeclarationBase() {
        // 54 GB declared peak activation implies long runs — no sub-second exemption.
        XCTAssertTrue(CancellationConformance.longRunImplied(by: LensT2IPackage.manifest))
        let report = CancellationConformance.checkCadence(
            manifest: LensT2IPackage.manifest, posture: posture)
        XCTAssertTrue(report.passed, report.summary)
    }

    func testCANCadenceDeclarationTurbo() {
        XCTAssertTrue(CancellationConformance.longRunImplied(by: LensTurboT2IPackage.manifest))
        let report = CancellationConformance.checkCadence(
            manifest: LensTurboT2IPackage.manifest, posture: posture)
        XCTAssertTrue(report.passed, report.summary)
    }
}

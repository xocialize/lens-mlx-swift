// RunProgressTests.swift — Lens's end of the V2 run-phase progress plane (AB-A-0139): the core
// reports into its own engine-free `LensProgress` sink, and the wrapper forwards those events
// into the engine's `RunProgress`. `LensProgressBridge.forward` IS the wrapper's entire mapping;
// the loop test drives the real LensPipeline denoise loop through it with a tiny random-init DiT
// on the CPU (no weights), so the per-step reports are observed, not just declared.

import Foundation
import Lens
import MLX
import MLXToolKit
import XCTest

@testable import MLXLens

final class RunProgressTests: XCTestCase {

    private final class Box: @unchecked Sendable {
        private let lock = NSLock()
        private var storage: [RunPhaseReport] = []
        func append(_ r: RunPhaseReport) { lock.lock(); storage.append(r); lock.unlock() }
        var items: [RunPhaseReport] { lock.lock(); defer { lock.unlock() }; return storage }
    }

    /// Bind both task-locals the way `run()` does — the engine's sink outermost, the core's
    /// bridge inside it — and return what the engine plane observed.
    private func forwarded(_ body: () -> Void) -> [RunPhaseReport] {
        let box = Box()
        RunProgress.$sink.withValue({ box.append($0) }) {
            LensProgress.$sink.withValue(LensProgressBridge.forward, operation: body)
        }
        return box.items
    }

    // MARK: - forwarding

    func testCoreEventsReachTheEnginePlane() {
        let seen = forwarded {
            LensProgress.report(.encode)
            LensProgress.report(.denoise)
            LensProgress.report(.denoise, step: 1, totalSteps: 4)
            LensProgress.report(.denoise, step: 4, totalSteps: 4)
            LensProgress.report(.decode)
        }
        XCTAssertEqual(seen.map(\.phase), [.encode, .denoise, .denoise, .denoise, .decode])
        XCTAssertEqual(seen[2], RunPhaseReport(phase: .denoise, step: 1, totalSteps: 4))
        XCTAssertEqual(seen[3].step, 4)
        XCTAssertNil(seen[0].step)   // phases without steps carry no invented counts
    }

    /// The forward is a raw-value pass-through, which is only correct while the core's names
    /// ARE the canonical constants.
    func testCorePhaseNamesMatchCanonicalRunPhases() {
        XCTAssertEqual(RunPhase(rawValue: LensProgress.Phase.encode.rawValue), .encode)
        XCTAssertEqual(RunPhase(rawValue: LensProgress.Phase.denoise.rawValue), .denoise)
        XCTAssertEqual(RunPhase(rawValue: LensProgress.Phase.decode.rawValue), .decode)
    }

    func testUnboundCoreSinkIsANoOp() {
        // The CLI and the parity gates run with nothing bound — reporting must stay safe there.
        LensProgress.report(.denoise, step: 1, totalSteps: 4)
        // And a core event outside the wrapper's binding must not reach a bound engine plane.
        let box = Box()
        RunProgress.$sink.withValue({ box.append($0) }) {
            LensProgress.report(.decode)
        }
        XCTAssertTrue(box.items.isEmpty)
    }

    // MARK: - the real denoise loop

    /// LensPipeline.denoise on a one-layer, one-head DiT over a 2×2 patch grid, 3 steps: the
    /// engine plane sees loop entry, then step 1/3 … 3/3.
    func testDenoiseLoopReportsEveryStep() {
        _ = Device.defaultDevice()
        Device.setDefault(device: Device(.cpu))
        defer { Device.setDefault(device: Device(.gpu)) }
        let seen: [RunPhaseReport] = Device.withDefaultDevice(Device(.cpu)) {
            let dit = LensTransformer2DModel(numLayers: 1, attentionHeadDim: 64,
                                             numAttentionHeads: 1, encHiddenDim: 8)
            let features = (0..<4).map { _ in MLXRandom.normal([2, 3, 8]) }
            let mask = concatenated([MLXArray.ones([1, 3], dtype: .int32),
                                     MLXArray.zeros([1, 3], dtype: .int32)], axis: 0)
            return forwarded {
                let out = LensPipeline.denoise(
                    transformer: dit, latents: MLXRandom.normal([1, 4, 128]),
                    encoderFeatures: features, encoderMask: mask, imgShape: (1, 2, 2),
                    numInferenceSteps: 3)
                XCTAssertEqual(out.shape, [1, 4, 128])
            }
        }
        XCTAssertEqual(seen, [
            RunPhaseReport(phase: .denoise),
            RunPhaseReport(phase: .denoise, step: 1, totalSteps: 3),
            RunPhaseReport(phase: .denoise, step: 2, totalSteps: 3),
            RunPhaseReport(phase: .denoise, step: 3, totalSteps: 3),
        ])
    }
}

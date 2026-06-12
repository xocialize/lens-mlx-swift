import XCTest
@testable import Lens

/// P1 gates — pure math, CPU-only, no MLX arrays needed beyond trivial ones.
final class SchedulerTests: XCTestCase {
    /// Golden config: 512×512 → latent 32×32 → seq 1024, 4 steps.
    /// Python: compute_empirical_mu(1024, 4) — verified against the reference run.
    func testEmpiricalMuMatchesPython() {
        // Reference values computed from the verbatim Python formula.
        XCTAssertEqual(computeEmpiricalMu(imageSeqLen: 1024, numSteps: 4), 2.0306897, accuracy: 1e-4)
        XCTAssertEqual(computeEmpiricalMu(imageSeqLen: 4096, numSteps: 20), 2.1980221, accuracy: 1e-4)
        // > 4300 branch: a2*s + b2
        XCTAssertEqual(computeEmpiricalMu(imageSeqLen: 8100, numSteps: 20), 1.8277537, accuracy: 1e-4)
    }

    func testSchedulerSigmasAndTimesteps() {
        // sigmas[0]=1.0 -> shift = 1.0 -> timestep 1000 (the golden dit_in_timestep is t/1000 = 1.0)
        let s = FlowMatchEulerDiscreteScheduler()
        let N = 4
        let sigmas = (0..<N).map { 1.0 - Double($0) * (1.0 - 1.0 / Double(N)) / Double(N - 1) }
        s.setTimesteps(sigmas: sigmas, mu: Double(computeEmpiricalMu(imageSeqLen: 1024, numSteps: N)))
        XCTAssertEqual(s.timesteps.count, N)
        XCTAssertEqual(s.sigmas.count, N + 1)
        XCTAssertEqual(s.timesteps[0], 1000.0, accuracy: 1e-9)
        XCTAssertEqual(s.sigmas.last!, 0.0)
        // monotone decreasing
        for i in 1..<s.timesteps.count {
            XCTAssertLessThan(s.timesteps[i], s.timesteps[i - 1])
        }
    }

    func testResolutionBuckets() throws {
        let hw = try LensResolution.resolve(baseResolution: 1024, aspectRatio: "16:9")
        XCTAssertEqual(hw.height, 768)
        XCTAssertEqual(hw.width, 1376)
        XCTAssertThrowsError(try LensResolution.resolve(baseResolution: 512, aspectRatio: "1:1"))
    }
}

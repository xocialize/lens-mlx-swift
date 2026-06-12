// FlowMatchEulerDiscreteScheduler + the bespoke empirical mu (T6) — Swift port.
//
// Isomorphic to lens_mlx/scheduler.py + compute_empirical_mu in pipeline_mlx.py.
// Sanity vs golden: sigmas[0]=1.0 -> shift(mu,1.0)=exp(mu)/exp(mu)=1.0 -> timestep
// 1000 -> the DiT sees timestep/1000 = 1.0 (== golden dit_in_timestep).

import Foundation
import MLX

/// Empirical `mu` for the dynamic exponential shift (T6 — NOT the stock FLUX mu).
/// Ported verbatim from upstream pipeline.py; constants calibrated for Lens.
public func computeEmpiricalMu(imageSeqLen: Int, numSteps: Int) -> Float {
    let a1: Double = 8.73809524e-05, b1: Double = 1.89833333
    let a2: Double = 0.00016927, b2: Double = 0.45666666
    let s = Double(imageSeqLen)
    if imageSeqLen > 4300 {
        return Float(a2 * s + b2)
    }
    let m200 = a2 * s + b2
    let m10 = a1 * s + b1
    let a = (m200 - m10) / 190.0
    let b = m200 - 200.0 * a
    return Float(a * Double(numSteps) + b)
}

public final class FlowMatchEulerDiscreteScheduler {
    public let numTrainTimesteps: Int
    public private(set) var sigmas: [Double] = []
    public private(set) var timesteps: [Double] = []
    private var stepIndex = 0

    public init(numTrainTimesteps: Int = 1000) {
        self.numTrainTimesteps = numTrainTimesteps
    }

    /// diffusers exponential time shift: exp(mu) / (exp(mu) + (1/t - 1)**sigma)
    static func timeShiftExponential(mu: Double, sigma: Double, t: Double) -> Double {
        exp(mu) / (exp(mu) + pow(1.0 / t - 1.0, sigma))
    }

    public func setTimesteps(sigmas: [Double], mu: Double) {
        let shifted = sigmas.map { Self.timeShiftExponential(mu: mu, sigma: 1.0, t: $0) }
        self.timesteps = shifted.map { $0 * Double(numTrainTimesteps) }
        self.sigmas = shifted + [0.0]  // terminal 0
        self.stepIndex = 0
    }

    public func step(modelOutput: MLXArray, sample: MLXArray) -> MLXArray {
        let sigma = sigmas[stepIndex]
        let sigmaNext = sigmas[stepIndex + 1]
        let prev = sample + Float(sigmaNext - sigma) * modelOutput
        stepIndex += 1
        return prev
    }
}

// MARK: - Resolution buckets (isomorphic to lens_mlx/resolution.py)

/// Bucket maps: aspect ratio "W:H" -> (height, width); all divisible by 16.
public enum LensResolution {
    public static let buckets: [Int: [String: (height: Int, width: Int)]] = [
        1024: [
            "1:2": (1472, 736), "9:16": (1376, 768), "2:3": (1248, 832),
            "3:4": (1152, 864), "1:1": (1024, 1024), "4:3": (864, 1152),
            "3:2": (832, 1248), "16:9": (768, 1376), "2:1": (736, 1472),
        ],
        1440: [
            "1:2": (2080, 1040), "9:16": (1936, 1088), "2:3": (1760, 1168),
            "3:4": (1616, 1216), "1:1": (1440, 1440), "4:3": (1216, 1616),
            "3:2": (1168, 1760), "16:9": (1088, 1936), "2:1": (1040, 2080),
        ],
    ]

    public static func resolve(baseResolution: Int, aspectRatio: String) throws
        -> (height: Int, width: Int)
    {
        guard let table = buckets[baseResolution] else {
            throw LensError.config(
                "Unsupported base_resolution=\(baseResolution). Supported: \(buckets.keys.sorted())")
        }
        guard let hw = table[aspectRatio] else {
            throw LensError.config(
                "Unsupported aspect_ratio=\(aspectRatio). Supported: \(table.keys.sorted())")
        }
        return hw
    }
}

public enum LensError: Error {
    case config(String)
    case loading(String)
    case generation(String)
}

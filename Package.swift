// swift-tools-version: 6.2
// lens-mlx-swift — Swift/MLX mirror of the completed Python lens-mlx port
// (microsoft/Lens, MIT: GPT-OSS-conditioned 3.8B t2i DiT + FLUX.2 VAE decode).
// Reference = the Python MLX port at /Volumes/DEV_ARCHIVE/lens-mlx (DiT cosine
// 0.999999 vs PT, e2e 45 dB). See PORTING-SPEC.md — phases gate on the shared
// fp32/CPU goldens; never advance on a red gate.

import PackageDescription

let package = Package(
    name: "Lens",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(name: "Lens", targets: ["Lens"]),
        // MLXEngine wrapper: the conformant `textToImage` ModelPackage over LensGenerator.
        .library(name: "MLXLens", targets: ["MLXLens"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.30.0"),
        // GPTOSS (text encoder backbone) + tokenizer plumbing.
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", from: "3.31.3"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
        // FLUX.2 VAE — neutral package shared with other t2i backers (ERNIE-Image); net dep.
        .package(url: "https://github.com/xocialize/flux2-vae-mlx-swift", from: "0.1.0"),
        // MLXEngine contract (MLXToolKit) for the wrapper target only; the core `Lens`
        // target stays engine-agnostic.
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.3.0"),
    ],
    targets: [
        .target(
            name: "Lens",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "Flux2VAE", package: "flux2-vae-mlx-swift"),
            ],
            path: "Sources/Lens"
        ),
        .target(
            name: "MLXLens",
            dependencies: [
                "Lens",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
            ],
            path: "Sources/MLXLens"
        ),
        // GPU validation CLI (`swift run lens-cli`) — the reliable GPU-gate path; drives the
        // real ModelPackage surface (register→load→run) for base or Turbo via ditRepoPath.
        .executableTarget(
            name: "lens-cli",
            dependencies: [
                "MLXLens",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
            ],
            path: "Sources/LensCLI"
        ),
        .testTarget(
            name: "LensTests",
            dependencies: ["Lens"],
            path: "Tests/LensTests"
        ),
        .testTarget(
            name: "MLXLensTests",
            dependencies: ["MLXLens"],
            path: "Tests/MLXLensTests"
        ),
    ]
)

// swift-tools-version: 6.0
// lens-mlx-swift — Swift/MLX mirror of the completed Python lens-mlx port
// (microsoft/Lens, MIT: GPT-OSS-conditioned 3.8B t2i DiT + FLUX.2 VAE decode).
// Reference = the Python MLX port at /Volumes/DEV_ARCHIVE/lens-mlx (DiT cosine
// 0.999999 vs PT, e2e 45 dB). See PORTING-SPEC.md — phases gate on the shared
// fp32/CPU goldens; never advance on a red gate.

import PackageDescription

let package = Package(
    name: "Lens",
    platforms: [
        .macOS(.v15)
    ],
    products: [
        .library(name: "Lens", targets: ["Lens"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.30.0"),
        // GPTOSS (text encoder backbone) + tokenizer plumbing.
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", from: "3.31.3"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
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
            ],
            path: "Sources/Lens"
        ),
        .testTarget(
            name: "LensTests",
            dependencies: ["Lens"],
            path: "Tests/LensTests"
        ),
    ]
)

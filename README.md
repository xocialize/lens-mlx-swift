# lens-mlx-swift

Swift/MLX mirror of the completed Python [`lens-mlx`](https://github.com/xocialize/lens-mlx)
port — `microsoft/Lens` (MIT): GPT-OSS-conditioned 3.8B text-to-image DiT + FLUX.2 VAE
decode — targeting MLXEngine's image-generation capability.

> **Status:** P1–P3 ✅ (scheduler/mu/resolution exact vs Python · **DiT max_abs 6.9e-06,
> cosine 1.0000 vs the PT golden** · **VAE decode 120 dB PSNR**). Next: P4 GPT-OSS
> encoder wrapper (MLXLLM + YaRN injection + MXFP4), P5 e2e generate(), P6 engine wrap.
> See `PORTING-SPEC.md`. Parity gates: `LENS_PARITY=1 swift test` (XCTest; Cmlx metallib
> bundle in `.build/debug/`; goldens at `VideoResearch/lens-mlx-models/goldens/`).

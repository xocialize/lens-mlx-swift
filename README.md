# lens-mlx-swift

Swift/MLX mirror of the completed Python [`lens-mlx`](https://github.com/xocialize/lens-mlx)
port — `microsoft/Lens` (MIT): GPT-OSS-conditioned 3.8B text-to-image DiT + FLUX.2 VAE
decode — targeting MLXEngine's image-generation capability.

> **Status:** P1–P5 ✅ — **the port generates images.** Parity vs the PT goldens:
> DiT max_abs 6.9e-06 (cosine 1.0000) · VAE decode 120 dB · encoder per-layer ≥0.997
> (mirror vs the Python MLX port ≥0.9974; structure sentinel L5 0.9998; tokens/template
> EXACT) · **full-pipeline e2e 51.9 dB** (the Python port's own number: 45.26). Demo:
> 1024×1024/20-step bf16 renders a sharp, prompt-faithful image in 73 s
> (`LENS_DEMO=1`). Next: P6 MLXEngine wrap + MXFP4 encoder for production memory.
> Gates: `LENS_PARITY=1 / LENS_E2E=1 swift test` (XCTest; Cmlx metallib bundle in
> `.build/debug/`; goldens at `VideoResearch/lens-mlx-models/goldens/`).

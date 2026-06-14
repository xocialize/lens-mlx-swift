# lens-mlx-swift

Swift/MLX mirror of the completed Python [`lens-mlx`](https://github.com/xocialize/lens-mlx)
port — `microsoft/Lens` (MIT): GPT-OSS-conditioned 3.8B text-to-image DiT + FLUX.2 VAE
decode. Ships MLXEngine's **`textToImage`** capability.

Two products:
- **`Lens`** — the engine-agnostic generator core (DiT + FLUX.2 VAE decode via the neutral
  `flux2-vae-mlx-swift` package).
- **`MLXLens`** — the conformant MLXEngine wrapper (`LensT2IPackage`, `textToImage`).

> **Status: complete · wrapped · in-app validated.** Parity vs the PT goldens: DiT max_abs
> 6.9e-06 (cosine 1.0000) · VAE decode 120 dB · encoder per-layer ≥0.997 (structure sentinel
> L5 0.9998; tokens/template EXACT) · **full-pipeline e2e 51.9 dB** (the Python port's own number:
> 45.26). 1024×1024/20-step bf16 renders a sharp, prompt-faithful image in ~73 s. Wrapped as the
> `textToImage` ModelPackage and validated live in the MLXEngine proving-ground app — the second
> public T2I backer alongside ERNIE-Image-Turbo, co-resident and selectable by PackageID.

## Consuming it
Public + version-tagged on github.com/xocialize — add by tagged URL:
`.package(url: "https://github.com/xocialize/lens-mlx-swift", from: "0.1.0")`, then import
`MLXLens` (the conformant package) or `Lens` (the bare generator). The FLUX.2 VAE is the neutral
`flux2-vae-mlx-swift` net dependency (shared with ERNIE-Image, not vendored here), and the wrapper
takes the engine contract (`MLXToolKit`) by tagged URL — so it builds standalone, no local
checkouts.

Gates: `LENS_PARITY=1 / LENS_E2E=1 swift test` (XCTest; goldens at
`VideoResearch/lens-mlx-models/goldens/`).

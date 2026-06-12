# lens-mlx-swift — Porting Spec

**Goal:** Swift/MLX mirror of the COMPLETED Python port (`/Volumes/DEV_ARCHIVE/lens-mlx`
— DiT cosine 0.999999 vs PT, e2e 45.26 dB, published `mlx-community/Lens-3.8B-{bf16,4bit,8bit}`),
serving text-to-image for MLXEngine. Lens = microsoft/Lens (MIT): GPT-OSS-20B multi-layer
text features → 3.8B 48-layer double-stream flow-matching DiT → FLUX.2 VAE decode.

**Reference = the Python MLX port, run live** (parity doctrine: gate against the reference
you ported, never its grandparent). PT goldens captured fp32/CPU ride along:
`/Volumes/DEV_VOL1/VideoResearch/lens-mlx-models/goldens/lens_goldens.safetensors`
(512×512, 4 steps, cfg 4.0, seed 42 — manifest copied to this repo). All traps T0–T8 in
`/Volumes/DEV_ARCHIVE/lens-mlx/Lens-MLX-Port-Handoff.md`; the Python sources already
encode their resolutions — port them verbatim.

## Component map (Swift)

| Component | Source (Python, LOC) | Swift action |
|---|---|---|
| `LensTransformer2DModel` (DiT) | `lens_mlx/model/transformer.py` (453) | **Port isomorphic** — all ops exist in mlx-swift (`MLXFast.scaledDotProductAttention`, RMSNorm, `LayerNorm(affine:false)`). Keep upstream names. |
| Axial RoPE (T2) | `LensEmbedRope` + `apply_rotary_emb_lens` | Port exactly: real interleaved rotation, (cos,sin) tables, `scale_rope=true` neg/pos split. Plain class, NOT a Module (tables must not become parameters). |
| Scheduler + empirical mu (T6) | `scheduler.py` (41) + `compute_empirical_mu` | Port verbatim (pure math; exponential time shift; sigmas linspace(1, 1/N, N) + terminal 0). |
| Resolution buckets | `resolution.py` (62) | Port verbatim. |
| Text encoder | `model/text_encoder.py` (115) wrapping mlx-lm `gpt_oss` | **Wrap MLXLLM's `GPTOSS`** (mlx-swift-lm) with: capture at `[5,11,17,23]` (T0, 0-indexed), early-exit after layer 23 (skip final norm + head), alternating sliding(128)/full causal masks (T8). If internals are fileprivate → copy-adapt into `Adapted/` (lance ViT pattern, MIT attribution). |
| **YaRN rope injection (F9)** | `GPT_OSS_YARN_ROPE` dict in text_encoder.py | The checkpoint config does NOT serialize rope_scaling (HF class default). Verify whether MLXLLM's GPTOSS hardcodes YaRN; if it reads config like mlx-lm, inject: yarn, factor 32, beta_fast 32, beta_slow 1, orig_max_pos 4096, rope_theta 150000, truncate false. **Silent plain-rope fallback = uniform divergence at every position (cosine ~0.94).** |
| FLUX.2 VAE **decoder** | mflux `flux2_vae/` (~480 LOC total, tiny blocks) | **Port decoder path only** (t2i never encodes): resnet_block_2d (33), attention (36), upsample_2d (16), unet_mid_block (22), up_decoder_block (35), batch_norm_stats (11), vae.decode (58). NHWC conv layout; weights transpose PT (O,I,kH,kW)→(O,kH,kW,I) at load. |
| Pipeline | `pipeline_mlx.py` (247) | Port: chat template (T5 — `_CHAT_SYSTEM` + canned assistant thinking turn, `DEFAULT_TXT_OFFSET=97`), CFG (check the norm-rescale `‖cond‖/‖comb‖` variant — pitfall #11), bn de-norm in patchified space (T1: `x = x/scale - shift` with shift=-running_mean, scale=1/sqrt(var+eps)), unpatchify (T4 axis order), decode. |
| Weights loader | `utils/weights.py` (121) | Sanitize `img_mod.1.→img_mod.`/`txt_mod.1.→txt_mod.`; strict load; quantized-repo rebuild (group 64, bits per config, `keep_hi_precision` substrings); VAE key map (`to_out.0.→to_out.`, drop `num_batches_tracked`, conv transpose). |
| Tokenizer | GPT-OSS harmony via tokenizer.json | swift-transformers `AutoTokenizer` from the encoder snapshot. |

## Weights

- DiT: `mlx-community/Lens-3.8B-bf16` (+ `-4bit`/`-8bit`) — published by the Python phase 5.
- Encoder: gpt-oss-20b MXFP4 (mlx-community) — **T7: known `quantization.mode` config parse
  mismatch in mlx-swift-examples #386; mlx-swift-lm 3.31.3 may carry the fix — verify, else shim.**
- VAE: from the local `weights/Lens/vae` snapshot (license question for redistribution — load
  from source, don't republish).
- Local dev mirror: `/Volumes/DEV_VOL1/VideoResearch/lens-mlx-models/` (goldens there now).

## Phases & gates (CPU stream for all parity; never advance on red)

1. **P1 — Pure math**: scheduler (+mu), resolution, RoPE tables + apply. Gate: micro-parity vs
   the Python functions on injected inputs, ≤1e-6 fp32.
2. **P2 — DiT**: port + strict-load bf16. Gate: `dit_in_hidden`+`dit_in_timestep`+text feats →
   output vs `dit_out_noise`, fp32 CPU, cosine ≥0.99999 / max_abs <1e-2 (golden is fp32-CPU PT;
   Python-MLX matched at 0.999999).
3. **P3 — VAE decoder**: port + load. Gate: `final_latent` → bn-denorm → unpatchify → decode vs
   `decoded_image` (PSNR ≥ 55 dB, matching Python's 57.65). Plus the noise-path smoke (pitfall #7).
4. **P4 — Encoder**: GPTOSS wrapper + YaRN + chat template + tokenizer. Gate: tokenized golden
   prompt → features vs `text_feat_{0..3}` (bf16 encoder golden: max_abs <1e-2 / cosine ≥0.999).
5. **P5 — E2E**: full `generate()` 512×512/4-step/seed-42 — semantic gate (image valid, sharp,
   subject correct) + cross-check vs Python e2e on the SAME seed noise (inject noise from a file;
   MLX Swift/Python RNG are not seed-compatible). Then 1024×1024/20-step listen-equivalent: the
   eye check.
6. **P6 — MLXEngine wrap**: `imageGenerate` capability package (check the contract's canonical
   name + schema), Testing-app window, APP-VALIDATION entry. Quant tier: DiT 4bit gate per the
   doctrine (per-pass cosine ≥0.99, NOT e2e PSNR — Lens int4 e2e PSNR is ~15 dB by design).

## Known Swift-side rakes (from this workspace's history)

- GPU fp32 matmul ≈8e-4/op accumulation — parity ONLY on the CPU stream (`Device.cpu`).
- MLXArray subscript-setters are silent no-ops for scatter patterns — slice+concat only.
- SPM metallib: GPU tests = XCTest only (swift-testing helper process breaks the colocated
  bundle lookup); copy `mlx-swift_Cmlx.bundle` into `.build/debug/`.
- BSD sed has no `\b`.
- `Linear(bias: false)` defaults differ from Python — set explicitly everywhere.

# Efficiency Adoption Brief — `lens-mlx-swift` (Lens, `textToImage`)

> **For a session-specific agent.** Adopt engine 1.14 efficiency (engine 0.15.0+). Load the
> `mlx-swift-integration` skill; read references/package-efficiency.md (four levers + **"Measurement
> findings"**, esp. *in-app phys vs smoke MLX-peak*) + references/memory-harness.md. Template: the
> **Qwen-Image-Edit** brief (multi-wrapper, multi-component text-to-image, encoder-evict headline).
> Audited 2026-06-30.

## Package at a glance
- **Two wrappers, shared core:** `MLXLens` has `LensT2IPackage` + `LensTurboT2IPackage`, both over core
  `Lens` (`Pipeline.swift`). Capability `textToImage`. Multi-component: **text encoder + DiT + VAE** (the
  62 GB flat footprint implies a large text encoder — confirm which component dominates).
- **Footprints today (flat):** both wrappers `QuantFootprint(.bf16, 62 GB)`. No split, no `QuantConfigured`.
- Config `LensConfiguration: PackageConfiguration, ModelStorable`. Engine pinned `from: "0.3.0"`.

## Audit vs. the four levers
| Lever | State | Finding | Priority |
|---|---|---|---|
| Engine dep | 🟡 | from 0.3.0 → 0.15.0 | **P0** |
| 1. Split footprint | ❌ | flat 62 GB (both wrappers) | **P1** |
| 2. Per-stage evict | ❌ (likely) | encoder used once then idle through denoise (verify in `Pipeline.swift`) — the 62 GB suggests a big evictable encoder | **P2 (headline, high upside)** |
| 3. mmap/lazy | 🟡 verify | confirm lazy/mmap (floor ≈ on-disk) | note |
| 4. BudgetAware | ➖ | single bf16 (Turbo is a separate wrapper, not a quant lever) | defer |

## Plan (mirror Qwen-Image-Edit; both wrappers share the core → one refactor benefits both)
- **P0:** `swift package update` → 0.15.0; build + fix any drift.
- **P2 (headline):** the 62 GB is the biggest of the image-gen set — if a large text encoder is held
  resident and idle through the denoise (Qwen-Image-Edit / LTX-Gemma pattern), staging it (load→encode→
  `nil`+`Memory.clearCache()` before denoise) is a big win. Refactor the shared core; both wrappers
  inherit. Swift 6 `#isolation` gotcha if the staged path goes async (use `isolated (any Actor)? =
  #isolation`). If components interleave, P2 is N/A — note the reason.
- **P1:** `QuantConfigured`. residentBytes = weights floor; peakActivationBytes = the **measured** transient.
- **P3** mmap (note). **P4** defer.
- **`unload()` must `MLX.Memory.clearCache()`** (eviction-frees-RSS rule; both wrappers' unloads).

## Measurement — IMPORTANT (in-app-phys lesson)
Same as ERNIE: the smoke measures MLX working-set peak, which UNDER-reads process `phys_footprint`
(R-MEM-1/admission basis) by ~2.7×. Declare `residentBytes` from the measured weight floor (solid) +
a **flagged** best-effort `peakActivationBytes` from the smoke, pending an in-app phys re-baseline once
Lens is registered in MLXEngineImage. Land P2 + the split; flag the activation.

## Definition of done
- [ ] engine 0.15.0; `QuantConfigured`; P2 (encoder-evict or N/A-with-reason); both wrappers' `unload()` clearCache.
- [ ] Split declared (`residentBytes` weights + `peakActivationBytes` smoke estimate, flagged) on both wrappers.
- [ ] Smoke green (valid coherent image); split recorded; activation flagged for in-app re-baseline.
- [ ] Registry: lens row Eff ⬜→✅ (note "activation = smoke est, phys re-baseline pending"), Eng→0.15.0.

## Report back
flat→split (both wrappers), the encoder-evict effect (which component the 62 GB is), the smoke transient
(flagged for phys), drift, effort, SHAs. STAY IN SCOPE — four-lever adoption + brief + registry row only;
no testing-app/shell changes; stop-and-report if bigger.

---

## Adoption outcome (executed 2026-06-30, engine 0.16.0)

**P0 — engine 0.16.0.** `swift package update mlx-engine-swift` moved the resolved engine to **0.16.0** (the
latest published; the `from: "0.3.0"` floor admits it, no manifest edit). **Zero API drift** — `MLXLens`
(both wrappers) + `lens-cli` built green against 0.16.0 unchanged. The `textToImage` / `T2IRequest` /
`PackageManifest` / `QuantFootprint` surface is stable from 0.3.0 → 0.16.0.

**P2 — per-stage encoder eviction (the headline, biggest win of the image-gen set).** Refactored the shared
core `LensGenerator` (`Sources/Lens/Pipeline.swift`): it no longer holds the GPT-OSS-20B encoder resident —
it owns an async `encoderProvider` closure (the wrapper's loader). `generate(...)` is now `async`: load
encoder → `encode(prompt:encoder:)` (the offset-sliced, CFG-batched multi-layer features) → `eval` the
features+mask → drop the encoder ref + `Memory.clearCache()` → then the 3.8B DiT denoise loop + FLUX.2 VAE
decode. Only the DiT + fp32 VAE stay resident. **Swift 6 isolation:** `generate` / `loadEncoder` take
`isolation: isolated (any Actor)? = #isolation`, inheriting the wrapper's `@InferenceActor` (the canonical
fix). A back-compat `init(transformer:vae:encoder:tokenizer:)` keeps the encoder resident
(`keepEncoderResident = true`) for the parity/demo tests. **Both wrappers share the core**, so
`LensTurboT2IPackage` (delegates to the inner `LensT2IPackage`) inherits the eviction from one refactor.
**Parity preserved:** encode/denoise/decode math byte-identical; only `eval()` + eviction added. The
`run()` + test `generate` call sites updated to `try await`.

**P1 — split footprint (residentBytes MEASURED from on-disk weight floor; activation FLAGGED), both wrappers:**

| Wrapper | OLD flat resident | Resident floor (declared) | Activation (declared) | Encoder now |
|---|---|---|---|---|
| LensT2IPackage (bf16) | 62 GB | **8 GB** (bf16 DiT ~7.6 + fp32 VAE ~0.3) | **54 GB** (worst peak − floor) | transient |
| LensTurboT2IPackage (bf16) | 62 GB | **8 GB** (= base; shared core) | **54 GB** | transient |

On-disk: encoder (GPT-OSS-20B dense bf16) **39 GB**, transformer 15 GB fp32 → bf16 ~7.6 GB, VAE 321 MB fp32.
**The 62 GB is dominated by the encoder (~39 GB).** The headline: the **~39 GB encoder moved from resident
into the transient bucket** — the persistent floor drops from the flat 62 GB to **8 GB** (the bf16 DiT +
fp32 VAE). The activation is large (~54 GB) because the encoder-load transient (during encode, co-resident
with the DiT) is itself the worst peak; after evict the denoise peak is far lower. The engine now reserves
ONE ~54 GB transient across co-residents instead of baking the full 62 GB peak into each resident.
`LensConfiguration` conforms to `QuantConfigured` (`quant` = bf16, the single declared variant per wrapper).

**Measurement path:** the `lens-cli` / demo smoke could **not** be driven to a real render here —
`xcodebuild test` does not propagate the `LENS_DEMO` gate env into the xctest runner (it skips, even passed
as an xcodebuild arg), and the MLXEngineImage category app is not stood up for Lens (no app-autorun path).
So per the brief's fallback: `residentBytes` is the **measured on-disk weight floor** (solid) and
`peakActivationBytes` is a **smoke/derived estimate** (old flat peak − floor), FLAGGED in both manifests +
the registry for a clean in-app phys re-baseline once Lens is registered in MLXEngineImage (the smoke
MLX-peak under-reads `phys_footprint` ~2.7×, the BiRefNet lesson). The async refactor's correctness rests on
full clean compilation of all targets (wrapper + CLI + both test targets) + byte-identical math.

**P3 — mmap/lazy: verified, no change.** `loadDiTFromPT` / `loadDiTRepo` rebuild a per-key dict with lazy
`asType` / packed-quant tensors and `eval(model)` once — no full eager copy; the resident floor tracks the
on-disk DiT bytes. The MXFP4 encoder path remains the tracked follow-up to shrink the ~39 GB transient.

**P4 — BudgetAware: deferred.** No load-time adaptive dtype lever (Turbo is a separate wrapper, not a quant
knob; the `ditRepoPath` 4-bit DiT is config-chosen).

**`unload()`** (base wrapper) now calls `MLX.Memory.clearCache()` (eviction-frees-RSS rule; the Turbo wrapper
delegates to it); the `MLXLens` wrapper target gained an explicit `.product(name: "MLX", package: "mlx-swift")`
link so the call doesn't rely on a transitive import via the `Lens` target.

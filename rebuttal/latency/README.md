# Latency Benchmark: SCDD vs GIDD+

This folder contains the code and instructions to reproduce the latency benchmark comparing SCDD and GIDD+ generation speed.

## Files

| File | Description |
|------|-------------|
| `diffusion.py` | Modified SCDD diffusion module (replaces `scdd/diffusion.py`) |
| `dit.py` | Modified DiT backbone (replaces `scdd/models/dit.py`) |
| `bench_latency_scdd.py` | SCDD latency benchmark script |
| `bench_latency_gidd.py` | GIDD+ latency benchmark script |
| `generate_scdd.py` | SCDD sample generation with compiled sampler |
| `pbs_bench_latency.sh` | Shell script to run both benchmarks (template) |

## Changes to the Codebase

To enable `torch.compile` for SCDD sampling, two source files need to be updated. The changes are minimal and do not affect training or model outputs.

### 1. Replace `scdd/models/dit.py`

Copy `dit.py` from this folder to `scdd/models/dit.py`. Changes made:

- **Removed `@torch.jit.script` decorators** from `bias_dropout_add_scale_fused_train`, `bias_dropout_add_scale_fused_inference`, and `modulate_fused`. These legacy JIT decorators conflict with `torch.compile` (TorchDynamo cannot trace through JIT-scripted functions).
- **Replaced `einops.rearrange` with native `view`/`reshape`** in `DDiTBlock.forward`. TorchDynamo has limited support for `einops` operations; native tensor reshaping allows full graph capture.
- **Removed duplicate `modulate` function** (the overload with `.unsqueeze(1)` that was never called by `modulate_fused`).
- **Removed `autocast(enabled=False)` wrapper** around rotary embedding application in `DDiTBlock.forward`. This is unnecessary since rotary embeddings operate on the already-cast qkv tensor.

### 2. Replace `scdd/diffusion.py`

Copy `diffusion.py` from this folder to `scdd/diffusion.py`. Changes made:

- **Added `ScddDenoisingStep` class** (before `class Diffusion`): An `nn.Module` wrapper around `_scdd_update` that `torch.compile` can trace. This mirrors GIDD+'s `DenoisingStep` pattern.
- **Added `compile_sampler()` method** to `Diffusion`: Lazily creates and compiles the `ScddDenoisingStep`, stored as `self._compiled_scdd_step`.
- **Modified `_sample()` loop**: The `scdd` branch now uses `_compiled_scdd_step` when available, falling back to `_scdd_update` otherwise. This means existing code that does not call `compile_sampler()` is unaffected.
- **Removed `assert` statements** in `_scdd_update`: `assert sigma_t.ndim == 1` causes graph breaks in `torch.compile`.
- **Replaced data-dependent `.any()` guards with unconditional `.clamp()`** in `_scdd_update`: Data-dependent control flow (`if (eff_t < 0).any()`) causes graph breaks. The unconditional `eff_t.clamp(min=0.0)` is semantically equivalent and compile-friendly.

## Running the Benchmark

### 1. Update the codebase

```bash
cp dit.py       <REPO>/scdd/models/dit.py
cp diffusion.py <REPO>/scdd/diffusion.py
```

### 2. Run the latency benchmark

Edit `pbs_bench_latency.sh` to fill in the placeholder paths (`<REPO_DIR>`, `<DATA_DIR>`) and checkpoint locations, then run:

```bash
bash pbs_bench_latency.sh
```

### 3. Generate samples with compiled sampler

```bash
python generate_scdd.py \
    --checkpoint_path /path/to/scdd.ckpt \
    --num_steps 1024 --batch_size 16 --seq_len 512 \
    --nucleus_p 0.9 \
    --generated_seqs_path generated_seqs.json
```

The first call to `restore_model_and_sample` will be slower due to `torch.compile` warmup. Subsequent calls (or the per-step time within the same call after the first few steps) run at compiled speed.

## Using `compile_sampler()` in Your Own Code

To enable compiled sampling in any script that uses the `Diffusion` model:

```python
model = load_your_model(...)
model.compile_sampler()  # one-time compilation
samples = model.restore_model_and_sample(num_steps=128)
```

No other changes are needed. If `compile_sampler()` is not called, the model falls back to the original uncompiled path automatically.

# Correction Ablation Experiments: SCDD

This folder contains the code and instructions to reproduce the two correction ablation experiments demonstrating that SCDD's correction mechanism is beneficial.

## Files

| File | Description |
|------|-------------|
| `diffusion.py` | Modified diffusion module with `_disable_corrections` support (replaces `scdd/diffusion.py`) |
| `ablation_correction.py` | Experiment 1: Gen-PPL with/without corrections |
| `corruption_recovery.py` | Experiment 2: Corruption recovery at last step |
| `pbs_ablation_correction.sh` | Shell script for Experiment 1 (template) |
| `pbs_corruption_recovery.sh` | Shell script for Experiment 2 (template) |

## Changes to the Codebase

### Replace `scdd/diffusion.py`

Copy `diffusion.py` from this folder to `scdd/diffusion.py`. The only changes relative to the latency-optimized version are in `_scdlm_update`:

1. **Added `x_input = x`** at the top of `_scdlm_update` to save the original input tokens before sampling.

2. **Added `_disable_corrections` check** before `return x`:
   ```python
   if getattr(self, '_disable_corrections', False):
       was_unmasked = (x_input != self.mask_index)
       x = torch.where(was_unmasked, x_input, x)
   ```
   When `model._disable_corrections = True`, this freezes all already-unmasked positions, allowing only `[MASK]` → token transitions (no corrections). Default behavior is unchanged.

These changes do not affect training, normal generation, or model outputs when `_disable_corrections` is not set.

## Running the Experiments

### 1. Update the codebase

```bash
cp diffusion.py <REPO>/scdd/diffusion.py
```

### 2. Experiment 1: No-Correction Ablation

Generates text with corrections enabled and disabled, compares Gen-PPL (GPT-2-large). Reports per-batch PPL with mean ± standard error.

```bash
python ablation_correction.py \
    --checkpoint_path /path/to/scdd.ckpt \
    --num_steps 128 \
    --batch_size 16 \
    --num_batches 16 \
    --nucleus_p 0.9 \
    --output_path ablation_correction.json
```

Or edit `pbs_ablation_correction.sh` to fill in `<REPO_DIR>` and `<DATA_DIR>`, then:

```bash
bash pbs_ablation_correction.sh
```

### 3. Experiment 2: Corruption Recovery

Corrupts K tokens in clean validation text, runs one SCDD denoising step at the last-step noise level, measures touch/recovery/damage rates. Saves clean, corrupted, and corrected text per sample.

```bash
python corruption_recovery.py \
    --checkpoint_path /path/to/scdd.ckpt \
    --num_steps 128 \
    --batch_size 16 \
    --num_batches 8 \
    --corrupt_counts 5,10,20,50 \
    --nucleus_p 0.9 \
    --data_cache_dir /path/to/data \
    --output_path corruption_recovery.json
```

Or edit `pbs_corruption_recovery.sh` to fill in `<REPO_DIR>` and `<DATA_DIR>`, then:

```bash
bash pbs_corruption_recovery.sh
```

**Note:** The `--data_cache_dir` argument overrides the `data.cache_dir` stored in the checkpoint config, which may point to a path that does not exist on your system.

## Output Format

### Experiment 1

JSON with per-variant results:
- `gen_ppl_mean`, `gen_ppl_stderr`, `gen_ppl_per_batch`: per-batch PPL values
- `text_samples`: all generated text

### Experiment 2

JSON with per-corruption-level results:
- `touch_rate_mean/stderr`, `exact_recovery_rate_mean/stderr`, `damage_rate_mean/stderr`: per-batch metrics
- `ppl_clean/corrupted/corrected_mean/stderr`: perplexity before and after correction
- `samples`: list of per-sample records, each containing:
  - `clean_text`, `corrupted_text`, `corrected_text`
  - `corrupted_positions`, `touched_positions`, `recovered_positions`

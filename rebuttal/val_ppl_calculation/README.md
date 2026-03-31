# Official GIDD Val PPL

This folder contains the minimal changes needed to evaluate official
`dvruette/gidd` checkpoints and compute validation perplexity.

## Files

- `gidd/eval/loss.py`
  Replaces the upstream `gidd/eval/loss.py`.
  Adds support for:
  - loading official HuggingFace checkpoints via `+hf_model=...`
  - loading the validation split from local cache / arrow files
  - writing a single `metrics.json` output for scripted runs

- `eval_loss_gidd_official.sh`
  Example single-GPU launcher for running the modified `loss.py` on official
  checkpoints.

## How To Adapt The Official Repo

1. Clone the official repo:

```bash
git clone https://github.com/dvruette/gidd.git
```

2. Copy the modified evaluator into the same path inside that repo:

```bash
cp gidd/eval/loss.py <your-gidd-repo>/gidd/eval/loss.py
```

3. Copy the launcher script somewhere convenient:

```bash
cp eval_loss_gidd_official.sh <your-gidd-repo>/
```

4. Edit the placeholders in `eval_loss_gidd_official.sh`:
   - `<REPO_DIR>`
   - `<FAST_STORAGE_DIR>`
   - `<EVAL_OUTPUT_DIR>`

5. Run the script in your cluster environment.

## Output

For each official checkpoint, the script writes:

```text
<EVAL_OUTPUT_DIR>/<model-slug>/loss_ppl/loss_ppl.json
```

The JSON contains:

- `elbo`
- `ppl`
- `num_samples`
- `path`

## Default Official Checkpoints

The launcher evaluates:

- `dvruette/gidd-small-p_unif-0.2`
- `dvruette/gidd-small-p_unif-0.1`

You can edit `HF_MODELS` in the script to change this list.

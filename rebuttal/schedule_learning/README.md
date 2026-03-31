# Learnable SCDD Adaptation

This bundle contains the minimal files needed to add the learnable corruption schedule into the current SCDD codebase.

## Files

- `noise_schedule.py`
  Adds `LearnablePolynomialNoise`, `get_sigma_max_gamma`, and `get_sigma_max_rho`.
- `diffusion.py`
  Threads the learnable `rho/gamma` schedule through SCDD training, sampling, optimizer setup, and EMA.
- `configs/learnable_scdd.yaml`
  Small override file that enables the learnable schedule and sets a separate schedule learning rate.
- `scripts/run_scdd_learnable.sh`
  Generic launcher template with placeholders only.

## How To Apply

Copy the bundled files into the matching locations in your SCDD checkout:

```bash
cp rebuttal/schedule_learning/noise_schedule.py scdd/noise_schedule.py
cp rebuttal/schedule_learning/diffusion.py scdd/diffusion.py
cp rebuttal/schedule_learning/configs/learnable_scdd.yaml scdd/configs/learnable_scdd.yaml
cp rebuttal/schedule_learning/scripts/run_scdd_learnable.sh scdd/scripts/run_scdd_learnable.sh
```

## What Changes

The learnable setting keeps the existing time-conditioning noise (`self.noise`) but replaces the SCDD corruption schedule with two learnable functions:

- `gamma(t)`: probability of staying non-mask
- `rho(t)`: probability of staying uncorrupted conditioned on staying non-mask

These define:

- `alpha_bar(t) = rho(t) * gamma(t)`
- `beta_bar(t) = (1 - rho(t)) * gamma(t)`

Everything else in the model stays the same.

## Important Notes

- The bundled `diffusion.py` defaults to `forward.schedule_version=base` when the field is missing, so existing configs keep the original behavior.
- To enable the learnable schedule, you must set `forward.schedule_version=learnable`.
- A smaller LR for the schedule parameters is recommended. A good default is `schedule_lr = 0.02 * lr`. The bundled YAML uses `1e-5`, which matches `lr=5e-4`.

## Example

After copying the files, either:

1. merge `configs/learnable_scdd.yaml` into an existing config, or
2. pass the same overrides on the command line:

```bash
python -u -m main \
  --config-name <BASE_CONFIG> \
  parameterization=scdd \
  forward=mix \
  forward.schedule_version=learnable \
  optim.schedule_lr=1e-5
```
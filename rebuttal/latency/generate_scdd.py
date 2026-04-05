"""Generate real text samples from SCDD using restore_model_and_sample."""

import argparse

import torch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mdlm_v2', 'mdlm'))

import dataloader
import diffusion as diffusion_module


def load_model(ckpt_path, device):
    import tokenizers as _tk
    _orig = _tk.Tokenizer.__setstate__
    _tk.Tokenizer.__setstate__ = lambda self, state: None
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    _tk.Tokenizer.__setstate__ = _orig

    config = checkpoint["hyper_parameters"]["config"]
    tokenizer = dataloader.get_tokenizer(config)

    model = diffusion_module.Diffusion(config, tokenizer=tokenizer)
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    # Load EMA shadow params (saved by on_save_checkpoint)
    if model.ema and 'ema' in checkpoint:
        model.ema.load_state_dict(checkpoint['ema'])
        model.ema.move_shadow_params_to_device(device)
        print("EMA weights loaded.")
    elif model.ema:
        print("WARNING: EMA configured but not found in checkpoint.")

    model.to(device)
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--nucleus_p", type=float, default=None)
    parser.add_argument("--generated_seqs_path", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {args.checkpoint_path}")
    model, tokenizer = load_model(args.checkpoint_path, device)
    print(f"Model: {model.config.model.name}, parameterization={model.parameterization}")

    # Preserve checkpoint-trained sampling/forward config, only overriding
    # the runtime generation knobs requested by the caller.
    model.config.loader.eval_batch_size = args.batch_size
    model.config.model.length = args.seq_len
    if args.nucleus_p is not None:
        model.config.sampling.nucleus_p = args.nucleus_p

    print(
        f"Generating {args.batch_size} samples with {args.num_steps} steps, "
        f"seq_len={args.seq_len}, nucleus_p={model.config.sampling.nucleus_p}\n")

    model.compile_sampler()
    print("Compiled SCDD denoising step with torch.compile.")

    model.gen_ppl_metric.reset()
    samples = model.restore_model_and_sample(num_steps=args.num_steps)

    entropies = []
    text_samples = []
    for row in samples:
        counts = torch.unique(row, return_counts=True, sorted=True)[1]
        entropies.append(
            torch.special.entr(counts.float() / counts.sum()).sum().item())
        text_samples.append(tokenizer.batch_decode(row.unsqueeze(0))[0])

    model.compute_generative_perplexity(text_samples)
    gen_ppl = model.gen_ppl_metric.compute().cpu().item()

    if args.generated_seqs_path is not None:
        import json
        with open(args.generated_seqs_path, "w") as f:
            json.dump({
                "gen_ppl": gen_ppl,
                "entropy": sum(entropies) / len(entropies),
                "entropies": entropies,
                "text_samples": text_samples,
            }, f, indent=4)

    print(f"gen_ppl: {gen_ppl:.4f}")
    print(f"entropy: {sum(entropies) / len(entropies):.4f}\n")
    for i, text in enumerate(text_samples):
        print(f"=== Sample {i+1} ===")
        print(text)
        print()


if __name__ == "__main__":
    main()

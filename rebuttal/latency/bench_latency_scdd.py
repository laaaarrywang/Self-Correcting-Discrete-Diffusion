"""Benchmark SCDD generation latency across different step counts."""

import argparse
import json
import time

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
    model.eval()
    model.to(device)
    return model, config


def benchmark_steps(denoising_step, model, num_steps, batch_size, seq_len,
                    num_warmup, num_timed, device, eps=1e-5):
    """Run compiled generation loop and return per-batch latencies (seconds)."""
    def _run():
        x = model._sample_prior(batch_size, seq_len).to(device)
        timesteps = torch.linspace(1, eps, num_steps + 1, device=device)
        dt = (1 - eps) / num_steps
        for i in range(num_steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
            x = denoising_step(x, t, dt)
        return x

    # Warmup (triggers torch.compile on first call)
    for _ in range(num_warmup):
        _run()
        torch.cuda.synchronize()

    # Timed runs
    latencies = []
    for _ in range(num_timed):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _run()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies.append(t1 - t0)

    return latencies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--num_warmup", type=int, default=3)
    parser.add_argument("--num_timed", type=int, default=10)
    parser.add_argument("--steps", type=str, default="32,64,128,256,512,1024",
                        help="Comma-separated list of step counts")
    parser.add_argument("--output_path", type=str, default="bench_scdd.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high')
    torch.set_grad_enabled(False)

    print(f"Loading SCDD checkpoint: {args.checkpoint_path}")
    model, config = load_model(args.checkpoint_path, device)
    print(f"Model: {config.model.name}, parameterization={model.parameterization}")

    denoising_step = diffusion_module.ScddDenoisingStep(model)
    denoising_step = torch.compile(denoising_step)
    print("torch.compile applied to ScddDenoisingStep")

    step_counts = [int(s) for s in args.steps.split(",")]
    results = {
        "model": "scdd",
        "checkpoint": args.checkpoint_path,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "num_warmup": args.num_warmup,
        "num_timed": args.num_timed,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "benchmarks": [],
    }

    for ns in step_counts:
        print(f"\n=== SCDD steps={ns} (warmup={args.num_warmup}, timed={args.num_timed}) ===")
        latencies = benchmark_steps(
            denoising_step, model, ns, args.batch_size, args.seq_len,
            args.num_warmup, args.num_timed, device,
        )
        mean_lat = sum(latencies) / len(latencies)
        std_lat = (sum((x - mean_lat) ** 2 for x in latencies) / len(latencies)) ** 0.5
        total_tokens = args.batch_size * args.seq_len
        throughput = total_tokens / mean_lat

        entry = {
            "num_steps": ns,
            "latencies": latencies,
            "mean_latency": mean_lat,
            "std_latency": std_lat,
            "throughput_tok_per_sec": throughput,
            "per_step_ms": mean_lat / ns * 1000,
        }
        results["benchmarks"].append(entry)
        print(f"  Latency: {mean_lat:.3f} ± {std_lat:.3f} s")
        print(f"  Throughput: {throughput:.0f} tok/s")
        print(f"  Per-step: {entry['per_step_ms']:.2f} ms")

    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_path}")


if __name__ == "__main__":
    main()

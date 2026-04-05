"""Benchmark GIDD generation latency across different step counts."""

import argparse
import json
import time

import torch
import tqdm

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gidd'))

from gidd.checkpoints import load_checkpoint
from gidd.sampling import get_sampler
from gidd.utils import parse_dtype


def benchmark_steps(sampler, num_steps, batch_size, seq_len, num_warmup, num_timed, dtype, device):
    """Run generation and return per-batch latencies (seconds)."""
    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad(), torch.autocast(device.type, dtype=dtype):
            sampler.generate(batch_size, num_steps, max_length=seq_len,
                             decode=False, show_progress=False)
        torch.cuda.synchronize()

    # Timed runs
    latencies = []
    for _ in range(num_timed):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad(), torch.autocast(device.type, dtype=dtype):
            sampler.generate(batch_size, num_steps, max_length=seq_len,
                             decode=False, show_progress=False)
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
    parser.add_argument("--output_path", type=str, default="bench_gidd.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high')
    torch.set_grad_enabled(False)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"Loading GIDD checkpoint: {args.checkpoint_path}")
    model, noise_schedule, tokenizer, config = load_checkpoint(
        args.checkpoint_path, device=device)
    model.eval()
    dtype = parse_dtype(config.training.dtype)

    # Use torch.compile (GIDD's default) for real-world comparison.
    # Extra warmup batches absorb the one-time compilation cost.
    sampler = get_sampler(config, model, tokenizer, noise_schedule,
                          compile_step=True)

    step_counts = [int(s) for s in args.steps.split(",")]
    results = {
        "model": "gidd",
        "checkpoint": args.checkpoint_path,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "num_warmup": args.num_warmup,
        "num_timed": args.num_timed,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "benchmarks": [],
    }

    for ns in step_counts:
        print(f"\n=== GIDD steps={ns} (warmup={args.num_warmup}, timed={args.num_timed}) ===")
        latencies = benchmark_steps(
            sampler, ns, args.batch_size, args.seq_len,
            args.num_warmup, args.num_timed, dtype, device,
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

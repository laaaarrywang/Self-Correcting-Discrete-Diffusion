import functools
import glob
import hashlib
import json
import os

import hydra
import numpy as np
import torch
import tqdm
from datasets import Dataset, concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from gidd.checkpoints import load_checkpoint
from gidd.data import (
    cached_dataset,
    get_dataloaders,
    pretokenized_collator,
    tokenize_dataset,
)
from gidd.loss import get_loss
from gidd.trainer import get_trainer
from gidd.utils import parse_dtype


def _load_test_split_from_arrows(data_dir, test_size):
    arrow_files = sorted(glob.glob(os.path.join(data_dir, "**/*.arrow"),
                                   recursive=True))
    if not arrow_files:
        return None
    print(f"[DATA] Loading from {len(arrow_files)} local arrow files")
    ds = concatenate_datasets([Dataset.from_file(f) for f in arrow_files])
    start = max(0, len(ds) - test_size)
    return ds.select(range(start, len(ds)))


def get_test_dataloader(config, tokenizer, eval_batch_size,
                        cache_dir=None, data_dir=None,
                        shard_id=None, num_shards=None):
    test_size = int(config.data.test_size)

    test_ds = None
    if data_dir is not None:
        test_ds = _load_test_split_from_arrows(data_dir, test_size)

    if test_ds is None:
        if config.data.dataset_subset in ["wikitext-103-raw-v1",
                                          "wikitext-2-raw-v1"]:
            test_ds = load_dataset(
                config.data.dataset_name,
                config.data.dataset_subset,
                split="test",
                trust_remote_code=config.data.trust_remote_code,
            )
        else:
            test_ds = load_dataset(
                config.data.dataset_name,
                config.data.dataset_subset,
                split=f"train[-{test_size}:]",
                trust_remote_code=config.data.trust_remote_code,
            )

    if config.data.pre_tokenize:
        max_seq_len = config.model.max_seq_len
        sequence_packing = config.data.sequence_packing
        cache_key = hashlib.sha256(
            json.dumps(
                {
                    "dataset_name": config.data.dataset_name,
                    "subset": config.data.dataset_subset,
                    "tokenizer_name": config.data.tokenizer_name,
                    "max_seq_len": max_seq_len,
                    "sequence_packing": sequence_packing,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        if cache_dir is None:
            cache_dir = hydra.utils.to_absolute_path(config.data.cache_dir)
        test_cache_file = (
            f"cache-{config.data.dataset_name.replace('/', '--')}-test-{cache_key}"
        )
        print(f"[DATA] Cache directory: {cache_dir}")
        print(f"[DATA] Test cache file: {test_cache_file}")

        test_ds = cached_dataset(
            cache_dir=cache_dir,
            file_name=test_cache_file,
            generate_fn=functools.partial(
                tokenize_dataset,
                ds=test_ds,
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
                sequence_packing=sequence_packing,
            ),
        )
        collate_fn = functools.partial(
            pretokenized_collator,
            pad_token_id=tokenizer.pad_token_id,
            tokens_key="input_ids",
        )
    else:
        from gidd.data import subsample_collator

        collate_fn = functools.partial(
            subsample_collator, config, tokenizer, text_key="text")

    if shard_id is not None and num_shards is not None and num_shards > 1:
        total_len = len(test_ds)
        shard_size = (total_len + num_shards - 1) // num_shards
        start = shard_id * shard_size
        end = min(start + shard_size, total_len)
        indices = list(range(start, end))
        test_ds = torch.utils.data.Subset(test_ds, indices)

    return DataLoader(
        test_ds,
        collate_fn=collate_fn,
        batch_size=eval_batch_size,
        drop_last=False,
        num_workers=min(os.cpu_count(), 4),
        shuffle=False,
        pin_memory=True,
    )


def _load_hf_model(hf_model, device, hf_cache_dir=None):
    from gidd.pipeline import GiddPipeline
    from omegaconf import OmegaConf

    kwargs = {"trust_remote_code": True}
    if hf_cache_dir is not None:
        kwargs["cache_dir"] = hf_cache_dir
    pipe = GiddPipeline.from_pretrained(hf_model, **kwargs)
    model = pipe.model
    noise_schedule = pipe.noise_schedule
    tokenizer = pipe.tokenizer
    hf_cfg = pipe.config

    config = OmegaConf.create({
        "model": {
            "type": "diffusion",
            "diffusion_process": "gidd",
            "p_uniform": hf_cfg.p_uniform,
            "t_eps": getattr(hf_cfg, "t_eps", 1e-4),
            "max_seq_len": hf_cfg.max_seq_len,
        },
        "data": {
            "dataset_name": "Skylion007/openwebtext",
            "dataset_subset": None,
            "tokenizer_name": "gpt2",
            "test_size": 100000,
            "pre_tokenize": True,
            "sequence_packing": True,
            "trust_remote_code": True,
            "cache_dir": "./cache",
        },
        "loss": {
            "loss_type": "gidd",
            "loss_weighting": "dynamic",
            "min_loss_weight": 0.0,
            "max_loss_weight": 2.0,
            "loss_scale": 1.0,
            "reduction": "tokenmean",
        },
        "training": {
            "eval_batch_size": 16,
            "dtype": "bf16",
            "low_discrepancy_sampling": True,
        },
    })

    if device is not None:
        model.to(device)
        noise_schedule.to(device)

    return model, noise_schedule, tokenizer, config


@hydra.main(config_path="../configs", config_name="eval", version_base="1.1")
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    torch.set_grad_enabled(False)

    hf_model = args.get("hf_model", None)
    hf_cache_dir = args.get("hf_cache_dir", None)
    cache_dir = args.get("cache_dir", None)
    data_dir = args.get("data_dir", None)
    shard_id = args.get("shard_id", None)
    num_shards = args.get("num_shards", None)

    if hf_model is not None:
        print(f"Loading HuggingFace model: {hf_model}")
        model, noise_schedule, tokenizer, config = _load_hf_model(
            hf_model, device, hf_cache_dir=hf_cache_dir)
        ckpt_path = hf_model
    else:
        ckpt_path = hydra.utils.to_absolute_path(args.path)
        model, noise_schedule, tokenizer, config = load_checkpoint(
            ckpt_path, device=device)

    if args.use_gpt2:
        model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()
    config.training.eval_batch_size = args.batch_size
    dtype = parse_dtype(config.training.dtype)

    loss_fn = get_loss(config, tokenizer, noise_schedule)
    if hf_model is not None:
        test_dl = get_test_dataloader(
            config,
            tokenizer,
            args.batch_size,
            cache_dir=cache_dir,
            data_dir=data_dir,
            shard_id=shard_id,
            num_shards=num_shards,
        )
    else:
        _, test_dl = get_dataloaders(config, tokenizer)

    trainer = get_trainer(config, model, tokenizer, noise_schedule, loss_fn,
                          dtype)
    trainer.to(device)
    trainer = torch.compile(trainer)
    model.eval()

    eval_metrics = {}
    with torch.no_grad():
        eval_loss = 0
        num_eval_samples = 0
        for test_batch in tqdm.tqdm(test_dl, desc="Eval", dynamic_ncols=True):
            bs = test_batch["input_ids"].size(0)
            test_batch = {
                k: v.to(device, non_blocking=True) for k, v in test_batch.items()
            }
            loss, metrics = trainer(test_batch)

            for k, v in metrics.items():
                value = v.item() if isinstance(v, torch.Tensor) else v
                eval_metrics[k] = eval_metrics.get(k, 0) + value * bs

            eval_loss += loss.item() * bs
            num_eval_samples += bs

    eval_metrics = {
        "loss": eval_loss / num_eval_samples,
        **{k: v / num_eval_samples for k, v in eval_metrics.items()},
    }
    eval_metrics["ppl"] = np.exp(eval_metrics["elbo"])
    eval_metrics["path"] = ckpt_path
    eval_metrics["num_samples"] = num_eval_samples
    if shard_id is not None:
        eval_metrics["shard_id"] = shard_id
        eval_metrics["num_shards"] = num_shards

    print(json.dumps(eval_metrics, indent=2))

    out_name = "metrics.json"
    if shard_id is not None and num_shards is not None and num_shards > 1:
        out_name = f"metrics_shard{shard_id}.json"
    with open(out_name, "w") as f:
        json.dump(eval_metrics, f, indent=2)


if __name__ == "__main__":
    main()

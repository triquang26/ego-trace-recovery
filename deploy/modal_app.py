import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
VOLUME_PATH = Path("/vol")
BUCKET = os.environ.get("TWE_HF_BUCKET", "twanghcmut/trace-world-expert")
GPU = os.environ.get("TWE_GPU", "A100-40GB")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .uv_pip_install("torch==2.8.0", "transformers==5.19.0", "numpy>=1.26", "pyyaml>=6",
                    "huggingface_hub>=1.0", "av>=12", "h5py>=3.10", "matplotlib>=3.8")
    .env({"HF_HOME": "/root/hf_cache", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(ROOT / "configs", "/root/configs")
    .add_local_python_source("twe")
)
app = modal.App("trace-world-expert", image=image)
volume = modal.Volume.from_name("trace-world-expert", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])


def sync_to_bucket(local: Path, prefix: str) -> None:
    from huggingface_hub import HfApi

    HfApi(token=os.environ["HF_TOKEN"]).sync_bucket(str(local), f"hf://buckets/{BUCKET}/{prefix}", quiet=True)


@app.function(volumes={VOLUME_PATH: volume}, timeout=3600, cpu=4)
def make_synthetic(name: str = "synthetic", per_shard: int = 256) -> str:
    from twe.data.synthetic import write_synthetic

    root = VOLUME_PATH / "data" / name
    root.mkdir(parents=True, exist_ok=True)
    write_synthetic(root, per_shard=per_shard)
    volume.commit()
    return str(root)


@app.function(volumes={VOLUME_PATH: volume}, timeout=3600, cpu=4)
def compute_normalizer(name: str) -> dict:
    from twe.preprocess.normalizer import normalizer_from_dataset

    record = normalizer_from_dataset(VOLUME_PATH / "data" / name)
    volume.commit()
    return record


@app.function(gpu=GPU, volumes={VOLUME_PATH: volume}, secrets=[hf_secret], timeout=24 * 3600, cpu=12,
              memory=65536)
def train_world(data: str, run: str, config: str = "configs/stage1.yaml", overrides: dict | None = None) -> dict:
    from twe.config import load_stage1_config
    from twe.training.pretrain_world import train

    cfg = load_stage1_config(Path("/root") / config, overrides or {})
    out = VOLUME_PATH / "runs" / run

    def on_checkpoint(directory: Path) -> None:
        volume.commit()
        sync_to_bucket(directory, f"runs/{run}")

    return train(cfg, VOLUME_PATH / "data" / data, out, device="cuda", on_checkpoint=on_checkpoint)


@app.function(gpu=GPU, volumes={VOLUME_PATH: volume}, secrets=[hf_secret], timeout=3600, cpu=8, memory=32768)
def demo(data: str, run: str, count: int = 200, steps: int = 4, split: str = "validation",
         config: str = "configs/stage1.yaml") -> dict:
    from twe.config import load_stage1_config
    from twe.data.dataset import QuerySampling, WorldWindowDataset
    from twe.evaluation.demo import DemoBuilder, build_demo
    from twe.models.world_module import build_world_module
    from twe.preprocess.normalizer import load_normalizer
    from twe.training.checkpoint import artifact_world_config, load_world
    from twe.training.pretrain_world import make_fitter

    volume.reload()
    cfg = load_stage1_config(Path("/root") / config)
    root, out = VOLUME_PATH / "data" / data, VOLUME_PATH / "runs" / run
    space = artifact_world_config(out / "world_latest.pt").target_space
    sigma = load_normalizer(root / "normalizer.json")[f"sigma_{space}"]
    fitter = make_fitter(cfg)
    weights = out / "world_best.pt" if (out / "world_best.pt").exists() else out / "world_latest.pt"
    module = build_world_module(artifact_world_config(weights))
    load_world(weights, module)
    sampling = QuerySampling(cfg.world.num_anchors, cfg.min_moving_points, False)
    dataset = WorldWindowDataset(root, split, sigma, fitter, sampling, space)
    name = "demo" + ("" if steps == 4 else f"_steps{steps}") + ("" if split == "validation" else f"_{split}")
    record = build_demo(DemoBuilder(module.to("cuda"), fitter, dataset, sigma, "cuda", steps), out / name, count)
    volume.commit()
    sync_to_bucket(out / name, f"runs/{run}/{name}")
    return record["summary"]


@app.function(gpu=GPU, volumes={VOLUME_PATH: volume}, secrets=[hf_secret], timeout=3600, cpu=8, memory=32768)
def evaluate_run(data: str, run: str, batches: int = 40, config: str = "configs/stage1.yaml") -> dict:
    from dataclasses import replace

    import torch

    from twe.config import load_stage1_config
    from twe.data.dataset import WorldWindowDataset
    from twe.models.world_module import build_world_module
    from twe.preprocess.normalizer import load_normalizer
    from twe.training.checkpoint import artifact_world_config, load_world
    from twe.training.evaluate import evaluate
    from twe.training.pretrain_world import make_fitter, query_sampling

    volume.reload()
    out = VOLUME_PATH / "runs" / run
    world = artifact_world_config(out / "world_best.pt")
    cfg = replace(load_stage1_config(Path("/root") / config), world=world, eval_batches=batches)
    module = build_world_module(world)
    artifact = load_world(out / "world_best.pt", module)
    root = VOLUME_PATH / "data" / data
    sigma = load_normalizer(root / "normalizer.json")[f"sigma_{world.target_space}"]
    fitter = make_fitter(cfg)
    dataset = WorldWindowDataset(root, "validation", sigma, fitter, query_sampling(cfg, False), world.target_space)
    autocast = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    metrics = evaluate(module.to("cuda"), dataset, cfg, fitter, "cuda", autocast)
    return {"update": artifact.get("update"), **metrics}


@app.local_entrypoint()
def main(action: str = "smoke", data: str = "synthetic", run: str = "smoke", overrides: str = "", steps: int = 4,
         split: str = "validation") -> None:
    import json

    parsed = json.loads(overrides) if overrides else {}
    if action == "smoke":
        make_synthetic.remote(data, 256)
        parsed = {"optimizer_updates": 200, "eval_every": 100, "checkpoint_every": 100, "log_every": 10,
                  **parsed}
        print(train_world.remote(data, run, overrides=parsed))
    elif action == "normalizer":
        print(compute_normalizer.remote(data))
    elif action == "demo":
        print(demo.remote(data, run, steps=steps, split=split))
    elif action == "evaluate":
        print(json.dumps(evaluate_run.remote(data, run)))
    elif action == "train":
        print(train_world.remote(data, run, overrides=parsed))
    else:
        raise ValueError(f"unknown action {action}")

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
    .uv_pip_install("torch==2.8.0", "transformers>=4.44", "sentencepiece>=0.2", "numpy>=1.26", "pyyaml>=6",
                    "huggingface_hub>=1.0", "av>=12")
    .env({"HF_HOME": "/vol/hf_cache", "PYTHONUNBUFFERED": "1"})
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


@app.local_entrypoint()
def main(action: str = "smoke", data: str = "synthetic", run: str = "smoke", overrides: str = "") -> None:
    import json

    parsed = json.loads(overrides) if overrides else {}
    if action == "smoke":
        make_synthetic.remote(data, 256)
        parsed = {"optimizer_updates": 200, "eval_every": 100, "checkpoint_every": 100, "log_every": 10,
                  **parsed}
        print(train_world.remote(data, run, overrides=parsed))
    elif action == "normalizer":
        print(compute_normalizer.remote(data))
    elif action == "train":
        print(train_world.remote(data, run, overrides=parsed))
    else:
        raise ValueError(f"unknown action {action}")

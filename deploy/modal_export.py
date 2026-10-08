import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
VOLUME_PATH = Path("/vol")
GPU = os.environ.get("TWE_EXPORT_GPU", "A100-80GB")
CONTAINERS = int(os.environ.get("TWE_EXPORT_CONTAINERS", "4"))
SPATRACKER = "/opt/SpaTrackerV2"
SPATRACKER_COMMIT = "7e12274c52077860cebfe007a6290777db43b63c"
EGODEX_URL = "https://ml-site.cdn-apple.com/datasets/egodex/{part}.zip"
CAPTIONS = VOLUME_PATH / "raw" / "egodex" / "molmomotion_clips.json"

BUCKET = os.environ.get("TWE_HF_BUCKET", "twanghcmut/trace-world-expert")
download_image = modal.Image.debian_slim(python_version="3.11").apt_install("curl", "unzip")
upload_image = modal.Image.debian_slim(python_version="3.11").uv_pip_install("huggingface_hub>=1.0")
teacher_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0", "build-essential")
    .pip_install("torch==2.4.1", "torchvision==0.19.1", "xformers==0.0.28.post1",
                 index_url="https://download.pytorch.org/whl/cu124")
    .run_commands(f"git clone https://github.com/henry123-boy/SpaTrackerV2 {SPATRACKER}",
                  f"git -C {SPATRACKER} checkout {SPATRACKER_COMMIT}")
    .pip_install("opencv-python-headless", "einops", "einx", "easydict", "decord", "moviepy==1.0.0", "safetensors",
                 "scikit-learn", "scikit-image", "hydra-core", "omegaconf", "pycolmap==3.11.1", "pyceres==2.4",
                 "kornia==0.8.1", "timm", "jaxtyping", "rich", "evo", "flow_vis", "plotly", "matplotlib", "mediapy",
                 "prettytable", "huggingface_hub<1.0", "transformers>=4.44,<5", "h5py", "av>=12",
                 "pyyaml", "numpy<2",
                 "git+https://github.com/EasternJournalist/utils3d.git@d3a577acf0a9ad7e513a1416449a07b6f47d967f")
    .env({"HF_HOME": "/root/hf_cache", "PYTHONPATH": SPATRACKER, "PYTHONUNBUFFERED": "1"})
    .add_local_python_source("twe")
)
app = modal.App("trace-world-expert-export")
volume = modal.Volume.from_name("trace-world-expert", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])


@app.function(image=download_image, volumes={VOLUME_PATH: volume}, timeout=12 * 3600, cpu=8, memory=16384,
              ephemeral_disk=800 * 1024)
def download_egodex(part: str, tasks: str = "") -> str:
    import subprocess

    target = VOLUME_PATH / "raw" / "egodex"
    wanted = [t for t in tasks.split(",") if t]
    if (target / part).exists() and all((target / part / t).exists() for t in wanted):
        return f"{target / part} exists"
    archive = Path("/tmp") / f"{part}.zip"
    command = ["curl", "-sSfL", "-C", "-", "--retry", "20", "--retry-all-errors", "--retry-delay", "10", "-o",
               str(archive), EGODEX_URL.format(part=part)]
    for _ in range(30):
        if subprocess.run(command).returncode == 0:
            break
    else:
        raise RuntimeError(f"download of {part} did not finish")
    target.mkdir(parents=True, exist_ok=True)
    patterns = [f"{part}/{t}/*" for t in wanted]
    subprocess.run(["unzip", "-q", "-o", str(archive), *patterns, "-d", str(target)], check=True)
    volume.commit()
    return str(target / part)


@app.function(image=teacher_image, gpu=GPU, volumes={VOLUME_PATH: volume}, timeout=24 * 3600, cpu=8, memory=65536,
              max_containers=CONTAINERS)
def export_egodex(dataset: str, part: str, split: str, start: int, count: int, stride: float,
                  every: int = 1, frame_step: int = 2, chunk_seconds: float = 10.0, captioned: bool = False) -> dict:
    from twe.preprocess.build_dataset import export_egodex_shard

    volume.reload()
    report = export_egodex_shard(VOLUME_PATH / "raw" / "egodex", VOLUME_PATH / "data" / dataset,
                                 CAPTIONS if captioned else None, part, split, start, count, stride, every,
                                 frame_step, chunk_seconds)
    volume.commit()
    return report


@app.function(image=teacher_image, timeout=24 * 3600)
def orchestrate(args: list[tuple]) -> int:
    finished = 0
    for result in export_egodex.starmap(args, return_exceptions=True):
        print(result, flush=True)
        finished += 1
    return finished


@app.function(image=teacher_image, volumes={VOLUME_PATH: volume}, timeout=600)
def count_episodes(part: str, every: int = 1, captioned: bool = False) -> int:
    from twe.data.egodex import count_egodex, load_captions

    volume.reload()
    return count_egodex(VOLUME_PATH / "raw" / "egodex" / part, every, load_captions(CAPTIONS) if captioned else None)


@app.function(image=teacher_image, volumes={VOLUME_PATH: volume}, timeout=3600)
def finalize(dataset: str, moving_threshold_px: float = 10.0) -> dict:
    from twe.data.manifest import Manifest
    from twe.preprocess.build_dataset import relabel_moving
    from twe.preprocess.normalizer import normalizer_from_dataset
    from twe.preprocess.spatracker_teacher import teacher_revision

    volume.reload()
    root = VOLUME_PATH / "data" / dataset
    manifest = Manifest.collect(root, teacher_revision())
    moving = relabel_moving(root, moving_threshold_px)
    record = normalizer_from_dataset(root)
    volume.commit()
    return {"shards": len(manifest.shards), "windows": sum(e.count for e in manifest.shards), "moving_points": moving,
            **record}


@app.function(image=teacher_image, volumes={VOLUME_PATH: volume}, timeout=3600, cpu=8, memory=32768)
def teacher_videos(dataset: str, count: int = 8, frame_step: int = 2) -> list[dict]:
    from twe.evaluation.teacher_video import render_dataset

    volume.reload()
    records = render_dataset(VOLUME_PATH / "data" / dataset, VOLUME_PATH / "raw" / "egodex",
                             VOLUME_PATH / "viz" / dataset / "teacher", count, frame_step)
    volume.commit()
    return records


@app.function(image=upload_image, volumes={VOLUME_PATH: volume}, secrets=[hf_secret], timeout=6 * 3600, cpu=4)
def upload(dataset: str) -> str:
    from huggingface_hub import HfApi

    volume.reload()
    target = f"hf://buckets/{BUCKET}/data/{dataset}"
    HfApi(token=os.environ["HF_TOKEN"]).sync_bucket(str(VOLUME_PATH / "data" / dataset), target, quiet=True)
    return target


@app.function(image=upload_image, volumes={VOLUME_PATH: volume}, timeout=1800)
def fetch_captions() -> int:
    import json
    import shutil

    from huggingface_hub import hf_hub_download

    path = hf_hub_download("allenai/molmo-motion-1m", "egodex/annotations/egodex_clips.json", repo_type="dataset")
    CAPTIONS.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(path, CAPTIONS)
    volume.commit()
    return len(json.loads(CAPTIONS.read_text()))


@app.local_entrypoint()
def main(action: str = "export", part: str = "test", dataset: str = "egodex_v1", split: str = "",
         start: int = 0, episodes: int = 0, per_shard: int = 16, stride: float = 0.5, val_every: int = 0,
         every: int = 1, tasks: str = "", captioned: bool = False) -> None:
    if action == "download":
        print(download_egodex.remote(part, tasks))
    elif action == "count":
        print(count_episodes.remote(part, every, captioned))
    elif action == "export":
        episodes = episodes or count_episodes.remote(part, every, captioned) - start
        starts = list(range(start, start + episodes, per_shard))
        fixed = split or ("validation" if part == "test" else "train")
        splits = [("validation" if n % val_every == val_every - 1 else "train") if val_every else fixed
                  for n in range(len(starts))]
        args = [(dataset, part, sp, s, min(per_shard, start + episodes - s), stride, every, 2, 10.0, captioned)
                for sp, s in zip(splits, starts)]
        call = orchestrate.spawn(args)
        print(f"export of {len(args)} shards running on Modal: {call.object_id}")
    elif action == "teacher-videos":
        for record in teacher_videos.remote(dataset):
            print(record)
    elif action == "captions":
        print(fetch_captions.remote())
    elif action == "upload":
        print(upload.remote(dataset))
    elif action == "finalize":
        print(finalize.remote(dataset))
    else:
        raise ValueError(f"unknown action {action}")

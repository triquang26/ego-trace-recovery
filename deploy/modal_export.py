import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
VOLUME_PATH = Path("/vol")
GPU = os.environ.get("TWE_EXPORT_GPU", "A100-80GB")
SPATRACKER = "/opt/SpaTrackerV2"
SPATRACKER_COMMIT = "7e12274c52077860cebfe007a6290777db43b63c"
EGODEX_URL = "https://ml-site.cdn-apple.com/datasets/egodex/{part}.zip"

download_image = modal.Image.debian_slim(python_version="3.11").apt_install("curl", "unzip")
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
                 "prettytable", "huggingface_hub<1.0", "transformers>=4.44,<5", "sentencepiece", "h5py", "av>=12",
                 "pyyaml", "numpy<2",
                 "git+https://github.com/EasternJournalist/utils3d.git@d3a577acf0a9ad7e513a1416449a07b6f47d967f")
    .env({"HF_HOME": "/vol/hf_cache", "PYTHONPATH": SPATRACKER, "PYTHONUNBUFFERED": "1"})
    .add_local_python_source("twe")
)
app = modal.App("trace-world-expert-export")
volume = modal.Volume.from_name("trace-world-expert", create_if_missing=True)


@app.function(image=download_image, volumes={VOLUME_PATH: volume}, timeout=12 * 3600, cpu=8, memory=16384,
              ephemeral_disk=800 * 1024)
def download_egodex(part: str) -> str:
    import subprocess

    target = VOLUME_PATH / "raw" / "egodex"
    if (target / part).exists():
        return f"{target / part} exists"
    archive = Path("/tmp") / f"{part}.zip"
    subprocess.run(["curl", "-sSfL", "--retry", "5", "-o", str(archive), EGODEX_URL.format(part=part)], check=True)
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(["unzip", "-q", str(archive), "-d", str(target)], check=True)
    volume.commit()
    return str(target / part)


@app.function(image=teacher_image, gpu=GPU, volumes={VOLUME_PATH: volume}, timeout=24 * 3600, cpu=8, memory=65536)
def export_egodex(dataset: str, part: str, split: str, start: int, count: int, stride: float,
                  every: int = 1, frame_step: int = 2, chunk_seconds: float = 10.0) -> dict:
    from twe.config import WorldConfig
    from twe.data.egodex import egodex_recordings
    from twe.models.visual_encoder import DinoVisualEncoder
    from twe.preprocess.build_dataset import dino_selector, export_shard
    from twe.preprocess.export_windows import ExportSettings
    from twe.preprocess.spatracker_teacher import load_spatracker

    volume.reload()
    cfg = WorldConfig()
    visual = DinoVisualEncoder(cfg.visual_encoder, cfg.visual_encoder_revision, cfg.patch_grid).to("cuda")
    teacher = load_spatracker("cuda")
    recordings = egodex_recordings(VOLUME_PATH / "raw" / "egodex" / part, start, count, frame_step, every)
    root = VOLUME_PATH / "data" / dataset
    root.mkdir(parents=True, exist_ok=True)
    settings = ExportSettings(chunk_seconds=chunk_seconds)
    report = export_shard(recordings, teacher, dino_selector(visual, cfg, "cuda"), cfg, settings, root,
                          f"egodex-{part}-{start:06d}", "human_nominal", split, stride)
    volume.commit()
    return {k: v for k, v in report.items() if k != "skipped"} | {"skipped": len(report.get("skipped", []))}


@app.function(image=teacher_image, volumes={VOLUME_PATH: volume}, timeout=600)
def count_episodes(part: str, every: int = 1) -> int:
    from twe.data.egodex import egodex_episodes

    return len(egodex_episodes(VOLUME_PATH / "raw" / "egodex" / part)[::every])


@app.function(image=teacher_image, volumes={VOLUME_PATH: volume}, timeout=3600)
def finalize(dataset: str) -> dict:
    from twe.data.manifest import Manifest
    from twe.preprocess.normalizer import normalizer_from_dataset
    from twe.preprocess.spatracker_teacher import teacher_revision

    volume.reload()
    root = VOLUME_PATH / "data" / dataset
    manifest = Manifest.collect(root, teacher_revision())
    record = normalizer_from_dataset(root)
    volume.commit()
    return {"shards": len(manifest.shards), "windows": sum(e.count for e in manifest.shards), **record}


@app.function(image=teacher_image, volumes={VOLUME_PATH: volume}, timeout=3600, cpu=8, memory=32768)
def teacher_videos(dataset: str, count: int = 8, frame_step: int = 2) -> list[dict]:
    import numpy as np

    from twe.data.egodex import egodex_recording
    from twe.evaluation.teacher_video import most_dynamic, render_window, write_index

    volume.reload()
    root = VOLUME_PATH / "data" / dataset
    out = VOLUME_PATH / "viz" / dataset / "teacher"
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for n, (reader, row) in enumerate(most_dynamic(root, count)):
        meta = reader.metas[row]
        part, task, index = meta["recording_id"].split("/")
        raw = VOLUME_PATH / "raw" / "egodex" / part / task
        recording = egodex_recording(part, task, raw / f"{index}.hdf5", raw / f"{index}.mp4", frame_step)
        start = int(np.argmin(np.abs(recording.timestamps - meta["current_timestamp_seconds"])))
        stop = int(np.searchsorted(recording.timestamps, meta["current_timestamp_seconds"] + 2.0, side="right"))
        frames = recording.read_frames(np.arange(start, stop))
        times = recording.timestamps[start:stop] - recording.timestamps[start]
        records.append(render_window(reader, row, frames, times, 224, out / f"teacher_{n}.gif"))
    write_index(out, records)
    volume.commit()
    return records


@app.local_entrypoint()
def main(action: str = "export", part: str = "test", dataset: str = "egodex_v1", split: str = "",
         start: int = 0, episodes: int = 0, per_shard: int = 16, stride: float = 1.0, val_every: int = 0,
         every: int = 1) -> None:
    if action == "download":
        print(download_egodex.remote(part))
    elif action == "count":
        print(count_episodes.remote(part, every))
    elif action == "export":
        episodes = episodes or count_episodes.remote(part, every) - start
        starts = list(range(start, start + episodes, per_shard))
        fixed = split or ("validation" if part == "test" else "train")
        splits = [("validation" if n % val_every == val_every - 1 else "train") if val_every else fixed
                  for n in range(len(starts))]
        args = [(dataset, part, sp, s, min(per_shard, start + episodes - s), stride, every)
                for sp, s in zip(splits, starts)]
        for result in export_egodex.starmap(args):
            print(result)
    elif action == "teacher-videos":
        for record in teacher_videos.remote(dataset):
            print(record)
    elif action == "finalize":
        print(finalize.remote(dataset))
    else:
        raise ValueError(f"unknown action {action}")

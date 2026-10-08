import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent))
import modal_app
import modal_export

VOLUME_PATH = Path("/vol")
MU0_REPO = "furonghuang-lab/mu0"
MU0_ROOT = VOLUME_PATH / "ext" / "mu0"
app = modal.App("trace-world-expert-tools")


@app.function(image=modal_export.upload_image, volumes={VOLUME_PATH: modal_app.volume}, secrets=[modal_app.hf_secret],
              timeout=3 * 3600, cpu=8, memory=16384)
def fetch_mu0() -> list[str]:
    import os
    import tarfile

    from huggingface_hub import hf_hub_download

    if not (MU0_ROOT / "test_set").exists():
        archive = hf_hub_download(MU0_REPO, "test_set.tar", local_dir="/tmp/mu0", token=os.environ["HF_TOKEN"])
        with tarfile.open(archive) as tar:
            tar.extractall(MU0_ROOT)
        modal_app.volume.commit()
    return sorted(p.name for p in (MU0_ROOT / "test_set").iterdir())


@app.function(image=modal_export.teacher_image.add_local_python_source("modal_app", "modal_export"), gpu="A100-80GB",
              volumes={VOLUME_PATH: modal_app.volume}, timeout=3 * 3600, cpu=8, memory=65536)
def compare_teacher(group: str = "test_dataset_egodex", limit: int = 20) -> list[dict]:
    from twe.evaluation.teacher_compare import compare_traceextract
    from twe.preprocess.spatracker_teacher import load_spatracker

    episodes = sorted(p for p in (MU0_ROOT / "test_set" / group).iterdir() if (p / "samples").exists())[:limit]
    out = VOLUME_PATH / "viz" / "teacher_compare" / group
    rows = compare_traceextract(load_spatracker("cuda"), episodes, out)
    modal_app.volume.commit()
    return rows


@app.function(image=modal_app.image.add_local_python_source("modal_app", "modal_export"), gpu=modal_app.GPU,
              volumes={VOLUME_PATH: modal_app.volume}, secrets=[modal_app.hf_secret], timeout=3600, cpu=8,
              memory=32768)
def prompts(data: str, run: str, scenes: int = 3, config: str = "configs/stage1.yaml") -> dict:
    from twe.config import load_stage1_config
    from twe.data.dataset import QuerySampling, TaskSelection, WorldWindowDataset
    from twe.evaluation.demo import DemoBuilder
    from twe.evaluation.prompt_sweep import build_prompt_sweep
    from twe.models.world_module import build_world_module
    from twe.training.checkpoint import artifact_world_config, load_world
    from twe.training.pretrain_world import make_fitter

    cfg = load_stage1_config(Path("/root") / config)
    out = VOLUME_PATH / "runs" / run
    world = artifact_world_config(out / "world_best.pt")
    module = build_world_module(world)
    sigma = load_world(out / "world_best.pt", module)["normalizer"][f"sigma_{world.target_space}"]
    fitter = make_fitter(cfg)
    dataset = WorldWindowDataset(VOLUME_PATH / "data" / data, "validation", sigma, fitter,
                                 QuerySampling(world.num_anchors, cfg.min_moving_points, False), world.target_space,
                                 TaskSelection(frozenset(cfg.heldout_tasks)))
    record = build_prompt_sweep(DemoBuilder(module.to("cuda"), fitter, dataset, sigma, "cuda"), out / "prompts", scenes)
    modal_app.volume.commit()
    modal_app.sync_to_bucket(out / "prompts", f"runs/{run}/prompts")
    return record


@app.function(image=modal_app.image.add_local_python_source("modal_app", "modal_export"), gpu=modal_app.GPU,
              volumes={VOLUME_PATH: modal_app.volume}, secrets=[modal_app.hf_secret], timeout=2 * 3600, cpu=8,
              memory=32768)
def rollout(run: str, groups: str = "test_dataset_egodex,test_dataset_custom_dataset_human", per_group: int = 8,
            horizon: float = 8.0, stride: float = 1.0, config: str = "configs/stage1.yaml") -> dict:
    from twe.config import load_stage1_config
    from twe.evaluation.rollout import TraceRollout
    from twe.evaluation.rollout_eval import rollout_benchmark
    from twe.models.world_module import build_world_module
    from twe.training.checkpoint import artifact_world_config, load_world
    from twe.training.pretrain_world import make_fitter

    out = VOLUME_PATH / "runs" / run
    world = artifact_world_config(out / "world_best.pt")
    module = build_world_module(world)
    sigma = load_world(out / "world_best.pt", module)["normalizer"][f"sigma_{world.target_space}"]
    engine = TraceRollout(module.to("cuda"), make_fitter(load_stage1_config(Path("/root") / config)), sigma, world,
                          "cuda")
    episodes = [p for g in groups.split(",") for p in sorted((MU0_ROOT / "test_set" / g).iterdir())[:per_group]
                if (p / "samples").exists()]
    report = rollout_benchmark(engine, episodes, out / "rollout", horizon=horizon, stride=stride)
    modal_app.volume.commit()
    modal_app.sync_to_bucket(out / "rollout", f"runs/{run}/rollout")
    return report


@app.local_entrypoint()
def main(action: str, data: str = "egodex_v4", run: str = "", group: str = "test_dataset_egodex") -> None:
    if action == "fetch-mu0":
        print(fetch_mu0.remote())
    elif action == "compare-teacher":
        for row in compare_teacher.remote(group):
            print(row)
    elif action == "rollout":
        import json

        print(json.dumps(rollout.remote(run), indent=1))
    elif action == "prompts":
        print(prompts.remote(data, run))
    else:
        raise ValueError(f"unknown action {action}")

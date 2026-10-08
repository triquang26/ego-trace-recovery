import hashlib
import io
from dataclasses import asdict
from pathlib import Path

import torch

from twe.config import WorldConfig
from twe.contracts import COORDINATE_CONTRACT, NOISE_PROTOCOL, PREPROCESSING_REVISION
from twe.models.world_module import WorldModule
from twe.preprocess.bspline_targets import BSplineTargets


def world_artifact(module: WorldModule, fitter: BSplineTargets, normalizer: dict, manifest_teacher: str) -> dict:
    return {
        "world_config": asdict(module.cfg),
        "visual_encoder": getattr(module.visual_encoder, "revision", "custom"),
        "grounding_encoder": getattr(module.grounding_encoder, "revision", "custom"),
        "trainable_state": module.trainable_state(),
        "normalizer": normalizer,
        "bspline_knots": fitter.knots.tolist(),
        "bspline_free_basis": fitter.free_basis.tolist(),
        "coordinate_contract": COORDINATE_CONTRACT,
        "preprocessing_revision": PREPROCESSING_REVISION,
        "noise_protocol": NOISE_PROTOCOL,
        "teacher_revision": manifest_teacher,
    }


def artifact_hash(artifact: dict) -> str:
    buffer = io.BytesIO()
    torch.save(artifact.get("trainable_state", artifact.get("expert_state")), buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()[:16]


def save_world(path: Path, artifact: dict) -> str:
    digest = artifact_hash(artifact)
    torch.save({**artifact, "model_revision": digest}, path)
    return digest


def artifact_world_config(path: Path) -> WorldConfig:
    return WorldConfig(**torch.load(path, map_location="cpu", weights_only=False)["world_config"])


def load_world(path: Path, module: WorldModule) -> dict:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    for key, expected in (("preprocessing_revision", PREPROCESSING_REVISION),
                          ("coordinate_contract", COORDINATE_CONTRACT),
                          ("noise_protocol", NOISE_PROTOCOL)):
        if artifact[key] != expected:
            raise ValueError(f"{key} mismatch: {artifact[key]} != {expected}")
    if WorldConfig(**artifact["world_config"]) != module.cfg:
        raise ValueError("world config mismatch")
    if "trainable_state" in artifact:
        module.load_trainable_state(artifact["trainable_state"])
    else:
        module.expert.load_state_dict(artifact["expert_state"])
    module.model_revision = artifact["model_revision"]
    return artifact

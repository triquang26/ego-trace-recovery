import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from twe.config import Stage1Config, load_stage1_config
from twe.data.dataset import QuerySampling, TaskSelection, WorldWindowDataset, collate
from twe.data.sampler import MixtureBatchSampler
from twe.models.world_module import WorldModule, build_world_module
from twe.preprocess.bspline_targets import BSplineTargets
from twe.preprocess.normalizer import load_normalizer
from twe.training.checkpoint import save_world, world_artifact
from twe.training.ema import ExponentialAverage, averaged
from twe.training.evaluate import evaluate
from twe.training.objective import (masked_flow_loss, motion_loss, noisy_controls, sample_flow_time, validity_loss,
                                    warmup_cosine)


def make_fitter(cfg: Stage1Config) -> BSplineTargets:
    w = cfg.world
    return BSplineTargets(w.future_steps, w.free_control_points, w.bspline_degree, cfg.fit_regularization,
                          cfg.fit_min_valid_steps, cfg.fit_max_condition)


def query_sampling(cfg: Stage1Config, randomize: bool) -> QuerySampling:
    return QuerySampling(cfg.world.num_anchors, cfg.min_moving_points, randomize)


def task_selection(cfg: Stage1Config, split: str) -> TaskSelection:
    fraction = cfg.train_shard_fraction if split == "train" else 1.0
    return TaskSelection(frozenset(cfg.heldout_tasks), None, fraction)


def parameter_groups(module: WorldModule, weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    for part in module.trainable().values():
        for param in part.parameters():
            if param.requires_grad:
                (decay if param.ndim >= 2 else no_decay).append(param)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def parameter_report(module: WorldModule) -> dict:
    count = lambda m, trainable: sum(p.numel() for p in m.parameters() if p.requires_grad or not trainable)
    return {
        "visual_encoder": count(module.visual_encoder, False),
        "grounding_encoder": count(module.grounding_encoder, False),
        "trace_expert": count(module.expert, False),
        "fusion": count(module.fusion, False) if module.fusion is not None else 0,
        "total": count(module, False),
        "trainable": count(module, True),
    }


def train_step(module, context, target, cfg: Stage1Config, autocast) -> dict[str, torch.Tensor]:
    with autocast():
        inputs = module.encode(context)
        s = sample_flow_time(target.controls.shape[0], cfg.endpoint_time_probability, target.controls.device)
        noise = torch.randn_like(target.controls)
        outputs = module(inputs, noisy_controls(target.controls, noise, s), s)
        guidance = module(inputs, torch.randn_like(noise), torch.ones_like(s))
    flow = masked_flow_loss(outputs.velocity, noise - target.controls, context.anchor_mask & target.fit_valid)
    valid = validity_loss(outputs.validity_logits, target.trace_valid, context.anchor_mask)
    moving = motion_loss(guidance.motion_logits, target.moving, context.anchor_mask)
    loss = flow + cfg.validity_loss_weight * valid + cfg.motion_loss_weight * moving
    return {"loss": loss, "flow": flow, "validity": valid, "motion": moving}


def train(cfg: Stage1Config, data_root: Path, out_dir: Path, module: WorldModule | None = None,
          device: str | None = None, on_checkpoint: Callable[[Path], None] | None = None) -> dict:
    torch.manual_seed(cfg.seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir.mkdir(parents=True, exist_ok=True)
    normalizer = load_normalizer(data_root / "normalizer.json")
    fitter = make_fitter(cfg)
    space = cfg.world.target_space
    sigma = normalizer[f"sigma_{space}"]
    train_set = WorldWindowDataset(data_root, "train", sigma, fitter, query_sampling(cfg, True), space,
                                   task_selection(cfg, "train"))
    val_set = WorldWindowDataset(data_root, "validation", sigma, fitter, query_sampling(cfg, False), space,
                                 task_selection(cfg, "validation"))
    module = (module or build_world_module(cfg.world)).to(device)
    (out_dir / "parameters.json").write_text(json.dumps(parameter_report(module), indent=2))
    optimizer = torch.optim.AdamW(parameter_groups(module, cfg.weight_decay), lr=cfg.learning_rate,
                                  betas=cfg.betas)
    total = cfg.optimizer_updates
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda u: warmup_cosine(u, total, cfg.warmup_fraction))
    ema = ExponentialAverage(module.trainable_parameters(), cfg.ema_decay) if cfg.ema_decay else None
    state_path = out_dir / "train_state.pt"
    start = 0
    best = float("inf")
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        module.load_trainable_state(state["trainable"])
        if ema is not None:
            ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start = state["update"]
        best = state.get("best_flow", best)
    accum = cfg.accumulation_steps
    sampler = MixtureBatchSampler([train_set.meta(i) for i in range(len(train_set))], cfg.mixture,
                                  cfg.micro_batch_size, cfg.max_null_text_fraction, (total - start) * accum,
                                  cfg.seed + start)
    loader = DataLoader(train_set, batch_sampler=sampler, num_workers=cfg.num_workers, collate_fn=collate,
                        pin_memory=device.type == "cuda", persistent_workers=cfg.num_workers > 0)
    use_bf16 = cfg.precision == "bfloat16" and device.type == "cuda"
    autocast = lambda: torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16)
    log = open(out_dir / "log.jsonl", "a")
    log.write(json.dumps({"event": "start", "update": start, "mixture": sampler.realized_mixture(),
                          "train_windows": len(train_set), "val_windows": len(val_set)}) + "\n")
    batches = iter(loader)
    module.train()
    metrics: dict = {}
    for update in range(start, total):
        tick = time.time()
        sums = {"loss": 0.0, "flow": 0.0, "validity": 0.0, "motion": 0.0}
        for _ in range(accum):
            context, target = next(batches)
            losses = train_step(module, context.to(device), target.to(device), cfg, autocast)
            (losses["loss"] / accum).backward()
            for key in sums:
                sums[key] += losses[key].item() / accum
        grad_norm = torch.nn.utils.clip_grad_norm_(module.trainable_parameters(), cfg.gradient_clip_norm)
        optimizer.step()
        if ema is not None:
            ema.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        done = update + 1
        if done % cfg.log_every == 0 or done == total:
            record = {"update": done, **sums, "grad_norm": float(grad_norm), "lr": scheduler.get_last_lr()[0],
                      "seconds_per_update": time.time() - tick}
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)
        if done % cfg.eval_every == 0 or done == total:
            with averaged(ema):
                metrics = evaluate(module, val_set, cfg, fitter, device, autocast)
                log.write(json.dumps({"update": done, "eval": metrics}) + "\n")
                print(json.dumps({"update": done, "eval": metrics}), flush=True)
                if metrics.get("flow", best) < best:
                    best = metrics["flow"]
                    artifact = world_artifact(module, fitter, normalizer, train_set.manifest.teacher_revision)
                    save_world(out_dir / "world_best.pt", {**artifact, "update": done, "eval": metrics})
            module.train()
        if done % cfg.checkpoint_every == 0 or done == total:
            with averaged(ema):
                artifact = world_artifact(module, fitter, normalizer, train_set.manifest.teacher_revision)
                module.model_revision = save_world(out_dir / "world_latest.pt", artifact)
            torch.save({"trainable": module.trainable_state(), "ema": ema.state_dict() if ema else None,
                        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "update": done,
                        "best_flow": best}, state_path)
            if on_checkpoint:
                on_checkpoint(out_dir)
    log.close()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage1.yaml")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--set", nargs="*", default=[], help="key=value overrides")
    args = parser.parse_args()
    overrides = {k: json.loads(v) for k, v in (item.split("=", 1) for item in args.set)}
    train(load_stage1_config(args.config, overrides), args.data, args.out)


if __name__ == "__main__":
    main()

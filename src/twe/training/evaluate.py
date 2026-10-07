import torch
from torch.utils.data import DataLoader

from twe.config import Stage1Config
from twe.data.dataset import collate
from twe.evaluation.trace_metrics import reconstruction_error, summarize, trace_errors, validity_calibration
from twe.training.objective import masked_flow_loss, noisy_controls, validity_loss


@torch.no_grad()
def evaluate(module, dataset, cfg: Stage1Config, fitter, device, autocast) -> dict[str, float]:
    module.eval()
    loader = DataLoader(dataset, batch_size=cfg.micro_batch_size, shuffle=False, num_workers=cfg.num_workers,
                        collate_fn=collate)
    generator = torch.Generator().manual_seed(cfg.seed)
    totals: dict[str, torch.Tensor] = {}
    flow_sum, valid_sum, batches = 0.0, 0.0, 0
    for index, (context, target) in enumerate(loader):
        if index >= cfg.eval_batches:
            break
        context, target = context.to(device), target.to(device)
        batch = target.controls.shape[0]
        s = torch.rand(batch, generator=generator).to(device)
        noise = torch.randn(target.controls.shape, generator=generator).to(device)
        with autocast():
            inputs = module.encode(context)
            outputs = module(inputs, noisy_controls(target.controls, noise, s), s)
            controls = module.sample_controls(inputs, cfg.world.trace_eval_solver_steps, generator)
        flow_sum += masked_flow_loss(outputs.velocity, noise - target.controls,
                                     context.anchor_mask & target.fit_valid).item()
        valid_sum += validity_loss(outputs.validity_logits, target.trace_valid, context.anchor_mask).item()
        batches += 1
        pred = fitter.decode(controls.double()).float()
        mask = context.anchor_mask[..., None].expand_as(target.trace_valid)
        stats = {**trace_errors(pred, target, context.anchor_mask, cfg.dynamic_threshold),
                 **reconstruction_error(fitter, target, context.anchor_mask),
                 **validity_calibration(outputs.validity_logits, target.trace_valid, mask)}
        for key, value in stats.items():
            totals[key] = totals.get(key, 0) + value
    if not batches:
        return {}
    return {"flow": flow_sum / batches, "validity": valid_sum / batches, **summarize(totals)}

import torch
import torch.nn.functional as F
from torch import Tensor


def patch_validity(image_valid: Tensor, grid: int) -> Tensor:
    kernel = image_valid.shape[-1] // grid
    return F.avg_pool2d(image_valid[:, None].float(), kernel)[:, 0]


def pool_visual(patch_features: Tensor, weight: Tensor, pooled_grid: int) -> tuple[Tensor, Tensor]:
    batch, grid, _, dim = patch_features.shape
    factor = grid // pooled_grid
    features = patch_features.permute(0, 3, 1, 2) * weight[:, None]
    summed = F.avg_pool2d(features, factor) * factor * factor
    total = F.avg_pool2d(weight[:, None], factor)[:, 0] * factor * factor
    pooled = summed / total.clamp_min(1e-6)[:, None]
    tokens = pooled.permute(0, 2, 3, 1).reshape(batch, pooled_grid * pooled_grid, dim)
    return tokens, total.reshape(batch, -1) > 0


def sample_anchor_features(patch_features: Tensor, uv: Tensor) -> Tensor:
    grid = patch_features.permute(0, 3, 1, 2)
    coords = (uv * 2 - 1)[:, :, None]
    sampled = F.grid_sample(grid, coords.to(grid.dtype), mode="bilinear", padding_mode="border", align_corners=False)
    return sampled[..., 0].transpose(1, 2)


def candidate_grid(image_valid: Tensor, resolution: int) -> tuple[Tensor, Tensor]:
    centers = (torch.arange(resolution, dtype=torch.float32) + 0.5) / resolution
    yy, xx = torch.meshgrid(centers, centers, indexing="ij")
    uv = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
    kernel = image_valid.shape[-1] // resolution
    cell = F.avg_pool2d(image_valid[None, None].float(), kernel)
    interior = -F.max_pool2d(-cell, 3, stride=1, padding=1)
    return uv, interior[0, 0].flatten() >= 1.0


def farthest_init(points: Tensor, count: int, start: int) -> Tensor:
    picked = [start]
    nearest = (points - points[start]).norm(dim=-1)
    for _ in range(count - 1):
        index = int(torch.argmax(nearest))
        picked.append(index)
        nearest = torch.minimum(nearest, (points - points[index]).norm(dim=-1))
    return points[torch.tensor(picked)].clone()


def lloyd(points: Tensor, centers: Tensor, iterations: int) -> Tensor:
    for _ in range(iterations):
        labels = torch.cdist(points, centers).argmin(dim=1)
        for k in range(len(centers)):
            members = points[labels == k]
            if len(members):
                centers[k] = members.mean(dim=0)
    return centers


def kmeans(points: Tensor, clusters: int, iterations: int) -> Tensor:
    centers = farthest_init(points, min(clusters, len(points)), int(torch.argmax(points.norm(dim=-1))))
    return torch.cdist(points, lloyd(points, centers, iterations)).argmin(dim=1)


def entity_quotas(sizes: Tensor, total: int, minimum: int, power: float) -> Tensor:
    weight = sizes.float().pow(power)
    quota = torch.minimum(torch.clamp(torch.floor(total * weight / weight.sum()), min=minimum), sizes.float())
    quota = quota.long()
    while quota.sum() > total:
        quota[int(torch.argmax(torch.where(quota > 1, quota, torch.zeros_like(quota))))] -= 1
    while quota.sum() < total and bool((quota < sizes).any()):
        spare = torch.where(quota < sizes, weight / (quota + 1).float(), torch.full_like(weight, -1.0))
        quota[int(torch.argmax(spare))] += 1
    return quota


def spread_points(uv: Tensor, count: int, iterations: int) -> Tensor:
    start = int(torch.argmin((uv - uv.mean(dim=0)).norm(dim=-1)))
    centers = lloyd(uv, farthest_init(uv, count, start), iterations)
    taken = torch.zeros(len(uv), dtype=torch.bool)
    picked = []
    for center in centers:
        distance = torch.where(taken, torch.full((len(uv),), float("inf")), (uv - center).norm(dim=-1))
        index = int(torch.argmin(distance))
        taken[index] = True
        picked.append(index)
    return uv[torch.tensor(picked)]


@torch.no_grad()
def select_anchors(patch_features: Tensor, image_valid: Tensor, num_anchors: int, entities: int,
                   min_per_entity: int, spatial_weight: float, area_power: float,
                   iterations: int = 10) -> tuple[Tensor, Tensor]:
    batch, grid = patch_features.shape[:2]
    features = patch_features.detach().float().cpu()
    valid = image_valid.cpu().bool()
    uv = torch.zeros(batch, num_anchors, 2)
    mask = torch.zeros(batch, num_anchors, dtype=torch.bool)
    for b in range(batch):
        cand_uv, alive = candidate_grid(valid[b], grid * 2)
        cand_uv = cand_uv[alive]
        desc = F.normalize(sample_anchor_features(features[b : b + 1], cand_uv[None])[0], dim=-1)
        labels = kmeans(torch.cat([desc, spatial_weight * cand_uv], dim=-1), entities, iterations)
        present = torch.unique(labels)
        sizes = torch.stack([(labels == k).sum() for k in present])
        quotas = entity_quotas(sizes, min(num_anchors, len(cand_uv)), min_per_entity, area_power)
        chosen = torch.cat([spread_points(cand_uv[labels == k], int(q), iterations)
                            for k, q in zip(present, quotas) if q > 0])
        uv[b, : len(chosen)] = chosen
        mask[b, : len(chosen)] = True
    return uv.to(patch_features.device), mask.to(patch_features.device)

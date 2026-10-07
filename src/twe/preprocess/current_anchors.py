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


def _grid_shape(count: int, aspect: float) -> tuple[int, int]:
    pairs = [(rows, count // rows) for rows in range(1, count + 1) if count % rows == 0]
    return min(pairs, key=lambda pair: abs(pair[1] / pair[0] - aspect))


def _coverage(valid: Tensor, count: int) -> Tensor:
    size_y, size_x = valid.shape
    ys = torch.nonzero(valid.any(dim=1)).flatten()
    xs = torch.nonzero(valid.any(dim=0)).flatten()
    x0, x1 = xs[0].item() / size_x, (xs[-1].item() + 1) / size_x
    y0, y1 = ys[0].item() / size_y, (ys[-1].item() + 1) / size_y
    rows, cols = _grid_shape(count, (x1 - x0) / max(y1 - y0, 1e-6))
    gx = x0 + (torch.arange(cols, dtype=torch.float32) + 0.5) / cols * (x1 - x0)
    gy = y0 + (torch.arange(rows, dtype=torch.float32) + 0.5) / rows * (y1 - y0)
    yy, xx = torch.meshgrid(gy, gx, indexing="ij")
    return torch.stack([xx.flatten(), yy.flatten()], dim=-1)


def _diversity(features: Tensor, weight: Tensor, chosen_uv: Tensor, count: int, min_distance: float) -> Tensor:
    grid = features.shape[0]
    centers = (torch.arange(grid, dtype=torch.float32) + 0.5) / grid
    yy, xx = torch.meshgrid(centers, centers, indexing="ij")
    cand_uv = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
    cand_desc = F.normalize(features.reshape(grid * grid, -1).float(), dim=-1)
    alive = weight.flatten() >= 1.0
    alive &= torch.cdist(cand_uv, chosen_uv).min(dim=1).values >= min_distance
    chosen_desc = F.normalize(sample_anchor_features(features[None].float(), chosen_uv[None])[0], dim=-1)
    nearest = 1 - (cand_desc @ chosen_desc.T).max(dim=1).values
    picked = []
    for _ in range(count):
        if not alive.any():
            break
        score = torch.where(alive, nearest, torch.full_like(nearest, -float("inf")))
        index = int(torch.argmax(score))
        picked.append(index)
        alive &= torch.linalg.norm(cand_uv - cand_uv[index], dim=-1) >= min_distance
        nearest = torch.minimum(nearest, 1 - cand_desc @ cand_desc[index])
    if not picked:
        return cand_uv[:0]
    return cand_uv[torch.tensor(picked)]


@torch.no_grad()
def select_anchors(
    patch_features: Tensor, image_valid: Tensor, num_anchors: int, coverage: int, min_distance: float
) -> tuple[Tensor, Tensor]:
    batch = patch_features.shape[0]
    weight = patch_validity(image_valid, patch_features.shape[1]).cpu()
    features = patch_features.detach().float().cpu()
    valid = image_valid.cpu().bool()
    uv = torch.zeros(batch, num_anchors, 2)
    mask = torch.zeros(batch, num_anchors, dtype=torch.bool)
    for b in range(batch):
        grid_uv = _coverage(valid[b], coverage)
        extra = _diversity(features[b], weight[b], grid_uv, num_anchors - coverage, min_distance)
        points = torch.cat([grid_uv, extra])
        uv[b, : len(points)] = points
        mask[b, : len(points)] = True
    return uv.to(patch_features.device), mask.to(patch_features.device)

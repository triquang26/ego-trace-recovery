import torch.nn.functional as F
from torch import Tensor, nn


class Attention(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.query = nn.Linear(width, width)
        self.key_value = nn.Linear(width, 2 * width)
        self.out = nn.Linear(width, width)

    def forward(self, x: Tensor, context: Tensor, key_mask: Tensor) -> Tensor:
        batch, length, width = x.shape
        q = self.query(x).view(batch, length, self.heads, -1).transpose(1, 2)
        k, v = self.key_value(context).view(batch, context.shape[1], 2, self.heads, -1).permute(2, 0, 3, 1, 4)
        mask = key_mask[:, None, None, :]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out(y.transpose(1, 2).reshape(batch, length, width))


class TraceBlock(nn.Module):
    def __init__(self, width: int, heads: int, ffn_width: int):
        super().__init__()
        self.norm_self = nn.LayerNorm(width)
        self.self_attn = Attention(width, heads)
        self.norm_cross = nn.LayerNorm(width)
        self.cross_attn = Attention(width, heads)
        self.norm_ffn = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, ffn_width), nn.GELU(), nn.Linear(ffn_width, width))

    def forward(self, x: Tensor, point_mask: Tensor, context: Tensor, context_mask: Tensor) -> Tensor:
        h = self.norm_self(x)
        x = x + self.self_attn(h, h, point_mask).float()
        x = x + self.cross_attn(self.norm_cross(x), context, context_mask).float()
        return x + self.ffn(self.norm_ffn(x)).float()


class Head(nn.Module):
    def __init__(self, width: int, output_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, output_dim))

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)

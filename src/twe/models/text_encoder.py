from collections import OrderedDict

import torch
from torch import Tensor, nn


class T5TextEncoder(nn.Module):
    def __init__(self, name: str, revision: str, max_length: int, cache_size: int = 4096):
        super().__init__()
        from transformers import AutoTokenizer, T5EncoderModel

        self.tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
        self.model = T5EncoderModel.from_pretrained(name, revision=revision).eval().requires_grad_(False)
        self.max_length = max_length
        self.revision = f"{name}@{revision}"
        self.cache: OrderedDict[str, tuple[Tensor, Tensor]] = OrderedDict()
        self.cache_size = cache_size

    def train(self, mode: bool = True) -> "T5TextEncoder":
        return super().train(False)

    def _encode(self, texts: list[str], device: torch.device) -> list[tuple[Tensor, Tensor]]:
        tokens = self.tokenizer(texts, padding="max_length", max_length=self.max_length, truncation=False,
                                return_tensors="pt")
        if tokens.input_ids.shape[1] > self.max_length:
            lengths = tokens.attention_mask.sum(1).tolist()
            raise ValueError(f"instruction exceeds {self.max_length} tokens: {lengths}")
        ids = tokens.input_ids.to(device)
        mask = tokens.attention_mask.to(device)
        features = self.model(input_ids=ids, attention_mask=mask).last_hidden_state.float()
        return [(features[i], mask[i].bool()) for i in range(len(texts))]

    @torch.no_grad()
    def forward(self, instructions: list[str | None]) -> tuple[Tensor, Tensor, Tensor]:
        device = next(self.model.parameters()).device
        pending = sorted({t for t in instructions if t and t not in self.cache})
        if pending:
            for text, value in zip(pending, self._encode(pending, device)):
                self.cache[text] = value
        width = self.model.config.d_model
        empty = (torch.zeros(self.max_length, width, device=device),
                 torch.zeros(self.max_length, dtype=torch.bool, device=device))
        rows = []
        for text in instructions:
            if text:
                self.cache.move_to_end(text)
            rows.append(self.cache[text] if text else empty)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        features = torch.stack([r[0] for r in rows])
        mask = torch.stack([r[1] for r in rows])
        null = torch.tensor([not t for t in instructions], device=device)
        return features, mask, null

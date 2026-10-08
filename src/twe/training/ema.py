from contextlib import contextmanager

import torch
from torch import nn


class ExponentialAverage:
    def __init__(self, params: list[nn.Parameter], decay: float):
        self.params = list(params)
        self.decay = decay
        self.updates = 0
        self.shadow = [p.detach().clone() for p in self.params]

    def current_decay(self) -> float:
        return min(self.decay, (1 + self.updates) / (10 + self.updates))

    @torch.no_grad()
    def update(self) -> None:
        weight = 1.0 - self.current_decay()
        for shadow, param in zip(self.shadow, self.params):
            shadow.lerp_(param.detach(), weight)
        self.updates += 1

    @contextmanager
    def applied(self):
        backup = [p.detach().clone() for p in self.params]
        with torch.no_grad():
            for shadow, param in zip(self.shadow, self.params):
                param.copy_(shadow)
        try:
            yield
        finally:
            with torch.no_grad():
                for saved, param in zip(backup, self.params):
                    param.copy_(saved)

    def state_dict(self) -> dict:
        return {"decay": self.decay, "updates": self.updates, "shadow": [s.cpu() for s in self.shadow]}

    def load_state_dict(self, state: dict) -> None:
        if len(state["shadow"]) != len(self.shadow):
            raise ValueError("ema state does not match parameters")
        self.updates = state["updates"]
        for shadow, saved in zip(self.shadow, state["shadow"]):
            shadow.copy_(saved)


@contextmanager
def averaged(ema: ExponentialAverage | None):
    if ema is None:
        yield
    else:
        with ema.applied():
            yield

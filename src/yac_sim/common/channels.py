
from __future__ import annotations
import numpy as np

class GilbertElliottChannel:
    """Two-state bursty loss model (Good/Bad) with per-state loss prob."""

    def __init__(self, p_good_to_bad: float, p_bad_to_good: float, loss_good: float, loss_bad: float, rng: np.random.Generator):
        self.p_gb = float(p_good_to_bad)
        self.p_bg = float(p_bad_to_good)
        self.loss_g = float(loss_good)
        self.loss_b = float(loss_bad)
        self.rng = rng
        self.state = 0  # 0=Good, 1=Bad

    def reset(self, state: int = 0) -> None:
        self.state = int(state)

    def step(self) -> None:
        u = float(self.rng.random())
        if self.state == 0:
            if u < self.p_gb:
                self.state = 1
        else:
            if u < self.p_bg:
                self.state = 0

    def deliver(self) -> bool:
        """Return True if a packet is delivered at current state."""
        u = float(self.rng.random())
        loss = self.loss_g if self.state == 0 else self.loss_b
        ok = u >= loss
        self.step()
        return ok

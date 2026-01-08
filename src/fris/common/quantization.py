
from __future__ import annotations
import numpy as np

def uniform_quantize(x: np.ndarray, bits: int, x_min: float = -10.0, x_max: float = 10.0) -> np.ndarray:
    """Uniform mid-tread quantizer (scalar applied elementwise)."""
    x = np.asarray(x, dtype=float)
    if bits is None or bits >= 32:
        return x.copy()
    bits = int(bits)
    levels = 2 ** bits
    # clip
    xc = np.clip(x, x_min, x_max)
    step = (x_max - x_min) / (levels - 1)
    q = np.round((xc - x_min) / step) * step + x_min
    return q

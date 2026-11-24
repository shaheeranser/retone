"""Exposure and white balance expressed as per-channel gains for linear-light RGB."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

EXPOSURE_EV_SPAN = 2.0
TEMPERATURE_EXPONENT = 0.4
TINT_EXPONENT = 0.3
_REC709_LUMA = np.array([0.2126, 0.7152, 0.0722])


def channel_gains(exposure: int, temperature: int, tint: int) -> NDArray[np.float32]:
    """RGB multipliers for linear light.

    White-balance gains are normalised to keep luminance, so temperature and tint
    never change exposure; only the exposure gain does.
    """
    warm = TEMPERATURE_EXPONENT * temperature / 100
    magenta = TINT_EXPONENT * tint / 100
    balance = np.array([2.0**warm, 2.0**-magenta, 2.0**-warm])
    balance = balance / float(balance @ _REC709_LUMA)
    exposure_gain = 2.0 ** (EXPOSURE_EV_SPAN * exposure / 100)
    return np.asarray(balance * exposure_gain, dtype=np.float32)

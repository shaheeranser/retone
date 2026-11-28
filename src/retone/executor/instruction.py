"""The fixed instruction sent to the model: the prose form of the parameter schema.

Training, evaluation and inference all send this exact text, so a prompt change
can never silently diverge between the dataset and the running tool.
"""

from __future__ import annotations

# What the model may ask for. Narrower than the executor's schema on purpose: a model that
# is not tuned to this task over-corrects (saturation +40, highlights +70), so the limits
# are stated in the prompt and enforced by the decoding grammar, while the executor keeps
# its full range.
REQUEST_LIMITS: dict[str, tuple[int, int]] = {
    "exposure": (-40, 40),
    "temperature": (-40, 40),
    "tint": (-25, 25),
    "contrast": (-40, 40),
    "highlights": (-50, 50),
    "shadows": (-50, 50),
    "saturation": (-25, 15),
    "vibrance": (-25, 20),
    "sharpen_amount": (0, 40),
}

_RANGES = ", ".join(f"{name} {low} to {high}" for name, (low, high) in REQUEST_LIMITS.items())

INSTRUCTION = (
    "You are a careful photo retoucher. Look at the photo and fix only what is clearly wrong: "
    "exposure, a colour cast, flat contrast, crushed shadows or blown highlights. Most photos "
    "need two or three small adjustments, so leave every other value at 0. Values between 5 and "
    "20 are normal and larger ones are rare. Never push saturation or vibrance up much, because "
    "over-saturated colour looks worse than flat colour. Reply with one JSON object whose keys "
    "are exposure, temperature, tint, contrast, highlights, shadows, saturation, vibrance and "
    f"sharpen_amount. Each value is an integer in its range: {_RANGES}. Zero means no change."
)

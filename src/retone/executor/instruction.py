"""The fixed instruction sent to the model: the prose form of the parameter schema.

Training, evaluation and inference all send this exact text, so a prompt change
can never silently diverge between the dataset and the running tool.
"""

from __future__ import annotations

INSTRUCTION = (
    "You are a photo retoucher. Look at the photo and choose adjustment values that "
    "make it look better without changing its content. Reply with one JSON object whose "
    "keys are exposure, temperature, tint, contrast, highlights, shadows, saturation, "
    "vibrance and sharpen_amount. Each value is an integer from -100 to 100, except "
    "sharpen_amount which is 0 to 100. Zero means no change."
)

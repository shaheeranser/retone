# 01 — Executor: Design

Read [`../architecture.md`](../architecture.md) §2 and §5 first.

## 1. Purpose

The executor is the single source of truth for what a parameter means. Label
fitting (`02`), evaluation (`04`) and the CLI (`05`) all import it. Because it is
the only code that edits pixels and contains no generative operation, the tool
cannot invent content.

## 2. Interface

```python
apply(image: NDArray, params: Params) -> NDArray
Params.from_mapping(raw: Mapping[str, object]) -> Params  # clamps, defaults, rejects unknown
SCHEMA: ParameterSchema                                   # names, ranges, defaults, identity
SCHEMA.to_json_schema() -> dict                           # grammar for llama-server
```

`Params` is a frozen dataclass of integers. `apply` takes already-validated
`Params`; validation lives in `from_mapping`.

## 3. Operation order

1. Convert to float32 RGB and linear light.
2. Exposure (EV) and white balance (temperature, tint) as channel multipliers in
   linear space.
3. Back to sRGB gamma.
4. Contrast, highlights, shadows through a tone curve.
5. Saturation and vibrance in a perceptual or hue-based space.
6. Unsharp mask on the lightness channel only; radius relative to the diagonal.
7. Clip and convert to the output bit depth (16-bit kept when present).

The order is fixed because exposure and white balance are only correct in linear
light, and sharpening after tone and colour keeps halos from being amplified.

## 4. Why a thumbnail is enough

Per-pixel colour and tone operations depend on the colour and tone distribution
of the image, which area downscaling preserves; for linear operations the result
is exact. Resolution-dependent operations (sharpening) use relative radii. The
thumbnail-validity check in `04` measures the residual.

## 5. Parameter set (starting proposal)

Integers, range -100 to 100 unless noted: `exposure`, `temperature`, `tint`,
`contrast`, `highlights`, `shadows`, `saturation`, `vibrance`, and a sharpening
`amount` (0 to 100) with a relative radius. Coarse bins (steps of 5) are the
fallback if the model's predictions collapse (see `03`). If the fitted ceiling
is too weak, add a parameter such as a tone-curve point.

## 6. Failure modes

- Out-of-range model output: clamped and reported (EXE-SCH-2).
- Extreme parameters clipping heavily: ranges are chosen so identity-adjacent
  values stay safe; augmentation avoids heavy clipping (`02`).
- OpenCV BGR/RGB mix-ups: conversion is isolated in two functions and covered by
  a golden colour-patch test.

## Open decisions

- **OD-01** — Final parameter list, integer ranges, and the mapping from
  integers to physical units (EV per step, kelvin shift per step).
- **OD-02** — Tone-curve formulation for contrast/highlights/shadows (parametric
  curve vs spline with fixed knots), since it affects differentiability if the
  fitter is later ported to PyTorch.
- **OD-03** — Colour space for saturation and vibrance (LAB vs HSV).
- **OD-04** — Colour handling beyond sRGB: assume sRGB input and ignore embedded
  ICC profiles, or convert. Affects ΔE comparability.

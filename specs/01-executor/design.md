# 01 — Executor: Design

Read [`../architecture.md`](../architecture.md) §2 and §5 first.

## 1. Purpose

The executor is the single source of truth for what a parameter means. Label
fitting (`02`), evaluation (`04`) and the CLI (`05`) all import it. Because it is
the only code that edits pixels and contains no generative operation, the tool
cannot invent content.

## 2. Interface

```python
apply(image: NDArray, params: Params) -> NDArray        # uint8/uint16 RGB in, same dtype out
parse_params(raw: Mapping[str, object]) -> ParsedParams  # defaults, clamp report, rejects unknown
SCHEMA: tuple[ParamSpec, ...]                            # names, ranges, defaults, identity
json_schema() -> dict                                    # grammar for llama-server
load_image(path) -> NDArray / save_image(path, image)    # the only BGR <-> RGB boundary
```

`Params` is a frozen dataclass of integers. `ParsedParams` carries the `Params`
and the names that were clamped. `apply` takes already-validated `Params`;
validation lives in `parse_params`. Every operation is skipped exactly when its
parameters are at identity, so identity parameters return the input unchanged.

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

## 5. Parameter set and maths (resolved OD-01 to OD-04)

All integers, default and identity 0, unless noted.

| Parameter | Range | Meaning |
|---|---|---|
| `exposure` | -100..100 | gain `2^(2*v/100)`, so +-2 EV, in linear light |
| `temperature` | -100..100 | red gain `2^(0.4*t)`, blue gain `2^(-0.4*t)`; positive is warmer |
| `tint` | -100..100 | green gain `2^(-0.3*t)`; positive is toward magenta |
| `contrast` | -100..100 | S-curve about 0.5, power `g = 2^(v/100)` |
| `highlights` | -100..100 | adds `0.12*v/100 * m(x)`, `m(x) = 27/4 * x^2 (1-x)` |
| `shadows` | -100..100 | adds `0.12*v/100 * m(1-x)` |
| `saturation` | -100..100 | LAB chroma scale `1 + v/100` (-100 gives grey) |
| `vibrance` | -100..100 | chroma scale `1 + v/100 * (1 - min(C/80, 1))`, boosting low chroma more |
| `sharpen_amount` | 0..100 | unsharp mask on LAB lightness, strength `1.5 * v/100` |

White-balance gains are normalised to keep luminance (Rec. 709 weights), so
temperature and tint do not change exposure. Sharpening radius is not a model
output: it is the fixed fraction `0.0003` of the image diagonal (sigma). The
tone curves act on sRGB-encoded values; the highlight and shadow strengths are
chosen so every curve stays monotonic, which a property test checks. Gamut
clipping after saturation is a per-channel clip. OpenCV's float LAB conversion
uses lookup tables, so a LAB round trip drifts by up to about 0.004 (one 8-bit
level); operations that use LAB are skipped at identity so identity stays exact.

Resolved decisions:

- **OD-01** — the nine parameters and ranges above (owner chose the proposed set,
  without a model-chosen sharpen radius).
- **OD-02** — smooth parametric curves, not splines, so the fitter can be ported
  to PyTorch if needed.
- **OD-03** — LAB chroma for saturation and vibrance.
- **OD-04** — assume sRGB and ignore embedded ICC profiles; warn on stderr when
  an input carries a non-sRGB profile, never convert.

The exact constants (EV span, gain exponents, curve strengths) are starting
values. They may be retuned during T-5.1 hand-tuning, but only before `02`
fits labels, because changing them afterwards invalidates every label.

## 6. Performance (measured 2026-11-24)

On the target laptop (i7-1185G7, 16 GB, other applications holding about 12 GB),
`apply()` on a synthetic 24 MP 8-bit image takes 0.4 s at identity and 11 to 31 s
with every operation active (the spread follows memory pressure), with about 2.1
GB peak RSS. Time is dominated by allocating and filling many 288 MB float
temporaries, not by any single formula. A table-interpolation variant of the
curves was tried and was slower (`np.interp` works in float64), so it was
discarded. Strip-wise processing with threads would likely cut this several
times, but sharpening needs overlap between strips, so it is left as OD-01.

## 7. Failure modes

- Out-of-range model output: clamped and reported (EXE-SCH-2).
- Extreme parameters clipping heavily: ranges are chosen so identity-adjacent
  values stay safe; augmentation avoids heavy clipping (`02`).
- OpenCV BGR/RGB mix-ups: conversion is isolated in two functions and covered by
  a golden colour-patch test.

## Open decisions

- **OD-01** — Render speed on 24 MP photos. Keep the current whole-array
  implementation (simple, 11 to 31 s), or process in horizontal strips with
  overlap and threads (several times faster, more code). Constants stay frozen
  for `02` either way, because strip processing must not change results beyond
  rounding.

OD-01 to OD-04 of the earlier draft (parameter set, tone curve, saturation
space, colour profiles) are resolved in section 5. Constant tuning is allowed
until `02-dataset` starts fitting; after that any change to an operation's
meaning requires regenerating the dataset and is a new decision.

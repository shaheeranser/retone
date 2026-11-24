# 02 — Dataset: Requirements

Depends on `01-executor`.

## Source

- **DAT-SRC-1** — THE SYSTEM SHALL build the dataset from MIT-Adobe FiveK using
  the downsized sRGB version, the standard 4,500 train / 500 test split, and
  Expert C as the target.
- **DAT-SRC-2** — THE SYSTEM SHALL record the dataset's license terms before any
  derived data is published.

## Label fitting

- **DAT-FIT-1** — THE SYSTEM SHALL, per image, find parameters such that
  `apply(original) ≈ expert` measured on 256 px copies, using the `01` executor.
- **DAT-FIT-2** — THE SYSTEM SHALL run fitting in parallel across cores with a
  bounded worker count and a per-image seed, so results are reproducible
  regardless of core count.
- **DAT-FIT-3** — THE SYSTEM SHALL record the final ΔE of each fit so poor fits
  can be filtered.
- **DAT-FIT-4** — WHEN the fitter cannot meet the time budget, THE SYSTEM MAY
  use a PyTorch port of the executor, AND THE SYSTEM SHALL include a test that
  both versions agree within a stated tolerance.

## Augmentation

- **DAT-AUG-1** — THE SYSTEM SHALL create extra examples by applying random
  "bad" parameters to well-edited images and labelling them with the correcting
  parameters.
- **DAT-AUG-2** — THE SYSTEM SHALL avoid augmentations that clip heavily.

## Sharpening (v1)

- **DAT-SHP-1** — THE SYSTEM SHALL set sharpening by a classical rule from a
  sharpness and noise measure on a full-resolution crop, not learned.

## Output

- **DAT-OUT-1** — THE SYSTEM SHALL emit training examples as thumbnail (448 px)
  plus optional 1:1 detail crop plus fixed instruction plus parameter JSON.
- **DAT-OUT-2** — THE SYSTEM SHALL upload the dataset as a private Hugging Face
  dataset, with the token read only from the environment.
- **DAT-OUT-3** — THE SYSTEM SHALL report, on the test set, the mean ΔE
  (CIEDE2000) of fitted parameters versus Expert C at full resolution: the
  quality ceiling of the parameter set.

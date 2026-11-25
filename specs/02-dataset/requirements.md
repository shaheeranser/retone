# 02 — Dataset: Requirements

Depends on `01-executor`.

> **Revised 2026-11-24 (owner decision).** MIT-Adobe FiveK and every comparable
> human-retouched set (PPR10K, MIT5K-derived sets) is licensed for non-commercial
> research only, which a hackathon project cannot satisfy. The dataset is built
> instead from permissively licensed Unsplash photographs with constructed
> (input, target, label) pairs. See `design.md` §1. The old source and fitting
> requirements are superseded; their IDs are not reused.

## Source

- **DAT-SRC-1** — THE SYSTEM SHALL build the dataset from photographs covered by
  the Unsplash License, recording each photo's id, photographer, source URL and
  license in a source manifest.
- **DAT-SRC-2** — THE SYSTEM SHALL record the dataset's license terms and the
  per-image provenance before any derived data is published.
- **DAT-SRC-3** — THE SYSTEM SHALL split photographs deterministically into
  train and test sets by a hash of the photo id and a seed, so the split does not
  depend on file order or worker count.
- **DAT-SRC-4** — THE SYSTEM SHALL collect an optional private set of
  human-edited before/after pairs from a folder tree, recognise the documented
  edit-suffix variants, rotate each edit to the original's orientation, copy
  them into a gitignored work folder and write a provenance manifest. It SHALL
  report rotations, skipped backup directories and unreadable files.

## Pair construction

- **DAT-PAIR-1** — THE SYSTEM SHALL build training pairs from a source photo `T`
  by one of three kinds: identity (`input = T`, label all identity), synthetic
  degradation (`input = apply(T, bad)`, label the inverse correction), or mild
  correction (`input = apply(T, mild)`, label the inverse correction).
- **DAT-PAIR-2** — THE SYSTEM SHALL sample degradation parameters per image from
  a seeded RNG keyed by the photo id and the run seed, so results are identical
  regardless of core count or processing order.
- **DAT-PAIR-3** — THE SYSTEM SHALL reject a sampled degradation when it causes
  more than the allowed fraction of newly clipped pixels, and resample.
- **DAT-PAIR-4** — THE SYSTEM SHALL record a reconstruction residual for each
  pair and drop pairs whose residual exceeds the configured threshold.
- **DAT-PAIR-5** — THE SYSTEM SHALL never label sharpening from a degradation,
  because sharpening cannot be inverted by the executor.
- **DAT-PAIR-6** — WHEN human-edited pairs are used for training, THE SYSTEM MAY
  recover parameter labels by fitting, and SHALL record each fit's residual and
  drop fits above the threshold. The fitting method is OD-05 and is not part of
  this milestone's implementation.

## Augmentation

- **DAT-AUG-1** — THE SYSTEM SHALL treat synthetic degradation as the
  augmentation mechanism and cap it at the configured ratio so examples do not
  drift from realistic corrections.
- **DAT-AUG-2** — THE SYSTEM SHALL avoid augmentations that clip heavily
  (DAT-PAIR-3).

## Sharpening (v1)

- **DAT-SHP-1** — THE SYSTEM SHALL provide a classical sharpening rule from
  sharpness and noise measured on a full-resolution crop, not a learned value;
  v1 training labels keep `sharpen_amount` at identity.

## Output

- **DAT-OUT-1** — THE SYSTEM SHALL emit training examples as a 448 px thumbnail,
  a fixed instruction and the parameter JSON (thumbnail only; no detail crop in
  v1).
- **DAT-OUT-2** — THE SYSTEM SHALL upload the dataset as a private Hugging Face
  dataset, with the token read only from the environment.
- **DAT-OUT-3** — THE SYSTEM SHALL report, on the test set, the mean CIEDE2000
  ΔE between `apply(input, label)` and the target at full resolution, as the
  reconstruction ceiling of the pair construction.
- **DAT-OUT-4** — THE SYSTEM SHALL report per-parameter label variance so label
  collapse is detected before training.
- **DAT-OUT-5** — THE SYSTEM SHALL write a deterministic JSONL manifest plus
  thumbnail images, and produce identical manifests for one worker and many.

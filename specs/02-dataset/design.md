# 02 — Dataset: Design

Read [`../01-executor/design.md`](../01-executor/design.md) first.

## 1. Source and why it changed

The original plan used MIT-Adobe FiveK, whose two license files both grant
rights "solely for your own research purposes" and forbid use "intended for or
directed toward commercial advantage or monetary compensation". A hackathon
submission, a public model link and a public demo do not fit that grant, and the
same restriction applies to PPR10K and every MIT5K-derived retouching set we
checked. The owner therefore replaced the source on 2026-11-24.

The replacement is the Unsplash corpus: the Unsplash License grants an
irrevocable, worldwide, free right to download, modify and use images
commercially, without attribution, and only bars compiling them into a competing
stock service. That is compatible with this project.

```text
Unsplash photo T (well-edited)  ->  target
        |                               ^
        | seeded degradation            | executor.apply
        v                               |
      input  --------------------> correction label
```

No permissive corpus ships human-retouched pairs, so the pairing is constructed.
Three kinds share a photo:

- **identity** — `input = T`, label all zeros. Teaches "leave a good photo
  alone", which is the failure mode of every auto-enhancer.
- **synthetic** — sample `bad` parameters, `input = apply(T, bad)`, label
  `-bad`. This is the augmentation mechanism and teaches large corrections.
- **mild** — the same with small magnitudes, teaching fine adjustments.

Because the label is analytic (the negated degradation), no optimiser is needed:
**OD-02 is resolved — there is no fitting step in v1.** Fitting is only
reintroduced if a future milestone needs true inversion of a clipped
degradation.

## 1a. Optional human-edited pairs

A private folder of the owner's own edits can complement or verify the synthetic
set. The collector pairs `name.ext` with `name(edit...).ext` in the same folder.
It accepts `(edit)`, `(edited)`, a software suffix such as
`(edited_lightroom)`, and a numeric variant written as `(edit)2`,
`(edited)(2)` or `(edited 2)`. Multiple edited versions share one original and
become separate pairs. Originals and edits are copied into separate `before/`
and `after/` folders under the gitignored `artifacts/data/human/`, with a
`pairs.jsonl` manifest; nothing from the owner's library is published. See
[`../data-layout.md`](../data-layout.md).

These edits carry no parameter values, so v1 uses them as an **evaluation set
only**: the model predicts parameters from the before thumbnail, the executor
renders them on the full-resolution before image, and human review judges the
result. No labels are fitted and nothing from this set enters training. Label
recovery and distillation are deferred; if these pairs ever enter training,
OD-05 reopens first. Mathematical ΔE to the human edit is reported as a
diagnostic, not as the acceptance signal.

The original is the orientation reference. When an edit's pixel dimensions are on
the other axis, the collector rotates it back, choosing 90 or 270 degrees by
structural similarity on a small gradient map so the decision survives tone
edits. Directories whose names match `_*` (backup or corrupt folders) are skipped
by default and `--exclude` overrides that. Rotations, skipped directories and
unreadable files are reported in the summary.

## 2. Pipeline

```text
source manifest -> deterministic split -> per photo:
    target = load(full res)
    thumb  = 448 px long edge
    for each kind: sample params -> apply at thumbnail size -> clipping guard
                   -> inverse label -> reconstruction residual -> filter
    -> Example(thumbnail, instruction, params) -> JSONL + images/
-> optional private Hugging Face upload (train entry script)
```

Operations other than sharpening are resolution-independent, so pairs are built
on the thumbnail for speed; the ceiling check re-renders at full resolution.

## 3. Splits and reproducibility

The test fraction is 0.1 by default. Membership is `sha256(f"{seed}:{photo_id}")`
mapped to `[0, 1)`; it depends on neither file order nor worker count.
Per-image RNG seeds use the same digest, so a pair is identical whether built
alone or in a process pool. This satisfies the spirit of the old DAT-FIT-2
without an optimiser.

## 4. Parameters and thresholds (OD-03 resolved)

All defaults live in one frozen `PairPolicy` so they are tunable in one place
before `03-finetune` freezes them:

| Setting | Default | Rationale |
|---|---|---|
| identity / synthetic / mild ratio | 0.3 / 0.5 / 0.2 | Identity guards against over-correction; synthetic dominates because that is the correction being taught; mild teaches restraint. |
| synthetic magnitude | 20..70 | Large enough to be visible, short of the ranges that clip. |
| mild magnitude | 3..15 | Subtle, below typical noise in perceived quality. |
| parameters per degradation | 1..4 | Avoids unrealistic all-at-once distortions. |
| max newly clipped fraction | 0.02 | Keeps the inverse label meaningful. |
| max reconstruction RMS | 0.02 of full scale | Drops pairs the executor cannot invert. |

Sharpening is excluded from deformations: the executor can only sharpen, so no
degradation corresponds to a negative label.

## 5. Sharpening (v1)

`classical_sharpen_amount()` crops the centre of the full-resolution image,
measures Laplacian variance (sharpness) and median-residual noise, and maps them
to `0..100`: low edge energy raises the amount, high noise lowers it. v1 training
labels keep `sharpen_amount = 0`; the CLI applies this rule at inference. A
learned variant stays deferred to `07-extensions`.

## 6. Output format

`examples.jsonl`, one object per line:

```json
{"id": "...-synthetic", "split": "train", "image": "images/....jpg",
 "instruction": "...", "answer": "{\"exposure\": 24, ...}",
 "params": {"exposure": 24, ...}, "kind": "synthetic",
 "source": {"photo_id": "...", "photographer": "...", "url": "...",
            "license": "Unsplash License"},
 "degradation": {"exposure": -24, ...}}
```

`params` is in schema order so the golden file is stable. `degradation` is kept
so the full-resolution ceiling can be reconstructed without refitting. The
instruction is one fixed string (golden-tested).

## 7. Failure modes

- **Labels are not exactly invertible** (clipping, LAB rounding, tone curves):
  guarded by the clipping limit and the residual filter; the residual
  distribution is recorded.
- **Unrealistic degradation** (e.g. -70 on four parameters): capped magnitudes
  and parameter count.
- **Label collapse**: per-parameter variance is reported before training.
- **License drift**: the manifest records the license per photo; the Unsplash
  License is recorded in `architecture.md` §7.
- **Non-reproducible output**: per-photo seeds and sorted emission.

## Open decisions

- **OD-01** — What may be published given the source terms. Resolved 2026-11-24:
  the Unsplash License permits commercial use and modification; derived
  thumbnails and weights may be published. Provenance is retained.
- **OD-02** — Optimiser. Resolved 2026-11-24 for constructed pairs: none; labels
  are analytic. Human-edited pairs are evaluation-only in v1, so no fitting step
  exists; OD-05 reopens only if they ever enter training.
- **OD-03** — Residual threshold and augmentation ratio. Resolved 2026-11-24 with
  the table in §4; retunable until `03-finetune` starts.
- **OD-04** — Detail crop. Resolved 2026-11-24: thumbnail only in v1.
- **OD-05** — Label recovery for human-edited pairs. Resolved 2026-11-24: not
  needed in v1; the pairs are evaluation-only and no labels are fitted. Reopen
  if they enter training; candidate methods remain bounded coordinate descent or
  distillation from a larger model.

The thresholds in §4 were set from the inversion guard and then checked against
measured residuals. Changing them after `03` training starts requires
regenerating the dataset.

First measurement (2026-11-24, 20 owner photos as a stand-in corpus, 60 sampled
pairs): residual median 0.0014, p95 0.015, max accepted 0.019, with 6 pairs
filtered at the 0.02 limit. The 19 held-out examples re-rendered at full
resolution have mean ΔE 0.56 and max 2.06. The limits therefore hold on real
images; they remain unchanged.

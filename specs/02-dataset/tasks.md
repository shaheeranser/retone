# 02 — Dataset: Tasks

Read [`design.md`](./design.md) first.

## 1. Source

- [x] **T-1.1** — Implement the Unsplash source manifest loader and the
      deterministic 90/10 split. *(DAT-SRC-1, DAT-SRC-3)*
- [x] **T-1.2** — Record the Unsplash License and per-image provenance in
      `architecture.md` §7 and the manifest. *(DAT-SRC-2)*
- [x] **T-1.3** — Script fetching a photo subset; commit only a small fixture.
      *(DAT-SRC-1)*
- [x] **T-1.4** — Collect the owner's before/after edits, including software and
      numeric variants, and rotate edits to the original orientation, into a
      gitignored folder with a manifest. *(DAT-SRC-4)*

## 2. Pair construction

- [x] **T-2.1** — Implement seeded degradation and mild-correction sampling with
      a maximum parameter count. *(DAT-PAIR-2, DAT-AUG-1)*
- [x] **T-2.2** — Implement the clipping guard: measure newly clipped pixels and
      resample. *(DAT-PAIR-3, DAT-AUG-2)*
- [x] **T-2.3** — Implement inverse labels and the reconstruction-residual
      filter; exclude sharpening. *(DAT-PAIR-1, DAT-PAIR-4, DAT-PAIR-5)*
- [x] **T-2.4** — Tests: identity pair is exact; labels negate the degradation;
      clipping guard rejects; reproducibility across worker counts.
- [x] **T-2.5** — Record the residual distribution and confirm the thresholds.
      First real run (20 owner photos, 60 pairs): median 0.0014, p95 0.015,
      max accepted 0.019, 6 pairs filtered at the 0.02 limit. *(DAT-PAIR-4;
      OD-03)*
- [ ] **T-2.6** — Deferred: human-edited pairs are evaluation-only in v1, so no
      labels are fitted. Reopens if they enter training. *(DAT-PAIR-6; OD-05)*

## 3. Sharpening

- [x] **T-3.1** — Implement the classical sharpening rule from a full-resolution
      crop, with direction tests. *(DAT-SHP-1)*

## 4. Output

- [x] **T-4.1** — Build 448 px thumbnails and deterministic JSONL examples with
      the fixed instruction and parameter JSON. *(DAT-OUT-1, DAT-OUT-5)*
- [x] **T-4.2** — Upload to a private Hugging Face dataset; token from the
      environment only. Implemented; not executed here (no `HF_TOKEN`).
      *(DAT-OUT-2)*
- [x] **T-4.3** — Compute and record the test-set reconstruction ceiling ΔE.
      19 held-out examples on 20 real photos: mean 0.56, max 2.06.
      *(DAT-OUT-3)*
- [x] **T-4.4** — Report per-parameter label variance for collapse risk.
      *(DAT-OUT-4)*
- [x] **T-4.5** — Golden tests for the instruction and one example record.

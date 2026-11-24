# 02 — Dataset: Tasks

Read [`design.md`](./design.md) first.

## 1. Source

- [ ] **T-1.1** — Download FiveK (downsized sRGB), implement the pair loader and
      the 4,500/500 split with Expert C targets. *(DAT-SRC-1)*
- [ ] **T-1.2** — Record license terms in `architecture.md` §7. *(DAT-SRC-2)*

## 2. Label fitting

- [ ] **T-2.1** — Implement the fitting objective (ΔE between `apply(original)`
      and expert on 256 px copies). *(DAT-FIT-1)*
- [ ] **T-2.2** — Implement the optimiser loop with per-image seeds. *(DAT-FIT-2;
      OD-02)*
- [ ] **T-2.3** — Parallelise with a bounded process pool. *(DAT-FIT-2)*
- [ ] **T-2.4** — Persist fitted params with residual ΔE. *(DAT-FIT-3)*
- [ ] **T-2.5** — Only if too slow: PyTorch executor port and agreement test.
      *(DAT-FIT-4)*
- [ ] **T-2.6** — Tests: fitting recovers known parameters on synthetic pairs;
      reproducible across worker counts.

## 3. Augmentation and sharpening

- [ ] **T-3.1** — Implement bad-parameter augmentation with a clipping guard.
      *(DAT-AUG-1, DAT-AUG-2; OD-03)*
- [ ] **T-3.2** — Implement the classical sharpening rule. *(DAT-SHP-1)*

## 4. Output

- [ ] **T-4.1** — Build training examples with thumbnails, instruction and
      parameter JSON. *(DAT-OUT-1; OD-04)*
- [ ] **T-4.2** — Upload to a private Hugging Face dataset; token from the
      environment only. *(DAT-OUT-2)*
- [ ] **T-4.3** — Compute and record test-set ceiling ΔE. *(DAT-OUT-3)*
- [ ] **T-4.4** — Check per-parameter label variance for collapse risk.

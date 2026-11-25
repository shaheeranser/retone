# 04 — Quantize and evaluate: Tasks

Read [`design.md`](./design.md) first.

## 1. Quantization

- [ ] **T-1.1** — Merge LoRA, convert to GGUF, produce Q4_K_M and Q8_0, keep
      `mmproj` f16. *(QE-QNT-1; OD-01)*
- [ ] **T-1.2** — Document and script launching `llama-server` with the JSON
      Schema grammar. *(QE-QNT-3)*

## 2. Metrics

- [x] **T-2.1** — CIEDE2000 ΔE at full resolution. *(QE-MET-1, QE-MET-3)*
- [x] **T-2.2** — Per-parameter MAE. *(QE-MET-2)*
- [x] **T-2.3** — Tests: ΔE of identical images is zero, known-colour fixtures,
      symmetric where expected.
- [x] **T-2.4** — PSNR and SSIM with a do-nothing baseline. *(QE-MET-4)*

## 3. Benchmark

- [ ] **T-3.1** — Latency and peak RAM harness run on the laptop. *(QE-BEN-1;
      OD-03)*
- [ ] **T-3.2** — Separate model and render timing on a 24 MP photo.
      *(QE-BEN-2)*
- [ ] **T-3.3** — Generate the results table from measured data. *(QE-BEN-3)*

## 4. Thumbnail validity

- [ ] **T-4.1** — Implement and run the thumbnail-validity check; record the
      mean ΔE. *(QE-THM-1; OD-02)*

## 5. Human-edited evaluation

- [x] **T-5.1** — Load `artifacts/data/human/pairs.jsonl`, render predictions on
      the full-resolution before image and score ΔE to the human after.
      *(QE-HUM-1, QE-HUM-2; OD-05)*
- [x] **T-5.2** — Build the HTML/CSV review bundle with original, model and
      reference folders and blind A/B labels. *(QE-HUM-3, QE-HUM-5)*
- [x] **T-5.3** — Wire the review command into `python -m retone.eval`.
      *(QE-HUM-3)*
- [ ] **T-5.4** — Tests and one real review bundle; record the owner's ratings.
      Tests and an identity smoke bundle are done; model predictions and owner
      ratings are pending. *(QE-HUM-4)*

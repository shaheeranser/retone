# 03 — Fine-tune: Tasks

Read [`design.md`](./design.md) first.

## 1. Toolchain gate (do first)

- [ ] **T-1.1** — Run one QLoRA step with image inputs on Qwen2.5-VL-3B, then on
      Qwen3.5-2B, in Unsloth on a GPU host. *(FT-GATE-1)*
- [ ] **T-1.2** — Convert to GGUF plus `mmproj` and run with an image in
      llama.cpp. *(FT-GATE-2)*
- [ ] **T-1.3** — Time one inference on the target laptop. *(FT-GATE-3)*
- [ ] **T-1.4** — Record the gate outcome for both candidates in the run report
      and lock the base model. *(FT-GATE-4; OD-01)*

## 2. Baseline

- [x] **T-2.1** — Prompted base model with a JSON-constrained grammar on the test
      set. Harness implemented (`train/baseline.py` with lm-format-enforcer);
      the GPU run is pending. *(FT-BASE-1)*

## 3. Training

- [x] **T-3.1** — Implement the example formatter (thumbnail, instruction, JSON,
      thinking off). *(FT-TRN-1)*
- [x] **T-3.2** — Implement the QLoRA training script with periodic Hugging Face
      checkpoints. *(FT-TRN-2, FT-TRN-3, FT-TRN-4)*
- [x] **T-3.3** — Add the prediction-variance check and the coarser-bin retrain
      path. *(FT-TRN-5)*
- [x] **T-3.4** — Make the entry script self-contained for any GPU host.
      *(FT-RUN-1, FT-RUN-2, FT-RUN-3)*
- [x] **T-3.5** — Resume from the latest local or Hugging Face checkpoint when
      `--resume` is passed. *(FT-TRN-6)*

## 4. Validation

- [ ] **T-4.1** — Compare fine-tuned ΔE against the baseline on the test set and
      record before/after on unseen images. *(FT-QLT-1)*
- [x] **T-4.2** — Tests: example formatter (golden), schema compliance of target
      JSON, no token reads outside the environment. Offline, no GPU.

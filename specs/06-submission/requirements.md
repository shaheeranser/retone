# 06 — Submission: Requirements

Depends on all earlier milestones. Source of truth for the rules:
`showcase/qwen-local-hackathon-2026/README.md` and `showcase/README.md` in
`junaiddshaukat/finetune-anywhere`, read on 2026-11-24. Re-read them before
submitting; they may change.

## Deadline

- **SUB-DL-1** — THE SYSTEM SHALL open the submission PR before Saturday 28
  November 2026, 23:59 PKT (18:59 UTC). A PR created at or after 19:00:00 UTC is
  late and not judged.

## PR shape

- **SUB-PR-1** — THE SUBMISSION SHALL be one PR from a fork containing only one
  entry folder, `showcase/qwen-local-hackathon-2026/<slug>/`, with a slug of
  lowercase letters, digits and hyphens.
- **SUB-PR-2** — THE ENTRY SHALL contain a completed `model.yaml` and `README.md`
  from `showcase/_template/`, where the slug matches `model.yaml` and `event`
  equals the parent folder.
- **SUB-PR-3** — THE PR SHALL be titled `Showcase: <Project Name>` and use the
  showcase PR template.
- **SUB-PR-4** — THE ENTRY SHALL contain no model weights and no file over 5 MB.

## Evidence (judging criteria, 25% each)

- **SUB-EVD-1** — Usefulness: THE README SHALL state the real problem and who it
  helps.
- **SUB-EVD-2** — Proof the fine-tune helped: THE README SHALL publish actual
  base-versus-tuned outputs on held-out examples, from the same inputs, plus the
  `04` results table.
- **SUB-EVD-3** — Runs locally: THE README SHALL state the hardware, the
  quantization, and measured latency and memory on a normal laptop.
- **SUB-EVD-4** — Demo and write-up: THE ENTRY SHALL link a public demo video,
  and the README SHALL let another person reproduce the run.

## Public artifacts

- **SUB-PUB-1** — THE ENTRY SHALL link a public Hugging Face or ModelScope model
  over HTTPS. THE CODE REPOSITORY SHALL be public.
- **SUB-PUB-2** — THE DATA SHALL be data the author has permission to use, and
  the base model's license SHALL be followed.
- **SUB-PUB-3** — THE AUTHOR SHALL have a ModelScope account to receive rewards.

## Writing

- **SUB-WRT-1** — THE README SHALL follow the starter repo's writing rules:
  direct English, short sentences, no em dash characters, why before
  non-obvious commands, and a source or a reproducible run for each technical
  claim.

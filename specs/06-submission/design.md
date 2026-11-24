# 06 — Submission: Design

## 1. Purpose

Package the work to meet the hackathon's four equal criteria and its mechanical
rules, so a technically good entry is not lost to a format error or lateness.

## 2. Mapping work to criteria

| Criterion | What proves it | Comes from |
|---|---|---|
| Usefulness | one-click enhance, offline, no hallucinated content | `architecture.md` §1 |
| Fine-tune helped | base vs tuned ΔE table and visual before/after on unseen photos | `03`, `04` |
| Runs locally | Q4_K_M/Q8_0 latency and RAM on i7-1185G7, 16 GB, no CUDA | `04` |
| Demo and write-up | public video, reproducible steps | `05`, this milestone |

## 3. Entry contents

`model.yaml` and `README.md` only (plus small images under 5 MB if allowed by
the template). Weights live on a public Hugging Face or ModelScope model; the
README links the model, the code repo, and the demo video.

## 4. Process and timing

The PR is opened from a fork of the starter repo with only the entry folder.
Automated checks run on the PR; fixes go to the same branch. The PR is opened
with margin before the deadline, since eligibility uses the PR's creation time.
Training must finish well before Saturday night.

## 5. Risks

- The plan keeps models and data in private Hugging Face repos, but the
  submission requires a public model link: see OD-01.
- The training corpus is Unsplash-licensed, so the derived dataset and
  fine-tuned weights may be published; see `02-dataset` design §1.
- A vision-language base model may not match the organizers' recommended text
  models: `00-scaffolding` OD-04.
- A late PR is not judged.

## Open decisions

- **OD-01** — Which artifacts become public (merged model, GGUF quants,
  adapter) and where (Hugging Face, ModelScope, both), reconciling the plan's
  "private repos" with the public-link rule.
- **OD-02** — What to publish if the source terms forbid redistributing the
  derived dataset. Resolved 2026-11-24: the Unsplash License permits it, so the
  dataset and weights may be published; provenance is retained.
- **OD-03** — Entry slug and project name, and the demo video format and host.
- **OD-04** — The hard internal deadline for finishing training and opening the
  PR.

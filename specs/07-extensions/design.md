# 07 — Extensions: Design (stub)

Deferred work, started only if time allows after `06-submission`. This file
holds only open decisions; no requirements or tasks exist until the owner
approves a scope.

## Candidates

1. **Desktop GUI.** Zoomable before/after at 100 percent, parameter sliders,
   export. The owner rejects Electron and browser-wrapped desktop apps; the
   tool should work on as many platforms as possible, Linux first. The CLI is a
   thin layer over a library (`architecture.md` §4) so a GUI can reuse it.
2. **Slider overrides.** The user adjusts model-chosen parameters and re-renders
   with the executor. Needs the GUI, or a CLI equivalent (`--set exposure=+10`).
3. **Learned sharpening.** Add sharpening to what the model predicts, if it can
   be done in a small fine-tuning step.
4. **Text instructions.** "Warmer, less contrast" with templated training data;
   deferred unless explicitly pulled in.

## Open decisions

- **OD-01** — GUI toolkit (a native toolkit such as Qt/PySide6, Dear PyGui,
  Slint or Kivy, versus none), judged against "no browser as desktop app" and
  cross-platform packaging.
- **OD-02** — Whether slider overrides ship first as a CLI feature.
- **OD-03** — Definition of learned sharpening (extra output field in the same
  run, or a separate stage) and its go/no-go: proposed to include only if it
  needs no data pipeline beyond `02` and does not delay the `04` results table.
- **OD-04** — Whether text instructions are in scope at all.

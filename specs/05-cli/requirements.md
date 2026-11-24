# 05 — CLI: Requirements

Depends on `01-executor` and `04-quantize-eval`.

## Command

- **CLI-CMD-1** — THE SYSTEM SHALL provide a `retone` command that takes an
  input image path and writes an enhanced image at the original resolution.
- **CLI-CMD-2** — WHEN no output path is given, THE SYSTEM SHALL write beside the
  input with a suffix, and SHALL never overwrite the input.
- **CLI-CMD-3** — THE SYSTEM SHALL support processing several inputs in one
  invocation.
- **CLI-CMD-4** — THE SYSTEM SHALL provide a `--params` option that prints the
  chosen parameters and does not require writing an image.
- **CLI-CMD-5** — THE SYSTEM SHALL provide a way to apply explicit parameters
  without invoking the model.

## Pipeline

- **CLI-PIP-1** — THE SYSTEM SHALL downscale the input to a 448 px thumbnail in
  linear light, query the model, then apply the validated parameters to the
  full-resolution original with the `01` executor.
- **CLI-PIP-2** — WHEN the model output is invalid or out of range, THE SYSTEM
  SHALL clamp or fall back to defaults, report it on stderr, and still produce
  an output.
- **CLI-PIP-3** — WHEN the model server is unreachable, THE SYSTEM SHALL fail
  with a clear message and a non-zero exit code, not a stack trace.
- **CLI-PIP-4** — THE SYSTEM SHALL keep the original file unchanged.

## Output

- **CLI-OUT-1** — THE SYSTEM SHALL write results only to stdout and diagnostics
  only to stderr.
- **CLI-OUT-2** — THE SYSTEM SHALL provide `--json` output of the chosen
  parameters.
- **CLI-OUT-3** — THE SYSTEM SHALL provide a before/after comparison image
  output option.
- **CLI-OUT-4** — THE SYSTEM SHALL use stable exit codes.

## Portability

- **CLI-PRT-1** — THE SYSTEM SHALL run on Linux, macOS and Windows, with Linux
  as the primary target.
- **CLI-PRT-2** — THE SYSTEM SHALL document how to obtain the model and server
  and how to run fully offline.

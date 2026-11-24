# 01 — Executor: Requirements

Acceptance criteria for the deterministic OpenCV executor and the parameter
schema. IDs are stable; tasks cite them.

## Schema

- **EXE-SCH-1** — THE SYSTEM SHALL define one parameter schema listing each
  parameter's name, integer range, default, and identity value.
- **EXE-SCH-2** — WHEN a parameter value lies outside its range, THE SYSTEM
  SHALL clamp it to the range and report which parameters were clamped.
- **EXE-SCH-3** — WHEN a parameter is missing, THE SYSTEM SHALL use its default.
- **EXE-SCH-4** — WHEN an input contains an unknown parameter name or a
  non-integer value, THE SYSTEM SHALL reject the input with a typed error naming
  every offending field.
- **EXE-SCH-5** — THE SYSTEM SHALL export the schema as JSON Schema for use as
  the constrained-output grammar of the model server.

## Apply

- **EXE-APP-1** — THE SYSTEM SHALL expose `apply(image, params) -> image` as the
  only function that edits pixels.
- **EXE-APP-2** — WHEN all parameters equal their identity values, THE SYSTEM
  SHALL return an image equal to the input within one quantization step.
- **EXE-APP-3** — THE SYSTEM SHALL never modify the input array.
- **EXE-APP-4** — THE SYSTEM SHALL apply operations in this fixed order: linear
  conversion, exposure and white balance, gamma encode, tone curve, saturation
  and vibrance, unsharp mask on lightness, clip and output conversion.
- **EXE-APP-5** — WHEN the input is 16-bit, THE SYSTEM SHALL return 16-bit
  output; WHEN it is 8-bit, 8-bit output.
- **EXE-APP-6** — THE SYSTEM SHALL compute internally in float32 and RGB channel
  order, converting from OpenCV's BGR only at load and save.
- **EXE-APP-7** — THE SYSTEM SHALL define sharpening radius as a fraction of the
  image diagonal and convert it to pixels at apply time.
- **EXE-APP-8** — THE SYSTEM SHALL be deterministic: the same input and
  parameters give bit-identical output.
- **EXE-APP-9** — THE SYSTEM SHALL contain no generative operation.

## Scale and boundaries

- **EXE-SCL-1** — THE SYSTEM SHALL produce a visually matching result when the
  same parameters are applied to an image and to its downscaled copy, with the
  difference measured and reported by the thumbnail-validity check (`04`).
- **EXE-BND-1** — THE `executor` package SHALL import nothing else from the
  project and no ML framework.

## Quality

- **EXE-QLT-1** — WHEN hand-tuned parameters are applied to a set of 20 test
  photos, THE SYSTEM SHALL make them look clearly better, judged by the owner
  and recorded in the run report.

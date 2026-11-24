# Contributing

Thanks for taking the time to contribute. This document covers how work is done
on `retone` and how to get a change merged.

## Before you open a PR

**Open an issue first for anything non-trivial.** A new adjustment operation, a
change to the parameter schema, or anything touching a module boundary should be
discussed in an issue before code is written, because it may conflict with the
design in `specs/`.

Small, obviously-correct fixes (typos, a clear bug with an obvious fix, a
missing test for existing behaviour) can go straight to a PR.

## Working in this repository

Read [`AGENTS.md`](./AGENTS.md) and the standards it links:
[`specs/style-guide.md`](./specs/style-guide.md) for how code reads and
[`specs/agent-practices.md`](./specs/agent-practices.md) for how work is done —
spec-first, open decisions, commit hygiene, run reports. These apply to human
contributors as much as to agents.

Build, test and lint commands are listed in `AGENTS.md` once `00-scaffolding`
fixes them; all of them must pass.

Enable the commit-message hook:

```sh
git config core.hooksPath scripts/hooks
```

## Tests

- New behaviour needs tests. Bug fixes need a regression test that fails before
  the fix.
- The executor's invariants are covered by property-based tests.
- Tests run offline and deterministically; the real model is never required.

## Submitting a pull request

1. Fork the repository and branch from `main`.
2. Make your change, including tests and any spec updates it implies.
3. Run the build, lint and test commands from `AGENTS.md`.
4. Open the PR and fill in the checklist. Link the issue it addresses.
5. Keep the PR focused. Unrelated refactors belong in their own PR.

## What gets closed

Low-effort and spam pull requests are closed rather than merged: reformat-only
changes without a reason, dependency bumps with no explanation, new behaviour
without tests, and anything that ignores the design in `specs/`.

## License

By contributing, you agree that your contributions are licensed under the MIT
License (see [LICENSE](./LICENSE)).

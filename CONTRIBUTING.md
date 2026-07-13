# Contributing

Contributions are welcome when they keep the runtime understandable, measurable, and honest about evidence.

## Development setup

Use Python 3.12 or newer. On PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
```

On POSIX, use `python3.12 -m venv .venv` and `.venv/bin/python`.

## Change guidelines

- Keep each pull request focused on one coherent change.
- Add a failing regression test before a bug fix.
- Preserve public APIs intentionally and document breaking changes.
- Update architecture or evaluation documents when runtime behavior or a claim boundary changes.
- Mark a feature as default, opt-in, library-only, evaluation-only, or backlog; “implemented” alone is not enough.
- Do not add generated traces, credentials, local paths, virtual environments, raw model transcripts, or unreviewed benchmark outputs.
- Review new dependencies, fixtures, prompts, and comparative references for provenance and licensing.

New agent subsystems should include a source-grounded design, implementation review, module contract, deterministic failure tests, and an evaluation plan. Avoid adding breadth that cannot be exercised or explained.

## Evaluation changes

Keep experimental design separate from results. Every result report should include:

- a glossary and plain-language TL;DR;
- exact conditions, commands, code version, and randomness/repeat settings;
- the controlled baseline and treatment when making a comparative claim;
- raw evidence and grader/protocol identity;
- `PASS`, `FAIL`, `INVALID`, `INCONCLUSIVE`, `ERROR`, or another declared status used consistently; and
- the credible boundary and what the result does not prove.

Do not turn a synthetic fixture result into a live-model, product-benefit, or official benchmark claim.

## Pull request checklist

- [ ] Tests cover the intended behavior and relevant failure path.
- [ ] `python -m pytest` passes; platform skips are explicit.
- [ ] Nested evaluation contract tests pass when their code changed.
- [ ] `python scripts/regression_gate.py offline` does not produce a false partial pass.
- [ ] Public docs and examples contain no machine-specific paths, private-source anchors, or secrets.
- [ ] New dependencies and external data have compatible notices and terms.
- [ ] User-facing claims link to evidence and state what that evidence cannot establish.

# SWE-bench adapters

This directory contains runners and result checkers. It does not distribute
SWE-bench problem statements, gold patches, test patches, Docker images, or
historical model predictions.

## Data boundary

Every runner requires an explicit `--instances` JSON file exported from the
benchmark distribution you are authorized to use. The file must be a JSON list
whose rows contain non-empty string values for:

- `instance_id`
- `repo`
- `base_commit`
- `problem_statement`
- `patch`
- `test_patch`

The loader rejects duplicate IDs and incomplete rows before starting Docker or
making a model call. Paths may point outside this repository; benchmark data
should remain outside Git.

Files under `suites/` are project-owned selection manifests. They contain IDs,
not benchmark rows:

```json
{
  "schema": "ace.swebench-suite.v1",
  "name": "example-suite",
  "source_dataset": "SWE-bench Verified",
  "external_dataset_required": true,
  "instance_ids": ["django__django-11087"]
}
```

When `--suite` is supplied, `eval.swebench.data` joins those IDs to the external
dataset, preserves manifest order, and fails if an ID is absent.

## Common commands

Run a repeatable resolved probe over an ID-only suite:

```powershell
python -m eval.swebench.run_resolved_probe `
  --instances C:\bench-data\swebench_verified.json `
  --suite eval\swebench\suites\verified_local38.json `
  --tag my-baseline --model-id <model-id> --repeat 3
```

Run one local proxy task or only validate clone access:

```powershell
python -m eval.swebench.run_swe `
  --instances C:\bench-data\swebench_lite.json `
  --clone-only django__django-11087
```

Other runners use the same explicit boundary:

```powershell
python -m eval.swebench.run_batch --instances C:\bench-data\swebench_lite.json 5
python -m eval.swebench.variance_probe --instances C:\bench-data\swebench_verified.json
python -m eval.swebench.probe_reach --instances C:\bench-data\swebench_verified.json --all
python -m eval.swebench.session_run --instances C:\bench-data\swebench_verified.json
```

Use `python -m <module> --help` for all options. The verification probe uses a
separate `--cases` file because it also needs caller-supplied prediction paths;
its benchmark rows still come only from `--instances`.

## Evidence boundary

`eval.swebench.checker` validates already-produced normalized artifacts. Its
bundled fixture is synthetic and is never benchmark evidence. Real resolved-rate
claims require the official harness, a fixed ID-only suite, complete coverage,
and a same-protocol repeated baseline.

# EvoClaw adapter and orchestration entry points

This directory connects the coding agent to an external EvoClaw checkout for
long-horizon compression and memory evaluations. The adapter code is versioned
here; `deploy.sh` copies and registers it in a checkout supplied by the caller.

These scripts are provider-neutral and machine-neutral. They do not load a
credential file, infer a local virtual environment, or choose a model/endpoint.
Export the required values in the current shell before starting paid work:

```bash
export EVOCLAW=/path/to/evoclaw
export EVOCLAW_PYTHON=python3
export EVOCLAW_REPO_NAME=<dataset-repository-id>
export EVOCLAW_IMAGE=<benchmark-image>
export EVOCLAW_DATA_ROOT=/path/to/dataset-root
export ANTHROPIC_API_KEY=<credential>
export ANTHROPIC_BASE_URL=<anthropic-compatible-endpoint>
export MODEL_ID=<model-id>
```

`env.sh` validates those exports. It never discovers or reads a file and never
prints the credential value, prefix, or length.

## Entry-point map

| Entry point | Capability | External effects |
|---|---|---|
| `deploy.sh` | install/register `myagent.py`; optional endpoint whitelist and CPU cap | modifies the supplied EvoClaw checkout |
| `run_chain.sh` | one chain/trial | provider calls, containers, trial data |
| `run_reps.sh` | detached repeat-N for one arm | provider calls; logs/PIDs under ignored results |
| `run_three_arms.sh` | none/pipeline/truncate run and collection | three trials plus traces/verdicts/curves |
| `run_sm_arms.sh` | pipeline without/with SessionMemory | two trials plus traces/verdicts/curves |
| `run_fork_seed.sh` | stop-at-context seed | detached trial for later cloning |
| `run_fork_resume.sh` | resume `fork_full` or `fork_sm` | provider-backed continuation |
| `pull_traces.sh` | copy JSONL traces from a named container | writes ignored local evidence |
| `smoke_container.sh` | adapter install, live run, durable session, resume | one container and provider calls |

Every entry point supports `--help` without checking Docker, data, credentials,
or network access. Generated artifacts live under `eval/evoclaw/results/` by
default and are ignored by Git.

## Typical flow

```bash
bash eval/evoclaw/deploy.sh "$EVOCLAW" --allow-domain <endpoint-host>

COMPACT_STRATEGY=none \
TRIAL_NAME=baseline_001 \
bash eval/evoclaw/run_chain.sh

bash eval/evoclaw/run_three_arms.sh run comparison_001 6 400
bash eval/evoclaw/run_sm_arms.sh run session_memory_001 6 400
```

Use a unique tag for each run. The multi-arm scripts refuse to overwrite a
discovered trial. `collect` mode is deterministic and does not make provider
calls:

```bash
bash eval/evoclaw/run_three_arms.sh collect comparison_001
```

## Fork-at-compact flow

Choose the stop threshold explicitly; it is part of the experiment protocol:

```bash
export MYAGENT_STOP_AT_CONTEXT=<positive-token-threshold>
TAG=fork_seed_001 bash eval/evoclaw/run_fork_seed.sh

python eval/evoclaw/prepare_fork_at_compact.py --help
bash eval/evoclaw/run_fork_resume.sh fork_full <prepared-trial-root>
bash eval/evoclaw/run_fork_resume.sh fork_sm <prepared-trial-root>
```

## Boundaries

- `deploy.sh` patches known EvoClaw registration anchors. Upstream layout drift
  fails closed instead of silently claiming deployment success.
- Endpoint DNS/proxy routing is an operator responsibility. The public scripts
  do not perform automatic DNS lookups or inject machine-specific IPs.
- Live scripts prove wiring only when their produced evidence is reviewed. A
  successful `--help`, import, or container start is not a benchmark result.
- Do not publish generated traces or tool-result payloads without a separate
  secret and data-rights review.

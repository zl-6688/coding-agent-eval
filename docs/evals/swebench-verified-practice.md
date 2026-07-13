---
title: SWE-bench Verified Engineering Practice
type: eval-report
status: complete
created: 2026-07-13
updated: 2026-07-13
authors: [zl-6688]
summary: Multi-round official-harness results, failure diagnosis, runtime fixes, and a Claude Code harness/model-boundary probe.
refs:
  - "evidence/swebench-local38-summary.json"
  - "../../eval/swebench/run_resolved_probe.py"
  - "../../scripts/regression_gate.py"
---

# SWE-bench Verified Engineering Practice

[English](./swebench-verified-practice.md) | [简体中文](./swebench-verified-practice.zh-CN.md)

## TL;DR

This work was not a single `19/38` run. It was a sequence of official-harness experiments:

1. DeepSeek V4 Flash resolved **16/38** in the first full-suite pass.
2. After fixing patch capture and strengthening the harness prompt, the 22 unresolved cases were rerun; **5/22** new cases resolved, bringing the two-stage Flash campaign to **21/38 unique cases covered**.
3. DeepSeek V4 Pro then ran the updated harness on all 38 cases in one fresh pass and resolved **19/38**.
4. Official artifacts split the 19 Pro failures into `10` incomplete target fixes, `5` incomplete fixes with regressions, and `4` no-patch runs.
5. Trace diagnosis exposed a Windows/Docker edit-transport bug; the targeted fix removed `WinError 206` from **5/5 to 0/5** and moved structured partial-edit adoption from **1/5 to 5/5**.
6. Full native Claude Code 2.1.207 with DeepSeek V4 Flash resolved **2/8** on a fixed hard slice; explicit max effort resolved **1/8** and rescued none of the six prior failures.

The `21/38` Flash figure is cumulative campaign coverage after a selected failure-pool rerun, not a fresh single-pass rate. It must not be compared directly with the Pro `19/38` single-pass result.

## Glossary

| Term | Definition |
|---|---|
| Agent harness | The coding-agent runtime: model loop, tools, prompt, editing path, context, and validation behavior. |
| Evaluation harness | SWE-bench containers, official tests, scorer, and result aggregation. |
| Official resolved | The official scorer accepts both required bug-fix tests and regression tests. |
| `FAIL_TO_PASS` | Tests that must change from failing on the base revision to passing after the patch. |
| `PASS_TO_PASS` | Existing tests that must remain passing. |
| Campaign coverage | Unique suite cases resolved across declared sequential stages. It is not a one-attempt pass rate. |
| False-green | The agent's local check looks green while the official harness still reports unresolved. |
| L3 gate | A fixed-suite, repeat-N benchmark check against a baseline generated with the same protocol. |

## Experiment timeline

All local38 stages used the same 38 SWE-bench Verified instance IDs. Conditions changed between stages, so each result is labeled by its role rather than flattened into one model ranking.

| Stage | Model and condition | Official result | Correct interpretation |
|---|---|---:|---|
| Flash full-suite pass | DeepSeek V4 Flash; earlier short-constraint prompt and harness | **16/38** | One complete single-pass baseline |
| Flash selected retry | Same model; patch-capture fix and expanded prompt; only the previous 22 failures | **5/22** new resolutions | Harness-iteration recovery on a selected failure pool |
| Flash campaign rollup | Union of first-pass successes and newly rescued cases | **21/38** unique | Cumulative coverage, not one-pass rate |
| Pro full-suite pass | DeepSeek V4 Pro; updated harness/prompt; fresh all-38 run, `max_turns=50` | **19/38** | Main comparable single-pass Pro result |
| Pro budget diagnostic | Selected retries with a 100-turn budget merged into the Pro view | **22/38** composite | Diagnostic convergence view, not a fresh all-38 baseline |

The multi-stage structure is itself an engineering result: the project changed the harness, reran the unresolved pool, then performed a fresh full-suite run with a different model. It did not silently relabel selected retries as a stronger single-run score.

## Flash campaign: 16 first-pass + 5 rescued = 21 unique

The first Flash pass produced 16 official resolutions. The follow-up reran exactly the 22 unresolved cases after two harness changes: untracked-file patch capture was fixed, and the task prompt/verification guidance was expanded. Five previously unresolved cases passed the official scorer:

- `astropy__astropy-14369`
- `matplotlib__matplotlib-25775`
- `sphinx-doc__sphinx-10673`
- `sympy__sympy-13031`
- `sympy__sympy-13091`

The sets do not overlap, so the campaign covered 21 unique cases. This supports a bounded claim that harness iteration recovered five additional cases on the same suite. It does not support a claim that Flash had a `21/38` fresh-pass rate or that it outperformed Pro.

## Pro full-suite result and official failure taxonomy

DeepSeek V4 Pro resolved **19/38 (50.0%)** in a fresh full-suite run under the updated harness. The useful work began after the score: every unresolved case was traced back to official `report.json` and `test_output.txt` artifacts.

| Official-evidence bucket | Count | Engineering meaning |
|---|---:|---|
| `FAIL_TO_PASS_STILL_FAILING` | 10 | A patch existed, but required target behavior was still failing. |
| `INCOMPLETE_PLUS_REGRESSION` | 5 | The fix remained incomplete and also broke behavior that previously passed. |
| `NO_PATCH` | 4 | The run ended without a scoreable solution patch. |
| **Total unresolved** | **19** | Exhaustive partition of the official unresolved set |

Twelve of the 19 unresolved runs hit the turn cap. Fifteen produced a patch, and 13 of those patches touched at least one gold-patch file. The dominant problem was therefore not simply file localization: many runs reached the correct neighborhood but missed the semantic contract, widened a shared behavior too far, or failed to converge before the budget ended.

## Trace-to-runtime closed loop

Trace analysis found a concrete harness defect in the large-file editing path. Full base64 content was placed in the host-side `docker exec` argument list, which hit the Windows command-line limit and raised `[WinError 206]`. The agent then fell back to shell-based source edits.

The fix streamed exact bytes through `docker exec -i`. On the same five targeted sentinels:

| Metric | Before | After |
|---|---:|---:|
| Runs with `WinError 206` | 5/5 | 0/5 |
| Runs adopting structured partial edit | 1/5 | 5/5 |
| Runs falling back to shell source edits | 5/5 | 0/5 |
| Gold-viable official resolved | 1/4 | 3/4 |

The strongest causal evidence is the known transport mechanism plus the deterministic disappearance of the error. The movement from 1/4 to 3/4 is supporting small-sample evidence, not a population-level solve-rate claim.

## Claude Code harness + DeepSeek boundary probe

A fixed eight-case hard slice was then executed with full native Claude Code 2.1.207 inside the official SWE-bench containers, using DeepSeek V4 Flash and the native Agent/Task/Skill/tool lifecycle. No gold patch, hidden test names, or custom runner hint was provided.

| Condition | Official resolved | Key observation |
|---|---:|---|
| Thinking enabled, no explicit max-effort setting | **2/8** | Mature harness organization and long-chain validation did not solve six hard cases |
| Explicit max effort | **1/8** | No old failure was rescued; one previous pass regressed across rounds |

This supports a deliberately bounded conclusion:

- Harness design matters: it determines how the model explores, edits, validates, and converges.
- Harness completeness is not a sufficient explanation for the remaining hard failures. Even a mature native harness left six cases unresolved with the same model.
- More effort is not monotonically better. Extra analysis can delay editing or widen the patch without satisfying the hidden semantic contract.
- The residual bottleneck on this selected slice includes model semantic understanding and convergence, while the local harness still has its own improvable failure modes.

This is not a synchronized head-to-head benchmark between this repository and Claude Code, and `n=8, k=1` cannot estimate either system's overall SWE-bench Verified performance.

## Evidence hierarchy and observability

The postmortem always reads evidence in this order:

1. official `report.json` for resolution and regression status;
2. official `test_output.txt` for the failing contract;
3. agent traces for search, tool, edit, and validation behavior;
4. runner-side labels only as navigation hints.

This order corrected early observation errors and made false-green measurable. A local test `PASS`, patch-file overlap, more turns, or a larger diff never overrides the official outcome.

The public, path-free aggregate is [swebench-local38-summary.json](evidence/swebench-local38-summary.json). Raw transcripts, third-party problem text, gold patches, and machine-specific harness paths are excluded.

## L3 repeat-N boundary

The repository implements repeat-aware result recording and same-protocol checking through [`run_resolved_probe.py`](../../eval/swebench/run_resolved_probe.py) and [`regression_gate.py`](../../scripts/regression_gate.py). Historical single runs and selected retries remain diagnostic evidence; they are not silently promoted into a repeat-3 release baseline.

The remaining operational step is a paid repeat-3 run on the frozen final protocol. Until then, this report demonstrates official-harness practice, iteration, and failure diagnosis—not a stable leaderboard estimate.

## Supported conclusions

- The project completed official SWE-bench Verified scoring on a fixed 38-case slice and retained a traceable multi-stage result history.
- Harness changes recovered five additional Flash failures in a selected rerun, producing 21/38 cumulative unique campaign coverage.
- A fresh Pro run produced 19/38 and an exhaustive official-artifact failure taxonomy.
- Trace evidence led to a concrete runtime transport fix and a successful same-sentinel recheck.
- The Claude Code + DeepSeek probe shows that a mature harness is necessary but not sufficient for the selected hard cases; model semantic/convergence limits remain material.

## What this report does not prove

It does not establish full-benchmark or leaderboard performance, a stable Flash-versus-Pro ranking, a population-level benefit from the edit-transport fix, universal superiority of either harness, or a formal repeat-3 release baseline. The Flash `21/38`, Pro `19/38`, Pro diagnostic `22/38`, and Claude Code `2/8`/`1/8` numbers have different protocols and are never merged into one comparative rate.

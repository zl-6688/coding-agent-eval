# Provenance and Publication Scope

[English](./provenance.md) | [简体中文](./provenance.zh-CN.md)

This repository is an educational and engineering project for building, observing, and evaluating a coding-agent harness.

## Design references

Some agent-runtime module designs draw on publicly available Claude Code technical documentation and learning materials. Those ideas were adapted, localized, implemented, and extended inside this project's runtime.

The observability and evaluation systems are separate project work. They are built around general tracing and evaluation techniques, including OpenTelemetry, OpenInference, Phoenix, controlled A/B experiments, and official benchmark interfaces.

## What the public repository contains

The public snapshot keeps the implemented capability surface rather than reducing it to a demo loop:

- the agent runtime, tools, sessions, context management, memory, skills, subagents, tasks, and optional MCP integration;
- JSONL/HTML/OpenTelemetry observability code;
- deterministic regressions plus context-compression, AutoMemory, EvoClaw, and SWE-bench evaluation harnesses;
- unit and integration tests for the published implementation; and
- concise bilingual architecture, evaluation, reproducibility, security, and contribution documents.

The source is aligned with the current development implementation, with small publication adaptations for portable paths, example configuration, privacy, and repository hygiene.

## What is intentionally excluded

- credentials, local environment files, generated traces, and model transcripts;
- raw third-party benchmark datasets, gold patches, containers, and problem text;
- large paid-run outputs and sample-level model content not selected as public evidence;
- internal planning, handoff, source-analysis, and process-heavy research notes; and
- local Git history and unrelated development artifacts.

External datasets and tools are obtained separately under their own terms. Bundled SWE-bench suite manifests contain identifiers only; synthetic fixtures are used only for local contract checks.

## Evaluation snapshots

Public reports preserve selected aggregate results and the protocol used at the time of each experiment. The runtime can evolve after a valid experiment; the report remains evidence for the measured snapshot, while new claims require a new run.

Generated evidence is kept separate from implementation code. Public aggregate JSON files use allowlisted fields and may include SHA-256 fingerprints of the internal artifacts used to prepare them, without publishing raw transcripts or machine-specific paths.

## Publication hygiene

The export excludes machine-local paths, credential-shaped literals, raw traces, and unselected benchmark outputs. Automated checks live in [`tests/test_public_release_policy.py`](../tests/test_public_release_policy.py), and dependency notices are recorded in [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

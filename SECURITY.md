# Security Policy

## Supported versions

Security fixes are applied to the latest `0.1.x` release and the default branch. Earlier development snapshots are not supported.

## Reporting a vulnerability

Please use the repository's private GitHub Security Advisory form instead of opening a public issue. Include:

- the affected version or commit;
- a minimal reproduction;
- expected and observed behavior;
- likely impact and required preconditions;
- any suggested mitigation.

Do not include real credentials, private model transcripts, or unrelated user data. Use synthetic examples and redact machine-specific paths.

Maintainers will acknowledge a complete report, reproduce it, assess severity, and coordinate a fix and disclosure. Exact response dates are not guaranteed for this volunteer-maintained project.

## Security boundaries

This project executes tools and may invoke local subprocesses. Permission rules and read-before-write checks reduce accidental misuse, but they are not an operating-system sandbox. Run the agent only in repositories and environments you are prepared to modify, use least-privilege credentials, and review enabled MCP servers.

The offline evaluation gate checks deterministic mechanism contracts and evidence integrity. It is not a penetration test, supply-chain attestation, or guarantee of safe autonomous operation.

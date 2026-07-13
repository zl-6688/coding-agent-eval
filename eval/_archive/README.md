# Archived evaluation code

This directory preserves historical experiment code for traceability. It is
not part of the default runtime, installed package, or offline release gate,
and it is not evidence for current performance claims.

The compact-evaluation CLI modules retain offline `--help` discovery and may
require provider credentials for live execution. Historical diagnostic files
whose names begin with `test_` are snapshots of earlier contracts; they are
not collected by the root test suite and may not match the current runtime
API. Maintained context and compression evaluation entrypoints live in
[`eval/context_eval`](../context_eval/) and
[`eval/compression_eval`](../compression_eval/).

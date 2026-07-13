# Third-Party Notices

The project license does not replace the licenses of separately installed dependencies. This repository does not vendor the packages below; installers may resolve additional transitive dependencies under their own licenses.

| Component | Declared use | License | Upstream |
|---|---|---|---|
| Anthropic Python SDK (`anthropic`) | Core provider client | MIT | [source and license](https://github.com/anthropics/anthropic-sdk-python) |
| python-dotenv (`python-dotenv`) | `.env` loading | BSD-3-Clause | [source and license](https://github.com/theskumar/python-dotenv) |
| prompt_toolkit (`prompt-toolkit`) | Interactive terminal input | BSD-3-Clause | [source and license](https://github.com/prompt-toolkit/python-prompt-toolkit) |
| Rich (`rich`) | Terminal rendering | MIT | [source and license](https://github.com/Textualize/rich) |
| MCP Python SDK (`mcp`) | Optional MCP runtime | MIT | [source and license](https://github.com/modelcontextprotocol/python-sdk) |
| OpenTelemetry Python | Optional trace export | Apache-2.0 | [source and license](https://github.com/open-telemetry/opentelemetry-python) |
| Arize Phoenix (`arize-phoenix`) | Optional local trace and experiment backend | Elastic-2.0 | [source and license](https://github.com/Arize-ai/phoenix) |
| HTTPX (`httpx`) | Test support | BSD-3-Clause | [source and license](https://github.com/encode/httpx) |
| pytest (`pytest`) | Test runner | MIT | [source and license](https://github.com/pytest-dev/pytest) |
| setuptools | Build backend | MIT | [source and license](https://github.com/pypa/setuptools) |
| EvoClaw | External evaluation harness and adapter contract; not bundled | MIT | [source and license](https://github.com/EvoClaw-Bench/EvoClaw) |

The version constraints are defined in [`pyproject.toml`](pyproject.toml). Consult the metadata installed in a concrete environment for the exact version and transitive dependency set.

## Comparative products

Public Claude Code technical documentation and learning materials informed selected agent-runtime design concepts; Claude Code is not a bundled dependency. The observability and evaluation layers are implemented in this project around the separately listed open tracing libraries and external benchmark interfaces.

## Evaluation adapters and data

The repository contains adapters and checkers that can operate with external
projects such as SWE-bench and EvoClaw. The EvoClaw adapter implements its
public `AgentFramework` contract and is deployed into a separately obtained
EvoClaw checkout. External datasets, problem statements, gold patches, images,
and harness environments are not vendored in this release and remain governed
by their upstream terms. Bundled SWE-bench suite manifests contain instance
identifiers only; users provide an authorized dataset copy at runtime.

Bundled toy tasks and explicitly synthetic integration/checker fixtures are project-owned test data. A synthetic checker fixture is not an official benchmark sample or score.

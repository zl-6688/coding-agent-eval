"""SessionMemory PaymentService edit-behavior probe.

This harness creates a multi-stage coding-session state, compacts the same
pre-compact messages through SessionMemory and full_compact, then asks the
agent to patch a real fixture test file.  The grader reads files and runs
pytest; it does not trust model self-report.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from agent import config, loop
from agent.context import compact
from agent.context.compact import CompactConfig
from agent.memory import session_memory as smmod
from agent.memory.forked_agent import ForkResult
from agent.memory.session_memory import SESSION_MEMORY_TEMPLATE, SessionMemory
from eval.compression_eval.sm_long_task_behavior import _merge_consecutive_same_role
from eval.compression_eval.sm_long_task_fidelity import (
    LONG_TASK_TAIL_FACT,
    _Response,
    _all_span_attrs,
    _fact_survives,
    _last_span_attrs,
    _message_text,
    _state_hash,
)
from obs import trace as trace_mod
from obs.trace import JsonlSink, SpanKind, span


PAYMENT_PROBE_ID = "paymentservice-edit"
PAYMENT_TEST_PATH = Path("tests/test_payment_service.py")
PAYMENT_SERVICE_PATH = Path("services/payment.py")
BILLING_PATH = Path("services/billing.py")
INVENTORY_PATH = Path("services/inventory.py")
RUNBOOK_PATH = Path("docs/runbook.md")

PAYMENT_RULE = (
    "PaymentService unit tests must use a StubGateway test double with endpoint "
    "disabled://payment-gateway, and must not instantiate the real PaymentGateway."
)
PAYMENT_DISTRACTOR_RULE = (
    "BillingGateway belongs to legacy invoice tests, FakeGateway belongs to inventory tests, "
    "and GatewayMock is stale QA-sandbox advice; none of those should be copied into "
    "PaymentService unit tests."
)
PAYMENT_BATCH_RULE = (
    "Importer batch BETA-27 already completed parse, normalize, and validate; only "
    "reconcile-ledger remains."
)
PAYMENT_TAIL_RULE = (
    "After the last SessionMemory extraction, the file-scope guardrail changed: edit only "
    "tests/test_payment_service.py and do not change service implementation files."
)

PAYMENT_EDIT_TASK = """Continue the compressed coding session.

Task: fix the PaymentService unit test in this workspace according to the remembered testing convention.

Constraints:
- Edit only tests/test_payment_service.py.
- Do not copy unrelated gateway examples from billing, inventory, or old QA notes.
- Keep the production service implementation unchanged.
- When done, summarize the patch in one sentence.
"""

PAYMENT_SERVICE_FIXTURE = """class PaymentGateway:
    def __init__(self, endpoint="https://payments.example/live"):
        self.endpoint = endpoint

    def charge(self, amount):
        if not self.endpoint.startswith("disabled://"):
            raise RuntimeError(f"real gateway call blocked for {self.endpoint}")
        return {"status": "stubbed", "amount": amount, "endpoint": self.endpoint}


class PaymentService:
    def __init__(self, gateway=None):
        self.gateway = gateway or PaymentGateway()

    def checkout(self, amount):
        receipt = self.gateway.charge(amount)
        return {
            "status": receipt["status"],
            "amount": receipt["amount"],
            "endpoint": receipt["endpoint"],
        }
"""

PAYMENT_TEST_FIXTURE = """from services.payment import PaymentGateway, PaymentService


def test_checkout_records_payment_receipt():
    service = PaymentService(PaymentGateway(endpoint="https://payments.example/live"))

    receipt = service.checkout(42)

    assert receipt["status"] == "stubbed"
    assert receipt["amount"] == 42
"""

BILLING_FIXTURE = """class BillingGateway:
    def charge_invoice(self, invoice_id):
        return {"invoice_id": invoice_id, "system": "legacy-billing"}
"""

INVENTORY_FIXTURE = """class FakeGateway:
    def reserve(self, sku):
        return {"sku": sku, "reserved": True}
"""

RUNBOOK_FIXTURE = """# Gateway notes

- Old QA sandbox docs mention GatewayMock.
- BillingGateway is only for legacy invoice tests.
- FakeGateway is only for inventory tests.
"""


@dataclass(frozen=True)
class PaymentFact:
    fact_id: str
    label: str
    statement: str
    required_terms: tuple[str, ...]


@dataclass
class PaymentServiceGrade:
    passed: bool
    uses_stub_gateway: bool
    uses_disabled_endpoint: bool
    uses_real_payment_gateway: bool
    uses_unrelated_gateway: bool
    payment_service_unchanged: bool
    unrelated_files_unchanged: bool
    pytest_passed: bool
    pytest_output: str
    error: str = ""


@dataclass
class PaymentArmResult:
    arm: str
    workspace: Path
    output: str
    test_text: str
    grade: PaymentServiceGrade
    status: str


@dataclass
class PaymentServiceEditResult:
    trace_path: Path
    sm_path: Path
    status: str
    mode: str
    edit_probe_id: str
    capture_gate: bool
    takeover_gate: bool
    same_state_gate: bool
    no_kept_tail_gate: bool
    tail_survival: bool
    sm_compact_status: str
    full_compact_statuses: list[str]
    sm_edit_pass: bool
    full_edit_passes: list[bool]
    sm_grade: PaymentServiceGrade | None
    full_grades: list[PaymentServiceGrade]
    sm_test_text: str
    full_test_texts: list[str]
    sm_workspace: Path
    full_workspaces: list[Path]
    edit_delta: float
    full_repeat_count: int
    pre_state_hash: str
    anchor_message_id: str | None
    extract_count: int
    extract_input_tokens: list[int]
    extract_output_tokens: list[int]
    extract_snapshot_tokens: list[int]
    precompact_tokens: int
    sm_post_compact_tokens: int
    full_post_compact_tokens: list[int]
    full_input_tokens: list[int]
    full_output_tokens: list[int]
    distractor_rounds: int
    payload_repeat: int
    compact_target_tokens: int
    summary_max_tokens: int
    max_turns: int
    error: str = ""


PAYMENT_FACTS: tuple[PaymentFact, ...] = (
    PaymentFact(
        fact_id="paymentservice_stub_gateway_rule",
        label="payment-stub-gateway",
        statement=PAYMENT_RULE,
        required_terms=("PaymentService", "StubGateway", "disabled://payment-gateway", "PaymentGateway"),
    ),
    PaymentFact(
        fact_id="paymentservice_gateway_boundaries",
        label="gateway-boundaries",
        statement=PAYMENT_DISTRACTOR_RULE,
        required_terms=("BillingGateway", "FakeGateway", "GatewayMock", "PaymentService"),
    ),
    PaymentFact(
        fact_id="paymentservice_batch_state",
        label="batch-state",
        statement=PAYMENT_BATCH_RULE,
        required_terms=("BETA-27", "parse", "normalize", "validate", "reconcile-ledger"),
    ),
)


def _append(messages: list[dict], role: str, content: str, message_id: str) -> None:
    messages.append({"role": role, "content": content, "id": message_id})


def _payment_rule_survives(text: str) -> bool:
    lower = text.lower()
    return all(term.lower() in lower for term in PAYMENT_FACTS[0].required_terms)


def _tail_rule_survives(text: str) -> bool:
    lower = text.lower()
    return "tests/test_payment_service.py" in lower and "do not change service implementation" in lower


def _payment_noise(round_idx: int) -> tuple[str, str]:
    samples = (
        (
            "Read services/billing.py; BillingGateway only serves legacy invoices.",
            "Kept BillingGateway out of PaymentService test plans.",
        ),
        (
            "Inventory tests still use FakeGateway for reservation retries.",
            "Marked FakeGateway as inventory-only and not relevant to payments.",
        ),
        (
            "Old QA sandbox runbook mentions GatewayMock and live PaymentGateway traces.",
            "Recorded GatewayMock as stale QA-sandbox advice.",
        ),
        (
            "Importer BETA-27 checkpoint shows parse, normalize, and validate done.",
            "Left only reconcile-ledger in the BETA-27 remaining-work note.",
        ),
        (
            "A retired smoke test hits the live payment endpoint during replay.",
            "Kept retired live-endpoint replay separate from unit-test conventions.",
        ),
        (
            "Invoice fixtures include a BillingGateway constructor in archived examples.",
            "Ignored archived invoice fixtures for current PaymentService unit tests.",
        ),
    )
    return samples[round_idx % len(samples)]


def _payment_length_payload(round_idx: int, payload_repeat: int) -> str:
    payload_repeat = max(0, payload_repeat)
    if payload_repeat == 0:
        return ""
    lines = []
    for item_idx in range(payload_repeat):
        lines.append(
            "Pressure packet "
            f"{round_idx + 1}.{item_idx + 1}: inspected legacy invoice traces, inventory reservation logs, "
            "QA sandbox notes, retry counters, pytest output fragments, old gateway diagrams, archived "
            "endpoint samples, and BETA-27 dry-run tables. This packet is distractor material; it does "
            "not change the early PaymentService testing convention, the gateway-boundary note, the "
            "BETA-27 remaining-work state, or the later file-scope guardrail."
        )
    return "\n" + "\n".join(lines)


def build_paymentservice_extract_snapshots(
    *,
    extract_count: int = 3,
    distractor_rounds: int = 8,
    payload_repeat: int = 0,
) -> tuple[list[list[dict]], list[dict]]:
    extract_count = max(1, extract_count)
    messages: list[dict] = []
    snapshots: list[list[dict]] = []

    _append(
        messages,
        "user",
        (
            "We are starting a long PaymentService test cleanup. Keep durable coding-session notes. "
            f"{PAYMENT_RULE} {PAYMENT_BATCH_RULE}"
        ),
        "payment-user-phase1",
    )
    _append(
        messages,
        "assistant",
        "Recorded the PaymentService test convention and the BETA-27 remaining-work state.",
        "payment-assistant-phase1",
    )
    snapshots.append(copy.deepcopy(messages))

    _append(
        messages,
        "user",
        (
            "Before more repo work, record the gateway boundaries. "
            f"{PAYMENT_DISTRACTOR_RULE}"
        ),
        "payment-user-boundaries",
    )
    _append(
        messages,
        "assistant",
        "Recorded gateway boundaries and marked GatewayMock as stale for PaymentService tests.",
        "payment-assistant-boundaries",
    )
    snapshots.append(copy.deepcopy(messages))

    for idx in range(distractor_rounds):
        user_noise, assistant_note = _payment_noise(idx)
        payload = _payment_length_payload(idx, payload_repeat)
        _append(messages, "user", f"PaymentService work log {idx + 1}: {user_noise}{payload}", f"payment-user-noise-{idx + 1}")
        _append(messages, "assistant", assistant_note, f"payment-assistant-noise-{idx + 1}")
        if len(snapshots) < extract_count and (idx + 1) >= max(1, distractor_rounds // 2):
            snapshots.append(copy.deepcopy(messages))

    while len(snapshots) < extract_count:
        idx = len(snapshots) + distractor_rounds
        user_noise, assistant_note = _payment_noise(idx)
        payload = _payment_length_payload(idx, payload_repeat)
        _append(messages, "user", f"Extra PaymentService work log {idx}: {user_noise}{payload}", f"payment-user-extra-{idx}")
        _append(messages, "assistant", assistant_note, f"payment-assistant-extra-{idx}")
        snapshots.append(copy.deepcopy(messages))

    precompact_messages = copy.deepcopy(messages)
    _append(
        precompact_messages,
        "user",
        f"Recent update after the last note extraction: {PAYMENT_TAIL_RULE}",
        "payment-user-tail",
    )
    _append(
        precompact_messages,
        "assistant",
        "Recorded the file-scope guardrail for the final PaymentService patch.",
        "payment-assistant-tail",
    )

    compact.ensure_runtime_message_ids(precompact_messages)
    snapshots = [copy.deepcopy(snapshot) for snapshot in snapshots[:extract_count]]
    for snapshot in snapshots:
        compact.ensure_runtime_message_ids(snapshot)
    return snapshots, precompact_messages


def setup_paymentservice_workspace(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    for directory in ("services", "tests", "docs"):
        (path / directory).mkdir(parents=True, exist_ok=True)
    (path / "services/__init__.py").write_text("", encoding="utf-8", newline="\n")
    (path / "tests/__init__.py").write_text("", encoding="utf-8", newline="\n")
    (path / PAYMENT_SERVICE_PATH).write_text(PAYMENT_SERVICE_FIXTURE, encoding="utf-8", newline="\n")
    (path / PAYMENT_TEST_PATH).write_text(PAYMENT_TEST_FIXTURE, encoding="utf-8", newline="\n")
    (path / BILLING_PATH).write_text(BILLING_FIXTURE, encoding="utf-8", newline="\n")
    (path / INVENTORY_PATH).write_text(INVENTORY_FIXTURE, encoding="utf-8", newline="\n")
    (path / RUNBOOK_PATH).write_text(RUNBOOK_FIXTURE, encoding="utf-8", newline="\n")


def _active_constructor(text: str, name: str) -> bool:
    pattern = re.compile(rf"(?<!class\s)\b{name}\s*\(")
    return bool(pattern.search(text))


def _run_pytest(workspace: Path) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", PAYMENT_TEST_PATH.as_posix(), "-q"],
            cwd=workspace,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except Exception as exc:  # pragma: no cover - env-specific failure path
        return False, f"{type(exc).__name__}: {exc}"
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, output[-4000:]


def grade_paymentservice_workspace(workspace: Path) -> PaymentServiceGrade:
    try:
        test_text = (workspace / PAYMENT_TEST_PATH).read_text(encoding="utf-8")
    except Exception as exc:
        return PaymentServiceGrade(
            passed=False,
            uses_stub_gateway=False,
            uses_disabled_endpoint=False,
            uses_real_payment_gateway=False,
            uses_unrelated_gateway=False,
            payment_service_unchanged=False,
            unrelated_files_unchanged=False,
            pytest_passed=False,
            pytest_output="",
            error=f"{type(exc).__name__}: {exc}",
        )

    uses_stub_gateway = "StubGateway" in test_text
    uses_disabled_endpoint = "disabled://payment-gateway" in test_text
    uses_real_payment_gateway = _active_constructor(test_text, "PaymentGateway")
    uses_unrelated_gateway = any(
        _active_constructor(test_text, name) for name in ("BillingGateway", "FakeGateway", "GatewayMock")
    )
    payment_service_unchanged = (workspace / PAYMENT_SERVICE_PATH).read_text(encoding="utf-8") == PAYMENT_SERVICE_FIXTURE
    unrelated_files_unchanged = (
        (workspace / BILLING_PATH).read_text(encoding="utf-8") == BILLING_FIXTURE
        and (workspace / INVENTORY_PATH).read_text(encoding="utf-8") == INVENTORY_FIXTURE
    )
    pytest_passed, pytest_output = _run_pytest(workspace)
    passed = all(
        (
            uses_stub_gateway,
            uses_disabled_endpoint,
            not uses_real_payment_gateway,
            not uses_unrelated_gateway,
            payment_service_unchanged,
            unrelated_files_unchanged,
            pytest_passed,
        )
    )
    return PaymentServiceGrade(
        passed=passed,
        uses_stub_gateway=uses_stub_gateway,
        uses_disabled_endpoint=uses_disabled_endpoint,
        uses_real_payment_gateway=uses_real_payment_gateway,
        uses_unrelated_gateway=uses_unrelated_gateway,
        payment_service_unchanged=payment_service_unchanged,
        unrelated_files_unchanged=unrelated_files_unchanged,
        pytest_passed=pytest_passed,
        pytest_output=pytest_output,
    )


def _fake_notes_for(messages: list[dict]) -> str:
    text = _message_text(messages)
    bullets = []
    if _payment_rule_survives(text):
        bullets.append(PAYMENT_RULE)
    if "BillingGateway" in text and "FakeGateway" in text and "GatewayMock" in text:
        bullets.append(PAYMENT_DISTRACTOR_RULE)
    if "BETA-27" in text:
        bullets.append(PAYMENT_BATCH_RULE)
    return SESSION_MEMORY_TEMPLATE + "\n\n# PaymentService Durable Notes\n" + "\n".join(f"- {item}" for item in bullets) + "\n"


def _fake_run_forked_agent(prompt, messages, **kwargs) -> ForkResult:  # noqa: ANN001
    del prompt, kwargs
    text = _message_text(messages)
    return ForkResult(
        final_text=_fake_notes_for(messages),
        written_paths=[],
        turns=1,
        input_tokens=max(1, len(text) // 4),
        output_tokens=333,
        stopped="finished",
    )


def _fake_full_summary(repeat_idx: int) -> str:
    return (
        f"PaymentService fake full summary {repeat_idx}: kept later gateway noise, old QA notes, "
        "and the file-scope guardrail, but omitted the early StubGateway and disabled endpoint rule. "
        "It still mentions PaymentGateway, BillingGateway, FakeGateway, and GatewayMock as noisy examples."
    )


def _continuation_messages(compacted_messages: list[dict]) -> list[dict]:
    api_messages = compact._compact_api_messages(compacted_messages)  # noqa: SLF001 - eval harness mirrors runtime view.
    api_messages.append({"role": "user", "content": PAYMENT_EDIT_TASK})
    return _merge_consecutive_same_role(api_messages)


def _fake_edit(compacted_messages: list[dict], workspace: Path) -> str:
    context = _message_text(compacted_messages)
    test_path = workspace / PAYMENT_TEST_PATH
    if _payment_rule_survives(context):
        test_path.write_text(
            "\n".join(
                [
                    "from services.payment import PaymentService",
                    "",
                    "",
                    "class StubGateway:",
                    "    endpoint = 'disabled://payment-gateway'",
                    "",
                    "    def charge(self, amount):",
                    "        return {'status': 'stubbed', 'amount': amount, 'endpoint': self.endpoint}",
                    "",
                    "",
                    "def test_checkout_records_payment_receipt():",
                    "    receipt = PaymentService(StubGateway()).checkout(42)",
                    "    assert receipt['status'] == 'stubbed'",
                    "    assert receipt['amount'] == 42",
                    "    assert receipt['endpoint'] == 'disabled://payment-gateway'",
                    "",
                ]
            ),
            encoding="utf-8",
            newline="\n",
        )
        return "Updated PaymentService test to use StubGateway."
    return "Left PaymentService test using the existing PaymentGateway."


def _run_edit_arm(
    *,
    arm: str,
    compacted_messages: list[dict],
    workspace: Path,
    live: bool,
    max_turns: int,
) -> PaymentArmResult:
    setup_paymentservice_workspace(workspace)
    if live:
        with config.using_workdir(workspace):
            output = loop.run_task(
                PAYMENT_EDIT_TASK,
                max_turns=max_turns,
                trace=False,
                initial_messages=_continuation_messages(compacted_messages),
                eval_hooks=loop.EvalHooks(compact_strategy="none"),
            )
    else:
        output = _fake_edit(compacted_messages, workspace)
    test_path = workspace / PAYMENT_TEST_PATH
    try:
        test_text = test_path.read_text(encoding="utf-8")
    except Exception:
        test_text = ""
    grade = grade_paymentservice_workspace(workspace)
    return PaymentArmResult(
        arm=arm,
        workspace=workspace,
        output=str(output or ""),
        test_text=test_text,
        grade=grade,
        status="PASS" if grade.passed else "FAIL",
    )


def _result_to_dict(result: PaymentServiceEditResult) -> dict:
    data = asdict(result)
    data["trace_path"] = result.trace_path.as_posix()
    data["sm_path"] = result.sm_path.as_posix()
    data["sm_workspace"] = result.sm_workspace.as_posix()
    data["full_workspaces"] = [path.as_posix() for path in result.full_workspaces]
    return data


def _status_for(
    *,
    error: str,
    capture_gate: bool,
    takeover_gate: bool,
    same_state_gate: bool,
    no_kept_tail_gate: bool,
    tail_survival: bool,
    full_compact_statuses: list[str],
    arm_results: list[PaymentArmResult],
) -> str:
    if error:
        return "ERROR"
    if not capture_gate:
        return "INVALID_CAPTURE"
    if not no_kept_tail_gate:
        return "INVALID_TAIL"
    if any(status != "ok" for status in full_compact_statuses):
        return "ERROR"
    if not all((takeover_gate, same_state_gate, tail_survival)):
        return "FAIL"
    if not arm_results:
        return "ERROR"
    return "PASS"


def run_paymentservice_edit_probe(
    out_dir: str | Path,
    *,
    live: bool = False,
    full_repeat_count: int = 3,
    extract_count: int = 3,
    distractor_rounds: int = 8,
    payload_repeat: int = 0,
    compact_target_tokens: int = 50_000,
    summary_max_tokens: int = compact.DEFAULT_SUMMARY_MAX_TOKENS,
    max_turns: int = 10,
) -> PaymentServiceEditResult:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mode = "live" if live else "fake"
    trace_path = out / f"sm_paymentservice_edit_behavior_{mode}.jsonl"
    sm_path = out / "session-memory.md"
    result_path = out / f"sm_paymentservice_edit_behavior_{mode}.json"
    workspaces_dir = out / "workspaces"
    for path in (trace_path, sm_path, result_path):
        if path.exists():
            path.unlink()
    if workspaces_dir.exists():
        shutil.rmtree(workspaces_dir)
    workspaces_dir.mkdir(parents=True, exist_ok=True)

    compact.reset_state()
    system = ""
    cfg = CompactConfig(
        keep_min_tokens=1,
        keep_min_msgs=1,
        keep_max_tokens=8_000,
        microcompact_clear_at_least=0,
        summary_max_tokens=summary_max_tokens,
    )
    extract_snapshots, precompact_messages = build_paymentservice_extract_snapshots(
        extract_count=extract_count,
        distractor_rounds=distractor_rounds,
        payload_repeat=payload_repeat,
    )
    pre_state_hash = _state_hash(precompact_messages, system, cfg)
    precompact_tokens = compact.estimate(precompact_messages, system)
    extract_snapshot_tokens = [compact.estimate(snapshot, system) for snapshot in extract_snapshots]
    sm = SessionMemory(sm_path)

    prior_sink = trace_mod._SINK
    original_fork = smmod.run_forked_agent
    original_chat = compact.llm.chat
    sink = JsonlSink(trace_path)
    trace_mod.set_sink(sink)

    extract_results: list[ForkResult] = []
    sm_messages: list[dict] = []
    full_messages: list[list[dict]] = []
    arm_results: list[PaymentArmResult] = []
    anchor_message_id = None
    error = ""

    if not live:
        smmod.run_forked_agent = _fake_run_forked_agent

    try:
        with span("sm_edit.paymentservice", SpanKind.INTERNAL, mode=mode):
            for snapshot in extract_snapshots:
                extract_results.append(sm.extract(copy.deepcopy(snapshot), system=system))
            anchor_message_id = sm.last_summarized_message_id

            sm_messages = compact.session_memory_compact(
                copy.deepcopy(precompact_messages),
                sm,
                system=system,
                cfg=cfg,
                auto_thr=compact_target_tokens,
            ) or []

            for repeat_idx in range(full_repeat_count):
                if not live:
                    def fake_chat(*args, _idx=repeat_idx, **kwargs):  # noqa: ANN001
                        del args, kwargs
                        return _Response(_fake_full_summary(_idx + 1), input_tokens=501 + _idx, output_tokens=71)

                    compact.llm.chat = fake_chat
                full_messages.append(
                    compact.full_compact(
                        copy.deepcopy(precompact_messages),
                        system=system,
                        cfg=cfg,
                        auto_thr=compact_target_tokens,
                    )
                )
            compact.llm.chat = original_chat

            arm_results.append(
                _run_edit_arm(
                    arm="sm",
                    compacted_messages=sm_messages,
                    workspace=workspaces_dir / "sm",
                    live=live,
                    max_turns=max_turns,
                )
            )
            for idx, messages in enumerate(full_messages):
                arm_results.append(
                    _run_edit_arm(
                        arm=f"full_{idx + 1}",
                        compacted_messages=messages,
                        workspace=workspaces_dir / f"full_{idx + 1}",
                        live=live,
                        max_turns=max_turns,
                    )
                )
    except Exception as exc:  # pragma: no cover - live/env failure path
        anchor_message_id = sm.last_summarized_message_id
        error = f"{type(exc).__name__}: {exc}"
    finally:
        compact.llm.chat = original_chat
        smmod.run_forked_agent = original_fork
        trace_mod._SINK = prior_sink

    sm_text = sm_path.read_text(encoding="utf-8") if sm_path.exists() else ""
    capture_gate = _payment_rule_survives(sm_text)
    sm_kept_text = _message_text(sm_messages[2:])
    no_kept_tail_gate = not _payment_rule_survives(sm_kept_text)
    tail_survival = _tail_rule_survives(sm_kept_text) or _fact_survives(sm_kept_text, LONG_TASK_TAIL_FACT)
    same_state_gate = all(
        _state_hash(precompact_messages, system, cfg) == pre_state_hash
        for _ in range(max(1, full_repeat_count + 1))
    )

    events = sink.events()
    sm_attrs = _last_span_attrs(events, "compact.session_memory_compact")
    sm_compact_status = str(sm_attrs.get("status", "missing"))
    full_attrs = _all_span_attrs(events, "compact.full_compact")
    full_compact_statuses = [str(attrs.get("status", "missing")) for attrs in full_attrs[-full_repeat_count:]]
    full_input_tokens = [
        int(attrs.get("compact_cost_input_tokens") or attrs.get("compact_cost_input") or 0)
        for attrs in full_attrs[-full_repeat_count:]
    ]
    full_output_tokens = [
        int(attrs.get("compact_cost_output_tokens") or attrs.get("compact_cost_output") or 0)
        for attrs in full_attrs[-full_repeat_count:]
    ]
    takeover_gate = sm_compact_status == "ok" and bool(sm_messages)

    sm_arm = arm_results[0] if arm_results else None
    full_arms = arm_results[1:]
    sm_edit_pass = bool(sm_arm and sm_arm.grade.passed)
    full_edit_passes = [arm.grade.passed for arm in full_arms]
    full_pass_rate = sum(1 for item in full_edit_passes if item) / max(1, len(full_edit_passes))
    edit_delta = (1.0 if sm_edit_pass else 0.0) - full_pass_rate if full_arms else 0.0

    status = _status_for(
        error=error,
        capture_gate=capture_gate,
        takeover_gate=takeover_gate,
        same_state_gate=same_state_gate,
        no_kept_tail_gate=no_kept_tail_gate,
        tail_survival=tail_survival,
        full_compact_statuses=full_compact_statuses,
        arm_results=arm_results,
    )

    result = PaymentServiceEditResult(
        trace_path=trace_path,
        sm_path=sm_path,
        status=status,
        mode=mode,
        edit_probe_id=PAYMENT_PROBE_ID,
        capture_gate=capture_gate,
        takeover_gate=takeover_gate,
        same_state_gate=same_state_gate,
        no_kept_tail_gate=no_kept_tail_gate,
        tail_survival=tail_survival,
        sm_compact_status=sm_compact_status,
        full_compact_statuses=full_compact_statuses,
        sm_edit_pass=sm_edit_pass,
        full_edit_passes=full_edit_passes,
        sm_grade=sm_arm.grade if sm_arm else None,
        full_grades=[arm.grade for arm in full_arms],
        sm_test_text=sm_arm.test_text if sm_arm else "",
        full_test_texts=[arm.test_text for arm in full_arms],
        sm_workspace=sm_arm.workspace if sm_arm else workspaces_dir / "sm",
        full_workspaces=[arm.workspace for arm in full_arms],
        edit_delta=round(edit_delta, 4),
        full_repeat_count=len(full_arms),
        pre_state_hash=pre_state_hash,
        anchor_message_id=anchor_message_id,
        extract_count=len(extract_results),
        extract_input_tokens=[result.input_tokens for result in extract_results],
        extract_output_tokens=[result.output_tokens for result in extract_results],
        extract_snapshot_tokens=extract_snapshot_tokens,
        precompact_tokens=precompact_tokens,
        sm_post_compact_tokens=compact.estimate(sm_messages, system),
        full_post_compact_tokens=[compact.estimate(messages, system) for messages in full_messages],
        full_input_tokens=full_input_tokens,
        full_output_tokens=full_output_tokens,
        distractor_rounds=distractor_rounds,
        payload_repeat=payload_repeat,
        compact_target_tokens=compact_target_tokens,
        summary_max_tokens=summary_max_tokens,
        max_turns=max_turns,
        error=error,
    )
    result_path.write_text(json.dumps(_result_to_dict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def render_paymentservice_report(result: PaymentServiceEditResult) -> str:
    lines = [
        "# SessionMemory PaymentService Edit Behavior Probe",
        "",
        "This probe asks the agent to patch tests/test_payment_service.py after SM compact and full_compact.",
        "",
        "## Gates",
        "",
        "| Gate | Value |",
        "|---|---:|",
        f"| Status | {result.status} |",
        f"| Mode | {result.mode} |",
        f"| edit probe | {result.edit_probe_id} |",
        f"| capture gate | {result.capture_gate} |",
        f"| takeover gate | {result.takeover_gate} |",
        f"| same-state gate | {result.same_state_gate} |",
        f"| no-kept-tail gate | {result.no_kept_tail_gate} |",
        f"| tail survival | {result.tail_survival} |",
        f"| full compact statuses | {result.full_compact_statuses} |",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| SM edit pass | {result.sm_edit_pass} |",
        f"| full edit passes | {result.full_edit_passes} |",
        f"| edit delta | {result.edit_delta:.2f} |",
        f"| precompact tokens | {result.precompact_tokens} |",
        f"| extract snapshot tokens | {result.extract_snapshot_tokens} |",
        f"| SM post-compact tokens | {result.sm_post_compact_tokens} |",
        f"| full post-compact tokens | {result.full_post_compact_tokens} |",
        f"| extract input tokens | {result.extract_input_tokens} |",
        f"| extract output tokens | {result.extract_output_tokens} |",
        f"| full input tokens | {result.full_input_tokens} |",
        f"| full output tokens | {result.full_output_tokens} |",
        f"| distractor rounds | {result.distractor_rounds} |",
        f"| payload repeat | {result.payload_repeat} |",
        f"| compact target tokens | {result.compact_target_tokens} |",
        "",
        "## Grader",
        "",
        "| Arm | passed | stub | disabled endpoint | real PaymentGateway | unrelated gateway | pytest |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    grades = []
    if result.sm_grade is not None:
        grades.append(("SM", result.sm_grade))
    grades.extend((f"full_compact {idx}", grade) for idx, grade in enumerate(result.full_grades, start=1))
    for name, grade in grades:
        lines.append(
            f"| {name} | {grade.passed} | {grade.uses_stub_gateway} | {grade.uses_disabled_endpoint} | "
            f"{grade.uses_real_payment_gateway} | {grade.uses_unrelated_gateway} | {grade.pytest_passed} |"
        )
    lines.extend(["", "## Edited Tests", "", "### SM", "", "```python", result.sm_test_text.strip(), "```"])
    for idx, test_text in enumerate(result.full_test_texts, start=1):
        lines.extend(["", f"### full_compact {idx}", "", "```python", test_text.strip(), "```"])
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Trace: `{result.trace_path.as_posix()}`",
            f"- SessionMemory file: `{result.sm_path.as_posix()}`",
            f"- SM workspace: `{result.sm_workspace.as_posix()}`",
            f"- Full workspaces: `{[path.as_posix() for path in result.full_workspaces]}`",
            f"- Pre-state hash: `{result.pre_state_hash}`",
            f"- Anchor message id: `{result.anchor_message_id or ''}`",
        ]
    )
    if result.error:
        lines.extend(["", "## Error", "", f"`{result.error}`"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SessionMemory PaymentService edit-behavior probe.")
    parser.add_argument("--out", default=".traces/sm_paymentservice_edit_behavior", help="Output directory.")
    parser.add_argument("--live", action="store_true", help="Call real configured LLMs and tools.")
    parser.add_argument("--full-repeat-count", type=int, default=3, help="Number of full_compact repeats.")
    parser.add_argument("--extract-count", type=int, default=3, help="Number of SessionMemory extract snapshots.")
    parser.add_argument("--distractor-rounds", type=int, default=8, help="Number of post-rule coding-noise rounds.")
    parser.add_argument("--payload-repeat", type=int, default=0, help="Length-pressure payload lines per distractor round.")
    parser.add_argument("--compact-target-tokens", type=int, default=50_000, help="Forced compact target/threshold for this probe.")
    parser.add_argument(
        "--summary-max-tokens",
        type=int,
        default=compact.DEFAULT_SUMMARY_MAX_TOKENS,
        help="full_compact summary max tokens.",
    )
    parser.add_argument("--max-turns", type=int, default=10, help="Max run_task turns per edit arm.")
    args = parser.parse_args()
    result = run_paymentservice_edit_probe(
        args.out,
        live=args.live,
        full_repeat_count=args.full_repeat_count,
        extract_count=args.extract_count,
        distractor_rounds=args.distractor_rounds,
        payload_repeat=args.payload_repeat,
        compact_target_tokens=args.compact_target_tokens,
        summary_max_tokens=args.summary_max_tokens,
        max_turns=args.max_turns,
    )
    print(render_paymentservice_report(result))
    return 0 if result.status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

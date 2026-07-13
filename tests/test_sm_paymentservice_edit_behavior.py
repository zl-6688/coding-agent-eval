from pathlib import Path

from eval.compression_eval.sm_paymentservice_edit_behavior import (
    grade_paymentservice_workspace,
    render_paymentservice_report,
    run_paymentservice_edit_probe,
    setup_paymentservice_workspace,
)


def test_paymentservice_grader_accepts_stub_gateway_and_rejects_real_gateway(tmp_path):
    workspace = tmp_path / "workspace"
    setup_paymentservice_workspace(workspace)

    initial_grade = grade_paymentservice_workspace(workspace)
    assert initial_grade.passed is False
    assert initial_grade.uses_real_payment_gateway is True

    test_path = workspace / Path("tests/test_payment_service.py")
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
                "def test_checkout_uses_stub_gateway():",
                "    receipt = PaymentService(StubGateway()).checkout(42)",
                "    assert receipt['status'] == 'stubbed'",
                "    assert receipt['endpoint'] == 'disabled://payment-gateway'",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )

    fixed_grade = grade_paymentservice_workspace(workspace)
    assert fixed_grade.passed is True
    assert fixed_grade.uses_stub_gateway is True
    assert fixed_grade.uses_disabled_endpoint is True
    assert fixed_grade.pytest_passed is True


def test_paymentservice_edit_fake_mode_detects_memory_dependent_patch(tmp_path):
    result = run_paymentservice_edit_probe(
        tmp_path,
        live=False,
        full_repeat_count=2,
        extract_count=3,
        distractor_rounds=4,
    )

    assert result.status == "PASS"
    assert result.mode == "fake"
    assert result.capture_gate is True
    assert result.takeover_gate is True
    assert result.same_state_gate is True
    assert result.no_kept_tail_gate is True
    assert result.tail_survival is True
    assert result.sm_edit_pass is True
    assert result.full_edit_passes == [False, False]
    assert result.edit_delta == 1.0
    assert "StubGateway" in result.sm_test_text
    assert "PaymentGateway(" in result.full_test_texts[0]


def test_paymentservice_report_is_self_describing(tmp_path):
    result = run_paymentservice_edit_probe(
        tmp_path,
        live=False,
        full_repeat_count=1,
        extract_count=3,
        distractor_rounds=2,
    )
    report = render_paymentservice_report(result)

    assert "PaymentService" in report
    assert "paymentservice-edit" in report
    assert "edit delta" in report
    assert "pytest" in report


def test_paymentservice_edit_allows_high_context_compact_target(tmp_path):
    result = run_paymentservice_edit_probe(
        tmp_path,
        live=False,
        full_repeat_count=1,
        extract_count=3,
        distractor_rounds=12,
        payload_repeat=130,
        compact_target_tokens=128_000,
    )

    assert result.status == "PASS"
    assert result.precompact_tokens > 128_000
    assert result.takeover_gate is True
    assert result.sm_post_compact_tokens < result.compact_target_tokens

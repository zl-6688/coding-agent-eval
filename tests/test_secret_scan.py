"""test_secret_scan.py — 单元测试 agent.memory.secret_scan。

每条规则给 1 个样例命中 + 干净文本放行；验证返回 rule_id 非明文。
所有测试不调 LLM（纯本地正则）。
"""
import pytest
from agent.memory import secret_scan


# ── 辅助 ──────────────────────────────────────────────────────────────────────

def assert_hits(content: str, expected_rule_id: str) -> None:
    """断言 content 命中指定规则，且返回值是 rule_id 字符串（非命中明文）。"""
    hits = secret_scan.scan(content)
    assert expected_rule_id in hits, (
        f"期望命中规则 {expected_rule_id!r}，实际命中：{hits}"
    )
    # 验证返回的是 rule_id（来自 SECRET_RULES 的 rule_id 字段），不是原始密钥文本
    for h in hits:
        assert isinstance(h, str), "scan 应返回字符串 rule_id"
        # rule_id 均由字母+连字符构成（无数字行敏感内容）
        assert h in {r.rule_id for r in secret_scan.SECRET_RULES}, (
            f"命中值 {h!r} 不是已知 rule_id"
        )
    # 最强断言：命中的文本不在返回值里
    # （不检查 content 本身是否含 rule_id 字符串——rule_id 比密钥更短、不会误报）


def assert_clean(content: str) -> None:
    """断言干净文本不触发任何规则。"""
    hits = secret_scan.scan(content)
    assert hits == [], f"干净文本误报：{hits}，内容：{content!r}"


# ── 逐规则测试 ────────────────────────────────────────────────────────────────

def test_anthropic_key_hits():
    pfx = "-".join(["sk", "ant", "api"])
    # 93 alnum/underscore/dash + "AA" suffix，覆盖 provider key 规则形态
    sample = pfx + "03-" + "a" * 93 + "AA"
    assert_hits(f"api_key = {sample}", "anthropic-api-key")


def test_aws_access_key_hits():
    # 组装 Amazon 公开文档形态的测试值，避免在仓库中保留连续 credential literal。
    sample = "AK" + "IAIOSFODNN7EXAMPLE"
    assert_hits(f"export AWS_ACCESS_KEY_ID={sample}", "aws-access-token")


def test_aws_temporary_key_hits():
    # ASIA = STS 临时凭证前缀（CC 新增覆盖）
    sample = "ASIA" + "A" * 16
    assert_hits(f"AWS_SESSION_KEY={sample}", "aws-access-token")


def test_databricks_token_hits():
    # hex-only 32 chars（Databricks token rule）
    sample = "dapi" + "abcdef01" * 4
    assert_hits(f"token = {sample}", "databricks-api-token")


def test_gcp_api_key_hits():
    sample = "AIza" + "B" * 35
    assert_hits(f"key={sample}", "gcp-api-key")


def test_github_pat_hits():
    # ghp_ = personal access token
    sample = "ghp_" + "x" * 36
    assert_hits(f"GITHUB_TOKEN={sample}", "github-pat")


def test_github_oauth_token_hits():
    # gho_ = OAuth access token — CC 将其拆为独立规则 "github-oauth"
    sample = "gho_" + "y" * 36
    assert_hits(f"token: {sample}", "github-oauth")


def test_huggingface_token_hits():
    # hf_ + 34 alpha-only chars（非 alnum，不是 40 位）
    sample = "hf_" + "A" * 34
    assert_hits(f"HF_TOKEN={sample}", "huggingface-access-token")


def test_npm_token_hits():
    sample = "npm_" + "N" * 36
    assert_hits(f"NPM_TOKEN={sample}", "npm-access-token")


def test_openai_key_hits():
    # T3BlbkFJ 是 base64("openAI")，是旧格式 key 的高置信指纹
    sample = "sk-" + "a" * 20 + "T3BlbkFJ" + "b" * 20
    assert_hits(f"OPENAI_API_KEY={sample}", "openai-api-key")


def test_pem_private_key_hits():
    # body 须 >= 64 chars（private-key rule: [\s\S-]{64,}?）
    body = "\n" + "A" * 64 + "\n"
    begin = "-----BEGIN " + "PRIVATE KEY-----"
    end = "-----END " + "PRIVATE KEY-----"
    assert_hits(f"{begin}{body}{end}",
                "private-key")


def test_pem_rsa_private_key_hits():
    body = "\n" + "A" * 64 + "\n"
    begin = "-----BEGIN RSA " + "PRIVATE KEY-----"
    end = "-----END RSA " + "PRIVATE KEY-----"
    assert_hits(f"{begin}{body}{end}",
                "private-key")


def test_postman_api_key_hits():
    # hex-only segments（Postman token rule）
    sample = "PMAK-" + "a" * 24 + "-" + "b" * 34
    assert_hits(f"key={sample}", "postman-api-token")


def test_sendgrid_key_hits():
    # SG. + 66-char single segment（SendGrid token rule）
    sample = "SG." + "s" * 22 + "." + "t" * 43  # 22+1+43 = 66 chars after SG.
    assert_hits(f"SENDGRID_API_KEY={sample}", "sendgrid-api-token")


def test_slack_bot_token_hits():
    sample = "xoxb-12345678901-12345678901-" + "a" * 24
    assert_hits(f"SLACK_BOT_TOKEN={sample}", "slack-bot-token")


def test_stripe_live_key_hits():
    sample = "sk_live_" + "s" * 24
    assert_hits(f"STRIPE_KEY={sample}", "stripe-access-token")


def test_twilio_api_key_hits():
    sample = "SK" + "f" * 32
    assert_hits(f"TWILIO_AUTH={sample}", "twilio-api-key")


# ── 干净文本放行 ──────────────────────────────────────────────────────────────

def test_clean_text_no_hits():
    assert_clean("这是一段普通文本，不含任何密钥信息。")


def test_clean_code_no_hits():
    assert_clean("def hello():\n    return 'world'\n\nx = hello()")


def test_clean_env_var_placeholder_no_hits():
    assert_clean("export API_KEY=<your-api-key-here>")


def test_clean_short_key_no_hits():
    # 短字符串不达正则长度要求，不应触发
    assert_clean("sk-abc123")
    assert_clean("npm_short")


def test_clean_uuid_no_hits():
    # UUID 虽然随机，但不匹配任何规则的独特前缀
    assert_clean("550e8400-e29b-41d4-a716-446655440000")


# ── 返回值类型不含明文 ────────────────────────────────────────────────────────

def test_scan_returns_rule_ids_not_matched_text():
    """scan() 返回 rule_id 列表，不含命中的明文密钥值。"""
    aws_key = "AK" + "IAIOSFODNN7EXAMPLE"
    hits = secret_scan.scan(f"key={aws_key}")
    assert "aws-access-token" in hits
    # 命中文本（密钥值）不在返回列表里
    assert aws_key not in hits
    assert not any(aws_key in h for h in hits)


def test_empty_content_returns_empty():
    assert secret_scan.scan("") == []
    assert secret_scan.scan("   ") == []


def test_multiple_rules_can_all_hit():
    """一段内容同时含多类密钥 → 两个 rule_id 都返回。"""
    aws_key = "AK" + "IAIOSFODNN7EXAMPLE"
    slack_token = "xoxb-12345678901-12345678901-" + "a" * 24
    hits = secret_scan.scan(f"{aws_key} and {slack_token}")
    assert "aws-access-token" in hits
    assert "slack-bot-token" in hits

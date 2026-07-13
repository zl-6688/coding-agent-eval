"""High-confidence secret scanner for memory write boundaries.

``scan()`` returns rule identifiers only, never matched text. Provider-key
prefixes are assembled at runtime so testable patterns do not become literal
credentials in the repository. Both shared and private AutoMemory paths are
scanned intentionally.
"""

import re
from typing import NamedTuple

_ANT_KEY_PFX = "-".join(["sk", "ant", "api"])
_B = r"""(?:[\x60'"\s;]|\\[nr]|$)"""


class _Rule(NamedTuple):
    rule_id: str
    pattern: re.Pattern


def _r(source: str, flags: int = 0) -> re.Pattern:
    return re.compile(source, flags | re.ASCII)


SECRET_RULES: list[_Rule] = [
    _Rule("aws-access-token",       _r(r'\b((?:A3T[A-Z0-9]|AKIA|ASIA|ABIA|ACCA)[A-Z2-7]{16})\b')),
    _Rule("gcp-api-key",            _r(rf'\b(AIza[\w\-]{{35}}){_B}')),
    _Rule("azure-ad-client-secret", _r(r"""(?:^|[\\'"\x60\s>=:(,)])([a-zA-Z0-9_~.]{3}\dQ~[a-zA-Z0-9_~.\-]{31,34})(?:$|[\\'"\x60\s<),])""")),
    _Rule("digitalocean-pat",       _r(rf'\b(dop_v1_[a-f0-9]{{64}}){_B}')),
    _Rule("digitalocean-access-token", _r(rf'\b(doo_v1_[a-f0-9]{{64}}){_B}')),
    _Rule("anthropic-api-key",      _r(rf'\b({_ANT_KEY_PFX}03-[a-zA-Z0-9_\-]{{93}}AA){_B}')),
    _Rule("anthropic-admin-api-key",_r(rf'\b(sk-ant-admin01-[a-zA-Z0-9_\-]{{93}}AA){_B}')),
    _Rule("openai-api-key",         _r(rf'\b(sk-(?:proj|svcacct|admin)-(?:[A-Za-z0-9_\-]{{74}}|[A-Za-z0-9_\-]{{58}})T3BlbkFJ(?:[A-Za-z0-9_\-]{{74}}|[A-Za-z0-9_\-]{{58}})|sk-[a-zA-Z0-9]{{20}}T3BlbkFJ[a-zA-Z0-9]{{20}}){_B}')),
    _Rule("huggingface-access-token", _r(rf'\b(hf_[a-zA-Z]{{34}}){_B}')),
    _Rule("github-pat",             _r(r'ghp_[0-9a-zA-Z]{36}')),
    _Rule("github-fine-grained-pat",_r(r'github_pat_\w{82}')),
    _Rule("github-app-token",       _r(r'(?:ghu|ghs)_[0-9a-zA-Z]{36}')),
    _Rule("github-oauth",           _r(r'gho_[0-9a-zA-Z]{36}')),
    _Rule("github-refresh-token",   _r(r'ghr_[0-9a-zA-Z]{36}')),
    _Rule("gitlab-pat",             _r(r'glpat-[\w\-]{20}')),
    _Rule("gitlab-deploy-token",    _r(r'gldt-[0-9a-zA-Z_\-]{20}')),
    _Rule("slack-bot-token",        _r(r'xoxb-[0-9]{10,13}-[0-9]{10,13}[a-zA-Z0-9\-]*')),
    _Rule("slack-user-token",       _r(r'xox[pe](?:-[0-9]{10,13}){3}-[a-zA-Z0-9\-]{28,34}')),
    _Rule("slack-app-token",        _r(r'xapp-\d-[A-Z0-9]+-\d+-[a-z0-9]+', re.IGNORECASE)),
    _Rule("twilio-api-key",         _r(r'SK[0-9a-fA-F]{32}')),
    _Rule("sendgrid-api-token",     _r(rf'\b(SG\.[a-zA-Z0-9=_\-.]{{66}}){_B}')),
    _Rule("npm-access-token",       _r(rf'\b(npm_[a-zA-Z0-9]{{36}}){_B}')),
    _Rule("pypi-upload-token",      _r(r'pypi-AgEIcHlwaS5vcmc[\w\-]{50,1000}')),
    _Rule("databricks-api-token",   _r(rf'\b(dapi[a-f0-9]{{32}}(?:-\d)?){_B}')),
    _Rule("hashicorp-tf-api-token", _r(r'[a-zA-Z0-9]{14}\.atlasv1\.[a-zA-Z0-9\-_=]{60,70}')),
    _Rule("pulumi-api-token",       _r(rf'\b(pul-[a-f0-9]{{40}}){_B}')),
    _Rule("postman-api-token",      _r(rf'\b(PMAK-[a-fA-F0-9]{{24}}-[a-fA-F0-9]{{34}}){_B}')),
    _Rule("grafana-api-key",        _r(rf'\b(eyJrIjoi[A-Za-z0-9+/]{{70,400}}={{0,3}}){_B}')),
    _Rule("grafana-cloud-api-token",_r(rf'\b(glc_[A-Za-z0-9+/]{{32,400}}={{0,3}}){_B}')),
    _Rule("grafana-service-account-token", _r(rf'\b(glsa_[A-Za-z0-9]{{32}}_[A-Fa-f0-9]{{8}}){_B}')),
    _Rule("sentry-user-token",      _r(rf'\b(sntryu_[a-f0-9]{{64}}){_B}')),
    _Rule("sentry-org-token",       _r(r'\bsntrys_eyJpYXQiO[a-zA-Z0-9+/]{10,200}(?:LCJyZWdpb25fdXJs|InJlZ2lvbl91cmwi|cmVnaW9uX3VybCI6)[a-zA-Z0-9+/]{10,200}={0,2}_[a-zA-Z0-9+/]{43}')),
    _Rule("stripe-access-token",    _r(rf'\b((?:sk|rk)_(?:test|live|prod)_[a-zA-Z0-9]{{10,99}}){_B}')),
    _Rule("shopify-access-token",   _r(r'shpat_[a-fA-F0-9]{32}')),
    _Rule("shopify-shared-secret",  _r(r'shpss_[a-fA-F0-9]{32}')),
    _Rule("private-key",            _r(r'-----BEGIN[ A-Z0-9_-]{0,100}PRIVATE KEY(?: BLOCK)?-----[\s\S-]{64,}?-----END[ A-Z0-9_-]{0,100}PRIVATE KEY(?: BLOCK)?-----', re.IGNORECASE | re.DOTALL)),
]


def scan(content: str) -> list[str]:
    """Return list of matched rule_ids (never returns matched plaintext). Empty = safe."""
    hits: list[str] = []
    seen: set[str] = set()
    for rule in SECRET_RULES:
        if rule.rule_id not in seen and rule.pattern.search(content):
            seen.add(rule.rule_id)
            hits.append(rule.rule_id)
    return hits

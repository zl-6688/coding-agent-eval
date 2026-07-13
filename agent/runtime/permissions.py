"""In-memory permission rules for tool execution."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Literal, Mapping

from agent.mcp.names import MCP_SEPARATOR, MCP_TOOL_PREFIX, build_mcp_tool_name

PermissionBehavior = Literal["allow", "deny", "ask", "passthrough"]
PermissionSource = Literal["session", "project", "user"]
AskBehavior = Literal["deny", "ask"]

_BEHAVIORS: set[str] = {"allow", "deny", "ask", "passthrough"}
_SOURCES: set[str] = {"session", "project", "user"}
_PRIORITY: dict[str, int] = {
    "deny": 3,
    "ask": 2,
    "allow": 1,
    "passthrough": 0,
}


@dataclass(frozen=True)
class PermissionDecision:
    behavior: PermissionBehavior
    message: str = ""
    updated_input: dict[str, Any] | None = None
    source: str = "default"
    reason: str = ""


InputMatcher = Callable[[dict[str, Any]], bool]
ToolChecker = Callable[[Any, dict[str, Any], "PermissionContext"], PermissionDecision | None]


@dataclass(frozen=True)
class PermissionContext:
    """Runtime context passed to tool-level permission checkers."""

    ask_behavior: AskBehavior = "deny"
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class PermissionRule:
    """A single in-memory permission rule.

    ``matcher`` scopes a tool rule to specific input without turning the
    PermissionManager into a persistence layer.
    """

    tool_name: str
    behavior: PermissionBehavior
    source: PermissionSource = "session"
    matcher: InputMatcher | None = None
    message: str = ""
    rule_id: str = ""

    def __post_init__(self) -> None:
        if self.behavior not in _BEHAVIORS:
            raise ValueError(f"unsupported permission behavior: {self.behavior}")
        if self.source not in _SOURCES:
            raise ValueError(f"unsupported permission rule source: {self.source}")
        if not self.tool_name:
            raise ValueError("permission rule tool_name must be non-empty")

    def matches(self, spec: Any, tool_input: dict[str, Any]) -> bool:
        if not _rule_matches_tool_name(self.tool_name, spec):
            return False
        if self.matcher is None:
            return True
        return bool(self.matcher(tool_input))

    def to_decision(self) -> PermissionDecision:
        return PermissionDecision(
            self.behavior,
            message=self.message,
            source=_rule_source(self.source, self.rule_id),
        )


class PermissionManager:
    """Minimal structured permission engine for tool execution."""

    def __init__(
        self,
        rules: tuple[PermissionRule, ...] | list[PermissionRule] | None = None,
        *,
        tool_checkers: Mapping[str, ToolChecker] | None = None,
        ask_behavior: AskBehavior = "deny",
        context: Mapping[str, Any] | None = None,
    ) -> None:
        if ask_behavior not in {"deny", "ask"}:
            raise ValueError(f"unsupported ask_behavior: {ask_behavior}")
        self._rules = list(rules or ())
        self._tool_checkers = dict(tool_checkers or {})
        self._ask_behavior = ask_behavior
        self._context_metadata = dict(context or {})

    @property
    def rules(self) -> tuple[PermissionRule, ...]:
        return tuple(self._rules)

    @property
    def ask_behavior(self) -> AskBehavior:
        return self._ask_behavior

    def add_rule(self, rule: PermissionRule) -> None:
        self._rules.append(rule)

    def set_tool_checker(self, tool_name: str, checker: ToolChecker | None) -> None:
        if not tool_name:
            raise ValueError("tool checker tool_name must be non-empty")
        if checker is None:
            self._tool_checkers.pop(tool_name, None)
            return
        self._tool_checkers[tool_name] = checker

    def decide(self, spec: Any, tool_input: dict[str, Any]) -> PermissionDecision:
        deny_rule = self._first_matching_rule_decision(spec, tool_input, "deny")
        if deny_rule is not None:
            return deny_rule

        ask_rule = self._first_matching_rule_decision(spec, tool_input, "ask")
        if ask_rule is not None:
            return self._finalize_ask(ask_rule)

        checker_decisions = self._run_checkers(spec, tool_input)
        checker_blocking = _highest_priority(
            [
                decision
                for decision in checker_decisions
                if decision.behavior in {"deny", "ask"}
            ]
        )
        if checker_blocking is not None:
            return self._finalize_ask(checker_blocking)

        allow_rule = self._first_matching_rule_decision(spec, tool_input, "allow")
        if allow_rule is not None:
            return allow_rule

        checker_fallback = _highest_priority(checker_decisions)
        if checker_fallback is not None:
            return checker_fallback
        return PermissionDecision("passthrough", source="default")

    def is_exposure_denied(self, spec: Any) -> bool:
        """Return True only for whole-tool deny rules safe for model pre-filtering."""

        for rule in self._rules:
            if rule.behavior != "deny" or rule.matcher is not None:
                continue
            if _rule_matches_tool_name(rule.tool_name, spec):
                return True
        return False

    def _first_matching_rule_decision(
        self,
        spec: Any,
        tool_input: dict[str, Any],
        behavior: PermissionBehavior,
    ) -> PermissionDecision | None:
        for rule in self._rules:
            if rule.behavior != behavior:
                continue
            try:
                matches = rule.matches(spec, tool_input)
            except Exception as exc:
                return PermissionDecision(
                    "deny",
                    message=_matcher_error_message(rule, exc),
                    source=_rule_source(rule.source, rule.rule_id),
                )
            if matches:
                return rule.to_decision()
        return None

    def _run_checkers(
        self,
        spec: Any,
        tool_input: dict[str, Any],
    ) -> list[PermissionDecision]:
        decisions: list[PermissionDecision] = []
        context = PermissionContext(
            ask_behavior=self._ask_behavior,
            metadata=dict(self._context_metadata),
        )

        for checker in self._matching_checkers(spec):
            decision = checker(spec, dict(tool_input), context)
            if decision is not None:
                decisions.append(decision)
        return decisions

    def _matching_checkers(self, spec: Any) -> tuple[ToolChecker, ...]:
        tool_name = _tool_name(spec)
        checkers: list[ToolChecker] = []
        wildcard = self._tool_checkers.get("*")
        if wildcard is not None:
            checkers.append(wildcard)
        checker = self._tool_checkers.get(tool_name)
        if checker is not None:
            checkers.append(checker)
        return tuple(checkers)

    def _finalize_ask(self, decision: PermissionDecision) -> PermissionDecision:
        if decision.behavior != "ask" or self._ask_behavior == "ask":
            return decision
        base = decision.message or f"permission ask from {decision.source}"
        return replace(
            decision,
            behavior="deny",
            reason="ask_unavailable",
            message=(
                f"{base}; denied because PermissionManager is noninteractive "
                'and ask_behavior="deny"'
            ),
        )


class PermissionEngine(PermissionManager):
    """Backward-compatible name used by ``agent.tools.runtime``."""


def _highest_priority(decisions: list[PermissionDecision]) -> PermissionDecision | None:
    if not decisions:
        return None
    return max(decisions, key=lambda decision: _PRIORITY.get(decision.behavior, -1))


def _tool_name(spec: Any) -> str:
    return str(getattr(spec, "name", ""))


def _rule_matches_tool_name(rule_tool_name: str, spec: Any) -> bool:
    if rule_tool_name == "*":
        return True

    mcp_permission_name = _mcp_permission_name(spec)
    if mcp_permission_name:
        if rule_tool_name == mcp_permission_name:
            return True
        server_rule = _mcp_server_rule_name(mcp_permission_name)
        if server_rule and rule_tool_name in {
            server_rule,
            f"{server_rule}{MCP_SEPARATOR}*",
        }:
            return True
        return False

    return rule_tool_name == _tool_name(spec)


def _mcp_permission_name(spec: Any) -> str:
    metadata = getattr(spec, "metadata", None)
    if isinstance(metadata, Mapping):
        mcp = metadata.get("mcp")
        if isinstance(mcp, Mapping):
            permission_name = mcp.get("permission_name")
            if permission_name:
                return str(permission_name)
            server_name = mcp.get("server_name")
            tool_name = mcp.get("tool_name")
            if server_name and tool_name:
                return build_mcp_tool_name(str(server_name), str(tool_name))

    server_name = getattr(spec, "server_name", None)
    tool_name = getattr(spec, "tool_name", None)
    if server_name and tool_name:
        return build_mcp_tool_name(str(server_name), str(tool_name))

    name = _tool_name(spec)
    if _looks_like_mcp_tool_name(name):
        return name
    return ""


def _looks_like_mcp_tool_name(name: str) -> bool:
    parts = name.split(MCP_SEPARATOR)
    return len(parts) >= 3 and parts[0] == MCP_TOOL_PREFIX and bool(parts[1])


def _mcp_server_rule_name(permission_name: str) -> str:
    parts = permission_name.split(MCP_SEPARATOR)
    if len(parts) < 3 or parts[0] != MCP_TOOL_PREFIX or not parts[1]:
        return ""
    return f"{MCP_TOOL_PREFIX}{MCP_SEPARATOR}{parts[1]}"


def _rule_source(source: str, rule_id: str) -> str:
    if rule_id:
        return f"{source}:{rule_id}"
    return source


def _matcher_error_message(rule: PermissionRule, exc: Exception) -> str:
    return (
        "PermissionRuleMatcherError: "
        f"rule_source={_rule_source(rule.source, rule.rule_id)} "
        f"tool_name={rule.tool_name}: {type(exc).__name__}: {exc}"
    )

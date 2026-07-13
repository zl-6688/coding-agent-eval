"""记忆模块 —— 对齐 Claude Code 记忆栈（设计见 docs/memory-design.md）。

分层：forked-agent 基础设施、Session Memory、AutoMemory 写/检索、
确定性 memory governance/consolidation、密钥扫描。consolidation
和 AutoDream daemon 第一版都刻意保持可注入、可测试。
"""

from .forked_agent import ForkResult, run_forked_agent
from .auto_memory import AutoMemory, AutoMemoryConfig, DEFAULT_AM_CONFIG
from .consolidation import (
    MemoryConsolidationPlan,
    MemoryConsolidationSkip,
    MemoryMergeGroup,
    consolidate_memories,
)
from . import secret_scan
from .auto_dream import (
    AutoDreamAcquireResult,
    AutoDreamConfig,
    AutoDreamDaemon,
    AutoDreamLock,
    AutoDreamRunContext,
    AutoDreamRunner,
    AutoDreamState,
    maybe_start_auto_dream_daemon,
    run_auto_dream_once,
)

__all__ = [
    "run_forked_agent", "ForkResult",
    "AutoMemory", "AutoMemoryConfig", "DEFAULT_AM_CONFIG",
    "MemoryMergeGroup", "MemoryConsolidationSkip",
    "MemoryConsolidationPlan", "consolidate_memories",
    "AutoDreamConfig", "AutoDreamState", "AutoDreamRunContext",
    "AutoDreamAcquireResult", "AutoDreamLock", "AutoDreamRunner",
    "AutoDreamDaemon", "run_auto_dream_once", "maybe_start_auto_dream_daemon",
    "secret_scan",
]

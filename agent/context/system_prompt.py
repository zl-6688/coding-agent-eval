"""agent/context/system_prompt.py — 分段式 system prompt 构建器。

忠实参照 CC `systemPromptSections.ts` 的分段+每段缓存范式；
这里保留 identity/tools/workspace/memory 这些 system 级段；CC 段更多。

设计对齐：
  CC `systemPromptSection()` 把 name+compute 封装成 section 对象，
  `resolveSystemPromptSections()` 统一 resolve + 按 name 取缓存。
  这里用更简单的函数式实现：每段一个 `_section_xxx`，模块级 dict 做缓存。
  WHY 不直接照抄 CC 的 async+Promise：Python 场景里 sync 更简洁，教学子集够用。

三类加载策略：
  identity  — 始终加载：角色 + 工作方法（传 identity 参数可切 profile）
  tools     — 始终加载：从 ToolPool 渲染可用工具清单，工具增减自动变
  workspace — 始终加载：工作目录（从 executor.cwd 取真实值）
  memory    — 按需加载：memory 启用 → 注入稳定 policy；动态 index 不进入 system

缓存：模块级 dict _CACHE，key = (section_key, 该段输入指纹)。
      输入不变直接取缓存；输入变（工具增减/workdir/memory mode 切换）→ 重算。
      对齐 CC `getSystemPromptSectionCache`（bootstrap/state.ts）的 per-section 缓存颗粒度。
"""

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Optional

from ..memory.prompts import build_memory_policy
from ..tools.pool import assemble_tool_pool


# ── 默认 identity 段 ──────────────────────────────────────────────────────────
# 默认产品行为使用已确认的 legacy identity；实验 identity 仅用于 eval before/after 对照。
LEGACY_IDENTITY = """你是一个 coding agent，在代码仓库里完成软件工程任务。

## 工作方法（按步骤）
1. 理解任务描述的问题与期望行为。
2. **定位（最关键）**：用 grep / glob 按关键词（函数名、类名、报错原文、符号）搜索，缩小到具体文件；用 read_file 读候选确认。仓库可能上千文件——靠**搜索关键词**定位，别逐个浏览、别臆测。
3. 读相关代码，想清根因在哪个函数 / 哪几行。
4. 用 edit_file 做改动。
5. 在环境支持的前提下，用 bash 跑相关测试 / 复现脚本验证你的修复。

## 原则
- 先搜索定位再动手，不臆测文件内容。
- 一次 grep 没命中，换关键词再试（同义词、报错原文、类名、相关概念）。
- 任务开头通常会给你仓库结构，据此挑搜索范围。
- 用 update_todos 维护计划（定位→诊断→修复→验证），每推进一步就更新。
- 该做的做完就收尾：用一两句话说清改了哪个文件、为什么（无需用满轮次）。"""

# WHY 这份实验 identity 曾同时改写工具名表达和验证契约，A/B 结果不可归因；
# 保留它用于复现实验，不作为默认产品行为。
EXPERIMENTAL_IDENTITY = """你是一个 coding agent，在代码仓库里完成软件工程任务。

## 工作方法（按步骤）
1. 理解任务描述的问题与期望行为。
2. **先定位，再改动**：围绕函数名、类名、报错原文、符号和用户描述关键词搜索，缩小到具体文件；读取候选代码确认。仓库可能上千文件，不要逐个浏览、不要臆测文件内容。
3. 读相关代码，想清根因在哪个函数 / 哪几行，以及改动会影响哪些调用方或行为面。
4. 做最小、直接、可解释的实现改动，不顺手扩展任务范围。
5. 完成前验证真实行为：优先用项目已有测试、构建命令、复现脚本或直接执行路径验证改动影响的行为；检查输出是否符合预期，不把命令结束当成成功。
6. 最后一次源码改动后，重跑最相关的验证。如果之前见过失败，不能用更窄或无关的通过结果覆盖它；要重跑同一失败检查，或说明为什么它已不相关。

## 原则
- 先搜索定位再动手，不臆测文件内容。
- 一次 grep 没命中，换关键词再试（同义词、报错原文、类名、相关概念）。
- 任务开头通常会给你仓库结构，据此挑搜索范围。
- 对需要多步处理的任务，维护一个短计划；每推进一步就更新。
- 验证强度匹配任务风险：文档/注释类小改可以说明无需运行测试；代码、配置、行为变更必须验证，或明确说出无法验证的具体原因。
- 该做的做完并完成验证后就收尾：用一两句话说清改了哪个文件、为什么、验证结果是什么。
"""

# Eval-only treatment derived from model-independent engineering rules in
# Claude Code's `constants/prompts.ts`. It intentionally stays Chinese and
# keeps the local tool vocabulary, so the experiment changes engineering
# discipline rather than language or runtime capabilities.
CC_CORE_IDENTITY_CN = """你是一个 coding agent，在代码仓库里完成软件工程任务。

## 工作方法
1. 理解任务描述的问题、边界与期望行为。
2. 先定位再动手：用 grep / glob 按函数名、类名、报错原文、符号和相关概念搜索；用 read_file 阅读候选文件。仓库可能很大，不要逐个浏览，也不要臆测文件内容。
3. 修改前先读懂相关实现、调用方和既有约束，确认根因所在；不要对没有读过的代码提出或实施改动。
4. 做满足任务所需的最小、直接改动。不要顺手添加功能、重构无关代码、增加假想需求的抽象或兼容层。
5. 修改源码优先使用 edit_file，创建文件使用 write_file，bash 只用于确实需要终端执行的命令。如果工具或方案失败，先读错误、检查假设并诊断原因，再做聚焦修正；不要盲目重复同一动作，也不要在一次失败后放弃仍然可行的方案。
6. 完成前验证真实行为：运行与改动相关的已有测试、复现脚本、构建或直接执行路径，并检查输出是否符合预期；不能只把命令结束当成成功。最后一次源码改动后，重新运行最相关的验证。

## 原则
- 多步骤任务用 update_todos 维护简短计划，完成一步就及时更新；简单任务不必为了形式增加计划。
- 失败过的相关检查没有重新通过，也没有证据说明它与改动无关时，不要用更窄或无关的通过结果宣称完成。
- 如实报告结果：测试失败就说明失败和关键输出；没有运行验证就明确说明；无法验证时给出具体原因。不要把未完成或损坏的工作描述成已完成。
- 该做的做完并完成必要验证后就收尾：简洁说明改了什么、为什么，以及实际验证结果。
"""

# 默认身份回退到用户确认过的 legacy 基线；更强验证契约留在 eval 显式开关里继续研究。
DEFAULT_IDENTITY = LEGACY_IDENTITY


@dataclass
class SystemState:
    """驱动 build_system 的真实运行时状态——输入变则段重算，输入稳则命中缓存。

    WHY dataclass：让调用方显式声明「哪些状态影响 prompt」，对齐 CC
    `getSystemPromptSectionCache` 的 per-section 缓存颗粒度。
    每个字段是一个「段输入」；字段不变 → 该段 fingerprint 不变 → 缓存命中。
    """
    # 工具清单；默认由 ToolPool 派生，调用方可显式传入受限工具集。
    tools: list = field(default_factory=lambda: assemble_tool_pool().prompt_tools_for_system())
    # 工作目录：从 executor.cwd 取真实值；兜底 os.getcwd()
    workdir: str = field(default_factory=os.getcwd)
    # 记忆目录只用于稳定 policy 中的工具路径说明；不会读取 MEMORY.md。
    memory_dir: Optional[str] = None
    memory_enabled: bool = False
    memory_recall_mode: str = "selector"


# ── 模块级缓存（对齐 CC getSystemPromptSectionCache）────────────────────────
# key = (section_key, 该段输入指纹)；value = 渲染后的段文本（或 None）。
# WHY 模块级：跨多次 build_system 调用共享缓存，不重复渲染不变的段（CC session 内单例缓存同义）。
_CACHE: dict[tuple, "str | None"] = {}


def _fingerprint(val) -> str:
    """对段输入做稳定短指纹：str 直接 hash；list（tools）序列化工具名后 hash。

    WHY MD5 截断：不需要密码学强度，只需碰撞率够低 + 计算轻量。
    """
    if isinstance(val, str):
        return hashlib.md5(val.encode("utf-8")).hexdigest()[:16]
    # list（ToolPool prompt view）：name + description 都是 tools section 输入。
    try:
        s = json.dumps(
            [
                {
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                }
                for t in val
            ],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except Exception:
        s = str(val)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:16]


# ── 各段实现 ──────────────────────────────────────────────────────────────────


def _section_identity(identity: str) -> str:
    """Identity 段：始终加载。输入 = identity 字符串本体；稳定则命中缓存。"""
    key = ("identity", _fingerprint(identity))
    if key not in _CACHE:
        # rstrip 去尾换行，让 "\n\n".join 接缝整齐
        _CACHE[key] = identity.rstrip("\n")
    return _CACHE[key]  # type: ignore[return-value]


def _section_tools(state: SystemState) -> str:
    """Tools 段：从 state.tools 渲染可用工具清单。工具增减 → 指纹变 → 重算。

    WHY 状态驱动而非硬编码：工具池（ToolPool）可在运行时扩展；
    这段跟着走，模型永远看到最新工具列表，无需手动同步 prompt。
    """
    key = ("tools", _fingerprint(state.tools))
    if key not in _CACHE:
        lines = ["## 可用工具"]
        for t in state.tools:
            name = t.get("name", "?")
            # description 可能多行，只取第一行（截断到单行），保持列表整洁
            desc = (t.get("description") or "").split("\n")[0].strip()
            lines.append(f"- {name}：{desc}")
        _CACHE[key] = "\n".join(lines)
    return _CACHE[key]  # type: ignore[return-value]


def _section_workspace(state: SystemState) -> str:
    """Workspace 段：始终加载工作目录。executor 切换（local↔docker）时 workdir 变 → 重算。"""
    key = ("workspace", _fingerprint(state.workdir))
    if key not in _CACHE:
        _CACHE[key] = f"## 工作目录\n{state.workdir}"
    return _CACHE[key]  # type: ignore[return-value]


def _section_memory(state: SystemState) -> "str | None":
    """Return stable memory behavior rules without dynamic index content.

    CC ``loadMemoryPrompt`` places ``buildMemoryLines`` in the system prompt,
    while AutoMem index data is routed through user context and skipped in
    selector mode.  The cache key therefore contains only stable policy inputs.
    """
    if not state.memory_enabled or not state.memory_dir:
        return None
    mode = (
        state.memory_recall_mode
        if state.memory_recall_mode in {"selector", "index"}
        else "selector"
    )
    key = ("memory", _fingerprint(f"{state.memory_dir}|{mode}"))
    if key not in _CACHE:
        _CACHE[key] = build_memory_policy(
            state.memory_dir,
            skip_index=mode == "selector",
        )
    return _CACHE[key]


# ── 公开接口 ──────────────────────────────────────────────────────────────────


def build_system(state: SystemState, identity: str = DEFAULT_IDENTITY) -> str:
    """按 identity→tools→workspace→memory 顺序构建完整 system prompt。

    每段独立缓存：输入不变直接取缓存，对齐 CC `getSystemPromptSectionCache`。
    memory 段缺席时自动跳过，不留空行。
    identity 可传入不同字符串，支持 profile 切换而不重启进程。
    """
    sections = [
        _section_identity(identity),
        _section_tools(state),
        _section_workspace(state),
        _section_memory(state),
    ]
    # 过滤 None（按需段缺席），段间双换行，对齐 CC section 拼接惯例
    return "\n\n".join(s for s in sections if s is not None)

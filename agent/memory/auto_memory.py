"""Auto Memory 写入路径（slice 3a）—— 跨会话经验记忆。

参考 Claude Code 公开技术资料的设计要点：
  - `executeExtractMemories`（`extractMemories.ts:415`）在每轮 query loop 终止（无 tool_call
    终态回复）时 fire-and-forget 调用，**仅主 agent**（`stopHooks.ts:149,154`）。
  - 内部跑 `runForkedAgent`：子 agent 用工具读/写 `~/.claude/projects/<git-root>/memory/`，
    写四类记忆文件 + 更新 MEMORY.md 索引（`extractMemories.ts:171-222` createAutoMemCanUseTool）。
  - 存储：平铺 frontmatter `name/description/type`。
  - MEMORY.md 双截：≤200 行 AND ≤25KB（`memdir.ts:35-38` + `truncateEntrypointContent`）。

⚠ 偏离诚实标注（要点，见各函数内就地注释）：
  1. **fork JSON + harness 落盘**（非 CC 的 fork 用工具写）：
     我们的工具受 LocalExecutor WORKDIR 沙箱限制、记忆目录在 workspace 外 → 工具写会越权
     （真实端到端实测；同 D-M8，`session_memory.py` 同款偏离）。
     故 fork 只做文本生成（不给工具）、harness 自己 Path.write_text 落盘。
  2. **同步执行**：CC fire-and-forget（不阻塞主线）；我们 eval 场景**同步**调用（简化、够用）。
     TODO：真接交互式 loop 时改异步（loop.py 接缝已留 TODO）。
  3. **平铺 frontmatter**（非嵌套 `metadata:`）：使用参考设计的平铺形态
     （`name/description/type` 顶层字段）。⚠ 本 harness 自身 auto-memory（`~/.claude/projects/.../memory/`
     由主 agent 写）用 `metadata: type:` 嵌套变体——那是本 harness 自己的约定。
     两种形态按各自目录隔离，并存不冲突。
  4. **无 prompt cache 省钱**（R-M1）：CC fork 共享主对话 prompt cache → 多数输入 cache hit
     按 cached token 计费；deepseek 代理无 prompt cache → 每次 fork 真金白银，成本要测。
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

from . import secret_scan
from .forked_agent import run_forked_agent
from .. import llm as _llm_mod

log = logging.getLogger(__name__)

# 四类记忆：每类的描述和何时使用
_MEMORY_TYPES: dict[str, str] = {
    "user":      "用户是谁、长期偏好、行为模式（如编程语言偏好、工作风格、目标）",
    "feedback":  "给 agent 的工作指正——被纠正的工作方式（正文须含 **Why:** 和 **How to apply:** 行）",
    "project":   "正在做的事、长期约束、里程碑（相对日期须转为绝对日期）",
    "reference": "外部资源指针（URL、dashboard、ticket、文档链接等）",
}


@dataclass
class AutoMemoryConfig:
    """记忆文件、索引和 fork 预算的配置。"""
    # MEMORY.md 双截上限（`memdir.ts:35-36` MAX_ENTRYPOINT_LINES / MAX_ENTRYPOINT_BYTES）
    max_entrypoint_lines: int = 200
    max_entrypoint_bytes: int = 25_000
    # 每个记忆文件只读前 N 行取 frontmatter（`memoryScan.ts:22` FRONTMATTER_MAX_LINES）
    frontmatter_max_lines: int = 30
    # 扫描文件数上限（`memoryScan.ts:21` MAX_MEMORY_FILES）
    max_memory_files: int = 200
    # fork 生成 JSON：max_tokens 须足以容纳完整 JSON 输出（含多条记忆 body）
    fork_max_tokens: int = 16_384
    # ⚠ 偏离：CC extractMemories maxTurns=5（工具循环 read-then-write）；
    #   我们塌缩成单次文本生成（不给工具）→ 2 轮足够（1 轮生成 + 1 轮容错）。
    max_turns: int = 2


DEFAULT_AM_CONFIG = AutoMemoryConfig()

# ── Tier-2 召回常量（3b-1）──────────────────────────────────────────────────────

# 单文件内容字节上限（对齐 CC attachments.ts MAX_MEMORY_BYTES）
_MAX_MEMORY_BYTES_PER_FILE = 4096

# 选择器默认模型：deepseek-v4-flash（便宜，选 ≤5 这种简单分类任务够用）。
# REPL 可通过 settings.models.recall / env.ACE_RECALL_MODEL 覆盖（见 llm_runtime.py）。
# ⚠ live 实测纠正：API 只有 deepseek-v4-pro / deepseek-v4-flash，无 "fresh"——
#   D-M3 原拍的 "deepseek-v4-fresh" 是不存在的名字，被 best-effort try/except 静默吞 400，
#   只有真打 API 才暴露。语义仍对齐 CC「便宜 LLM 选择器」范式。

# 选择器系统 prompt（对齐 CC findRelevantMemories.ts:18-24 SELECT_MEMORIES_SYSTEM_PROMPT）
# ⚠ 偏离：CC 用英文 Sonnet sideQuery；我们用 deepseek-v4-flash（见 _SELECTOR_MODEL_ID），语义对齐（D-M3）。
_SELECTOR_SYSTEM_PROMPT = (
    "You are selecting memories that will be useful as context for the current task. "
    "You will be given the query and a list of available memory files with their filenames and descriptions.\n\n"
    "Return a list of filenames for the memories that will clearly be useful (up to 5). "
    "Only include memories that you are certain will be helpful based on their name and description.\n"
    "- If you are unsure if a memory will be useful, do not include it. Be selective and discerning.\n"
    "- If there are no memories that would clearly be useful, return an empty list.\n"
    "- If a list of recently-used tools is provided, do not select memories that are usage reference "
    "or API documentation for those tools. DO still select memories containing warnings, gotchas, "
    "or known issues about those tools — active use is exactly when those matter.\n\n"
    'Respond with JSON only: {"selected": ["filename.md", ...]}'
)


class AutoMemory:
    """跨会话经验记忆的写入路径：forked-LLM 判断 + harness 落盘四类记忆文件 + MEMORY.md 索引。

    调用方构造时传入 memory_dir（该 project 的记忆目录），每轮 query loop 结束时调 write()。
    3a 只实现写入路径；检索路径（3b：Sonnet sideQuery 语义召回）另行实现。

    存储路径对齐 CC `paths.ts:223-235`：`<base>/projects/<sanitized-git-root>/memory/`。
    runtime 路径推导（从 git root 派生 key、拼 ~/.claude/projects/）留 3b；
    3a 由调用方/测试直接传入 memory_dir。
    """

    def __init__(self, memory_dir: str | Path, cfg: AutoMemoryConfig = DEFAULT_AM_CONFIG):
        self.memory_dir = Path(memory_dir)
        self.cfg = cfg
        # Phase-2 capture hooks: harness reads these after each session to surface
        # write-fork decisions and recall selections without invasive instrumentation.
        # WHY attributes not return values: write() / recall() already have stable
        # callers (loop.py) that we can't easily change; a side-channel attribute
        # is the minimal-diff approach.
        self.last_write_raw: str | None = None         # raw fork JSON text from write()
        self.last_recall_selected: list[str] | None = None  # filenames chosen by tier-2 sideQuery

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    def write(self, messages: list, system: str = "") -> dict:
        """对话结束后提取跨会话记忆：fork → 解析 JSON → 密钥拦截 → 落盘 → 更新索引。

        对齐 CC「runs once at the end of each complete query loop」（`extractMemories.ts:5-6`），
        仅主 agent 调用（fork/子 agent 不触发，由调用方 loop.py 保证）。

        ⚠ 偏离：CC fire-and-forget（不阻塞主线）；我们**同步**调用（eval 场景够用）。
        TODO：真接交互式 loop 时改异步（loop.py 接缝已留 TODO）。

        返回计数 dict: {written, skipped_secret, total}。
        """
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        # ① 扫现有 manifest（供 fork 去重/更新，免花一轮 ls）
        # 对齐 CC extractMemories.ts:398 预注入已存在 manifest
        manifest = self._scan_manifest()
        existing_index = self._read_index()

        # ② 构建 extraction prompt → 跑 fork（不给工具，fork 只做 JSON 文本生成）
        prompt = _build_extract_prompt(manifest, existing_index)
        # ⚠ 偏离：CC fork 调用 createAutoMemCanUseTool（允许 Read/Grep/Glob + 记忆目录 Edit/Write）；
        #   我们 allowed_tools=set()（不给工具）——理由见模块 docstring 偏离 1。
        res = run_forked_agent(
            prompt, messages,
            system=system,
            allowed_tools=set(),
            max_turns=self.cfg.max_turns,
            max_tokens=self.cfg.fork_max_tokens,
            label="auto_memory",
        )
        # Phase-2 capture: persist raw fork JSON so harness can read write_fork_decision.
        # This is the fork's full response — what it decided to remember (or not).
        self.last_write_raw = res.final_text

        # ③ 解析 fork 输出的 JSON（健壮：去代码块包裹、容错前后缀）
        memories = _parse_memories_json(res.final_text)
        if not memories:
            # 空 list = 本轮无值得跨会话记的，正常，不报错
            return {"written": 0, "skipped_secret": 0, "total": 0}

        # ④ 逐条落盘（密钥拦截 → 写文件 → 更新索引）
        written = skipped = 0
        for mem in memories:
            name = str(mem.get("name", "")).strip()
            description = str(mem.get("description", "")).strip()
            mem_type = str(mem.get("type", "")).strip()
            body = str(mem.get("body", "")).strip()

            if not name or not body:
                log.warning("auto_memory: 跳过缺 name/body 的记忆条目 %r", mem)
                continue
            if mem_type not in _MEMORY_TYPES:
                log.warning("auto_memory: 未知 type %r，默认 reference", mem_type)
                mem_type = "reference"

            # 密钥硬拦截（对齐 CC teamMemSecretGuard 硬阻断，只报 rule_id 不报值）
            # P1-1: 扫 name+description+body，description 进 MEMORY.md 索引同样有泄露风险
            hits = secret_scan.scan(f"{name}\n{description}\n{body}")
            if hits:
                log.warning(
                    "auto_memory: 记忆 %r body 含潜在密钥（%s），跳过写盘",
                    name, ", ".join(hits),
                )
                skipped += 1
                continue

            safe_name = _sanitize_name(name)
            self._write_memory_file(safe_name, name, description, mem_type, body)
            self._update_index(safe_name, description, mem_type)
            written += 1
            log.info("auto_memory: 写入 %s（%s）", safe_name, mem_type)

        return {"written": written, "skipped_secret": skipped, "total": len(memories)}

    # ── Tier-2 召回接口（3b-1）────────────────────────────────────────────────

    def recall(self, query: str, already_surfaced: set, recent_tools: list | None = None) -> list[dict]:
        """Tier-2 选择器召回：sideQuery 选 ≤5 相关记忆 → 读文件 → 返回 [{path, content}]。

        对齐 CC findRelevantMemories.ts（语义）：
          1. 扫 frontmatter manifest（_scan_manifest_entries）
          2. 过滤 already_surfaced（在选择器前过滤，节省 5-slot 预算于新候选）
          3. 调 deepseek-v4-flash sideQuery（⚠ 偏离：CC Sonnet；无 Sonnet key → v4-flash，D-M3）
          4. 解析 JSON → 过滤无效文件名 → 取 ≤5 条
          5. 读文件正文，字节上限 4096（对齐 CC MAX_MEMORY_BYTES，attachments.ts）

        任何环节失败 → 返 []，绝不抛到主循环（best-effort，对齐 CC 健壮性约定）。

        ⚠ 偏离 CC（P2，已知，本 slice 不补）：
          - JSON key 用 `selected`（CC 用 `selected_memories` + json_schema 强约束）；prompt 与
            parser 自洽，仅命名不同。
          - 未实现 CC 的 MAX_SESSION_BYTES（60KB 累计上限）；只做了每文件 4096 字节上限。
          - recent_tools 形参已接（选择器 prompt 含去噪规则），但 loop 暂不传 → 去噪规则当前不触发
            （MVP 后置 recentTools 去噪，见 memory-design backlog）。
        """
        try:
            return self._do_recall(query, already_surfaced, recent_tools or [])
        except Exception as e:
            log.warning("auto_memory.recall 内部错误（不影响主任务）: %s", e)
            return []

    def _do_recall(self, query: str, already_surfaced: set, recent_tools: list) -> list[dict]:
        """召回实现：任何子步允许抛出，由 recall() 统一兜底为 []。"""
        if not self.memory_dir.exists():
            return []

        entries = self._scan_manifest_entries()
        # 过滤已浮现路径（对齐 CC findRelevantMemories.ts:47-49 alreadySurfaced 前置过滤）
        entries = [e for e in entries if str(e["path"]) not in already_surfaced]
        if not entries:
            return []

        # 构建 manifest 文本（必修-2：filename 领头，对齐 CC memoryScan.ts:84-94 formatMemoryManifest）
        # 格式：- {filename} [{type}] ({ISO mtime}): {description}
        # WHY filename 绝对最前：CC 是 `[type] filename` 领头（memoryScan.ts:90 `- ${tag}${m.filename}...`）；
        # 我们把 filename 提到绝对最前 → 主键=校验键（valid_names={filename}），弱选择器照抄领头 token 必中。
        manifest_text = "\n".join(
            f"- {e['path'].name} [{e['type']}] ({_iso_mtime(e['mtime'])}): {e['description']}"
            for e in entries
        )
        tools_section = (
            f"\n\nRecently used tools: {', '.join(recent_tools)}" if recent_tools else ""
        )
        user_content = f"Query: {query}\n\nAvailable memories:\n{manifest_text}{tools_section}"

        # sideQuery：独立轻量 LLM 调用，非 fork、非主模型
        # ⚠ 偏离：CC 用 Sonnet（getDefaultSonnetModel()）；我们用 deepseek-v4-flash（D-M3，无 Sonnet key）
        resp = _llm_mod.chat(
            messages=[{"role": "user", "content": user_content}],
            system=_SELECTOR_SYSTEM_PROMPT,
            tools=None,
            max_tokens=256,
            purpose="memory_recall",
        )
        text = "".join(
            getattr(b, "text", "") for b in resp.content
            if getattr(b, "type", None) == "text"
        )
        selected_filenames = _parse_selector_json(text)

        # 只取有效文件名（防选择器幻觉），上限 5（对齐 CC ≤5 返回）
        valid_names = {e["path"].name for e in entries}
        selected_filenames = [f for f in selected_filenames if f in valid_names][:5]
        # Phase-2 capture: persist selected filenames so harness can read recall_tier2_files.
        # WHY here and not after the empty-list check: even an empty selection is informative
        # ("sideQuery ran but selected nothing"), whereas None means "recall never ran."
        self.last_recall_selected = list(selected_filenames)
        if not selected_filenames:
            return []

        by_name = {e["path"].name: e for e in entries}
        result = []
        for fname in selected_filenames:
            path = by_name[fname]["path"]
            try:
                raw = path.read_bytes()
                if len(raw) > _MAX_MEMORY_BYTES_PER_FILE:
                    # 在最后一个换行处截（避免切断 UTF-8 多字节序列中间）
                    truncated = raw[:_MAX_MEMORY_BYTES_PER_FILE]
                    nl = truncated.rfind(b"\n")
                    if nl > 0:
                        truncated = truncated[:nl]
                    content = truncated.decode("utf-8", errors="replace")
                    content += f"\n[已截断到 {_MAX_MEMORY_BYTES_PER_FILE} 字节]"
                else:
                    content = raw.decode("utf-8", errors="replace")
            except OSError:
                log.warning("auto_memory.recall: 跳过无法读取的文件 %s", path)
                continue
            # 必修-1a：mtime 穿到结果，供 loop 渲染新鲜度头（对齐 CC attachments.ts:2327-2332）
            result.append({"path": str(path), "content": content, "mtime": by_name[fname]["mtime"]})

        return result

    # ── helpers ───────────────────────────────────────────────────────────────

    def _scan_manifest(self) -> str:
        """扫现有记忆文件，取前 frontmatter_max_lines 行提取 name/description/type，
        拼成 manifest 文本注入 extraction prompt。

        对齐 CC `memoryScan.ts:21-22`（MAX_MEMORY_FILES / FRONTMATTER_MAX_LINES）：
        每个文件只读前 30 行——frontmatter 一定在前几行，避免读完整大文件拖慢速度。
        扫描上限 200 个文件（防目录里文件过多导致 prompt 过大）。
        """
        files = sorted(self.memory_dir.glob("*.md"))
        # 排除 MEMORY.md 索引本身（对齐 CC `memoryScan.ts:42` MEMORY.md 不进 scan）
        files = [f for f in files if f.name != "MEMORY.md"]
        files = files[: self.cfg.max_memory_files]

        lines = []
        for f in files:
            fm = _read_frontmatter(f, self.cfg.frontmatter_max_lines)
            name = fm.get("name", f.stem)
            desc = fm.get("description", "")
            mtype = fm.get("type", "?")
            lines.append(f"- [{mtype}] {name}: {desc} ({f.name})")

        return "\n".join(lines)

    def _scan_manifest_entries(self) -> list[dict]:
        """扫现有记忆文件，返回结构化条目列表 [{path, name, description, type, mtime}]。

        与 _scan_manifest() 逻辑相似，但返回结构化数据供 recall() 选择器使用。
        对齐 CC memoryScan.ts scanMemoryFiles 返回 MemoryHeader 数组的设计。

        必修-2 新增：
        - 每条带 mtime（float 秒级，via f.stat().st_mtime）。
        - 按 mtime 新→旧排序取前 max_memory_files（对齐 CC memoryScan.ts:72-73 sort newest-first）。
        """
        files = [f for f in self.memory_dir.glob("*.md") if f.name != "MEMORY.md"]

        entries = []
        for f in files:
            try:
                mtime = f.stat().st_mtime   # float seconds since epoch（OS mtime）
            except OSError:
                mtime = 0.0
            fm = _read_frontmatter(f, self.cfg.frontmatter_max_lines)
            entries.append({
                "path": f,
                "name": fm.get("name", f.stem),
                "description": fm.get("description", ""),
                "type": fm.get("type", "?"),
                "mtime": mtime,
            })

        # 对齐 CC memoryScan.ts:72-73：newest-first 排序，取前 max_memory_files
        entries.sort(key=lambda e: e["mtime"], reverse=True)
        return entries[: self.cfg.max_memory_files]

    def _read_index(self) -> str:
        """读现有 MEMORY.md（注入 extraction prompt 供 fork 参考）。不存在则返空串。"""
        idx = self.memory_dir / "MEMORY.md"
        if not idx.exists():
            return ""
        return idx.read_text(encoding="utf-8")

    def _write_memory_file(
        self, safe_name: str, name: str, description: str, mem_type: str, body: str
    ) -> None:
        """落盘记忆文件。

        frontmatter 格式**平铺**：
          ---
          name: kebab-slug
          description: one-line summary
          type: user|feedback|project|reference
          ---
        本 harness 另有 `metadata: type:` 嵌套变体；两者按目录隔离。
        """
        # P2-4: 控制字符进 YAML 会产生无效 frontmatter
        name = _strip_ctl(name).strip()
        description = _strip_ctl(description).strip()
        content = (
            f"---\nname: {name}\ndescription: {description}\ntype: {mem_type}\n---\n\n{body}\n"
        )
        path = self.memory_dir / f"{safe_name}.md"
        path.write_text(content, encoding="utf-8")

    def _update_index(self, safe_name: str, description: str, mem_type: str) -> None:
        """更新 MEMORY.md 索引：已存在同名条目则替换，否则追加。

        行格式（对齐 CC MEMORY.md 只做索引，无 Type: 前缀）：
          `- [safe-name](safe-name.md) — description`

        P0：磁盘 MEMORY.md 永远写完整内容。CC truncateEntrypointContent 是**读时注入**
        （memdir.ts:296 + claudemd.ts:384 两处），从不写截断结果回盘。
        截断逻辑仅在 Index mode 构建 conversation-cached user context 时调用。
        """
        idx_path = self.memory_dir / "MEMORY.md"
        existing = idx_path.read_text(encoding="utf-8") if idx_path.exists() else "# Memory Index\n"
        lines = existing.splitlines(keepends=True)

        # P2-7: 去掉 Type: 前缀（CC MEMORY.md 是纯索引，无类型标签）
        # R1: description 来自 LLM，含换行会破坏索引「一条一行」不变量 → 剥控制字符
        new_line = f"- [{safe_name}]({safe_name}.md) — {_strip_ctl(description)}\n"
        link_token = f"({safe_name}.md)"

        replaced = False
        for i, line in enumerate(lines):
            if link_token in line:
                lines[i] = new_line
                replaced = True
                break
        if not replaced:
            lines.append(new_line)

        # P0: 写完整内容，不截断
        idx_path.write_text("".join(lines), encoding="utf-8")

    def truncate_index_for_injection(self, content: str) -> str:
        """对齐 CC `truncateEntrypointContent`（`memdir.ts:57-103`）的双截断逻辑。

        P0: 此方法是**读时注入**截断，仅在 Index mode 创建 user-context 快照时调用。
        CC 中 truncateEntrypointContent 有且仅有两处调用（memdir.ts:296 + claudemd.ts:384），
        均在读取路径，从不写截断结果回磁盘。

        先按行数（max_entrypoint_lines=200）截；再按字节数（max_entrypoint_bytes=25KB）截
        （在最后一个换行处切，避免切断 UTF-8 多字节序列）；触发任一截断就追加 WARNING。

        ⚠ 偏离 CC（R2，读侧/3b 才用，优先级低，TODO 接 3b 时校准）：
          ① 字节度量用 UTF-8 字节数；CC 用 `trimmed.length`（JS UTF-16 code unit，memdir.ts:61）
             → CJK 描述下我们更早触发字节截断。
          ② `wasByteTruncated` CC 在**原始** byteCount 上判（memdir.ts:66）；我们在**行截断后**的
             内容上判 → WARNING 触发条件与 CC 略不一致。
        """
        triggered = []
        lines = content.splitlines(keepends=True)

        # 第一截：按行数
        if len(lines) > self.cfg.max_entrypoint_lines:
            lines = lines[: self.cfg.max_entrypoint_lines]
            triggered.append(f"max_entrypoint_lines={self.cfg.max_entrypoint_lines}")

        content = "".join(lines)

        # 第二截：按字节数（UTF-8 编码后字节数）
        encoded = content.encode("utf-8")
        if len(encoded) > self.cfg.max_entrypoint_bytes:
            truncated = encoded[: self.cfg.max_entrypoint_bytes]
            # 在最后一个换行处切（避免切断多字节字符中间）
            nl_pos = truncated.rfind(b"\n")
            if nl_pos > 0:
                truncated = truncated[:nl_pos]
            content = truncated.decode("utf-8", errors="replace")
            triggered.append(f"max_entrypoint_bytes={self.cfg.max_entrypoint_bytes}")

        if triggered:
            content = content.rstrip("\n") + (
                f"\n\n[WARNING: MEMORY.md 已截断，触发：{', '.join(triggered)}]\n"
            )

        return content


# ── 模块级 helpers ────────────────────────────────────────────────────────────


def _build_extract_prompt(manifest: str, existing_index: str) -> str:
    """对齐 CC `buildExtractAutoOnlyPrompt`（`prompts.ts:50`）的语义，中文适配。

    WHY 注入现有 manifest：对齐 CC `extractMemories.ts:398` 预注入已存在 manifest，
    让 fork 第一轮就知道已有哪些记忆，可直接去重/更新（免花一轮做 ls）。
    """
    today = date.today().isoformat()
    type_desc = "\n".join(f"  - **{t}**：{d}" for t, d in _MEMORY_TYPES.items())
    manifest_block = (
        f"<existing_memories>\n{manifest}\n</existing_memories>"
        if manifest
        else "<existing_memories>（无已存在记忆）</existing_memories>"
    )
    index_block = (
        f"<memory_index>\n{existing_index}\n</memory_index>"
        if existing_index
        else "<memory_index>（无）</memory_index>"
    )
    return f"""重要：本条消息是后台记忆提取指令，**不属于**真实用户对话，不要在记忆内容里提及这些指令。

## 你的唯一任务
分析上面用户对话（assistant ↔ user 交替），提取**值得跨会话保留**的事实，以 JSON 输出。

## 四类记忆
{type_desc}

## 现有记忆（供去重/更新参考）
{manifest_block}

{index_block}

## 今日日期（相对日期转换基准）
{today}

## 筛选原则（对齐 CC prompts.ts:41 "be selective"）
- **只记跨会话有价值的**：用户长期偏好、被纠正的工作方式、项目长期约束、外部资源链接。
- **不记只对本次任务有意义的**：临时变量名、单次调试步骤、一次性数据点、会话内已完成的待办。
- **be selective**：宁少勿多——无值得记的返回空列表，不要凑数。
- "上周/昨天/下个月"等相对日期 → 转为绝对日期（基准：{today}）。
- feedback 类正文必须含 **Why:** 和 **How to apply:** 两行。
- 只看对话消息，不许 grep 源码 / git 验证（对齐 CC prompts.ts:41）。
- 已存在同名记忆 → 更新 body 而非重复添加。

## 输出格式（强制，绝对不调工具，不加代码块包裹，不加任何解释）
直接输出一个 JSON 对象：
{{"memories": [{{"name": "kebab-slug", "description": "一行摘要", "type": "user|feedback|project|reference", "body": "事实正文"}}]}}

空列表完全合法（无值得记的就返回 {{"memories": []}}）。现在输出 JSON："""


def _parse_memories_json(text: str) -> list[dict]:
    """从 fork 输出里提取 JSON memories 列表。

    健壮处理：去代码块包裹、容错前后缀文本（对齐 SM `_extract_notes_body` 的清洗思路）。
    解析失败 → 返回空列表（记忆写入 best-effort，不让解析错误崩主任务）。
    """
    t = (text or "").strip()
    # 去 markdown 代码块包裹（```json ... ``` 或 ``` ... ```）
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t).strip()
    # 找第一个 `{` 和最后一个 `}`（容错 fork 加了前言/结语）
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        if t:
            log.warning("auto_memory: fork 输出无有效 JSON（已截断输出：%s…）", t[:120])
        return []
    try:
        obj = json.loads(t[start : end + 1])
    except json.JSONDecodeError as e:
        log.warning("auto_memory: JSON 解析失败 (%s)；已截断输出：%s…", e, t[:120])
        return []
    mems = obj.get("memories", [])
    if not isinstance(mems, list):
        return []
    # P2-5: 过滤非 dict 条目（LLM 偶尔输出 null/string 混入列表）
    return [m for m in mems if isinstance(m, dict)]


def _parse_selector_json(text: str) -> list[str]:
    """从选择器 LLM 输出里提取 {"selected": [...]} 文件名列表。

    健壮处理：去代码块包裹、找首尾括号、失败返 []（对齐 _parse_memories_json 健壮策略）。
    """
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t).strip()
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        if t:
            log.warning("auto_memory: 选择器输出无有效 JSON（已截断：%s…）", t[:80])
        return []
    try:
        obj = json.loads(t[start: end + 1])
    except json.JSONDecodeError as e:
        log.warning("auto_memory: 选择器 JSON 解析失败 (%s)；输出：%s…", e, t[:80])
        return []
    sel = obj.get("selected", [])
    if not isinstance(sel, list):
        return []
    return [s for s in sel if isinstance(s, str)]


def _read_frontmatter(path: Path, max_lines: int) -> dict[str, str]:
    """只读文件前 max_lines 行，提取平铺 YAML frontmatter（name/description/type）。

    对齐 CC `memoryScan.ts:22` FRONTMATTER_MAX_LINES=30：只读前 30 行取 frontmatter，
    不读完整文件——frontmatter 一定在文件头几行，大文件无需全量读。
    """
    try:
        with path.open(encoding="utf-8") as f:
            head = [next(f, None) for _ in range(max_lines)]
        head = [l for l in head if l is not None]
    except OSError:
        return {}

    in_fm = False
    fm: dict[str, str] = {}
    for line in head:
        stripped = line.rstrip("\n")
        if stripped == "---":
            if not in_fm:
                in_fm = True
                continue
            else:
                break   # 第二个 --- = frontmatter 结束
        if in_fm and ":" in stripped:
            key, _, val = stripped.partition(":")
            fm[key.strip()] = val.strip()
    return fm


def _iso_mtime(mtime: float) -> str:
    """float 秒级时间戳 → ISO 8601 UTC 字符串（对齐 CC `new Date(m.mtimeMs).toISOString()` 格式）。

    用于 manifest 行的时间戳字段（memoryScan.ts:84-94 `const ts = new Date(m.mtimeMs).toISOString()`）。
    """
    return datetime.fromtimestamp(mtime, tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.000Z')


def _strip_ctl(s: str) -> str:
    """剥控制字符（替为空格）。值来自 LLM、不可信：进 YAML frontmatter 产出无效结构，
    进 MEMORY.md 索引行的换行会破坏「一条记忆一行」不变量（R1）。不碰 ≥0x80 的 CJK/可见字符。"""
    return re.sub(r'[\x00-\x1f\x7f]', ' ', s)


# P1-3: Windows 保留文件名 + "memory" → MEMORY.md 冲突（Windows 大小写不敏感）
_RESERVED_NAMES = frozenset({
    "memory",
    "con", "prn", "aux", "nul",
    *[f"com{i}" for i in range(1, 10)],
    *[f"lpt{i}" for i in range(1, 10)],
})


def _sanitize_name(name: str, max_len: int = 80) -> str:
    """name 字段 → 安全文件名：小写、非字母数字/连字符替换成连字符、去首尾连字符。

    P1-3 补丁：
    - 控制字符先替换为 '-'（防 YAML 嵌入漏洞）
    - 长度上限 80（防过长路径）
    - fallback "untitled-memory"（旧 "memory" → MEMORY.md 碰撞，Windows 大小写不敏感）
    - 命中 _RESERVED_NAMES → 前缀 "note-"（Windows CON/PRN/AUX/NUL/COM*/LPT* 保留名）
    """
    name = re.sub(r'[\x00-\x1f\x7f]', '-', name)
    safe = re.sub(r'[^a-z0-9-]', '-', name.lower())
    safe = re.sub(r'-+', '-', safe).strip('-')[:max_len].rstrip('-')
    if not safe:
        return "untitled-memory"
    if safe in _RESERVED_NAMES:
        return f"note-{safe}"
    return safe

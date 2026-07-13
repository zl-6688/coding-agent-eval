"""大结果落盘，设计参考 Claude Code 公开技术资料。

CC 的"大结果落盘"层（4 层压缩梯度里最轻的一层，发模型前最先跑）：
工具输出超阈值 → 把**完整内容写到磁盘** → 上下文里**替换成"指针 + 预览"**，
模型需要全文时用 read_file 按指针重读。

这是 CC "上下文=对外部真相(文件)的 cache" 哲学的直接体现：
**不可恢复地删（micro/full）用得最少；优先用"落盘留指针/可重读"这种可恢复手段。**
解决我们一直的 "recoverable≠usable" 缺口的另一半（大输出不撑爆上下文）。

── 诚实定位（关于 "可恢复≠可用"，2026-06-23）──
落盘是**软指针**：可用性不保证 100%（模型可能需要全文却不重读）。**CC 亦然**——
参考设计对落盘的工具输出**也没有确定性自动注入**（只保证指针路径可读，读不读是
模型的选择）。CC 不靠"软提示赌模型读"，而是用**一套设计把"需读却没读"的概率压低**
（见下 _PERSIST_MSG 各字段的哲学）。这是工业级"概率工程"：不追求理论确定性，靠多层
设计叠加降失败率。（真正的确定性兜底是 CC 的 Attachment System——每轮自动重注入变更
文件，那是**独立子系统、非压缩层**，我们排在 memory 之后再评估。）

参考实现使用的常量：
  DEFAULT_MAX_RESULT_SIZE_CHARS = 50_000   PREVIEW_SIZE_BYTES = 2000
我们规模小（任务上下文才几十 K），阈值取 8_000 字符更易触发、可观测。
"""

from pathlib import Path

from .. import config

PERSIST_THRESHOLD_CHARS = 8_000     # 超此字符数的工具输出落盘（CC 是 50K；我们规模小取 8K）
# 预览大小 = "答案命中率 vs 上下文节省" 的权衡：太小→答案常不在预览里→退化成纯路径被迫重读；
#   太大→省的上下文少、落盘失去意义。CC 取 2KB（占其 50K 阈值的 4%），覆盖"多数查询的答案在
#   输出头部"这一经验。对齐之。数值最优点本可用 eval 找（不同预览大小下命中率 vs 节省），暂不做。
PREVIEW_CHARS = 2_000
PERSIST_OPEN = "<persisted-output>"
PERSIST_CLOSE = "</persisted-output>"
# 不落盘的工具：它们的输出本就是"引用/路径"，落盘没意义或会循环
_SKIP_TOOLS = {"read_file", "glob", "grep", "update_todos"}


def _store_dir() -> Path:
    d = config.TRACES_DIR / ".tool_results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def is_persisted_path(path: str) -> bool:
    """判断一个 read 的路径是不是落盘文件（用于观测：agent 是否主动重读落盘内容）。"""
    try:
        return ".tool_results" in str(path)
    except Exception:
        return False


def maybe_persist(tool_name: str, tool_id: str, output: str) -> tuple[str, bool, str]:
    """大输出落盘 → 返回 (给模型看的内容, 是否落盘, 落盘文件路径或"")。

    落盘内容写到 TRACES_DIR/.tool_results（**仓库外**，不污染 agent 的 git diff/patch）；
    上下文替换成指针 + 预览。落盘失败时退化为原样返回（绝不掀翻 agent）。
    """
    if tool_name in _SKIP_TOOLS or output.startswith("Error"):
        return output, False, ""
    if len(output) <= PERSIST_THRESHOLD_CHARS:
        return output, False, ""
    try:
        fp = _store_dir() / f"{tool_id}.txt"
        fp.write_text(output, encoding="utf-8")
        preview = output[:PREVIEW_CHARS]
        more = "\n..." if len(output) > PREVIEW_CHARS else ""
        # 指针消息的每个字段都对应一个"降低需读却没读概率"的设计（对照 CC buildLargeToolResultMessage）：
        #   ① <persisted-output> 标签：结构化边界，模型对标签敏感、知道这是被特殊处理的块（非普通输出）
        #   ② 报字符数：给模型"值不值得读回来"的量化判断依据（而非模糊"已保存"）
        #   ③ 绝对路径 + 明确"用 read_file 读"：可操作性，零歧义降低重读摩擦
        #   ④ 预览：让多数情况"不用读就拿到答案"（答案常在头部）= 概率性可用的第一道命中
        #   ⑤ "..."：诚实信号，告诉模型这不是全部、别把预览当完整输出
        msg = (f"{PERSIST_OPEN}\n"
               f"输出过大（{len(output):,} 字符）。完整内容已存到：{fp}\n"
               f"需要完整内容时用 read_file 读这个路径。\n\n"
               f"预览（前 {PREVIEW_CHARS} 字符）：\n{preview}{more}\n"
               f"{PERSIST_CLOSE}")
        return msg, True, str(fp)
    except Exception:
        return output, False, ""   # 落盘失败不影响主流程

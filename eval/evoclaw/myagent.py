"""BYO-agent framework: coding-agent-eval 的压缩 agent（"myagent"）。

★ 部署位置：本文件是 **EvoClaw 侧** adapter，运行时必须落在 EvoClaw 仓库的
   `harness/e2e/agents/myagent.py`。这里（coding-agent-eval/eval/evoclaw/）是**可评审的源真相 +
   部署源**——worktree 隔离不能直接写 EvoClaw 仓库，故在本仓库版本管理，部署时 cp 进 EvoClaw clone
   （见同目录 deploy.sh）。作为本仓库文件它 import 不了 harness.*，正常——它只在 EvoClaw 里被 import。

DEVIATION (coding-agent-eval, 2026-06-26): 本文件是我们为 compression-eval 新增的 EvoClaw
agent adapter，非 EvoClaw 上游内容。仿 harness/e2e/agents/claude_code.py 实现 AgentFramework
的 4 抽象 + 2 可选。架构 = B1（agent-in-container + LocalExecutor，cwd=/testbed）。
设计权威：docs/compression-eval/01-design-framework.md §2.4 + docs/evoclaw-validation.md §7b.4。
TODO（升级冲突）：本文件是 EvoClaw 新增文件，上游升级不直接冲突；但 base.py 契约若变需同步。

它与 claude-code 的关键差异：
- 不依赖 claude.ai 安装源（被 Cloudflare 墙，validation §7b.7）；agent 只需 `pip install
  anthropic python-dotenv`（PyPI 白名单可达）+ 只读挂载我们的源码进容器。
- 上下文压缩在**我们的 loop 内**（micro/full_compact/pipeline），由 COMPACT_STRATEGY 切三臂——
  这正是用 EvoClaw 测我们压缩的前提（claude-code 用的是 CC 自己的 compaction，测不到我们的）。

宿主侧需提供的环境变量（launch 脚本注入，密钥只经 .env→env、绝不写进本文件/commit）：
- MYAGENT_REPO       ：宿主上 coding-agent-eval 检出根（含 agent/ + obs/）的路径，**必填**。
                       注意要指向**带 evoclaw_cli.py 的那个检出**（如本次的 worktree）。
- ANTHROPIC_API_KEY  ：deepseek key（或 UNIFIED_API_KEY 兜底）。
- ANTHROPIC_BASE_URL ：默认 https://api.deepseek.com/anthropic（deepseek 自带 Anthropic 兼容端点）。
- MODEL_ID           ：默认 deepseek-v4-flash。
- COMPACT_STRATEGY   ：none / pipeline / truncate（三臂开关）。
- MYAGENT_SESSION_MEMORY：1/true/on 时在容器内 run_task 挂 SessionMemory。
- MYAGENT_ARM_LABEL  ：可选 trace arm 标签；SM 对照用 pipeline_full / pipeline_sm。
- MYAGENT_INSTANCE_ID / MYAGENT_MILESTONE：透传给 trace meta（链/repo id、里程碑 id）。
"""

import logging
import os
from typing import List, Optional

from harness.e2e.agents.base import AgentFramework, register_framework

logger = logging.getLogger(__name__)

# 容器内路径常量。
_SRC_RO = "/opt/myagent-src"    # 只读挂载点（宿主源码）
_SRC_RW = "/opt/myagent"        # init 拷到此可写处（agent 要写 .traces/__pycache__/~/.myagent）
_WRAPPER = "/usr/local/bin/myagent"

# deepseek 自带 Anthropic 兼容端点（validation §7b.5 R-6）；白名单已加 api.deepseek.com（§E-2）。
_DEFAULT_BASE_URL = "https://api.deepseek.com/anthropic"
_DEFAULT_MODEL = "deepseek-v4-flash"


@register_framework("myagent")
class MyAgentFramework(AgentFramework):
    """coding-agent-eval 压缩 agent 的容器内 CLI 适配器。"""

    FRAMEWORK_NAME = "myagent"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 宿主源码根：必须指向带 evoclaw_cli.py 的检出（本次是 worktree）。缺失 → 早失败、别静默跑错。
        self._repo = os.environ.get("MYAGENT_REPO")
        # 密钥/端点/模型：优先专用名，UNIFIED_* 兜底（与 EvoClaw 习惯对齐）。
        self._api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("UNIFIED_API_KEY")
        self._base_url = (os.environ.get("ANTHROPIC_BASE_URL")
                          or os.environ.get("UNIFIED_BASE_URL") or _DEFAULT_BASE_URL)
        self._model_id = (os.environ.get("MODEL_ID")
                          or os.environ.get("UNIFIED_MODEL") or _DEFAULT_MODEL)
        # 三臂开关（none / pipeline / truncate）。透传给容器内 run_task(compact_strategy=)。
        self._compact_strategy = os.environ.get("COMPACT_STRATEGY", "none")
        # trace meta 透传（链/repo id、里程碑 id）。
        self._instance_id = os.environ.get("MYAGENT_INSTANCE_ID", "")
        self._milestone = os.environ.get("MYAGENT_MILESTONE", "")

    # ── 4 抽象方法 ──────────────────────────────────────────────────────────

    def get_container_mounts(self) -> List[str]:
        """只读挂载我们的源码进容器（供 init 拷到可写处）。

        ⚠ 偏离设计（§2.4 表写"挂 agent/ + .env"）：agent 包 import 了**同级 obs 包**
        （loop/tools `from obs.trace import ...`），只挂 agent/ 会缺 obs → ImportError。
        故挂**整个检出根**（含 agent/ + obs/）。.env **不挂**：密钥经 -e 注入（见 env_vars），
        worktree 本就无 .env、config.load_dotenv 空跑、os.environ 生效 → 单一来源、key 不落挂载源码。
        """
        if not self._repo:
            raise ValueError(
                "MYAGENT_REPO 未设置：需指向宿主上 coding-agent-eval 检出根（含 agent/+obs/，"
                "且带 evoclaw_cli.py 的那个，如本次 worktree）。launch 脚本应 export 它。")
        repo = os.path.abspath(self._repo)
        if not os.path.isdir(os.path.join(repo, "agent")):
            raise ValueError(f"MYAGENT_REPO={repo} 下无 agent/ 目录，路径不对？")
        return ["-v", f"{repo}:{_SRC_RO}:ro"]

    def get_container_init_script(self, agent_name: str) -> str:
        """root 下跑一次：装依赖 + 拷源码到可写处 + chown fakeroot + 写 myagent wrapper（LF）。

        WHY 拷而非直接用只读挂载：agent 运行要写 REPO_ROOT/.traces、__pycache__、~/.myagent；
        只读挂载会 EROFS。拷到 /opt/myagent 可写、chown fakeroot 后让 fakeroot 能写。
        """
        # wrapper 用三引号串：Python 读源码时做 universal-newline 翻译，串内是 LF（即使本文件存成
        # CRLF）。容器是 Linux，open(...,'w') 落 LF → shebang 无 \r、可执行（避开 validation 的 CRLF 坑）。
        wrapper = (
            "#!/usr/bin/env python3\n"
            "import sys\n"
            f"sys.path.insert(0, {_SRC_RW!r})\n"
            "from agent.evoclaw_cli import main\n"
            "main()\n"
        )
        # 注意：本方法返回的是一段**在容器内以 root 执行的 Python 脚本字符串**（container_setup
        # `docker exec python3 -c <script>`）。下面用 subprocess/shutil/os 完成装包+拷贝+落 wrapper。
        return f'''
import os, sys, shutil, subprocess, pwd

def _run(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return r.returncode == 0, (r.stdout or "") + (r.stderr or "")
    except Exception as e:
        return False, str(e)

print("[myagent-init] start")

# === 1) 确保 pip 可用，装 anthropic + python-dotenv（PyPI 白名单可达）===
# WHY apt-get update 先行：standalone smoke 不跑 EvoClaw base init（它会 apt update），apt 缓存可能空
#   → install python3-pip 失败。先 update 兜底；真实 flow 里 base init 已 update、再 update 无害。
ok, _ = _run([sys.executable, "-m", "pip", "--version"])
if not ok:
    _run(["apt-get", "update"])
    if not _run(["apt-get", "install", "-y", "-qq", "python3-pip"])[0]:
        if not _run(["apk", "add", "--no-cache", "py3-pip"])[0]:
            _run([sys.executable, "-m", "ensurepip", "--upgrade"])
# WHY --break-system-packages：Debian bookworm(py3.11)+ PEP 668 把系统环境标 externally-managed，
#   直接 pip install 系统 site-packages 会被拒。带 flag 装进系统 python（wrapper 用的就是它）。
#   老 pip 不认该 flag → 去掉重试（两序兼容）。
base = [sys.executable, "-m", "pip", "install", "--no-input", "--root-user-action=ignore"]
pkgs = ["anthropic>=0.39.0", "python-dotenv>=1.0.0"]
ok, out = _run(base + ["--break-system-packages"] + pkgs)
if not ok:
    ok, out = _run(base + pkgs)
print("[myagent-init] pip install anthropic/dotenv:", "OK" if ok else "FAIL")
if not ok:
    print(out[-1500:])

# === 2) 拷只读源码 → 可写处，chown fakeroot ===
src, dst = {_SRC_RO!r}, {_SRC_RW!r}
try:
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    # 只拷需要的包，跳过大目录（.git/.venv*/镜像数据等）省时间。
    os.makedirs(dst, exist_ok=True)
    for name in ("agent", "obs"):
        s = os.path.join(src, name)
        if os.path.isdir(s):
            shutil.copytree(s, os.path.join(dst, name),
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    # ★ **故意不拷 .env**（即使挂载源码根带 .env）：config.py 用 load_dotenv(REPO_ROOT/.env,
    #   override=True)，若容器内 /opt/myagent/.env 存在会**覆盖 -e 注入**。主仓库 .env 里有
    #   HTTPS_PROXY=<your-proxy>（容器内不可达）→ httpx 走它 → LLM 调用必崩；且会把 COMPACT_STRATEGY
    #   之外的值锁死。单一来源 = adapter 的 -e 注入（已砍内联注释的干净 key/端点/模型 + 三臂开关），
    #   容器内无 .env → load_dotenv 空跑 → os.environ(-e) 生效、无代理。
    fake = pwd.getpwnam("fakeroot")
    for root, dirs, files in os.walk(dst):
        os.chown(root, fake.pw_uid, fake.pw_gid)
        for fn in files:
            os.chown(os.path.join(root, fn), fake.pw_uid, fake.pw_gid)
    print("[myagent-init] copied src -> ", dst)
except Exception as e:
    print("[myagent-init] copy FAIL:", e)

# === 3) 写 /usr/local/bin/myagent wrapper（LF）===
try:
    with open({_WRAPPER!r}, "w", newline="\\n") as f:
        f.write({wrapper!r})
    os.chmod({_WRAPPER!r}, 0o755)
    print("[myagent-init] wrapper ready at", {_WRAPPER!r})
except Exception as e:
    print("[myagent-init] wrapper FAIL:", e)

# === 4) 冒烟：**纯 import 检查**（绝不调 main/不发模型）===
# ⚠ 重要修复：旧版这里 `echo '' | myagent run` 其实会跑一次 **完整 run_task("")**（空 prompt +
#   默认 max_turns=200）→ 真发 deepseek、可能空转很久 → 卡死 init（_run timeout=600s）。
#   改为纯 import：只验 wrapper 的 import 链通（agent.evoclaw_cli + anthropic + obs），不执行 main()。
ok, out = _run([sys.executable, "-c",
                "import sys; sys.path.insert(0, {_SRC_RW!r}); import agent.evoclaw_cli"])
print("[myagent-init] import check:", "OK" if ok else ("FAIL " + out[-300:]))
print("[myagent-init] done")
'''

    def build_run_command(self, model: str, session_id: str, prompt_path: str) -> str:
        """首跑里程碑序列：prompt 从 stdin。model 由 env MODEL_ID 驱动（我们 CLI 不收 --model）。"""
        return f"{_WRAPPER} run --session-id {session_id} < {prompt_path}"

    def build_resume_command(self, model: str, session_id: str, message_path: str) -> str:
        """续作：新消息从 stdin。session 跨 exec 持久在 ~/.myagent/<sid>.json。"""
        return f"{_WRAPPER} resume --session-id {session_id} < {message_path}"

    # ── 可选方法 ────────────────────────────────────────────────────────────

    def get_container_env_vars(self) -> List[str]:
        """每次 docker exec 注入。三臂只差一个 COMPACT_STRATEGY（§2.1 同构钉死）。

        密钥经 -e 注入（不落挂载源码/commit）；config.py 在容器内读 os.environ（无 .env 时 load_dotenv
        空跑、os.environ 生效）。AGENT_WORKDIR=/testbed 让 LocalExecutor 落在 /testbed。
        """
        env: List[str] = []
        if self._api_key:
            env += ["-e", f"ANTHROPIC_API_KEY={self._api_key}"]
        else:
            logger.warning("myagent: 无 ANTHROPIC_API_KEY/UNIFIED_API_KEY，容器内模型调用会失败")
        env += ["-e", f"ANTHROPIC_BASE_URL={self._base_url}"]
        env += ["-e", f"MODEL_ID={self._model_id}"]
        env += ["-e", f"COMPACT_STRATEGY={self._compact_strategy}"]
        env += ["-e", "AGENT_WORKDIR=/testbed"]
        # trace meta 透传（CoderA 曲线脚本按 instance_id/milestone 聚合）。
        if self._instance_id:
            env += ["-e", f"MYAGENT_INSTANCE_ID={self._instance_id}"]
        if self._milestone:
            env += ["-e", f"MYAGENT_MILESTONE={self._milestone}"]
        # 单 exec 轮数上限透传（多里程碑自驱时放大；不设则 CLI 默认 200）。EvoClaw 仍会在
        # 无进展时 resume 续作，故这只是单 exec 的天花板、不限制总里程碑数。
        for k in (
            "MYAGENT_MAX_TURNS",
            "MYAGENT_COMPACT_WINDOW",
            "MYAGENT_COMPACT_THRESHOLD",
            "MYAGENT_STOP_AT_CONTEXT",
            "MYAGENT_SESSION_MEMORY",
            "MYAGENT_ARM_LABEL",
        ):
            v = os.environ.get(k)
            if v is not None:
                env += ["-e", f"{k}={v}"]
        # quarantine 模式：与其它 agent 一致走离线 wheelhouse（base 提供的共享 helper）。
        env += self.get_quarantine_env_vars()
        return env

    def extract_session_id_from_container(self, container_name: str) -> Optional[str]:
        """照 claude-code：session_id 由 harness 外部生成并传入，无需从容器回捞 → None。"""
        return None

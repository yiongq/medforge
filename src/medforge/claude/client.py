"""Claude Code CLI provider:用本机已登录的 Claude 订阅当推理后端,免 API key。

为什么走 CLI 而不是 Anthropic API:订阅制(Max)的额度只能通过本机 `claude` 登录态使用,
没有可编程的 API key。headless 模式(`claude -p ... --output-format json`)把一次问答
压成一次进程调用,stdout 是一份干净 JSON——足够当同步的「一问一答」后端使。

调用形态(每个参数都是必要的,别删):
    claude -p <prompt> --model <id> --output-format json --max-turns 1 \
      --tools "" --setting-sources "" --disable-slash-commands --no-session-persistence \
      --system-prompt <system> [--json-schema <schema>] [--effort <low|medium|high|xhigh|max>]
  --tools "" / --setting-sources "" / --disable-slash-commands:判卷是纯文本推理,
  不给它文件系统与项目配置,免得判分口径被某个 CLAUDE.md 悄悄改写。
  --max-turns 1:一问一答,不让它自己续轮。--no-session-persistence:不落会话文件。

三个必须遵守的环境约束:
  1. 进程环境里必须清掉 CLAUDECODE 与所有 CLAUDE_* 变量,否则 CLI 认为自己被嵌套调用而拒绝;
  2. 同时清掉 ANTHROPIC_API_KEY / _AUTH_TOKEN / _BASE_URL:留着的话 CLI 会拿 API key 计费,
     而这条 provider 的全部意义是「用订阅额度」——文档承诺不花 API 费,环境就不能偷偷改口;
  3. cwd 必须是空目录,否则它会加载仓库的 CLAUDE.md / memory,把无关上下文塞进判卷提示词。

已知代价与限制(2026-09-05 用本模块的参数实测,claude-sonnet-5):
  - 每次调用有约 31.6k 输入 token 的基线上下文;首次调用是 cache_creation,随后连续调用
    命中缓存(cache_read ≈ 31.6k、cache_creation = 0),单次约 5 秒。
    per-call 开销可接受,但必须限流(ThreadPool 4~8 足够),额度按 Max 的 5 小时窗口计。
  - 扩展思考的过程文本**不在** JSON 结果里,只拿得到最终答案——所以用它评测时
    存档答卷里没有 </think>,必须按 thinking=off 记账(见 medforge.eval.run)。
  - 模型 ID 要写全(claude-opus-5 / claude-sonnet-5 / claude-haiku-4-5-20251001);
    裸别名 opus/sonnet/haiku 被 CLI 钉在旧代,拿到的不是你以为的模型。

政策边界:蒸馏教师**不得**换成这个 provider。Anthropic 消费者条款禁止用 Claude 输出训练
其他模型,教师角色(medforge.data.build_distill)继续用 DeepSeek(其条款 §4.2 明确允许蒸馏)。

冒烟自测(验证本机登录态,先 `claude auth status` / `claude auth login`):
    uv run python -m medforge.claude.client --model claude-sonnet-5
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field

CLI = "claude"
DEFAULT_TIMEOUT = 180.0
# 与 v2.1.252 的 `claude --help` 逐字对齐;少写一档会让合法配置在 _build_argv 里被拒
EFFORTS = ("low", "medium", "high", "xhigh", "max")


class ClaudeCodeError(RuntimeError):
    """CLI 调用失败:非零退出 / 超时 / subtype != success / stdout 不是 JSON。"""


@dataclass
class ClaudeCodeResult:
    text: str  # 最终答案文本(不含扩展思考)
    structured: dict | None  # --json-schema 时 CLI 解析好的对象
    output_tokens: int | None  # 含思考 token,用来当 completion_tokens 记账
    cost_usd: float | None
    raw: dict = field(repr=False)  # 原始 JSON,审计用


# 会让 CLI 改用 API key 计费的变量:这条 provider 承诺走订阅额度,所以一律不传给子进程
_AUTH_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL")


def _clean_env() -> dict[str, str]:
    """去掉嵌套会话标记与 API key:CLAUDECODE 没有下划线,单独判一次。"""
    return {
        k: v
        for k, v in os.environ.items()
        if k != "CLAUDECODE" and not k.startswith("CLAUDE_") and k not in _AUTH_VARS
    }


def cli_available() -> bool:
    """`claude` 在不在 PATH 上:给调用方做开跑前体检——CLI 找不到时每一条都会失败,
    而判卷侧的契约是「失败即弃权」,弃权计错,整卷会无声塌下去。"""
    return shutil.which(CLI) is not None


def _build_argv(
    prompt: str, *, model: str, system_prompt: str, json_schema: dict | None, effort: str | None
) -> list[str]:
    # fmt: off
    argv = [
        CLI, "-p", prompt, "--model", model,
        "--output-format", "json", "--max-turns", "1",
        "--tools", "", "--setting-sources", "",
        "--disable-slash-commands", "--no-session-persistence",
        "--system-prompt", system_prompt,
    ]
    # fmt: on
    if json_schema is not None:
        argv += ["--json-schema", json.dumps(json_schema, ensure_ascii=False)]
    if effort:
        if effort not in EFFORTS:
            raise ValueError(f"effort 只能是 {EFFORTS},收到 {effort!r}")
        argv += ["--effort", effort]
    return argv


def claude_code_query(
    prompt: str,
    *,
    model: str,
    system_prompt: str = "",
    json_schema: dict | None = None,
    effort: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    cwd: str | None = None,
    retries: int = 0,
) -> ClaudeCodeResult:
    """同步问一次 Claude Code,失败一律抛 ClaudeCodeError(调用方决定是弃权还是重试)。

    cwd 不给就临时开一个空目录:必须避开项目的 CLAUDE.md/memory(见模块 docstring)。
    retries>0 时对 ClaudeCodeError 重试(短退避):OpenAI SDK 那条路有 max_retries=2 兜底,
    CLI 这条路一次抖动(额度窗口、spawn 失败)就是一条硬失败,整臂失败率一超阈值就不出表。
    配置类错误(effort 非法)在 _build_argv 里先抛 ValueError,不进重试。
    """
    argv = _build_argv(
        prompt, model=model, system_prompt=system_prompt, json_schema=json_schema, effort=effort
    )
    for attempt in range(retries):
        try:
            return _run_once(argv, timeout=timeout, cwd=cwd)
        except ClaudeCodeError:
            time.sleep(min(2.0 * (attempt + 1), 10.0))
    return _run_once(argv, timeout=timeout, cwd=cwd)


def _run_once(argv: list[str], *, timeout: float, cwd: str | None) -> ClaudeCodeResult:
    env = _clean_env()
    with tempfile.TemporaryDirectory(prefix="medforge-cc-") as tmp:
        try:
            # argv 固定、不过 shell:注入面在提示词内容里,而提示词是我们自己拼的
            proc = subprocess.run(
                argv, capture_output=True, text=True, timeout=timeout, cwd=cwd or tmp, env=env, check=False
            )
        except subprocess.TimeoutExpired as e:
            raise ClaudeCodeError(f"claude 调用超时({timeout}s)") from e
        except OSError as e:  # CLI 没装 / 不可执行
            raise ClaudeCodeError(f"无法执行 {CLI}: {e}") from e
    if proc.returncode != 0:
        raise ClaudeCodeError(f"claude 退出码 {proc.returncode}: {(proc.stderr or '')[-300:].strip()}")
    try:
        data = json.loads(proc.stdout)
    except (json.JSONDecodeError, TypeError) as e:
        raise ClaudeCodeError(f"claude 输出不是 JSON: {(proc.stdout or '')[:200]!r}") from e
    if not isinstance(data, dict):
        raise ClaudeCodeError(f"claude 输出不是 JSON 对象: {str(data)[:200]!r}")
    if data.get("subtype") != "success":
        raise ClaudeCodeError(
            f"claude 未成功: subtype={data.get('subtype')!r} {str(data.get('result'))[:180]!r}"
        )
    usage = data.get("usage") or {}
    structured = data.get("structured_output")
    return ClaudeCodeResult(
        text=data.get("result") or "",
        structured=structured if isinstance(structured, dict) else None,
        output_tokens=usage.get("output_tokens"),
        cost_usd=data.get("total_cost_usd"),
        raw=data,
    )


def parse_json_object(text: str) -> dict | None:
    """从自由文本里刮出第一个 JSON 对象;刮不出或非法返回 None。

    结构化输出(--json-schema)偶尔为空时的兜底,与 verifier 里的正则口径保持一致。
    """
    import re

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def main() -> None:
    """--smoke:打一次最小请求,把解析结果与用量打出来,用来验证本机 `claude auth login` 的登录态是否可用。"""
    import argparse

    ap = argparse.ArgumentParser(description="Claude Code provider 冒烟自测")
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--effort", choices=EFFORTS, default=None)
    ap.add_argument("--prompt", default="用一句中文回答:阿司匹林的主要药理作用是什么?")
    ap.add_argument("--smoke", action="store_true", help="兼容写法;不加也是冒烟")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    args = ap.parse_args()

    from rich import print as rprint

    try:
        r = claude_code_query(
            args.prompt,
            model=args.model,
            system_prompt="你是简洁的中文助手。",
            json_schema={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
            effort=args.effort,
            timeout=args.timeout,
        )
    except ClaudeCodeError as e:
        rprint(f"[red]✗ {e}[/]")
        raise SystemExit(1) from e
    rprint(f"[green]✓[/] model={args.model} text={r.text[:200]!r}")
    rprint(f"  structured={r.structured}")
    rprint(f"  output_tokens={r.output_tokens} cost_usd={r.cost_usd} usage={r.raw.get('usage')}")


if __name__ == "__main__":
    main()

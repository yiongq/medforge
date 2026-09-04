"""验证器:判定模型输出是否答对。整个项目的灵魂部件。

三层结构:
  0. 截断守卫(免费):没写完的答卷直接判「未收尾」,不进规则层也不进 LLM 层。
     思考型模型撞上 max_tokens 时末段多半是复读循环,规则层会从循环里刮出「答案:X」
     判硬分(base-v2/cmexam 实测 133 题)——「没交卷」必须与「答错」「弃权」分开计数。
     两条腿:finish_reason=="length" 无条件;缺 finish_reason 时只能靠思考型口径
     (thinking=True:没有 </think> 即未收尾)——存档答卷没有 finish_reason,重判必须显式开它。
  1. 规则层(免费,覆盖大多数):extract 抽答案 → 与 gold 比对
  2. LLM 层(兜底,按次付费):规则层弃权时,调 OpenAI 兼容接口判分

用途横跨两处,判分口径必须唯一:
  - DPO 数据构造:采样多解 → 判对错 → 对/错配偏好对
  - 评测:开放题的对错判定

provider 无关,两条后端都由环境变量切(选谁由 200 题人工校准集的一致率决定,见 docs/adr/001):
  - MEDFORGE_JUDGE_PROVIDER=openai(默认):MEDFORGE_JUDGE_BASE_URL / _API_KEY / _MODEL,
    走 OpenAI 兼容接口,可指 DeepSeek/智谱/任何兼容层;
  - MEDFORGE_JUDGE_PROVIDER=claude-code:走本机已登录的 Claude Code CLI(用订阅额度,不用 API key),
    只需 MEDFORGE_JUDGE_MODEL(如 claude-opus-5),可选 MEDFORGE_JUDGE_EFFORT=low|medium|high|max。
    细节见 medforge.verify.claude_code 与 docs/claude-code-provider.md。

注意:判卷员可以用 Claude,蒸馏教师不行——Anthropic 消费者条款禁止用 Claude 输出训练其他模型,
data/build_distill 的教师角色继续用 DeepSeek(其条款 §4.2 允许蒸馏)。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from medforge.data.schema import Sample
from medforge.verify.extract import extract

_PUNCT_RE = re.compile(r"[\s\W_]+", re.UNICODE)
THINK_END = "</think>"


@dataclass
class Verdict:
    correct: bool | None   # None = 无法判定(未收尾 / 未配 LLM key / LLM 也拿不准)
    method: str            # "rule" | "llm" | "abstain" | "unfinished"
    detail: str = ""


def split_answer(
    output: str, *, finish_reason: str | None = None, thinking: bool | None = None,
) -> tuple[str, str | None]:
    """把答卷切成 (作答段, 未收尾原因);原因非 None 时作答段不得判分。

    作答段 = 最后一个 </think> 之后(没有就是全文;多个 </think> 实测只见「把收尾写了两遍」,取最后一个)。
    未收尾只有两条腿:
      - finish_reason == "length":端点报告撞上 max_tokens,无条件——哪怕末段能刮出答案;
      - thinking=True 且没有 </think>:思考型口径,从未收尾的思考流不得判分。
    thinking=None 是自动模式:含 </think> 就按思考型处理,否则当作非思考型输出全文判分——
    它挡不住「缺 finish_reason 又没写出 </think>」的存档答卷,评测思考型模型必须显式 thinking=True
    (eval CLI 默认 --thinking on)。Qwen3.5 的 chat 模板吃掉了 <think> 开标签,输出里通常只剩 </think>。
    """
    n = output.count(THINK_END)
    segment = output.rsplit(THINK_END, 1)[1] if n else output
    if finish_reason == "length":
        return segment, "finish_reason=length"
    if thinking is None:
        thinking = n > 0
    if thinking and n == 0:
        return segment, "no-think-close"
    return segment, None


def _norm(text: str) -> str:
    return _PUNCT_RE.sub("", text).lower()


def verify_by_rule(sample: Sample, output: str) -> Verdict | None:
    """规则层。返回 None 表示弃权(不是判错),由上层决定是否调 LLM。"""
    ext = extract(output, sample.is_choice, options=sample.options)
    if ext is None:
        return None
    if ext.kind == "abstain":
        # 主动弃权:计错进分母(与验证器弃权同列),但 detail 标 declared 以便分开统计;不再花钱问 judge
        return Verdict(None, "abstain", "declared")
    if sample.is_choice:
        gold = "".join(sorted(sample.gold.upper()))
        return Verdict(ext.value == gold, "rule", f"extracted={ext.value} gold={gold}")
    # 开放题:规则层只认归一化后精确相等。
    # 包含匹配已被审查实测证伪:「阿莫西林」⊂「阿莫西林克拉维酸钾」(不同药)、
    # 「手术治疗」⊂「无需手术治疗」(意思相反)、「急性心肌梗死」⊂「可排除急性心肌梗死」——
    # 这些假阳性会直接进 DPO 正例。语义等价的判定是 LLM 层的职责,规则层一律弃权。
    gold_n, out_n = _norm(sample.gold), _norm(ext.value)
    if not gold_n:
        return None
    # 抽取值可能带尾随评论(「阿司匹林,不过我不确定」):取首子句作第二候选,
    # 仍只做精确相等——「阿莫西林克拉维酸钾」无标点截不短,不会误配「阿莫西林」
    first_clause = _norm(re.split(r"[,,。;;!!??]", ext.value, maxsplit=1)[0])
    if gold_n in (out_n, first_clause):
        return Verdict(True, "rule", f"matched={ext.value!r}")
    return None


_JUDGE_PROMPT = """你是医学考试判卷员。判断考生的最终答案与标准答案是否一致(只看结论,不看过程)。

题目:{question}
标准答案:{gold}
考生作答(截尾):{output}

只输出 JSON:{{"correct": true/false, "reason": "一句话依据"}}
拿不准时输出 {{"correct": null, "reason": "..."}},不要硬判。"""


# claude-code 后端用结构化输出,省掉「模型在 JSON 外面裹一段话」的解析风险;
# correct 允许 null 与提示词里的「拿不准」口径一致
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"correct": {"type": ["boolean", "null"]}, "reason": {"type": "string"}},
    "required": ["correct", "reason"],
}
_JUDGE_SYSTEM = "你是严格的医学考试判卷员,只输出 JSON,不要输出任何其他文字。"
JUDGE_PROVIDERS = ("openai", "claude-code")


def judge_provider() -> str:
    """判卷后端:默认 openai(历史行为),claude-code 走本机订阅。非法值按 openai 处理并不报错——
    判分链的契约是「永不抛」,配置写错顶多退回默认后端,不该把整卷炸掉。"""
    p = (os.environ.get("MEDFORGE_JUDGE_PROVIDER") or "openai").strip().lower()
    return p if p in JUDGE_PROVIDERS else "openai"


def judge_effort() -> str | None:
    """扩展思考档位,归一成小写;非法值不在这里兜底(会被 missing_judge_env 在开跑前拦下),
    因为「悄悄降到默认档」等于指纹说谎。"""
    return (os.environ.get("MEDFORGE_JUDGE_EFFORT") or "").strip().lower() or None


def missing_judge_env() -> list[str]:
    """按当前 provider 返回「开跑前就该拦下」的问题清单(缺失变量名或一句人话),空 = 可以跑。

    judge 没配时 verify_by_llm 只会静默弃权,而弃权计错——整卷分数会无声塌一半,
    所以 eval/DPO 入口宁可开跑前就退出。claude-code 不需要 base_url/api_key,只要模型名;
    但它还有两件同样会「每条都失败 → 全卷弃权」的事必须一起体检:CLI 不在 PATH、effort 写错。
    这两件在运行时都被 _judge_by_claude_code 的 except 吞成弃权,不在这里拦就没人拦了。
    """
    if judge_provider() != "claude-code":
        keys = ("MEDFORGE_JUDGE_BASE_URL", "MEDFORGE_JUDGE_API_KEY", "MEDFORGE_JUDGE_MODEL")
        return [k for k in keys if not os.environ.get(k)]

    from medforge.verify import claude_code as cc

    problems = [k for k in ("MEDFORGE_JUDGE_MODEL",) if not os.environ.get(k)]
    if not cc.cli_available():
        problems.append(f"`{cc.CLI}` 不在 PATH(claude-code 后端要本机 CLI:先 `claude auth login`)")
    effort = judge_effort()
    if effort and effort not in cc.EFFORTS:
        problems.append(f"MEDFORGE_JUDGE_EFFORT={effort!r} 非法(只能是 {'|'.join(cc.EFFORTS)})")
    return problems


def _verdict_from_judge(data: dict) -> Verdict:
    correct = data.get("correct")
    if not isinstance(correct, bool):
        correct = None
    return Verdict(correct, "llm", str(data.get("reason", ""))[:200])


def _judge_by_claude_code(prompt: str) -> Verdict:
    """Claude Code CLI 后端:只需 MEDFORGE_JUDGE_MODEL;失败一律弃权(与 openai 路径同契约)。"""
    from medforge.verify.claude_code import claude_code_query, parse_json_object

    model = os.environ.get("MEDFORGE_JUDGE_MODEL")
    if not model:
        return Verdict(None, "abstain", "judge 未配置(claude-code 需要 MEDFORGE_JUDGE_MODEL)")
    effort = judge_effort()
    try:
        r = claude_code_query(
            prompt, model=model, system_prompt=_JUDGE_SYSTEM, json_schema=JUDGE_SCHEMA,
            effort=effort, timeout=180, retries=1,  # 一次抖动就弃权太贵:弃权按错算
        )
    except Exception as e:  # noqa: BLE001  CLI 挂了/超时/effort 写错:弃权而不是炸掉整卷
        return Verdict(None, "llm", f"judge 调用失败: {type(e).__name__}: {str(e)[:120]}")
    data = r.structured or parse_json_object(r.text)
    if data is None:
        return Verdict(None, "llm", f"无法解析判分输出: {r.text[:100]!r}")
    return _verdict_from_judge(data)


def verify_by_llm(sample: Sample, output: str) -> Verdict:
    from medforge.env import load_env

    load_env()
    prompt = _JUDGE_PROMPT.format(
        question=sample.render_question()[:1500],
        gold=sample.gold[:500],
        output=output[-1500:],  # 结论在结尾;截尾控成本
    )
    if judge_provider() == "claude-code":
        return _judge_by_claude_code(prompt)

    base_url = os.environ.get("MEDFORGE_JUDGE_BASE_URL")
    api_key = os.environ.get("MEDFORGE_JUDGE_API_KEY")
    model = os.environ.get("MEDFORGE_JUDGE_MODEL")
    if not (base_url and api_key and model):
        return Verdict(None, "abstain", "judge 未配置(MEDFORGE_JUDGE_* 环境变量)")

    from openai import OpenAI  # 延迟 import:纯规则路径不依赖网络配置

    # timeout 必须显式给:SDK 默认 10 分钟 × 重试,一次挂起就是半小时;
    # 评测的 scored 文件是 "w" 模式,整卷炸掉会连上一版判分一起丢——所以失败只弃权不抛
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=60, max_retries=2)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
        )
    except Exception as e:  # noqa: BLE001  网络/限流/5xx:弃权而不是炸掉整卷
        return Verdict(None, "llm", f"judge 调用失败: {type(e).__name__}: {str(e)[:120]}")
    text = resp.choices[0].message.content or ""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return Verdict(None, "llm", f"无法解析判分输出: {text[:100]!r}")
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return Verdict(None, "llm", f"判分输出非法 JSON: {text[:100]!r}")
    return _verdict_from_judge(data)


def verify(
    sample: Sample,
    output: str,
    allow_llm: bool = True,
    *,
    finish_reason: str | None = None,
    thinking: bool | None = None,
) -> Verdict:
    """完整判分链:截断守卫 → 规则层 → LLM 兜底。守卫在最前,不看 extract 结果——
    extract 能从未收尾的答卷里刮出字母,恰恰是它要拦的情形。thinking 的语义见 split_answer。"""
    answer, unfinished = split_answer(output, finish_reason=finish_reason, thinking=thinking)
    if unfinished is not None:
        return Verdict(None, "unfinished", unfinished)
    v = verify_by_rule(sample, answer)
    if v is not None:
        return v
    if allow_llm:
        return verify_by_llm(sample, answer)
    return Verdict(None, "abstain", "规则层弃权且未启用 LLM 兜底")

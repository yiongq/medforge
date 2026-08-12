"""验证器:判定模型输出是否答对。整个项目的灵魂部件。

两层结构:
  1. 规则层(免费,覆盖大多数):extract 抽答案 → 与 gold 比对
  2. LLM 层(兜底,按次付费):规则层弃权时,调 OpenAI 兼容接口判分

用途横跨两处,判分口径必须唯一:
  - DPO 数据构造:采样多解 → 判对错 → 对/错配偏好对
  - 评测:开放题的对错判定

provider 无关:MEDFORGE_JUDGE_BASE_URL / MEDFORGE_JUDGE_API_KEY /
MEDFORGE_JUDGE_MODEL 三个环境变量即可切 DeepSeek/智谱/Claude,
选谁由 200 题人工校准集的一致率决定(见 docs/adr/001)。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from medforge.data.schema import Sample
from medforge.verify.extract import extract

_PUNCT_RE = re.compile(r"[\s\W_]+", re.UNICODE)


@dataclass
class Verdict:
    correct: bool | None   # None = 无法判定(未配 LLM key 或 LLM 也拿不准)
    method: str            # "rule" | "llm" | "abstain"
    detail: str = ""


def _norm(text: str) -> str:
    return _PUNCT_RE.sub("", text).lower()


def verify_by_rule(sample: Sample, output: str) -> Verdict | None:
    """规则层。返回 None 表示弃权(不是判错),由上层决定是否调 LLM。"""
    ext = extract(output, sample.is_choice)
    if ext is None:
        return None
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
    if out_n == gold_n:
        return Verdict(True, "rule", f"matched={ext.value!r}")
    return None


_JUDGE_PROMPT = """你是医学考试判卷员。判断考生的最终答案与标准答案是否一致(只看结论,不看过程)。

题目:{question}
标准答案:{gold}
考生作答(截尾):{output}

只输出 JSON:{{"correct": true/false, "reason": "一句话依据"}}
拿不准时输出 {{"correct": null, "reason": "..."}},不要硬判。"""


def verify_by_llm(sample: Sample, output: str) -> Verdict:
    from medforge.env import load_env

    load_env()
    base_url = os.environ.get("MEDFORGE_JUDGE_BASE_URL")
    api_key = os.environ.get("MEDFORGE_JUDGE_API_KEY")
    model = os.environ.get("MEDFORGE_JUDGE_MODEL")
    if not (base_url and api_key and model):
        return Verdict(None, "abstain", "judge 未配置(MEDFORGE_JUDGE_* 环境变量)")

    from openai import OpenAI  # 延迟 import:纯规则路径不依赖网络配置

    client = OpenAI(base_url=base_url, api_key=api_key)
    prompt = _JUDGE_PROMPT.format(
        question=sample.render_question()[:1500],
        gold=sample.gold[:500],
        output=output[-1500:],  # 结论在结尾;截尾控成本
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=200,
    )
    text = resp.choices[0].message.content or ""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return Verdict(None, "llm", f"无法解析判分输出: {text[:100]!r}")
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return Verdict(None, "llm", f"判分输出非法 JSON: {text[:100]!r}")
    correct = data.get("correct")
    if not isinstance(correct, bool):
        correct = None
    return Verdict(correct, "llm", str(data.get("reason", ""))[:200])


def verify(sample: Sample, output: str, allow_llm: bool = True) -> Verdict:
    v = verify_by_rule(sample, output)
    if v is not None:
        return v
    if allow_llm:
        return verify_by_llm(sample, output)
    return Verdict(None, "abstain", "规则层弃权且未启用 LLM 兜底")

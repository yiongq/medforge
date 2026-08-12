"""校准集构造 CLI(ADR 口径 3 的前置工序)。

两步用法:
    # ① 分层抽题 → data/calibration/pending.jsonl(output 为空待生成)
    uv run python -m medforge.verify.sample_calibration

    # ② 用 OpenAI 兼容 API 生成模型作答(复用 MEDFORGE_JUDGE_* 环境变量指向的服务)
    uv run python -m medforge.verify.sample_calibration --generate

之后人工逐条填 human_correct(true/false),再跑 calibrate.py 出一致率。

抽样设计:开放题 120(验证器最难判的形态,DPO 数据构造的主战场)+
选择题 80(评测判分的主形态,从 cmexam-validation 抽——刻意避开 test,
不给「对着考卷调判卷程序」留任何口实)。seed 固定,抽样可复现。
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

from rich import print as rprint

from medforge.data.normalize import ADAPTERS
from medforge.data.schema import Sample

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "calibration" / "pending.jsonl"
SEED = 42
N_OPEN, N_CHOICE = 120, 80

_GEN_PROMPT = "你是医学助手,回答下面的问题。先给出推理过程,最后一行以「最终答案:」开头给出结论。\n\n{question}"


def sample() -> None:
    rng = random.Random(SEED)
    rows = [json.loads(line) for line in (RAW / "med-o1-verifiable.train.jsonl").open(encoding="utf-8")]
    open_qs = list(ADAPTERS["med-o1-verifiable"](rows, ""))
    rows = [json.loads(line) for line in (RAW / "cmexam.validation.jsonl").open(encoding="utf-8")]
    choice_qs = list(ADAPTERS["cmexam"](rows, "validation"))

    picked = rng.sample(open_qs, N_OPEN) + rng.sample(choice_qs, N_CHOICE)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for s in picked:
            f.write(json.dumps(
                {"sample": s.to_dict(), "output": None, "human_correct": None},
                ensure_ascii=False) + "\n")
    rprint(f"[green]✓[/] 抽样 {len(picked)} 题(开放 {N_OPEN} + 选择 {N_CHOICE})→ {OUT}")


def generate() -> None:
    base_url = os.environ.get("MEDFORGE_JUDGE_BASE_URL")
    api_key = os.environ.get("MEDFORGE_JUDGE_API_KEY")
    model = os.environ.get("MEDFORGE_JUDGE_MODEL")
    if not (base_url and api_key and model):
        rprint("[red]需配置 MEDFORGE_JUDGE_BASE_URL / _API_KEY / _MODEL[/]")
        sys.exit(2)
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key)
    lines = OUT.read_text(encoding="utf-8").splitlines()
    done: list[str] = []
    for i, line in enumerate(lines):
        row = json.loads(line)
        if row["output"] is None:
            s = Sample(**row["sample"])
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": _GEN_PROMPT.format(question=s.render_question())}],
                temperature=0.7,  # 刻意要多样性:校准集需要覆盖对/错/格式混乱三种作答形态
                max_tokens=1024,
            )
            row["output"] = resp.choices[0].message.content or ""
            rprint(f"  [{i + 1}/{len(lines)}] {s.id} ✓")
        done.append(json.dumps(row, ensure_ascii=False))
        # 每条即写:API 中断不丢已生成的(按条计费,重跑只补空缺)
        OUT.write_text("\n".join(done + lines[len(done):]) + "\n", encoding="utf-8")
    rprint(f"[green]✓[/] 作答生成完毕 → {OUT},下一步人工填 human_correct 后跑 calibrate")


def main() -> None:
    if "--generate" in sys.argv:
        generate()
    else:
        sample()


if __name__ == "__main__":
    main()

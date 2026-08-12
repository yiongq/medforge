"""验证器校准 CLI(ADR 口径 3:一致率 ≥95% 才可上岗)。

输入 jsonl 每行:{"sample": {...Sample 字段...}, "output": "模型作答", "human_correct": true/false}
用法:
    uv run python -m medforge.verify.calibrate data/calibration/human200.jsonl          # 仅规则层
    uv run python -m medforge.verify.calibrate data/calibration/human200.jsonl --llm    # 含 LLM 兜底

口径:验证器弃权不计入一致率分母(弃权是诚实行为,不是判错),但弃权率单独公布——
弃权率过高意味着 DPO 数据构造阶段的 LLM 判分成本会失控,同样要治。
退出码:一致率 <95% 或有效样本 <50 → 1(禁止上岗),否则 0。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rich import print as rprint

from medforge.data.schema import Sample
from medforge.verify.verifier import verify

THRESHOLD = 0.95
MIN_SAMPLES = 50  # 样本不足时的一致率没有统计意义,同样禁止上岗


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        rprint("[red]用法: python -m medforge.verify.calibrate <human_labels.jsonl> [--llm][/]")
        sys.exit(2)
    allow_llm = "--llm" in sys.argv

    agree = disagree = abstain = 0
    mismatches: list[str] = []
    for line in Path(args[0]).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sample = Sample(**row["sample"])
        v = verify(sample, row["output"], allow_llm=allow_llm)
        if v.correct is None:
            abstain += 1
        elif v.correct == bool(row["human_correct"]):
            agree += 1
        else:
            disagree += 1
            if len(mismatches) < 10:
                mismatches.append(f"{sample.id}: 人判 {row['human_correct']} / 机判 {v.correct} ({v.method}: {v.detail})")

    judged = agree + disagree
    rate = agree / judged if judged else 0.0
    total = judged + abstain
    rprint(f"样本 {total} | 判定 {judged} | 一致 {agree} | 不一致 {disagree} | 弃权 {abstain}")
    rprint(f"一致率 [bold]{rate * 100:.1f}%[/](阈值 {THRESHOLD:.0%}) | 弃权率 {abstain / total * 100 if total else 0:.1f}%")
    for m in mismatches:
        rprint(f"  [yellow]✗[/] {m}")

    if judged < MIN_SAMPLES or rate < THRESHOLD:
        rprint("[red]✗ 校准未通过,验证器禁止用于 DPO 数据构造与评测判分[/]")
        sys.exit(1)
    rprint("[green]✓ 校准通过[/]")


if __name__ == "__main__":
    main()

"""验证器校准 CLI(ADR 口径 3:一致率 ≥95% 才可上岗)。

输入 jsonl 每行:{"sample": {...Sample 字段...}, "output": "模型作答", "human_correct": true/false, ...}
用法:
    uv run python -m medforge.verify.calibrate data/calibration/pending.jsonl              # 仅规则层
    uv run python -m medforge.verify.calibrate data/calibration/pending.jsonl --llm        # 含 LLM 兜底
    uv run python -m medforge.verify.calibrate data/calibration/pending-mcq.jsonl --llm --label-field proxy_correct

口径:验证器弃权不计入一致率分母(弃权是诚实行为,不是判错),但弃权率单独公布——
弃权率过高意味着 DPO 数据构造阶段的 LLM 判分成本会失控,同样要治。
W2 审查后改为**按判分路径分层**报:规则层与 LLM 层各自的一致率、Cohen's κ(标签不均衡时原始一致率会虚高)、
false-accept / false-reject 各几条;上岗门槛作用在 **LLM 层**(它才是校准的对象;规则层由单测锁住)。
退出码:LLM 层(无 LLM 层时取总体)一致率 <95% 或有效样本 <50 → 1(禁止上岗),否则 0。
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from rich import print as rprint

from medforge.data.schema import Sample
from medforge.verify.verifier import verify

THRESHOLD = 0.95
MIN_SAMPLES = 50  # 样本不足时的一致率没有统计意义,同样禁止上岗


@dataclass
class Bucket:
    agree: int = 0
    false_accept: int = 0   # 人判错、机判对(对 DPO 是毒样本的方向)
    false_reject: int = 0   # 人判对、机判错
    tp: int = 0             # 机判对 ∧ 人判对(算 κ 用)
    tn: int = 0
    mismatches: list[str] = field(default_factory=list)

    @property
    def judged(self) -> int:
        return self.agree + self.false_accept + self.false_reject

    @property
    def rate(self) -> float:
        return self.agree / self.judged if self.judged else 0.0

    @property
    def kappa(self) -> float:
        """Cohen's κ:机判 vs 人判两个二值标注者的 chance-corrected 一致率。"""
        n = self.judged
        if not n:
            return 0.0
        po = self.agree / n
        m_pos, h_pos = (self.tp + self.false_accept) / n, (self.tp + self.false_reject) / n
        pe = m_pos * h_pos + (1 - m_pos) * (1 - h_pos)
        return (po - pe) / (1 - pe) if pe < 1 else 1.0


def evaluate(rows: list[dict], *, allow_llm: bool, label_field: str = "human_correct") -> tuple[dict[str, Bucket], int, int]:
    """返回 (按 method 分桶的结果, 弃权数, 无标签跳过数)。"""
    buckets: dict[str, Bucket] = {}
    abstain = skipped = 0
    for row in rows:
        label = row.get(label_field)
        if label is None:
            skipped += 1
            continue
        sample = Sample(**row["sample"])
        v = verify(sample, row["output"], allow_llm=allow_llm)
        if v.correct is None:
            abstain += 1
            continue
        b = buckets.setdefault(v.method, Bucket())
        human = bool(label)
        if v.correct == human:
            b.agree += 1
            b.tp += human
            b.tn += not human
        else:
            if v.correct:
                b.false_accept += 1
            else:
                b.false_reject += 1
            if len(b.mismatches) < 10:
                b.mismatches.append(f"{sample.id}: 人判 {human} / 机判 {v.correct} ({v.method}: {v.detail[:80]})")
    return buckets, abstain, skipped


def main() -> None:
    argv = sys.argv[1:]
    files = [a for a in argv if not a.startswith("--") and (argv.index(a) == 0 or argv[argv.index(a) - 1] != "--label-field")]
    if len(files) != 1:
        rprint("[red]用法: python -m medforge.verify.calibrate <labels.jsonl> [--llm] [--label-field human_correct|proxy_correct][/]")
        sys.exit(2)
    allow_llm = "--llm" in argv
    label_field = argv[argv.index("--label-field") + 1] if "--label-field" in argv else "human_correct"

    rows = [json.loads(line) for line in Path(files[0]).read_text(encoding="utf-8").splitlines() if line.strip()]
    buckets, abstain, skipped = evaluate(rows, allow_llm=allow_llm, label_field=label_field)
    judged = sum(b.judged for b in buckets.values())
    rprint(f"样本 {len(rows)} | 有标签 {len(rows) - skipped} | 判定 {judged} | 弃权 {abstain}(标签列 {label_field})")
    for method, b in sorted(buckets.items()):
        rprint(
            f"  {method:>5s} 层 | 判定 {b.judged} | 一致率 [bold]{b.rate * 100:.1f}%[/] | κ={b.kappa:.2f} "
            f"| false-accept {b.false_accept} | false-reject {b.false_reject}"
        )
        for m in b.mismatches:
            rprint(f"    [yellow]✗[/] {m}")
    gate = buckets.get("llm") if allow_llm and "llm" in buckets else None
    gate_rate = gate.rate if gate else (sum(b.agree for b in buckets.values()) / judged if judged else 0.0)
    gate_n = gate.judged if gate else judged
    rprint(f"上岗门槛作用于 {'LLM 层' if gate else '总体'}:一致率 {gate_rate * 100:.1f}%(阈值 {THRESHOLD:.0%}),样本 {gate_n}")
    if gate_n < MIN_SAMPLES or gate_rate < THRESHOLD:
        rprint("[red]✗ 校准未通过,验证器禁止用于 DPO 数据构造与评测判分[/]")
        sys.exit(1)
    rprint("[green]✓ 校准通过[/]")


if __name__ == "__main__":
    main()

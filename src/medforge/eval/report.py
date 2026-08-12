"""对照实验报告层:逐 run 的判分结果 → 带置信区间的对照表。

EvalScope 负责跑题出分,本文件只做它不做的事:
多 run 对照、Wilson 95% 置信区间、弃权率透明化、markdown 报告。
用 Wilson 而不是正态近似:评测集按科目切分后单格 n 可能只有几十,
正态近似在小 n / 高准确率时区间会越界失真。
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunResult:
    name: str          # 如 "base" / "sft" / "sft-dpo"
    n: int
    correct: int
    abstained: int = 0  # 验证器弃权数:计错进分母,但必须单独可见(见 load_run)

    @property
    def acc(self) -> float:
        return self.correct / self.n if self.n else 0.0

    @property
    def abstain_rate(self) -> float:
        return self.abstained / self.n if self.n else 0.0

    def wilson_ci(self, z: float = 1.96) -> tuple[float, float]:
        if self.n == 0:
            return (0.0, 0.0)
        p, n = self.acc, self.n
        denom = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        return (max(0.0, center - half), min(1.0, center + half))


def load_run(path: Path, name: str) -> RunResult:
    """读取判分结果 jsonl:每行 {"id": ..., "correct": true/false/null}。

    correct=null(验证器弃权)计入分母按错处理——评测口径必须保守,
    弃权算对会给「输出格式混乱」的模型发免费分。但弃权必须单独计数:
    两个 run 的准确率差可能全部来自弃权率差(格式崩坏 vs 真答错),
    报告里看不见弃权率,读者就会把前者误读成后者。
    """
    n = correct = abstained = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        n += 1
        if row.get("correct") is True:
            correct += 1
        elif row.get("correct") is None:
            abstained += 1
    return RunResult(name, n, correct, abstained)


def markdown_table(runs: Iterable[RunResult], baseline: str = "base") -> str:
    runs = list(runs)
    base = next((r for r in runs if r.name == baseline), None)
    lines = [
        "| 配置 | n | 准确率 | 95% CI | 弃权率 | vs base |",
        "|---|---|---|---|---|---|",
    ]
    for r in runs:
        lo, hi = r.wilson_ci()
        delta = f"{(r.acc - base.acc) * 100:+.1f}pp" if base and r is not base else "—"
        lines.append(
            f"| {r.name} | {r.n} | {r.acc * 100:.1f}% | [{lo * 100:.1f}, {hi * 100:.1f}] "
            f"| {r.abstain_rate * 100:.1f}% | {delta} |"
        )
    return "\n".join(lines)

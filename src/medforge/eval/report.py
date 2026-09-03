"""对照实验报告层:逐 run 的判分结果 → 带置信区间的对照表 + 配对检验。

自写薄执行器(eval/run.py)负责跑题出分,本文件只做它不做的事:
多 run 对照、Wilson 95% 置信区间、弃权/未收尾率透明化、配对 McNemar、markdown 报告。
用 Wilson 而不是正态近似:评测集按科目切分后单格 n 可能只有几十,
正态近似在小 n / 高准确率时区间会越界失真。
两个 run 考的是同一批题,比较准确率要用配对检验(McNemar),
拿两个独立 Wilson 区间看「是否重叠」是弱得多的判据(W2 报告曾这样做)。
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
    unfinished: int = 0  # 未收尾数(撞 max_tokens / 没有 </think>):同样计错,但与弃权分列
    missing: int = 0     # 压根没生成(端点失败/空作答):同样计错;是评测故障不是模型行为,再分一列
    declared: int = 0    # 主动弃权(模型自己写「答案:不确定」),是 abstained 的子集:能力而非故障

    @property
    def acc(self) -> float:
        return self.correct / self.n if self.n else 0.0

    @property
    def abstain_rate(self) -> float:
        return self.abstained / self.n if self.n else 0.0

    @property
    def unfinished_rate(self) -> float:
        return self.unfinished / self.n if self.n else 0.0

    @property
    def missing_rate(self) -> float:
        return self.missing / self.n if self.n else 0.0

    @property
    def declared_rate(self) -> float:
        return self.declared / self.n if self.n else 0.0

    def wilson_ci(self, z: float = 1.96) -> tuple[float, float]:
        if self.n == 0:
            return (0.0, 0.0)
        p, n = self.acc, self.n
        denom = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        return (max(0.0, center - half), min(1.0, center + half))


def load_verdicts(path: Path) -> dict[str, dict]:
    """读取判分 jsonl 为 {id: row},保留文件顺序。"""
    rows: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["id"]] = row
    return rows


def load_run(path: Path, name: str) -> RunResult:
    """读取判分结果 jsonl:每行 {"id": ..., "correct": true/false/null, "method": ...}。

    correct=null 计入分母按错处理——评测口径必须保守,
    弃权算对会给「输出格式混乱」的模型发免费分。但它必须按 method 拆成三列单独计数:
    "unfinished" 是「没交卷」(模型行为),"missing" 是「没生成」(评测故障),
    其余 null 是「验证器判不了」(弃权)。两个 run 的准确率差可能全部来自这几列的差
    (格式崩坏 / 复读截断 / 端点掉线 vs 真答错),报告里看不见它们,读者就会把前者误读成后者。
    """
    n = correct = abstained = unfinished = missing = declared = 0
    for row in load_verdicts(path).values():
        n += 1
        if row.get("correct") is True:
            correct += 1
        elif row.get("correct") is None:
            method = row.get("method")
            if method == "unfinished":
                unfinished += 1
            elif method == "missing":
                missing += 1
            else:
                abstained += 1
                declared += row.get("detail") == "declared"
    return RunResult(name, n, correct, abstained, unfinished, missing, declared)


def markdown_table(runs: Iterable[RunResult], baseline: str = "base") -> str:
    runs = list(runs)
    base = next((r for r in runs if r.name == baseline), None)
    lines = [
        "| 配置 | n | 准确率 | 95% CI | 弃权率 | 未收尾率 | 缺失率 | vs base |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        lo, hi = r.wilson_ci()
        delta = f"{(r.acc - base.acc) * 100:+.1f}pp" if base and r is not base else "—"
        abst = f"{r.abstain_rate * 100:.1f}%" + (f"(主动 {r.declared_rate * 100:.1f}%)" if r.declared else "")
        lines.append(
            f"| {r.name} | {r.n} | {r.acc * 100:.1f}% | [{lo * 100:.1f}, {hi * 100:.1f}] "
            f"| {abst} | {r.unfinished_rate * 100:.1f}% | {r.missing_rate * 100:.1f}% | {delta} |"
        )
    return "\n".join(lines)


def mcnemar_exact(b: int, c: int) -> float:
    """配对 McNemar 精确检验(双侧):b = 只有 A 对的题数,c = 只有 B 对的题数。

    零假设下 b ~ Binomial(b+c, 0.5)。纯 math 手写,不引 scipy(本机刻意不装重依赖)。
    """
    m = b + c
    if m == 0:
        return 1.0
    k = min(b, c)
    p = 2 * sum(math.comb(m, j) for j in range(k + 1)) / 2**m
    return min(1.0, p)


@dataclass
class Paired:
    n: int          # 两个 run 共同作答的题数
    a_only: int     # 只有 A 对
    b_only: int     # 只有 B 对
    both: int
    neither: int

    @property
    def p_value(self) -> float:
        return mcnemar_exact(self.a_only, self.b_only)

    @property
    def delta(self) -> float:
        """B 相对 A 的准确率差(比例)。"""
        return (self.b_only - self.a_only) / self.n if self.n else 0.0


def paired_counts(a: dict[str, bool], b: dict[str, bool]) -> Paired:
    """按共同 id 配对:输入是 {id: 是否判对}(None 视为不对)。"""
    ids = a.keys() & b.keys()
    a_only = b_only = both = neither = 0
    for i in ids:
        ra, rb = a[i] is True, b[i] is True
        if ra and rb:
            both += 1
        elif ra:
            a_only += 1
        elif rb:
            b_only += 1
        else:
            neither += 1
    return Paired(len(ids), a_only, b_only, both, neither)

"""弃权验收:选择性预测(selective prediction)口径下的配对对照。

弃权训练不能用准确率验收。一个学会说「答案:不确定」的模型,准确率**必然**下降(弃权计错),
但它可能变得更有用——把不会的题标出来,调用方就能转人工而不是收到一个自信的错答。
所以要同时看四个数,缺一个都能被单独刷高:

  覆盖率 coverage        **拿到一个可用答案**的题占比 = (n − 主动弃权 − 未收尾/缺失) / n。
                         全弃权 → 覆盖率 0,选择性准确率 100%,毫无用处。
  未收尾/缺失 unusable   撞 max_tokens 没交卷 / 端点没生成。它不是弃权也不是答错,必须单列:
                         混进「答了但答错」,一条三成截断的臂会打印覆盖率 100%,再把截断
                         造成的准确率塌陷读成「弃权训练毁了作答能力」。
  选择性准确率 selective 只在**拿到可用答案**的题里算准确率,即「调用方拿到答案时答案对的概率」。
  严格准确率 strict      弃权计错的老口径,与 report.py / usability.py 同分母,防止上面两个数打架。
  弃权精度 precision     弃权掉的题里,**参照 run 本来就答错**的比例——弃权是不是弃在该弃的地方。
  弃权召回 recall        参照 run 答错的题里,被弃权掉的比例——该弃的弃掉了多少。

参照 run(--ref)必须是**同一个模型弃权训练之前**在同一批题上的 run(如 distill-v3-sample):
「本来就答错」只有相对于这个基准才有意义。拿基座或别的配置当参照,弃权精度会变成
「新模型弃权的题恰好也是别的模型不会的题」——一个关于题目难度的数,不是关于这次训练的数。

同理,--run 必须与 --ref **同提示词、同解码,只差权重**:弃权教材的 user 是 default 提示词
(build_abstain 用 eval.run.PROMPT_CHOICE 渲染),所以配对臂是 `<prefix>-v3-sample`。
不要拿 `<prefix>-v3-abstain` 当 --run:那条臂是贪心 8192 + `--prompt abstain`(见
scripts/eval_p2_arms.sh),与 distill-v3-sample 的采样 32768 + default 提示词差着解码协议、
提示词模板、权重三样东西,而且它测的是另一个问题——「换个提示词能不能白拿弃权」。
真想看那条臂,应与 base-v3-abstain / distill-v3-abstain 这类同协议的 run 配对。

用法:
    uv run python -m medforge.eval.abstain_report \
      --run abstain-v3-sample --ref distill-v3-sample \
      --sets cmexam,cmb-val,medxpertqa --out reports/abstain-selective.md

读 `reports/runs/<run>/<set>.scored.jsonl`(schema 见 report.load_verdicts),打一张 markdown 表。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from medforge.eval.report import load_verdicts


# 主动弃权的判据:验证器把「答案:不确定」判成 method=abstain + detail=declared
# (见 verify_by_rule)。method=abstain 但 detail 不是 declared 的,是**验证器**抽不出答案,
# 不是模型主动弃权——两者混在一起,弃权率就会被格式崩坏的输出灌水。
def is_declared_abstain(row: dict) -> bool:
    return row.get("method") == "abstain" and row.get("detail") == "declared"


def is_correct(row: dict) -> bool:
    return row.get("correct") is True


# 没交卷:撞 max_tokens(unfinished)或端点压根没生成(missing)。与 report.load_run 同一组
# method 取值——两张表必须用同一把尺,否则覆盖率和 RunResult 的未收尾率会互相矛盾。
def is_unusable(row: dict) -> bool:
    return row.get("method") in ("unfinished", "missing")


@dataclass
class Selective:
    """一套卷上的配对选择性预测结果。分母全部是两个 run 都作答过的共同题目。"""

    name: str
    n: int  # 共同题数
    abstained: int  # 新 run 主动弃权的题数
    correct: int  # 新 run 判对的题数(弃权题不可能判对)
    ref_correct: int  # 参照 run 判对的题数
    hit: int  # 弃权 ∧ 参照答错:弃对了地方
    ref_wrong: int  # 参照答错的题数
    unusable: int = 0  # 新 run 没交卷的题数(未收尾 / 缺失):不是弃权也不是答错,单列

    @property
    def answered(self) -> int:
        """真正交出了一个可用答案的题数。弃权与未收尾都不算——调用方两种情况下都没拿到答案。"""
        return self.n - self.abstained - self.unusable

    @property
    def coverage(self) -> float:
        return self.answered / self.n if self.n else 0.0

    @property
    def unusable_rate(self) -> float:
        return self.unusable / self.n if self.n else 0.0

    @property
    def abstain_rate(self) -> float:
        return self.abstained / self.n if self.n else 0.0

    @property
    def selective_acc(self) -> float:
        return self.correct / self.answered if self.answered else 0.0

    @property
    def strict_acc(self) -> float:
        """弃权计错的准确率:与 report.RunResult.acc 同口径,两张表的数必须对得上。"""
        return self.correct / self.n if self.n else 0.0

    @property
    def ref_acc(self) -> float:
        return self.ref_correct / self.n if self.n else 0.0

    @property
    def precision(self) -> float:
        return self.hit / self.abstained if self.abstained else 0.0

    @property
    def recall(self) -> float:
        return self.hit / self.ref_wrong if self.ref_wrong else 0.0


def selective(name: str, run: dict[str, dict], ref: dict[str, dict]) -> Selective:
    """两个 run 的判分表 → 配对选择性指标。只看共同 id(抽样卷 / 断点续跑都可能不齐)。"""
    ids = run.keys() & ref.keys()
    abstained = correct = ref_correct = hit = ref_wrong = unusable = 0
    for i in ids:
        a = is_declared_abstain(run[i])
        rw = not is_correct(ref[i])
        abstained += a
        unusable += is_unusable(run[i])
        correct += is_correct(run[i])
        ref_correct += not rw
        ref_wrong += rw
        hit += a and rw
    return Selective(name, len(ids), abstained, correct, ref_correct, hit, ref_wrong, unusable)


def load_pair(runs_dir: Path, run: str, ref: str, name: str) -> Selective | None:
    """一套卷的两份 scored.jsonl → Selective;任一缺失返回 None(调用方跳过这套卷)。"""
    a, b = runs_dir / run / f"{name}.scored.jsonl", runs_dir / ref / f"{name}.scored.jsonl"
    if not (a.exists() and b.exists()):
        return None
    return selective(name, load_verdicts(a), load_verdicts(b))


def markdown_table(rows: list[Selective]) -> str:
    lines = [
        (
            "| 卷 | n | 覆盖率 | 主动弃权 | 未收尾/缺失 | 选择性准确率 | 严格准确率 "
            "| 参照准确率 | 弃权精度 | 弃权召回 |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r.name} | {r.n} | {r.coverage * 100:.1f}% | {r.abstain_rate * 100:.1f}% "
            f"| {r.unusable_rate * 100:.1f}% | {r.selective_acc * 100:.1f}% "
            f"| {r.strict_acc * 100:.1f}% | {r.ref_acc * 100:.1f}% "
            f"| {r.precision * 100:.1f}% | {r.recall * 100:.1f}% |"
        )
    return "\n".join(lines)


def render_report(run: str, ref: str, rows: list[Selective]) -> str:
    return "\n".join(
        [
            f"# 弃权验收:{run} vs 参照 {ref}",
            "",
            (
                f"参照 run `{ref}` 是同一模型弃权训练之前在同一批题上的成绩;"
                "所有分母都取两个 run 的共同题目。两个 run 必须同提示词同解码、只差权重"
                "(如 `*-v3-sample` 配 `*-v3-sample`),否则量到的是解码差异不是弃权能力。"
            ),
            "",
            markdown_table(rows),
            "",
            (
                "- **覆盖率** = 交出了可用答案的题占比 =(n − 主动弃权 − 未收尾/缺失)/ n;"
                "**选择性准确率** = 这些题里的准确率,即「调用方拿到答案时答案对的概率」。"
            ),
            (
                "- **未收尾/缺失** 单列:它是「没交卷」不是「不会」。这一列高的臂(如贪心 8192 的 "
                "`*-v3-abstain`)不能拿来量弃权——选择性准确率的塌陷来自截断,与训练无关。"
            ),
            "- **严格准确率** 弃权计错,与 `report.py` / `usability.py` 同口径。",
            (
                "- **弃权精度** = 弃权题里参照 run 本来就答错的比例(随机弃权时它等于参照的错误率);"
                "**弃权召回** = 参照答错的题里被弃掉的比例。"
            ),
            (
                "- 验收标准:选择性准确率显著高于参照准确率,且弃权精度明显高于参照错误率——"
                "否则模型只是在随机拒答,拿覆盖率换了个假的准确率。"
            ),
            "",
        ]
    )


def main(argv: list[str] | None = None) -> None:
    from medforge.data.sources import ROOT

    ap = argparse.ArgumentParser(
        prog="medforge.eval.abstain_report",
        description="弃权验收:覆盖率 / 选择性准确率 / 弃权精度召回(与参照 run 配对)",
    )
    ap.add_argument("--run", required=True, help="弃权训练后的 run 名(reports/runs/<run>/)")
    ap.add_argument("--ref", required=True, help="参照 run:同一模型弃权训练之前,如 distill-v3-sample")
    ap.add_argument("--sets", default="cmexam,cmb-val,medxpertqa")
    ap.add_argument("--runs-dir", default=str(ROOT / "reports" / "runs"))
    ap.add_argument("--out", default="", help="markdown 落盘路径;留空只打印")
    args = ap.parse_args(argv)

    runs_dir = Path(args.runs_dir)
    rows = []
    for name in [s.strip() for s in args.sets.split(",") if s.strip()]:
        r = load_pair(runs_dir, args.run, args.ref, name)
        if r is None:
            print(f"  ! 跳过 {name}:{args.run} 或 {args.ref} 缺 scored.jsonl")
            continue
        rows.append(r)
    if not rows:
        raise SystemExit(f"✗ 没有任何一套卷同时有 {args.run} 与 {args.ref} 的判分结果")
    text = render_report(args.run, args.ref, rows)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"→ {out}")


if __name__ == "__main__":
    main()

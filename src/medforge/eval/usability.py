"""严格可用口径:把「没交卷」从分数里剥出来,重算存档答卷。

W2 审查(2026-09-03)发现:协议 v2 的 temperature=0 让思考型基座在 8192 token 内
大面积陷入复读循环、从未写出 </think>,而规则层只看末 2000 字,会从循环里刮出「答案:X」
判硬分。于是四个配置从来不在同一测量制度下比较,「弃权计错」这个口径也没覆盖「压根没写完」。

本脚本对每份存档答卷逐题打三个互相独立的标签,再按同一批题做配对检验:
  收尾  finished   输出里恰好一个 </think>(Qwen3.5 模板吃掉了开标签)
  声明  declared   </think> 之后的作答段能被规则层抽出答案(不从思考流里刮)
  严格  strict     收尾 ∧ 声明 ∧ 答对——「调用方拿到手的东西是对的」
  退化  degenerate 尾部 12-gram 重复率 ≥ 0.5(语言无关;标定见 REP_THRESHOLD 注释)
宽口径(as-scored)= 原 scored.jsonl 里的 correct,含 LLM 兜底与从复读段刮出的硬分,
用的是 W2 审查前的抽取器——所以它与严格口径不严格可比(新抽取器补抽的少数题会让严格 > 宽口径),只作对外数字的锚。
「规则层·全文」才是与严格口径同抽取器、只差守卫的一列:严格 ≤ 规则层·全文 恒成立。

用法:
    # 本机有原始答卷(reports/runs/<run>/<set>.outputs.jsonl):重新打标 + 出表
    uv run python -m medforge.eval.usability --runs base-v2,sft-v2,sft-r1-v2,dpo-v2 --baseline base-v2
    # 第三方(只有 git 里的逐题标签):直接出表,不碰标签文件
    uv run python -m medforge.eval.usability --from-tags
产出:
    reports/runs/<run>/<set>.usability.jsonl  逐题标签(小文件,入 git:第三方无需 300MB 原始答卷即可复算表格)
    reports/usability.md                       各集对照表 + 配对 McNemar + 三层分解
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median

from medforge.data.schema import Sample
from medforge.eval.report import Paired, benjamini_hochberg, holm, load_verdicts, paired_counts
from medforge.verify.extract import extract
from medforge.verify.verifier import split_answer, verify_by_rule

# 标定(reports/runs/*-v2 全部答卷,末 4000 字符、12-gram):写完的答卷重复率 p99 ≤ 0.37,
# 未收尾且复读的答卷中位 0.57~0.98;0.5 把两群干净分开,且对中英文同样成立。
# 不用 zlib 压缩比:英文与中文的基线压缩比不同(0.39 vs 0.34),同一阈值两种语言严苛程度不同。
REP_WINDOW = 4000
REP_N = 12
REP_THRESHOLD = 0.5
# 进多重比较校正族的臂:协议 v2 下训练出来的配方;存档 v1(base)只是协议对照
NON_ARCHIVE = {"sft-v2", "sft-r1-v2", "dpo-v2"}


def tail_repetition(text: str, n: int = REP_N, window: int = REP_WINDOW) -> float:
    """尾部字符 n-gram 重复率:1 − 唯一 n-gram 数 / 总 n-gram 数。太短的文本记 0。"""
    t = text[-window:]
    if len(t) < n * 4:
        return 0.0
    grams = [t[i : i + n] for i in range(len(t) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum("一" <= ch <= "鿿" for ch in text) / len(text)


@dataclass
class Tag:
    id: str
    finished: bool
    declared: bool
    strict: bool
    rule_full: bool     # 新抽取器看全文末段(允许从思考流刮)也判对——与 strict 的差就是「刮草稿」送的分
    wide: bool          # as-scored:原判分结果(旧抽取器 + LLM 兜底)
    method: str         # 原判分路径 rule / llm / abstain / missing
    chars: int
    cjk: float
    rep: float
    degenerate: bool
    extracted: str | None = None


def tag_output(sample: Sample, output: str | None, scored: dict) -> Tag:
    """一题一标签。output=None 表示存档里没有这题的作答。"""
    wide = scored.get("correct") is True
    method = str(scored.get("method", "missing"))
    if output is None:
        return Tag(sample.id, False, False, False, False, wide, method, 0, 0.0, 0.0, False)
    answer, unfinished = split_answer(output, thinking=True)
    finished = unfinished is None
    ext = extract(answer, sample.is_choice, options=sample.options) if finished else None
    v = verify_by_rule(sample, answer) if finished else None
    v_full = verify_by_rule(sample, output)  # 不设守卫、看全文末 2000 字:原口径的规则层行为
    rep = tail_repetition(output)
    return Tag(
        id=sample.id, finished=finished, declared=ext is not None,
        strict=bool(v is not None and v.correct is True),
        rule_full=bool(v_full is not None and v_full.correct is True),
        wide=wide, method=method,
        chars=len(output), cjk=round(cjk_ratio(output), 4), rep=round(rep, 4), degenerate=rep >= REP_THRESHOLD,
        extracted=ext.value if ext else None,
    )


def tag_run(run_dir: Path, eval_set: str, samples: dict[str, Sample]) -> list[Tag]:
    """给一个 run 的一套卷打标,顺序与 scored.jsonl 一致;同时落 <set>.usability.jsonl。"""
    scored = load_verdicts(run_dir / f"{eval_set}.scored.jsonl")
    out_file = run_dir / f"{eval_set}.outputs.jsonl"
    if not out_file.exists():
        # 没有原始答卷就不能重新打标——静默把全部题打成「未收尾」会覆盖掉入库的标签文件
        raise FileNotFoundError(f"{out_file} 不存在:用 --from-tags 直接读已入库的 usability.jsonl 出表")
    outputs: dict[str, str] = {}
    for line in out_file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row["id"] in scored:  # outputs 可能含早期会话残留,只认被判分的题
                outputs[row["id"]] = row["output"]
    tags = [tag_output(samples[sid], outputs.get(sid), row) for sid, row in scored.items() if sid in samples]
    with (run_dir / f"{eval_set}.usability.jsonl").open("w", encoding="utf-8") as f:
        for t in tags:
            f.write(json.dumps(asdict(t), ensure_ascii=False) + "\n")
    return tags


def load_tags(run_dir: Path, eval_set: str) -> list[Tag]:
    """从已落盘的 usability.jsonl 读回(没有原始答卷的机器也能复算表格)。"""
    return [Tag(**json.loads(line)) for line in (run_dir / f"{eval_set}.usability.jsonl").read_text("utf-8").splitlines() if line.strip()]


@dataclass
class SetStats:
    run: str
    n: int
    finished: float
    declared: float
    degenerate: float
    strict: float
    rule_full: float    # 新抽取器看全文(允许刮草稿)的判对率;严格 ≤ 它,差值 = 从思考流刮出的分
    wide: float         # 原判分(旧抽取器 + LLM 兜底 + 刮草稿)
    chars_p50: int
    cjk_p50: float

    @classmethod
    def of(cls, run: str, tags: list[Tag]) -> SetStats:
        n = len(tags) or 1
        return cls(
            run=run, n=len(tags),
            finished=sum(t.finished for t in tags) / n,
            declared=sum(t.declared for t in tags) / n,
            degenerate=sum(t.degenerate for t in tags) / n,
            strict=sum(t.strict for t in tags) / n,
            rule_full=sum(t.rule_full for t in tags) / n,
            wide=sum(t.wide for t in tags) / n,
            chars_p50=int(median(t.chars for t in tags)) if tags else 0,
            cjk_p50=median(t.cjk for t in tags) if tags else 0.0,
        )


def paired(a: list[Tag], b: list[Tag], attr: str) -> Paired:
    return paired_counts({t.id: getattr(t, attr) for t in a}, {t.id: getattr(t, attr) for t in b})


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _rate(tags: list[Tag], ids: set[str], attr: str) -> float:
    """只在 ids 子集上算比例(v1 全量卷与 v2 抽样卷配对时用)。"""
    return sum(getattr(t, attr) for t in tags if t.id in ids) / (len(ids) or 1)


def _vs(p: Paired, stat_a: float, stat_b: float, n_total: int, holm_ok: bool | None, bh_ok: bool | None) -> str:
    """一格里放齐 Miller 2024 要求的东西:配对差值 + 95% CI + 相关系数 + 配对 p + 多重比较校正结论。"""
    lo, hi = p.ci()
    n = f" · n={p.n}" if p.n != n_total else ""
    corr = "" if holm_ok is None else f" · Holm{'✓' if holm_ok else '✗'}/BH{'✓' if bh_ok else '✗'}"
    return (
        f"{(stat_b - stat_a) * 100:+.1f}pp [{lo * 100:+.1f}, {hi * 100:+.1f}]{n} · φ={p.phi:.2f}"
        f" · 独对 {p.a_only}/{p.b_only} · p={p.p_value:.4f}{corr}"
    )


def render(
    results: dict[str, dict[str, list[Tag]]], baseline: str, sets: list[str], run_notes: dict[str, str],
) -> str:
    """results[set][run] = tags。"""
    lines = [
        "# 严格可用口径 · 存档答卷重算",
        "",
        "由 `uv run python -m medforge.eval.usability` 生成;逐题标签在 `reports/runs/<run>/<set>.usability.jsonl`。",
        "",
        "## 口径",
        "",
        "- **收尾率**:输出含 `</think>`(作答段取最后一个之后)。v2 协议下未收尾的答卷约七成落入复读循环;",
        "  v1 存档(base)是 2048 token 硬截断,同样不该有分但机理不同,其退化率列不可与 v2 横比。",
        "- **声明率**:`</think>` 之后的作答段能被规则层抽出答案。不从思考流里刮——",
        "  原 scored 的规则层看的是全文末 2000 字,对未收尾答卷那正是循环体。",
        "- **严格准确率** = 收尾 ∧ 声明 ∧ 答对(选择题字母集合相等;开放题规则层精确匹配)。",
        "  这是「调用方每次拿到的结论是对的」的比例,是本表的主指标。",
        "- **规则层·全文**:同一抽取器不设守卫、看全文末 2000 字(原口径的规则层行为,允许从思考流刮)。",
        "  严格 ≤ 规则层·全文 恒成立,差值就是「刮草稿」送出去的分。",
        "- **宽口径(as-scored)**:原 scored.jsonl 的判分,含 LLM 兜底与刮草稿,用的是 W2 审查前的抽取器",
        "  (连写多选「答案:ABD」抽不出、跨行分隔误并),所以它与前两列不严格可比,只作对外数字的锚。",
        f"- **退化率**:末 {REP_WINDOW} 字符的 {REP_N}-gram 重复率 ≥ {REP_THRESHOLD}。语言无关;",
        "  标定(v2 存档):写完的答卷 p99 ≤ 0.37,0.5 之上几乎不含写完的答卷;未收尾答卷中位 0.57~0.98,",
        "  约三成未收尾答卷落在 0.5 以下(被掐断时尚未进入复读)——它是单侧指标,不是截断的代理。",
        "- **vs 基线**:同一批题配对。格内依次是:配对差值与其 95% CI(逐题差分的标准误,Miller 2024)、",
        "  φ(两臂逐题对错的相关系数,越高配对检验越有功效)、「独对 a/b」= 只有基线对 / 只有本 run 对、",
        "  McNemar 精确检验双侧 p、Holm / BH 多重比较校正是否通过。pp 差值按配对子集算",
        "  (v1 全量卷与 v2 抽样卷配对时取交集,n 标在格内),与整卷的严格准确率相减对不上是正常的。",
        "",
        "所有 run 考的是同一批固定种子抽样题,可逐题配对;v1 存档(base)是全量卷,配对时取交集。",
        "",
    ]
    # 先算完全部配对比较,再做多重比较校正:校正族 = 非存档臂 × 三卷(严格与宽口径各一族),
    # 存档 v1 的 base 行不进族(它是协议对照,不是「哪个配方更好」的假设)
    comps: dict[tuple[str, str, str], tuple[Paired, float, float, int]] = {}
    for s in sets:
        base_tags = results[s].get(baseline)
        if base_tags is None:
            continue
        for run, tags in results[s].items():
            if run == baseline:
                continue
            common = {t.id for t in tags} & {t.id for t in base_tags}
            for attr in ("strict", "wide"):
                comps[(s, run, attr)] = (
                    paired(base_tags, tags, attr), _rate(base_tags, common, attr), _rate(tags, common, attr), len(tags),
                )
    family = {attr: [k for k in comps if k[2] == attr and k[1] in NON_ARCHIVE] for attr in ("strict", "wide")}
    corrected: dict[tuple[str, str, str], tuple[bool, bool]] = {}
    for attr, keys in family.items():
        ps = [comps[k][0].p_value for k in keys]
        for k, h, b in zip(keys, holm(ps), benjamini_hochberg(ps)):
            corrected[k] = (h, b)
    lines += [
        (
            f"多重比较校正:严格口径与宽口径各自一族({len(family['strict'])} 项:非存档臂 × 三卷),"
            "Holm 控 FWER、BH 控 FDR,均取 α=0.05;存档 v1 行不进族。"
        ),
        "",
    ]
    for s in sets:
        runs = results[s]
        base_tags = runs.get(baseline)
        lines += [f"## {s}", ""]
        lines += [
            (
                "| 配置 | n | 收尾率 | 声明率 | 退化率 | **严格准确率** | 规则层·全文 | 宽口径 | 字符 p50 "
                f"| 中文占比 p50 | vs {baseline}(严格) | vs {baseline}(宽口径) |"
            ),
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        mdes = []
        for run, tags in runs.items():
            st = SetStats.of(run, tags)
            if base_tags is None or run == baseline:
                vs_s = vs_w = "—"
            else:
                cells = []
                for attr in ("strict", "wide"):
                    p, ra, rb, n_total = comps[(s, run, attr)]
                    h, b = corrected.get((s, run, attr), (None, None))
                    cells.append(_vs(p, ra, rb, n_total, h, b))
                    if attr == "strict" and run in NON_ARCHIVE:
                        mdes.append(p.mde())
                vs_s, vs_w = cells
            lines.append(
                f"| {run} | {st.n} | {_pct(st.finished)} | {_pct(st.declared)} | {_pct(st.degenerate)} "
                f"| **{_pct(st.strict)}** | {_pct(st.rule_full)} | {_pct(st.wide)} | {st.chars_p50:,} "
                f"| {_pct(st.cjk_p50)} | {vs_s} | {vs_w} |"
            )
        if mdes:
            lines.append("")
            lines.append(
                f"本卷严格口径的最小可检出差异(α=0.05,功效 0.8)约 {min(mdes) * 100:.1f}~{max(mdes) * 100:.1f}pp;"
                "「未检出差异」只表示效应小于这个量级。"
            )
        lines.append("")
    if run_notes:
        lines += ["## 各 run 说明", ""]
        lines += [f"- **{k}**:{v}" for k, v in run_notes.items()]
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    from medforge.data.sources import EVAL_SOURCES, ROOT, load_source

    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="base,base-v2,sft-v2,sft-r1-v2,dpo-v2")
    ap.add_argument("--sets", default=",".join(EVAL_SOURCES))
    ap.add_argument("--baseline", default="base-v2")
    ap.add_argument("--out", default=str(ROOT / "reports" / "usability.md"))
    ap.add_argument("--from-tags", action="store_true", help="不读原始答卷,直接用已落盘的 usability.jsonl 出表")
    args = ap.parse_args()
    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    sets = [s.strip() for s in args.sets.split(",") if s.strip()]
    runs_dir = ROOT / "reports" / "runs"

    results: dict[str, dict[str, list[Tag]]] = {}
    for s in sets:
        samples = {} if args.from_tags else {x.id: x for x in load_source(s)}
        results[s] = {}
        for run in runs:
            run_dir = runs_dir / run
            if not (run_dir / f"{s}.scored.jsonl").exists():
                continue
            if args.from_tags or not (run_dir / f"{s}.outputs.jsonl").exists():
                if not args.from_tags:
                    print(f"  ! {run}/{s} 没有原始答卷,改读已入库的 usability.jsonl")
                results[s][run] = load_tags(run_dir, s)
            else:
                results[s][run] = tag_run(run_dir, s, samples)
            st = SetStats.of(run, results[s][run])
            print(f"{s:11s} {run:10s} n={st.n:4d} 收尾 {_pct(st.finished):>6s} 严格 {_pct(st.strict):>6s} 宽 {_pct(st.wide):>6s}")
    notes = {
        "base": "协议 v1 存档:max_tokens 2048,全量卷。严格准确率主要受 2048 硬截断支配,不能当模型能力读;与 v2 抽样卷配对时取交集",
        "base-v2": "协议 v2:max_tokens 8192,temperature 0,固定种子抽样卷",
    }
    Path(args.out).write_text(render(results, args.baseline, sets, {k: v for k, v in notes.items() if k in runs}), "utf-8")
    print(f"✓ → {args.out}")


if __name__ == "__main__":
    main()

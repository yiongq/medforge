"""数据构建管线 CLI:raw → 归一 → 去污染 → 干净题池落盘 + 去污染报告。

用法:
    uv run python -m medforge.data.build                                  # W1 默认卷:三个开放题源
    uv run python -m medforge.data.build --source cmexam-train \
        --out-suffix cmexam-train                                         # CMExam 官方训练集,另存

产出(默认):
    data/processed/train_pool.jsonl   去污染后的训练题池(Sample 字典;SFT 成品由 W2 的
                                      build_sft 在此之上合成 CoT 并混通用数据)
    reports/decontamination.md        去污染报告:方法、数字、存疑样本清单(进 git)

产出(--out-suffix S 时):train_pool-S.jsonl / decontamination-S.md。
带 --source 就**必须**带 --out-suffix:已公开的 W1 数字不允许被另一批训练源静默改写,
所以覆盖默认产物这条路在 CLI 层直接堵死(见 _resolve_outputs)。

报告是一等公民:「查了什么、怎么查的、剔了多少」必须公开可追溯,
这是评测数字可信的前提(ADR 口径 2)。embedding 语义层接入后本报告会追加一节。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from rich import print as rprint

from medforge.data.decontaminate import (
    BOILERPLATE_DF,
    CONTAMINATED,
    NGRAM,
    SUSPICIOUS,
    ExactHit,
    Hit,
    RemovalPolicy,
    removal_policy,
    removed_train_ids,
    scan,
    scan_exact,
    short_stem_ids,
    unscannable,
)
from medforge.data.normalize import drop_counts
from medforge.data.schema import Sample
from medforge.data.sources import (
    DEFAULT_TRAIN_SOURCES,
    EVAL_SOURCES,
    ROOT,
    TRAIN_SOURCES,
    load_source,
)

PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

# 主力卷:CMExam test 的固定种子抽样,与 eval/run.py 的 `--samples cmexam=2000` 同构
# (那边是 random.Random(42).sample(load_source("cmexam"), 2000))。
# 它是 cmexam 全卷的子集,不额外产生剔除;单列一行是为了让报告里的数字能和
# REMOVAL_POLICIES 注释里引用的实测数字直接对上。
MAIN_PAPER_SET = "cmexam"
MAIN_PAPER_N = 2000
MAIN_PAPER_SEED = 42
MAIN_PAPER_NAME = f"{MAIN_PAPER_SET}-{MAIN_PAPER_N}(主力卷)"

SHORT_STEM_LIMIT = 30  # 报告里「短题干」的口径:归一化后不足 30 字符


def _resolve_outputs(suffix: str) -> tuple[Path, Path]:
    tag = f"-{suffix}" if suffix else ""
    return PROCESSED / f"train_pool{tag}.jsonl", REPORTS / f"decontamination{tag}.md"


def _pct(n: int, total: int) -> str:
    return f"{n / total:.2%}" if total else "0.00%"


def build(sources: list[str], suffix: str) -> None:
    out_pool, out_report = _resolve_outputs(suffix)

    rprint("[bold]== 加载数据源 ==[/]")
    train: list[Sample] = []
    for name in sources:
        samples = load_source(name)
        rprint(f"  {name}: {len(samples)} 样本")
        train.extend(samples)
    evals: dict[str, list[Sample]] = {}
    for name in EVAL_SOURCES:
        evals[name] = load_source(name)
        rprint(f"  {name}: {len(evals[name])} 样本")
    train_source = {s.id: s.source for s in train}
    # 精确通道(以及随它一起的主力卷这一行)只在「至少有一个源真的拿它当判据」时才跑:
    # 默认那一跑因此和 W1 同构,decontamination.md 的表格与数字不会被动改写。
    policies = {name: removal_policy(name) for name in sources}
    use_exact = any(p.stem_options_exact or p.stem_exact for p in policies.values())
    if use_exact and MAIN_PAPER_SET in evals:
        evals[MAIN_PAPER_NAME] = random.Random(MAIN_PAPER_SEED).sample(
            evals[MAIN_PAPER_SET], min(MAIN_PAPER_N, len(evals[MAIN_PAPER_SET]))
        )

    rprint(f"[bold]== 字面去污染扫描(字符 {NGRAM}-gram{'+ 精确通道' if use_exact else ''})==[/]")
    train_texts = [(s.id, s.question) for s in train]
    train_exact = [(s.id, s.question, s.options) for s in train]
    all_hits: dict[str, list[Hit]] = {}
    exact: dict[str, list[ExactHit]] = {}
    unscan: dict[str, int] = {}
    unscan_ids: dict[str, set[str]] = {}   # n-gram 结构性看不见的题:留 id 才能算「精确通道捞回多少」
    short: dict[str, int] = {}
    for name, samples in evals.items():
        eval_texts = [(s.id, s.question) for s in samples]
        hits = scan(train_texts, eval_texts)
        unscan_ids[name] = set(unscannable(eval_texts))
        unscan[name] = len(unscan_ids[name])
        short[name] = len(short_stem_ids(eval_texts, SHORT_STEM_LIMIT))
        n_cont = sum(1 for h in hits if h.level == "contaminated")
        all_hits[name] = hits
        line = (
            f"  vs {name}: n-gram 命中 {len(hits)}(污染 {n_cont} / 存疑 {len(hits) - n_cont})"
            f" | 短题干不可扫描 {unscan[name]}"
        )
        if use_exact:
            ex = scan_exact(train_exact, [(s.id, s.question, s.options) for s in samples])
            exact[name] = ex
            n_full = sum(1 for e in ex if e.channel == "stem_options_exact")
            line += f" | 精确 全等 {n_full} / 仅题干同 {len(ex) - n_full}"
        rprint(line)

    removed = set()
    for name, hits in all_hits.items():
        removed |= removed_train_ids(hits, exact.get(name, []), train_source)
    clean = [s for s in train if s.id not in removed]

    PROCESSED.mkdir(parents=True, exist_ok=True)
    with out_pool.open("w", encoding="utf-8") as f:
        for s in clean:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
    rprint(f"[green]✓[/] 题池 {len(train)} → 剔除 {len(removed)} → {len(clean)} 条 → {out_pool}")

    REPORTS.mkdir(exist_ok=True)
    lines = (
        _report_exact(
            sources, policies, train, clean, removed, evals, all_hits, exact, unscan_ids, short
        )
        if use_exact
        else _report_legacy(sources, train, clean, removed, evals, all_hits, unscan)
    )
    out_report.write_text("\n".join(lines), encoding="utf-8")
    rprint(f"[green]✓[/] 报告 → {out_report}")


def _report_legacy(
    sources: list[str],
    train: list[Sample],
    clean: list[Sample],
    removed: set[str],
    evals: dict[str, list[Sample]],
    all_hits: dict[str, list[Hit]],
    unscan: dict[str, int],
) -> list[str]:
    """W1 口径的报告(n-gram 单通道,命中数=配对数)。

    刻意与精确通道版分成两个函数、不做参数化合并:reports/decontamination.md 是
    已公开数字,任何为了复用而做的改写都可能悄悄挪动它一个字符。这里的正文与
    W1 逐字节一致,新通道的表达自由全放在 _report_exact 里。
    """
    lines = [
        "# 去污染报告(字面层)",
        "",
        f"方法:字符 {NGRAM}-gram 倒排索引,评测题 shingle 覆盖率 ≥{CONTAMINATED} 判污染(训练侧剔除)、",
        f"≥{SUSPICIOUS} 记存疑(仅报告);模板噪声按文档频率 >{BOILERPLATE_DF:.1%} 剔出索引。",
        "只比题干,不比选项。embedding 语义层接入后在此追加第二节。",
        "",
        f"训练题池:{len(train)} 条(来源:{', '.join(sources)})",
        f"剔除污染样本:{len(removed)} 条 → 干净题池 {len(clean)} 条",
        "",
        "| 评测集 | 题数 | 污染命中 | 存疑命中 | 短题干不可扫描 |",
        "|---|---|---|---|---|",
    ]
    for name, hits in all_hits.items():
        n_cont = sum(1 for h in hits if h.level == "contaminated")
        lines.append(
            f"| {name} | {len(evals[name])} | {n_cont} | {len(hits) - n_cont} | {unscan[name]} |"
        )
    lines += [
        "",
        "「短题干不可扫描」= 归一化后不足 10 字符的题干(如「甘味的作用特点是」),题意在选项中,",
        "对开放题训练池无字面泄漏面;字面层对其无能为力,如实公布计数。",
    ]
    lines += ["", "## 存疑清单(覆盖率降序,各集 top 20,供人工抽看)", ""]
    for name, hits in all_hits.items():
        sus = [h for h in hits if h.level == "suspicious"][:20]
        if not sus:
            continue
        lines.append(f"### vs {name}\n")
        lines += [f"- `{h.eval_id}` ← `{h.train_id}`(覆盖率 {h.ratio}" + ")" for h in sus]
        lines.append("")
    if drop_counts():
        lines.append(f"归一层丢弃统计:{drop_counts()}")
    return lines


def _report_exact(
    sources: list[str],
    policies: dict[str, RemovalPolicy],
    train: list[Sample],
    clean: list[Sample],
    removed: set[str],
    evals: dict[str, list[Sample]],
    all_hits: dict[str, list[Hit]],
    exact: dict[str, list[ExactHit]],
    unscan_ids: dict[str, set[str]],
    short: dict[str, int],
) -> list[str]:
    """三通道报告:每条通道各自计数、各自可复算,剔除判据按训练源公示。

    计数口径统一成「命中的评测题数」(去重后的 eval_id),而不是 W1 的配对数——
    只有这样 9.05% / 0.6% 这类占比才有分母可言;剔除列另按训练样本数计。
    """
    lines = [
        "# 去污染报告(字面层 · 三通道)",
        "",
        f"训练题池:{len(train)} 条(来源:{', '.join(sources)})",
        f"剔除样本:{len(removed)} 条 → 干净题池 {len(clean)} 条",
        "",
        "## 方法:三条通道并列,各自计数、各自可复算",
        "",
        (
            f"1. `ngram` — 字符 {NGRAM}-gram 倒排索引,评测题 shingle 覆盖率 ≥{CONTAMINATED} 判污染、"
            f"≥{SUSPICIOUS} 记存疑;模板噪声按文档频率 >{BOILERPLATE_DF:.1%} 剔出索引。"
            f"归一化后 <{NGRAM} 字符的题干直接跳过 —— 这类题在本通道里结构性不可见。"
        ),
        "2. `stem_exact` — 归一化题干精确相等。**不受**上面的短题干跳过限制,专补短题。",
        "3. `stem_options_exact` — 归一化「题干 + 按字母序拼接的选项文本」精确相等,即真重题。",
        "",
        "归一化三条通道共用一套(去空白/标点/大小写,见 `decontaminate.normalize_text`)。",
        "",
        "## 剔除判据按训练源选择",
        "",
        "| 训练源 | ngram 污染档 | stem_exact | stem_options_exact |",
        "|---|---|---|---|",
    ]
    mark = {True: "**剔除**", False: "仅报告"}
    for name in sources:
        p = policies[name]
        lines.append(f"| {name} | {mark[p.ngram]} | {mark[p.stem_exact]} | {mark[p.stem_options_exact]} |")
    lines += [
        "",
        "依据(cmexam-train × 主力卷 2000 题的实测):n-gram 通道判污染 181 题(9.05%)、存疑 194 题(9.70%),",
        "而「题干+选项完全一致且 gold 相同」的真重题只有约 12 道(0.6%);仅题干一致的 104 道里有 92 道选项不同,",
        "是同一考点模板下的不同题。照搬 n-gram 阈值当剔除判据会误剔约 88% 的好数据,所以 cmexam-train 只认",
        "`stem_options_exact` 剔除、`stem_exact` 只报存疑,n-gram 通道保留召回与报告用途。",
        "开放题三源(med-o1-verifiable / med-o1-sft-zh / med-r1-zh)的判据一个字节不改,仍是 n-gram 单通道。",
        "",
        "## 三通道数字",
        "",
        "计数口径 = 命中的**评测题数**(eval_id 去重);`stem_options_exact 剔除` 列另计被剔的**训练样本数**。",
        "",
        (
            "| 评测卷 | 题数 | ngram 污染 | ngram 存疑 | stem_exact 命中 | ├ 选项相同 | └ 选项不同 |"
            " stem_options_exact 剔除(训练样本) | 短题干不可扫描 |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, hits in all_hits.items():
        total = len(evals[name])
        cont_q = {h.eval_id for h in hits if h.level == "contaminated"}
        sus_q = {h.eval_id for h in hits if h.level == "suspicious"} - cont_q
        ex = exact.get(name, [])
        same_q = {e.eval_id for e in ex if e.channel == "stem_options_exact"}
        diff_q = {e.eval_id for e in ex if e.channel == "stem_exact"} - same_q
        rm_train = {e.train_id for e in ex if e.channel == "stem_options_exact"}
        lines.append(
            f"| {name} | {total} | {len(cont_q)}({_pct(len(cont_q), total)}) | "
            f"{len(sus_q)}({_pct(len(sus_q), total)}) | {len(same_q) + len(diff_q)} | "
            f"{len(same_q)}({_pct(len(same_q), total)}) | {len(diff_q)} | {len(rm_train)} | {len(unscan_ids[name])} |"
        )
    main = MAIN_PAPER_SET
    # 精确通道对「n-gram 结构性看不见的短题干」的补位效果:全部脚本算,不手填
    main_exact = exact.get(main, [])
    short_stem_hit = {e.eval_id for e in main_exact if e.eval_id in unscan_ids[main]}
    short_stem_dupe = {
        e.eval_id for e in main_exact
        if e.channel == "stem_options_exact" and e.eval_id in unscan_ids[main]
    }
    lines += [
        "",
        f"{MAIN_PAPER_NAME} = {main} test 用 `random.Random({MAIN_PAPER_SEED}).sample(..., {MAIN_PAPER_N})` 抽的固定卷,",
        "与 `eval/run.py --samples cmexam=2000` 同构;它是 cmexam 全卷的子集,不额外产生剔除,单列一行只为让",
        "上面「判据依据」引用的实测数字能被直接复算。",
        "",
        (
            f"CMExam test 有 {_pct(short[main], len(evals[main]))} 题干不足 {SHORT_STEM_LIMIT} 字符"
            f"({short[main]}/{len(evals[main])}),字面近似查重对它们基本无效;"
            "精确通道覆盖了其中的完全重合。"
        ),
        (
            f"其中归一化后不足 {NGRAM} 字符的 {len(unscan_ids[main])} 道在 n-gram 通道里被直接跳过(结构性不可见),"
            f"而精确通道在这批题里捞出 {len(short_stem_hit)} 道题干完全相同的训练样本、"
            f"其中 {len(short_stem_dupe)} 道连选项也完全一致 —— 后者已按判据剔除。"
        ),
        "",
        "## stem_options_exact 命中清单(真重题,全部;即剔除对象)",
        "",
    ]
    seen: set[tuple[str, str]] = set()
    rows = 0
    for name, ex in exact.items():
        if name == MAIN_PAPER_NAME:
            continue  # 主力卷是 cmexam 子集,清单里不重复列
        for e in ex:
            if e.channel != "stem_options_exact" or (e.eval_id, e.train_id) in seen:
                continue
            seen.add((e.eval_id, e.train_id))
            lines.append(f"- `{e.eval_id}` ← `{e.train_id}`(vs {name})")
            rows += 1
    if not rows:
        lines.append("(无)")
    lines += ["", "## stem_exact 存疑清单(题干同、选项不同 → 同模板不同题,不剔;各卷 top 20)", ""]
    shown = 0
    for name, ex in exact.items():
        if name == MAIN_PAPER_NAME:
            continue
        same_q = {e.eval_id for e in ex if e.channel == "stem_options_exact"}
        diff = [e for e in ex if e.channel == "stem_exact" and e.eval_id not in same_q][:20]
        if not diff:
            continue
        lines.append(f"### vs {name}\n")
        lines += [f"- `{e.eval_id}` ← `{e.train_id}`" for e in diff]
        lines.append("")
        shown += len(diff)
    if not shown:
        lines += ["(无)", ""]
    lines += ["## ngram 存疑清单(覆盖率降序,各卷 top 20,供人工抽看;不作为剔除判据)", ""]
    for name, hits in all_hits.items():
        sus = [h for h in hits if h.level == "suspicious"][:20]
        if not sus:
            continue
        lines.append(f"### vs {name}\n")
        lines += [f"- `{h.eval_id}` ← `{h.train_id}`(覆盖率 {h.ratio}" + ")" for h in sus]
        lines.append("")
    if drop_counts():
        lines.append(f"归一层丢弃统计:{drop_counts()}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description="raw → 归一 → 去污染 → 干净题池 + 报告")
    ap.add_argument(
        "--source", action="append", default=None,
        help=f"训练源(可重复);缺省为 W1 默认卷 {'/'.join(DEFAULT_TRAIN_SOURCES)}。可选:{'/'.join(TRAIN_SOURCES)}",
    )
    ap.add_argument(
        "--out-suffix", default="",
        help="产物后缀:train_pool-<S>.jsonl / decontamination-<S>.md;带 --source 时必填",
    )
    args = ap.parse_args()

    sources = args.source or list(DEFAULT_TRAIN_SOURCES)
    unknown = [s for s in sources if s not in TRAIN_SOURCES]
    if unknown:
        rprint(f"[red]✗ 未注册的训练源 {unknown};已注册:{list(TRAIN_SOURCES)}[/]")
        sys.exit(2)
    # 覆盖闸门:换了训练源却写回默认产物 = 静默改写已公开的 W1 数字
    if args.source and not args.out_suffix:
        rprint("[red]✗ --source 必须配 --out-suffix:默认产物 train_pool.jsonl / "
               "decontamination.md 对应已公开的 W1 数字,不接受被别的训练源覆盖[/]")
        sys.exit(2)
    build(sources, args.out_suffix)


if __name__ == "__main__":
    main()

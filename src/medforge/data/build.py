"""数据构建管线 CLI:raw → 归一 → 去污染 → 干净题池落盘 + 去污染报告。

用法:
    uv run python -m medforge.data.build

产出:
    data/processed/train_pool.jsonl   去污染后的训练题池(Sample 字典;SFT 成品由 W2 的
                                      build_sft 在此之上合成 CoT 并混通用数据)
    reports/decontamination.md        去污染报告:方法、数字、存疑样本清单(进 git)

报告是一等公民:「查了什么、怎么查的、剔了多少」必须公开可追溯,
这是评测数字可信的前提(ADR 口径 2)。embedding 语义层接入后本报告会追加第二节。
"""

from __future__ import annotations

import json
from pathlib import Path

from rich import print as rprint

from medforge.data.decontaminate import (
    BOILERPLATE_DF,
    CONTAMINATED,
    NGRAM,
    SUSPICIOUS,
    Hit,
    contaminated_train_ids,
    scan,
    unscannable,
)
from medforge.data.normalize import ADAPTERS, drop_counts
from medforge.data.schema import Sample

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

# 训练侧题池与评测集清单:新增数据源在这里挂接
TRAIN_SOURCES = [("med-o1-verifiable", "med-o1-verifiable.train.jsonl", ""),
                 ("med-o1-sft-zh", "med-o1-sft-zh.train.jsonl", "")]
EVAL_SOURCES = [("cmexam", "cmexam.test.jsonl", "test"),
                ("cmb-val", "cmb.val.jsonl", ""),
                ("medxpertqa", "medxpertqa.test.jsonl", "")]


def load_samples(name: str, filename: str, split: str) -> list[Sample]:
    rows = [json.loads(line) for line in (RAW / filename).open(encoding="utf-8")]
    samples = list(ADAPTERS[name](rows, split))
    rprint(f"  {name}: {len(rows)} 行 → {len(samples)} 样本")
    return samples


def main() -> None:
    rprint("[bold]== 加载数据源 ==[/]")
    train: list[Sample] = []
    for args in TRAIN_SOURCES:
        train.extend(load_samples(*args))
    evals: dict[str, list[Sample]] = {}
    for name, filename, split in EVAL_SOURCES:
        evals[name] = load_samples(name, filename, split)

    rprint("[bold]== 字面去污染扫描(字符 10-gram)==[/]")
    # 只比题干:选项文本会稀释覆盖率造成漏报;题干撞了就该人工看
    train_texts = [(s.id, s.question) for s in train]
    all_hits: dict[str, list[Hit]] = {}
    unscan: dict[str, int] = {}
    for name, samples in evals.items():
        eval_texts = [(s.id, s.question) for s in samples]
        hits = scan(train_texts, eval_texts)
        unscan[name] = len(unscannable(eval_texts))
        n_cont = sum(1 for h in hits if h.level == "contaminated")
        rprint(
            f"  vs {name}: 命中 {len(hits)}(污染 {n_cont} / 存疑 {len(hits) - n_cont})"
            f" | 短题干不可扫描 {unscan[name]}"
        )
        all_hits[name] = hits

    removed = set()
    for hits in all_hits.values():
        removed |= contaminated_train_ids(hits)
    clean = [s for s in train if s.id not in removed]

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / "train_pool.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for s in clean:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
    rprint(f"[green]✓[/] 题池 {len(train)} → 剔除 {len(removed)} → {len(clean)} 条 → {out}")

    REPORTS.mkdir(exist_ok=True)
    report = REPORTS / "decontamination.md"
    lines = [
        "# 去污染报告(字面层)",
        "",
        f"方法:字符 {NGRAM}-gram 倒排索引,评测题 shingle 覆盖率 ≥{CONTAMINATED} 判污染(训练侧剔除)、",
        f"≥{SUSPICIOUS} 记存疑(仅报告);模板噪声按文档频率 >{BOILERPLATE_DF:.1%} 剔出索引。",
        "只比题干,不比选项。embedding 语义层接入后在此追加第二节。",
        "",
        f"训练题池:{len(train)} 条(来源:{', '.join(n for n, _, _ in TRAIN_SOURCES)})",
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
    report.write_text("\n".join(lines), encoding="utf-8")
    rprint(f"[green]✓[/] 报告 → {report}")


if __name__ == "__main__":
    main()

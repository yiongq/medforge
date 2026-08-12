"""数据源注册表:训练侧/评测侧清单与统一加载入口。

build(去污染)与 eval(跑考卷)共用同一份清单——两边各抄一份的话,
新增数据源时漏改一边就会出现「训练查重了、评测却没跑」的静默错位。
"""

from __future__ import annotations

import json
from pathlib import Path

from medforge.data.normalize import ADAPTERS
from medforge.data.schema import Sample

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw"

# name -> (adapter 名, raw 文件, split)
TRAIN_SOURCES: dict[str, tuple[str, str, str]] = {
    "med-o1-verifiable": ("med-o1-verifiable", "med-o1-verifiable.train.jsonl", ""),
    "med-o1-sft-zh": ("med-o1-sft-zh", "med-o1-sft-zh.train.jsonl", ""),
}
EVAL_SOURCES: dict[str, tuple[str, str, str]] = {
    "cmexam": ("cmexam", "cmexam.test.jsonl", "test"),
    "cmb-val": ("cmb-val", "cmb.val.jsonl", ""),
    "medxpertqa": ("medxpertqa", "medxpertqa.test.jsonl", ""),
}


def load_source(name: str) -> list[Sample]:
    adapter, filename, split = (TRAIN_SOURCES | EVAL_SOURCES)[name]
    rows = [json.loads(line) for line in (RAW / filename).open(encoding="utf-8")]
    return list(ADAPTERS[adapter](rows, split))

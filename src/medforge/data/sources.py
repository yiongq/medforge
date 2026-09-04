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
    "med-r1-zh": ("med-r1-zh", "med-r1-zh.train.jsonl", ""),
    # CMExam 官方训练集(54,497 题):蒸馏 2.0 / 自采样的题源。
    # 复用 cmexam adapter,split="train" → id/source 前缀 "cmexam-train",
    # 与评测侧 "cmexam-test" 天然区分,不会在 id 空间相撞。
    # 注意:它与主力评测卷同源同分布,去污染判据另有一套(见 decontaminate.REMOVAL_POLICIES)。
    "cmexam-train": ("cmexam", "cmexam.train.jsonl", "train"),
}
# build 默认扫的训练源:W1 已公开数字对应的三个开放题源。
# cmexam-train 刻意不在其中——它必须显式 `--source cmexam-train` 另存产出,
# 否则一次默认 build 就会静默改写 train_pool.jsonl 与 reports/decontamination.md。
DEFAULT_TRAIN_SOURCES: tuple[str, ...] = ("med-o1-verifiable", "med-o1-sft-zh", "med-r1-zh")

EVAL_SOURCES: dict[str, tuple[str, str, str]] = {
    "cmexam": ("cmexam", "cmexam.test.jsonl", "test"),
    "cmb-val": ("cmb-val", "cmb.val.jsonl", ""),
    "medxpertqa": ("medxpertqa", "medxpertqa.test.jsonl", ""),
}


def load_source(name: str) -> list[Sample]:
    adapter, filename, split = (TRAIN_SOURCES | EVAL_SOURCES)[name]
    rows = [json.loads(line) for line in (RAW / filename).open(encoding="utf-8")]
    return list(ADAPTERS[adapter](rows, split))

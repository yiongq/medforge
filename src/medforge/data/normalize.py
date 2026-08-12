"""各数据源 → 统一 Sample 的 adapter 层。

每个数据源的字段名、选项编码、答案格式都不同,全部差异吸收在这里。
adapter 对字段做显式断言:上游数据集悄悄改列名时要在这里炸出来,
而不是产出一批空 question 的静默坏数据(参照 ai-edge 十条军规的教训)。
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable, Iterator
from typing import Any

from medforge.data.schema import Sample

_drop_counts: dict[str, int] = {}


def _warn_drop(source: str, idx: int, detail: str) -> None:
    """丢行必须留信号:前 3 条打样例,之后只计数——静默丢数据是本仓明令禁止的形态。"""
    _drop_counts[source] = _drop_counts.get(source, 0) + 1
    if _drop_counts[source] <= 3:
        print(f"[normalize] 丢弃 {source} 第 {idx} 行: {detail}", file=sys.stderr)


def drop_counts() -> dict[str, int]:
    return dict(_drop_counts)


def _require(row: dict[str, Any], *keys: str) -> None:
    missing = [k for k in keys if k not in row]
    if missing:
        raise KeyError(f"数据源字段缺失 {missing},实际字段: {sorted(row)[:20]}")


def norm_med_o1_verifiable(rows: Iterable[dict[str, Any]]) -> Iterator[Sample]:
    """FreedomIntelligence/medical-o1-verifiable-problem:4 万道开放式可验证医学题。"""
    for i, row in enumerate(rows):
        _require(row, "Open-ended Verifiable Question", "Ground-True Answer")
        yield Sample(
            id=f"med-o1-verifiable-{i}",
            source="med-o1-verifiable",
            question=str(row["Open-ended Verifiable Question"]).strip(),
            gold=str(row["Ground-True Answer"]).strip(),
        )


def norm_med_o1_sft_zh(rows: Iterable[dict[str, Any]]) -> Iterator[Sample]:
    """FreedomIntelligence/medical-o1-reasoning-SFT(zh):自带 CoT 的蒸馏数据。"""
    for i, row in enumerate(rows):
        _require(row, "Question", "Complex_CoT", "Response")
        yield Sample(
            id=f"med-o1-sft-zh-{i}",
            source="med-o1-sft-zh",
            question=str(row["Question"]).strip(),
            gold=str(row["Response"]).strip(),
            cot=str(row["Complex_CoT"]).strip(),
        )


_OPTION_KEYS = ("A", "B", "C", "D", "E")


def norm_cmexam(rows: Iterable[dict[str, Any]], split: str) -> Iterator[Sample]:
    """CMExam:中国执业医师资格考试选择题。

    上游选项列既出现过 "Options"(JSON 串/列表)也出现过 A-E 独立列,
    两种都接;答案列 "Answer" 可能是 "C" 或 "ACD"。
    """
    for i, row in enumerate(rows):
        if "Options" in row:
            raw = row["Options"]
            if isinstance(raw, str):
                import json

                raw = json.loads(raw)
            if isinstance(raw, list):  # 实测形态: [{"key": "A", "value": "..."}]
                options = {str(o["key"]).strip(): str(o["value"]).strip() for o in raw}
            elif isinstance(raw, dict):
                options = {str(k).strip(): str(v).strip() for k, v in raw.items()}
            else:
                raise TypeError(f"CMExam Options 列类型未知: {type(raw)}")
        elif all(k in row for k in ("A", "B")):
            options = {k: str(row[k]).strip() for k in _OPTION_KEYS if row.get(k)}
        else:
            raise KeyError(f"CMExam 行找不到选项列,实际字段: {sorted(row)[:20]}")
        _require(row, "Question", "Answer")
        # 分隔符一次性归一(实测数据里全角逗号/分号都出现过,漏一种就静默丢一批多选题),
        # 再按字符排序满足 schema 契约「多选按字典序连写」
        cleaned = re.sub(r"[\s,，、;;;/]+", "", str(row["Answer"]).upper())
        gold = "".join(sorted(set(cleaned)))
        # 合法字母集跟着选项键走(CMB/CMExam 都实测出现过 5 个以上选项的题)
        if not gold or any(c not in options for c in gold):
            _warn_drop(f"cmexam-{split}", i, repr(row["Answer"]))
            continue
        yield Sample(
            id=f"cmexam-{split}-{i}",
            source=f"cmexam-{split}",
            question=str(row["Question"]).strip(),
            gold=gold,
            options=options,
        )


def norm_cmb_val(rows: Iterable[dict[str, Any]]) -> Iterator[Sample]:
    """CMB-val(280 题,带答案)。实测字段:question/option/answer/question_type/exam_*。

    实测有 6 选项题(answer='BEF'):合法字母集必须跟着选项键走,不能硬编码 ABCDE。
    """
    for i, row in enumerate(rows):
        _require(row, "question", "option", "answer")
        options = {str(k).strip(): str(v).strip() for k, v in row["option"].items() if str(v).strip()}
        gold = "".join(sorted(set(str(row["answer"]).upper().strip())))
        if not gold or any(c not in options for c in gold):
            _warn_drop("cmb-val", i, repr(row["answer"]))
            continue
        yield Sample(
            id=f"cmb-val-{i}",
            source="cmb-val",
            question=str(row["question"]).strip(),
            gold=gold,
            options=options,
            meta={"exam_subject": row.get("exam_subject", ""), "question_type": row.get("question_type", "")},
        )


def norm_medxpertqa(rows: Iterable[dict[str, Any]]) -> Iterator[Sample]:
    """MedXpertQA-Text(2450 题,英文困难集)。实测字段:question/options/label/…。"""
    for i, row in enumerate(rows):
        _require(row, "question", "options", "label")
        options = {str(k).strip(): str(v).strip() for k, v in row["options"].items()}
        gold = "".join(sorted(set(str(row["label"]).upper().strip())))
        if not gold or any(c not in options for c in gold):
            _warn_drop("medxpertqa", i, repr(row["label"]))
            continue
        yield Sample(
            id=f"medxpertqa-{i}",
            source="medxpertqa",
            question=str(row["question"]).strip(),
            gold=gold,
            options=options,
            meta={"medical_task": row.get("medical_task", ""), "body_system": row.get("body_system", "")},
        )


# 统一签名 callable(rows, split) -> Iterator[Sample]:注册表消费方(build 管线)不关心
# 各源是否需要 split,差异全部吸收在 lambda 里
ADAPTERS = {
    "med-o1-verifiable": lambda rows, split="": norm_med_o1_verifiable(rows),
    "med-o1-sft-zh": lambda rows, split="": norm_med_o1_sft_zh(rows),
    "cmexam": norm_cmexam,
    "cmb-val": lambda rows, split="": norm_cmb_val(rows),
    "medxpertqa": lambda rows, split="": norm_medxpertqa(rows),
}

"""弃权训练数据(R-Tuning 式):蒸馏模型自采样 → 按「会 / 不会」分类 → 只在「不会」的题上教它说不确定。

为什么是自采样而不是再请老师:弃权是**模型自己的知识边界**,不是老师的。R-Tuning 的做法是先让
待训模型在训练题上作答,答对的题保留原答案、答错的题改教它拒答——教材里的「不确定」必须落在
这个模型真的不会的题上,否则教的是「随机拒答」,覆盖率掉了准确率还不涨。

输入:
  --samples  data/processed/abstain_samples.jsonl  蒸馏模型(output/sft_distill_v1/merged)在本地 vLLM 上
             对每题 K 次采样的结果,由 build_distill.sample_teacher 产出:
             {"id","k","reasoning"(vLLM 没挂 reasoning parser,恒为空),"answer","finish_reason","completion_tokens"}
             ——思考与作答都在 answer 里,靠 </think> 分段(与评测的严格口径同一把尺)。
  --pool     data/processed/abstain_pool.jsonl     题池(CMExam-train 去污染子集,与蒸馏训练题不相交)

判定(严格可用协议 v3,与 eval/usability.py 同口径,只用规则层,绝不调 LLM judge):
  收尾 finished  = split_answer 不报未收尾(finish_reason≠length 且有 </think>)
  声明 declared  = 规则层能从最后一个 </think> 之后的作答段抽出答案
  正确 correct   = 收尾 ∧ 声明 ∧ 规则层判对

分类(逐题,K 个样本):
  known    K 次全对             → 保留一条正解(中位长度,**不取最短**),当普通 SFT 样本
  unknown  0 次对,且未收尾 ≤ 1 → 把它自己的思考接上一句过渡,改成「答案:不确定」
  unstable 其余(半对半错 / 大面积截断 / 一条都没声明)→ 丢

  unstable 必须丢:半对半错的题上模型本来就有一半机会答对,教它拒答是净损失;而截断/没声明的
  样本只是「没写完」,不是「不会」——把它们标成 unknown 等于教模型「思考长了就放弃」,
  那是在训一个更早停下来的模型,不是一个更诚实的模型。

用法(纯本地,免费,可反复重跑调阈值):
    uv run python -m medforge.data.build_abstain \
      --samples data/processed/abstain_samples.jsonl \
      --pool data/processed/abstain_pool.jsonl \
      --out data/processed/sft_abstain_v1.jsonl \
      --report reports/abstain-dataset.md \
      --abstain-ratio 0.35 --general-ratio 0.15

产出:ms-swift SFT 文件 + reports/abstain-dataset.md(漏斗 / 分类直方图 / 配比 / 长度分布)。
训练:configs/sft_abstain_qwen35_4b_lora.yaml(第二阶段,基座 = 蒸馏合并权重)。
验收:python -m medforge.eval.abstain_report --run abstain-v3-sample --ref distill-v3-sample
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from rich import print as rprint

from medforge.data.build_distill import (
    CHARS_PER_TOKEN,
    CHAT_TEMPLATE_OVERHEAD,
    PROCESSED,
    SEED,
    TRAIN_MAX_LENGTH,
    _pct,
    est_tokens,
    load_samples_file,
    pick_median,
    render_prompt,
    to_message_row,
)
from medforge.data.schema import Sample
from medforge.data.sources import ROOT
from medforge.eval.run import git_describe
from medforge.verify.extract import extract
from medforge.verify.verifier import THINK_END, split_answer, verify_by_rule

DEFAULT_SAMPLES = PROCESSED / "abstain_samples.jsonl"
DEFAULT_POOL = PROCESSED / "abstain_pool.jsonl"
DEFAULT_OUT = PROCESSED / "sft_abstain_v1.jsonl"
DEFAULT_REPORT = ROOT / "reports" / "abstain-dataset.md"

# 过渡句:模块常量,因为它是这份教材里唯一「人写的」文本,调它就是调弃权的语气与触发条件。
# 要求 ① 承接模型自己的思考(前面已经推了一大段,不能显得突兀);② 说的是「证据不足」而不是
# 「我是个笨模型」——后者会把弃权训成人格特征,在会做的题上也触发;③ 短,不喧宾夺主。
BRIDGE_SENTENCE = "但以上分析不足以支撑一个有把握的结论,与其猜一个,不如如实说明。"
# 弃权声明:必须与 extract._ABSTAIN_RE 认的字面、以及 eval/run.py 弃权提示词要求的格式逐字一致
ABSTAIN_ANSWER_CHOICE = "答案:不确定"
ABSTAIN_ANSWER_OPEN = "最终答案:不确定"

_THINK_OPEN_RE = re.compile(r"^\s*<think>\s*")
# 「答案:」字面(与 build_distill 闸门 ⑤ 同一条):教材末段要与评测提示词要求的格式逐字一致
_ANSWER_LITERAL_RE = re.compile(r"答案\s*[::]")

CLASSES = ("known", "unknown", "unstable")

# 样本级漏斗:顺序 = 实际判定顺序,报告按这个顺序累减
SAMPLE_GATES: tuple[tuple[str, str], ...] = (
    ("pool_missing", "前置:id 不在题池里(样本文件比题池旧 / 换了 --source)"),
    ("unfinished", "① 收尾:finish_reason=length 或作答里没有 </think>"),
    ("undeclared", "② 声明:作答段抽不出答案(规则层弃权)"),
)
# 题目级分类里 unstable 的丢弃理由
UNSTABLE_REASONS: tuple[tuple[str, str], ...] = (
    ("u_too_few", "采样条数 < --min-samples"),
    ("u_partial", "半对半错(0 < 判对数 < 门槛):本来就有一半机会答对,教拒答是净损失"),
    ("u_unfinished", "未收尾条数 > --unknown-max-unfinished:是「没写完」不是「不会」"),
    ("u_no_declared", "没有一条既收尾又声明:全是截断 / 格式垃圾,不能当作不会"),
)
# 选样阶段的逐题剔除(known / unknown 共用)
PICK_GATES: tuple[tuple[str, str], ...] = (
    ("p_no_literal", "格式:作答段缺「答案:」字面(只对 known 行要求)"),
    ("p_too_long", "长度:提示词 + 思考 + 作答的估算总长超 --train-max-length"),
    ("p_empty", "选样:该题没有候选样本活到最后"),
)


# ---------------------------------------------------------------- 共用


def load_pool(pool_file: Path, source: str = "") -> list[Sample]:
    """题池 jsonl(每行 Sample.to_dict())→ 样本表。source 为空串表示不筛(整池同一个来源)。"""
    pool: list[Sample] = []
    with pool_file.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if not source or row.get("source") == source:
                pool.append(Sample(**row))
    return pool


def split_think(text: str) -> tuple[str, str]:
    """一条答卷 → (思考段, 作答段)。

    vLLM 不挂 reasoning parser,思考与作答都在 answer 字段里,靠最后一个 </think> 分家;
    Qwen3.5 的 chat 模板把 <think> 开标签吃进了生成提示,所以答卷里通常只有收尾标签——
    真出现开标签时一并剥掉,拼教材时统一补回,保证 <think>…</think> 恰好成对。
    """
    head, sep, tail = text.rpartition(THINK_END)
    if not sep:
        return "", text.strip()
    return _THINK_OPEN_RE.sub("", head).strip(), tail.strip()


def abstain_answer(sample: Sample) -> str:
    return ABSTAIN_ANSWER_CHOICE if sample.is_choice else ABSTAIN_ANSWER_OPEN


def to_abstain_row(sample: Sample, think: str, bridge: str = BRIDGE_SENTENCE) -> dict:
    """弃权样本一行:用模型**自己的思考**,只把结论换成不确定。

    为什么不写一段全新的「我不知道」:思考段是模型在这道题上真实的分布,替换掉它等于同时改了
    风格与结论两件事;只换结论,梯度才落在「这种推理该导向弃权」上。
    """
    body = f"{think.strip()}\n\n{bridge}" if think.strip() else bridge
    return {
        "messages": [
            {"role": "user", "content": render_prompt(sample)},
            {"role": "assistant", "content": f"<think>\n{body}\n</think>\n\n{abstain_answer(sample)}"},
        ]
    }


def _sample_tokens(row: dict, think: str, final: str, chars_per_token: float) -> tuple[int, int]:
    """(思考 token, 作答 token)。有 completion_tokens 就以端点报的真实值为准——
    英文思考按中文 1.6 字符/token 估会高估两倍多(build_distill 实测砍掉过 451 条正常样本)。"""
    ans_t = est_tokens(final, chars_per_token)
    if row.get("completion_tokens"):
        return max(0, int(row["completion_tokens"]) - ans_t), ans_t
    return est_tokens(think, chars_per_token), ans_t


def classify(
    flags: list[dict],
    *,
    known_ratio: float = 1.0,
    unknown_max_correct: int = 0,
    unknown_max_unfinished: int = 1,
    unknown_min_declared: int = 1,
    min_samples: int = 1,
) -> tuple[str, str]:
    """一道题的 K 个样本标记 → (类别, unstable 理由键)。

    flags 每项 {"finished","declared","correct"}。known_ratio=1.0 即「K 次全对」;
    unknown 的两条护栏(未收尾条数上限、至少一条既收尾又声明)是为了不把「没写完」当成「不会」。
    """
    n = len(flags)
    if n < min_samples:
        return "unstable", "u_too_few"
    n_correct = sum(f["correct"] for f in flags)
    n_finished = sum(f["finished"] for f in flags)
    n_declared = sum(f["finished"] and f["declared"] for f in flags)
    need = math.ceil(known_ratio * n - 1e-9)
    if n_correct >= max(need, 1):
        return "known", ""
    if n_correct > unknown_max_correct:
        return "unstable", "u_partial"
    if n - n_finished > unknown_max_unfinished:
        return "unstable", "u_unfinished"
    if n_declared < unknown_min_declared:
        return "unstable", "u_no_declared"
    return "unknown", ""


def downsample(
    known: list[str], unknown: list[str], ratio: float, seed: int = SEED
) -> tuple[list[str], list[str]]:
    """把弃权行占医疗行的比例压到 ratio,只砍多的一边(固定 seed,可复现)。

    砍多的一边而不是补少的一边:两类题都是采出来的,补不出来;而配比失衡的后果不对称——
    弃权行过多会把模型训成「一律说不确定」(覆盖率崩),过少则学不到弃权。0.35 是起手值,
    真实取值应由 abstain_report 的覆盖率 / 选择性准确率回头定。
    """
    if ratio <= 0:
        return known, []
    if ratio >= 1:
        return [], unknown
    n_k, n_u = len(known), len(unknown)
    if n_k == 0 or n_u == 0:
        return known, unknown
    rng = random.Random(seed)
    if n_u > ratio * (n_k + n_u):
        keep = max(1, round(ratio * n_k / (1 - ratio)))
        return known, sorted(rng.sample(unknown, min(keep, n_u)))
    keep = max(1, round(n_u * (1 - ratio) / ratio))
    return sorted(rng.sample(known, min(keep, n_k))), unknown


# ---------------------------------------------------------------- 构造


def _pick(
    sample: Sample,
    cands: list[dict],
    cls: str,
    *,
    chars_per_token: float,
    train_max_length: int,
) -> tuple[dict | None, str]:
    """从候选里挑一条(中位长度,**不取最短**)并渲染成教材行。返回 (行 | None, 剔除理由键)。

    最短的解多半是蒙对或跳步(build_dpo 的坑,见 build_distill.pick_median);对 unknown 更要紧——
    最短的错解常常是「没想就答」,拿它当弃权范例等于教模型不思考就说不确定。
    """
    prompt_t = est_tokens(render_prompt(sample), chars_per_token)
    ok: list[dict] = []
    reason = "p_empty"
    for c in cands:
        if cls == "known" and not _ANSWER_LITERAL_RE.search(c["_final"]):
            reason = "p_no_literal"
            continue
        # 弃权行的作答段被换成固定短句,长度只由提示词 + 思考决定
        ans_t = c["_ans_t"] if cls == "known" else est_tokens(abstain_answer(sample), chars_per_token)
        total = prompt_t + c["_think_t"] + ans_t + CHAT_TEMPLATE_OVERHEAD
        if total > train_max_length:
            reason = "p_too_long"
            continue
        ok.append({**c, "_total_t": total})
    if not ok:
        return None, reason
    best = pick_median(ok, 1)[0]
    msg = (
        to_message_row(sample, best["reasoning"], best["_final"])
        if cls == "known"
        else to_abstain_row(sample, best["reasoning"])
    )
    return {**best, "_msg": msg}, ""


def build_dataset(
    pool_file: Path,
    samples_file: Path,
    out_file: Path,
    report_file: Path,
    *,
    source: str = "",
    known_ratio: float = 1.0,
    unknown_max_correct: int = 0,
    unknown_max_unfinished: int = 1,
    unknown_min_declared: int = 1,
    min_samples: int = 1,
    abstain_ratio: float = 0.35,
    general_ratio: float = 0.15,
    chars_per_token: float = CHARS_PER_TOKEN,
    train_max_length: int = TRAIN_MAX_LENGTH,
    seed: int = SEED,
    general_loader=None,
) -> dict:
    """自采样 → 判定 → 分类 → 选样 → 配比 → SFT 文件 + 报告。返回统计字典(报告与测试都读它)。"""
    if not 0.0 <= abstain_ratio <= 1.0:
        raise ValueError(f"--abstain-ratio 必须在 [0,1],收到 {abstain_ratio}")
    pool = {s.id: s for s in load_pool(pool_file, source)}
    rows_by_id = load_samples_file(samples_file)
    counts: Counter[str] = Counter()
    classes: Counter[str] = Counter()
    n_rows = sum(len(v) for v in rows_by_id.values())

    # 逐题:判定 → 分类 → 挑一条候选(此时还不落盘:配比要等两类都数完)
    picked: dict[str, tuple[str, dict]] = {}   # qid → (类别, 教材行)
    lengths: dict[str, list[int]] = {"known": [], "unknown": []}
    for qid in sorted(rows_by_id):
        rows = rows_by_id[qid]
        sample = pool.get(qid)
        if sample is None:
            counts["pool_missing"] += len(rows)
            continue
        flags: list[dict] = []
        cands: list[dict] = []
        for r in rows:
            seg, unfinished = split_answer(
                r.get("answer", ""), finish_reason=r.get("finish_reason"), thinking=True
            )
            if unfinished is not None:
                counts["unfinished"] += 1
                flags.append({"finished": False, "declared": False, "correct": False})
                continue
            if extract(seg, sample.is_choice, options=sample.options) is None:
                counts["undeclared"] += 1
                flags.append({"finished": True, "declared": False, "correct": False})
                continue
            v = verify_by_rule(sample, seg)
            correct = v is not None and v.correct is True
            flags.append({"finished": True, "declared": True, "correct": correct})
            think, final = split_think(r.get("answer", ""))
            think_t, ans_t = _sample_tokens(r, think, final, chars_per_token)
            # pick_median 按 "reasoning" 长度排序,而 vLLM 的 reasoning 恒为空:把切出来的思考塞回去
            cands.append({**r, "reasoning": think, "_final": final, "_correct": correct,
                          "_think_t": think_t, "_ans_t": ans_t})

        cls, reason = classify(
            flags, known_ratio=known_ratio, unknown_max_correct=unknown_max_correct,
            unknown_max_unfinished=unknown_max_unfinished,
            unknown_min_declared=unknown_min_declared, min_samples=min_samples,
        )
        classes[cls] += 1
        if cls == "unstable":
            counts[reason] += 1
            continue
        usable = [c for c in cands if c["_correct"]] if cls == "known" else cands
        row, why = _pick(sample, usable, cls, chars_per_token=chars_per_token,
                         train_max_length=train_max_length)
        if row is None:
            counts[why] += 1
            counts["dropped_at_pick"] += 1
            continue
        picked[qid] = (cls, row["_msg"])
        lengths[cls].append(row["_total_t"])

    known_ids = sorted(q for q, (c, _) in picked.items() if c == "known")
    unknown_ids = sorted(q for q, (c, _) in picked.items() if c == "unknown")
    keep_known, keep_unknown = downsample(known_ids, unknown_ids, abstain_ratio, seed)
    counts["ratio_drop"] = (len(known_ids) - len(keep_known)) + (len(unknown_ids) - len(keep_unknown))
    med_rows = [picked[q][1] for q in sorted(keep_known + keep_unknown)]

    # 通用回放:与 build_distill 同一条口径(build_sft.mix 的公式 + 同一份加载器)
    out_rows = med_rows
    n_general = 0
    if general_ratio > 0 and med_rows:
        from medforge.data.build_sft import mix

        loader = general_loader
        if loader is None:
            from medforge.data.build_sft import general_rows as loader
        try:
            general = loader()
        except FileNotFoundError as e:
            raise SystemExit(
                f"✗ 通用回放数据缺失({e.filename});先跑 medforge.data.download 或加 --general-ratio 0"
            ) from e
        out_rows = mix(med_rows, general, general_ratio, seed)
        n_general = len(out_rows) - len(med_rows)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {
        "n_rows": n_rows,
        "n_questions": len(rows_by_id),
        "counts": dict(counts),
        "classes": {c: classes.get(c, 0) for c in CLASSES},
        "n_known_picked": len(known_ids),
        "n_unknown_picked": len(unknown_ids),
        "n_known_rows": len(keep_known),
        "n_unknown_rows": len(keep_unknown),
        "n_med": len(med_rows),
        "n_general": n_general,
        "n_total": len(out_rows),
        "abstain_share": len(keep_unknown) / len(med_rows) if med_rows else 0.0,
        "known_tokens": lengths["known"],
        "unknown_tokens": lengths["unknown"],
        "params": {
            "source": source or "(全部)", "known_ratio": known_ratio,
            "unknown_max_correct": unknown_max_correct,
            "unknown_max_unfinished": unknown_max_unfinished,
            "unknown_min_declared": unknown_min_declared, "min_samples": min_samples,
            "abstain_ratio": abstain_ratio, "general_ratio": general_ratio,
            "chars_per_token": chars_per_token, "train_max_length": train_max_length, "seed": seed,
        },
        "out_file": str(out_file),
    }
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(render_report(stats), encoding="utf-8")
    return stats


# ---------------------------------------------------------------- 报告


def render_report(stats: dict) -> str:
    p, counts, cls = stats["params"], stats["counts"], stats["classes"]
    n_q = max(stats["n_questions"], 1)
    in_pool = stats["n_rows"] - counts.get("pool_missing", 0)
    finished = in_pool - counts.get("unfinished", 0)
    declared = finished - counts.get("undeclared", 0)
    denom = max(in_pool, 1)
    lines = [
        "# 弃权数据集报告(sft_abstain_v1)",
        "",
        f"生成:{datetime.now(UTC).astimezone().isoformat(timespec='seconds')} · git {git_describe(ROOT)}",
        "",
        ("自采样:蒸馏模型(`output/sft_distill_v1/merged`)在去污染题池上每题 K 次作答;"
         "判定用严格可用协议 v3 的规则层(收尾 ∧ 声明 ∧ 判对),全程不调 LLM judge;"
         "user 提示词与评测共用 `eval.run.PROMPT_CHOICE/PROMPT_OPEN`,训练与评测同一份。"),
        "",
        "参数:" + " · ".join([
            f"source={p['source']}", f"known 判对占比≥{p['known_ratio']:.0%}",
            f"unknown 判对≤{p['unknown_max_correct']}", f"未收尾≤{p['unknown_max_unfinished']}",
            f"声明≥{p['unknown_min_declared']}", f"min-samples={p['min_samples']}",
            f"abstain-ratio={p['abstain_ratio']}", f"general-ratio={p['general_ratio']}",
            f"总长≤{p['train_max_length']}", f"seed={p['seed']}",
        ]),
        "",
        "## 1. 样本漏斗",
        "",
        f"采样条数 {stats['n_rows']} 条(覆盖 {stats['n_questions']} 道题)",
        "",
        "| 判定 | 剔除 | 剩余 |",
        "|---|---|---|",
    ]
    remain = stats["n_rows"]
    for key, label in SAMPLE_GATES:
        dropped = counts.get(key, 0)
        remain -= dropped
        lines.append(f"| {label} | {dropped} | {remain} |")
    lines += [
        "",
        (f"题池内 {in_pool} 条:收尾率 {finished / denom:.1%} · 声明率 {declared / denom:.1%}。"
         "这两条是**样本级**,分类是**题目级**,不要互相换算。"),
        "",
        "## 2. 题目分类直方图",
        "",
        "| 类别 | 题数 | 占比 | 含义 |",
        "|---|---|---|---|",
        f"| known | {cls['known']} | {cls['known'] / n_q:.1%} | K 次全对:保留正解,当普通 SFT 样本 |",
        f"| unknown | {cls['unknown']} | {cls['unknown'] / n_q:.1%} | 0 次对且基本写完了:改教「答案:不确定」 |",
        f"| unstable | {cls['unstable']} | {cls['unstable'] / n_q:.1%} | 其余,全部丢弃(理由见下) |",
        "",
        "| unstable 理由 | 题数 |",
        "|---|---|",
    ]
    for key, label in UNSTABLE_REASONS:
        lines.append(f"| {label} | {counts.get(key, 0)} |")
    lines += ["", "| 选样剔除(分完类之后) | 题数 |", "|---|---|"]
    for key, label in PICK_GATES:
        lines.append(f"| {label} | {counts.get(key, 0)} |")
    lines += [
        "",
        f"分类之后又在选样阶段掉了 {counts.get('dropped_at_pick', 0)} 道题。",
        "",
        "## 3. 配比与成品",
        "",
        (f"选样后 known {stats['n_known_picked']} / unknown {stats['n_unknown_picked']};"
         f"按 `--abstain-ratio {p['abstain_ratio']}` 下采样多的一边,丢 {counts.get('ratio_drop', 0)} 行。"),
        "",
        "| 类别 | 教材行数 | 占医疗行 |",
        "|---|---|---|",
        f"| known(正常作答) | {stats['n_known_rows']} | {1 - stats['abstain_share']:.1%} |",
        f"| unknown(答案:不确定) | {stats['n_unknown_rows']} | {stats['abstain_share']:.1%} |",
        "",
        (f"**医疗 {stats['n_med']} 行 + 通用回放 {stats['n_general']} 行 = {stats['n_total']} 行**"
         f" → `{stats['out_file']}`"),
        "",
        "## 4. 长度分布(token;有 completion_tokens 时以端点报的为准)",
        "",
        "| 类别 | p50 | p90 | p99 | max |",
        "|---|---|---|---|---|",
    ]
    for label, values in (
        ("known 单条总长", stats["known_tokens"]),
        ("unknown 单条总长", stats["unknown_tokens"]),
    ):
        mx = max(values) if values else 0
        lines.append(
            f"| {label} | {_pct(values, 0.5)} | {_pct(values, 0.9)} | {_pct(values, 0.99)} | {mx} |"
        )
    all_t = stats["known_tokens"] + stats["unknown_tokens"]
    lines += [
        "",
        (f"总长 max {max(all_t) if all_t else 0} ≤ 训练 max_length {p['train_max_length']}:"
         "超限的在数据侧已剔除,不交给框架 truncation(它截掉的正是末段那句「答案:…」)。"),
        "",
        "## 5. 待办",
        "",
        ("- 过渡句(`BRIDGE_SENTENCE`)只有一句、固定不变,模型可能把它当触发词背下来,"
         "在会做的题上一写出这句就滑向弃权。验收要看 `abstain_report` 的**选择性准确率**是否上升、"
         "覆盖率的跌幅是否小于弃权精度带来的收益;若否,先试多样化过渡句,再调 `--abstain-ratio`。"),
        ("- `--abstain-ratio 0.35` 是起手值,没有实证依据。跑完第一版后用 "
         "`python -m medforge.eval.abstain_report --run abstain-v3-sample --ref distill-v3-sample` "
         "看覆盖率 / 选择性准确率,再决定往哪边调。"),
        ("- unknown 的思考段是模型答错时的推理,里面可能含事实错误。本轮不做人工抽检——"
         "若发现模型学会「编一段错推理再说不确定」,应改为截短思考或换成模板化思考。"),
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="medforge.data.build_abstain",
        description="弃权训练数据:蒸馏模型自采样 → known/unknown 分类 → SFT 教材 + 报告",
    )
    ap.add_argument("--samples", default=str(DEFAULT_SAMPLES),
                    help="自采样落盘文件(build_distill sample 的形状)")
    ap.add_argument("--pool", default=str(DEFAULT_POOL), help="题池 jsonl(每行 Sample.to_dict())")
    ap.add_argument("--source", default="", help="只用题池里这个 source 的题;留空 = 整池")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--report", default=str(DEFAULT_REPORT))
    ap.add_argument("--known-ratio", type=float, default=1.0,
                    help="known 门槛:判对数 / 采样数 ≥ 该比例;1.0 = K 次全对")
    ap.add_argument("--unknown-max-correct", type=int, default=0, help="unknown 允许的最大判对数")
    ap.add_argument("--unknown-max-unfinished", type=int, default=1,
                    help="unknown 允许的最大未收尾条数;默认 1 即「至少 K-1 条写完了」")
    ap.add_argument("--unknown-min-declared", type=int, default=1,
                    help="unknown 至少要有几条既收尾又声明:全是格式垃圾的题不算「不会」")
    ap.add_argument("--min-samples", type=int, default=1, help="采样条数不足这个数的题直接丢")
    ap.add_argument("--abstain-ratio", type=float, default=0.35,
                    help="弃权行占医疗行的目标比例(下采样多的一边,固定 seed)")
    ap.add_argument("--general-ratio", type=float, default=0.15,
                    help="通用回放占比(build_sft.mix 同一公式);0 = 不混")
    ap.add_argument("--chars-per-token", type=float, default=CHARS_PER_TOKEN)
    ap.add_argument("--train-max-length", type=int, default=TRAIN_MAX_LENGTH,
                    help="训练配置的 max_length,数据侧硬筛")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)

    stats = build_dataset(
        Path(args.pool), Path(args.samples), Path(args.out), Path(args.report),
        source=args.source, known_ratio=args.known_ratio,
        unknown_max_correct=args.unknown_max_correct,
        unknown_max_unfinished=args.unknown_max_unfinished,
        unknown_min_declared=args.unknown_min_declared, min_samples=args.min_samples,
        abstain_ratio=args.abstain_ratio, general_ratio=args.general_ratio,
        chars_per_token=args.chars_per_token, train_max_length=args.train_max_length, seed=args.seed,
    )
    if not stats["n_unknown_rows"]:
        rprint("[red]✗ 0 条弃权样本——这份教材教不出弃权;对着报告的分类直方图调阈值后重跑[/]")
        raise SystemExit(1)
    rprint(
        f"[green]✓[/] 弃权教材 {stats['n_total']} 条(known {stats['n_known_rows']} + unknown "
        f"{stats['n_unknown_rows']}(占医疗 {stats['abstain_share']:.1%}) + 通用 {stats['n_general']})"
        f" → {args.out};报告 → {args.report}"
    )


if __name__ == "__main__":
    main()

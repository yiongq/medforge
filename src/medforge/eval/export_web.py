"""评测产物 → 前台回放数据(web/public/replay.json)。

用法:
    uv run python -m medforge.eval.export_web [--per-bucket 8] [--max-chars 7000]

设计:
- 回放模式的全部数据都来自已跑完的评测,不需要 GPU 也不需要模型在线;
  live 模式另走 vLLM 接口(W3b),两者共用同一份题面渲染逻辑
- 选题不是随机:按「故事桶」抽——降智案例(base 对、SFT 错)、DPO 修复案例、
  全员失守、全员正确各取若干。访客点开就能看到现象本身,而不是靠运气翻到
- 思考与结论在导出侧就切开:结论必须完整保留(它才是「答了什么」),
  只有超长的思考段才截断——从头部一刀切会把结尾的结论砍掉(实测基座 17.7k 字答卷踩过)
- 「思考长度的差异」本身是负结果的可视化证据,所以原始字数一律如实上报
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any

from rich import print as rprint

from medforge.data.sources import EVAL_SOURCES, ROOT, load_source
from medforge.eval.report import load_run

THINK_END = "</think>"


def split_thinking(text: str) -> tuple[str, str]:
    """(思考, 结论)。Qwen3.5 模板吃掉了开标签,输出里通常只剩收尾的 </think>。"""
    idx = text.rfind(THINK_END)
    if idx == -1:
        return "", text.strip()
    return text[:idx].removeprefix("<think>").strip(), text[idx + len(THINK_END):].strip()

RUNS_DIR = ROOT / "reports" / "runs"
WEB_PUBLIC = ROOT / "web" / "public"

# 展示顺序即训练故事的时间线;key = reports/runs/ 下的目录名
RUN_LABELS: list[tuple[str, str, str]] = [
    ("base-v2", "原装基座", "Qwen3.5-4B,未经任何训练"),
    ("sft-v2", "抄旧教材", "SFT · 2024 年 GPT-4o 蒸馏的医学 CoT"),
    ("sft-r1-v2", "抄新教材", "SFT · 2025 年 R1 蒸馏的医学 CoT"),
    ("dpo-v2", "自己刷题", "DPO · 基座自采样,验证器判分配对"),
]


def load_answers(run: str, eval_set: str) -> dict[str, dict[str, Any]]:
    """{sample_id: {"text":…, "correct":…, "method":…}};缺失的 run 返回空。"""
    out_f = RUNS_DIR / run / f"{eval_set}.outputs.jsonl"
    scored_f = RUNS_DIR / run / f"{eval_set}.scored.jsonl"
    if not out_f.exists():
        return {}
    verdicts = {}
    if scored_f.exists():
        for line in scored_f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                verdicts[r["id"]] = r
    answers = {}
    for line in out_f.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        v = verdicts.get(r["id"], {})
        answers[r["id"]] = {
            "text": r["output"],
            "correct": v.get("correct"),
            "method": v.get("method"),
        }
    return answers


def bucket_of(verdicts: dict[str, bool | None]) -> str:
    """按各 run 的对错组合给题目归类(故事桶)。"""
    base = verdicts.get("base-v2")
    sfts = [verdicts.get("sft-v2"), verdicts.get("sft-r1-v2")]
    dpo = verdicts.get("dpo-v2")
    if base is True and all(v is False for v in sfts if v is not None) and sfts.count(None) < 2:
        return "regression"          # 降智:原装会做,抄完不会了
    if dpo is True and base is not True:
        return "dpo_fix"             # DPO 修复:刷题后学会了
    if all(v is True for v in verdicts.values() if v is not None):
        return "all_correct"
    if all(v is not True for v in verdicts.values()):
        return "all_wrong"
    return "mixed"


BUCKET_LABELS = {
    "regression": "抄笔记后退步",
    "dpo_fix": "刷题后学会",
    "all_wrong": "全员失守",
    "all_correct": "全员正确",
    "mixed": "结果分化",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-bucket", type=int, default=8, help="每个故事桶每套卷最多取几题")
    ap.add_argument("--max-chars", type=int, default=7000, help="单份答卷保留的字符上限")
    args = ap.parse_args()

    runs = [(k, zh, desc) for k, zh, desc in RUN_LABELS if (RUNS_DIR / k).exists()]
    rprint(f"可用 run: {[r[0] for r in runs]}")

    summary: list[dict[str, Any]] = []
    questions: list[dict[str, Any]] = []

    for eval_set in EVAL_SOURCES:
        samples = {s.id: s for s in load_source(eval_set)}
        per_run_answers = {run: load_answers(run, eval_set) for run, _, _ in runs}

        for run, zh, _ in runs:
            scored = RUNS_DIR / run / f"{eval_set}.scored.jsonl"
            if not scored.exists():
                continue
            r = load_run(scored, run)
            lo, hi = r.wilson_ci()
            summary.append({
                "run": run, "label": zh, "set": eval_set, "n": r.n,
                "acc": round(r.acc * 100, 1),
                "ci": [round(lo * 100, 1), round(hi * 100, 1)],
                "abstain": round(r.abstain_rate * 100, 1),
            })

        # 只保留所有 run 都作答过的题目,保证并排比较公平
        common = set.intersection(*(set(a) for a in per_run_answers.values() if a)) if per_run_answers else set()
        buckets: dict[str, list[str]] = defaultdict(list)
        for sid in sorted(common):
            verdicts = {run: per_run_answers[run][sid]["correct"] for run, _, _ in runs if sid in per_run_answers[run]}
            buckets[bucket_of(verdicts)].append(sid)

        for bucket, ids in buckets.items():
            for sid in ids[: args.per_bucket]:
                s = samples.get(sid)
                if s is None:
                    continue
                answers = {}
                for run, _, _ in runs:
                    a = per_run_answers[run].get(sid)
                    if not a:
                        continue
                    thinking, conclusion = split_thinking(a["text"])
                    # 结论若本身超长(模型没写 </think> 时整段都算结论),保留头尾各一半,
                    # 中段省略——宁可断在中间,也不能丢掉结尾的答案声明
                    if len(conclusion) > args.max_chars:
                        half = args.max_chars // 2
                        conclusion = f"{conclusion[:half]}\n\n……（中段省略）……\n\n{conclusion[-half:]}"
                    answers[run] = {
                        "thinking": thinking[: args.max_chars],
                        "thinkingChars": len(thinking),
                        "thinkingTruncated": len(thinking) > args.max_chars,
                        "conclusion": conclusion,
                        "chars": len(a["text"]),
                        "correct": a["correct"],
                        "method": a["method"],
                    }
                questions.append({
                    "id": sid,
                    "set": eval_set,
                    "bucket": bucket,
                    "bucketLabel": BUCKET_LABELS[bucket],
                    "question": s.question,
                    "options": s.options,
                    "gold": s.gold,
                    "meta": s.meta,
                    "answers": answers,
                })

    payload = {
        "meta": {
            "protocol": "v2(temperature 0,max_tokens 8192,固定种子抽样卷)",
            "runs": [{"key": k, "label": zh, "desc": d} for k, zh, d in runs],
            "sets": {
                "cmexam": "CMExam · 中国执业医师考试真题(中文)",
                "cmb-val": "CMB-val · 中文医学综合(中文)",
                "medxpertqa": "MedXpertQA · 专科困难卷(英文,10 选项)",
            },
        },
        "summary": summary,
        "questions": questions,
    }
    WEB_PUBLIC.mkdir(parents=True, exist_ok=True)
    dst = WEB_PUBLIC / "replay.json"
    dst.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    size_mb = dst.stat().st_size / 1024 / 1024
    by_bucket = defaultdict(int)
    for q in questions:
        by_bucket[q["bucketLabel"]] += 1
    rprint(f"[green]✓[/] {len(questions)} 题 / {len(summary)} 条成绩 → {dst}({size_mb:.1f} MB)")
    rprint(f"  故事桶分布: {dict(by_bucket)}")


if __name__ == "__main__":
    main()

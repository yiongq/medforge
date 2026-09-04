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
- 成绩每行同时给严格口径(strict = 写完 ∧ 有结论 ∧ 答对,来自 <set>.usability.jsonl 的逐题标签)
  与宽口径(acc = scored.jsonl 的原判分,含从复读段刮出的分)。前台主数字用严格口径:
  W2 的降分大半是「没交卷」被当成「答错」,只报宽口径会把解码工件说成模型差异
- MedXpertQA 的题面、选项、标准答案与作答原文一律不导出(REDACTED_SETS):其论文附录 A 的
  Leakage Prevention Statement 要求不要以任何形式在线分享样例;模型思考流常整段复述题目,所以连作答文本
  也只留字数与判分。前台对这些题只展示「谁答对了」的格局。
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from typing import Any

from rich import print as rprint

from medforge.data.sources import EVAL_SOURCES, ROOT, load_source
from medforge.eval.report import load_run

THINK_END = "</think>"
# 许可不允许公开样例的评测集:只导出 id / 判分 / 字数,不导出任何文本
REDACTED_SETS = {"medxpertqa"}
REDACTED_NOTE = "按 MedXpertQA 许可不公开题面与作答原文(论文附录 A:请勿以任何形式在线分享样例)"


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
    ("base-v3-sample", "基座 · 正确解码", "同一份基座权重,不训练;官方采样参数 + 32k 预算(协议 v3)"),
]
# 协议按 run 名判定:v3 = 官方采样参数 + 32768 预算 + 截断守卫;其余为 v2(贪心 + 8192)
V2_RUNS = [k for k, _, _ in RUN_LABELS if "-v3" not in k]


def protocol_of(run: str) -> str:
    return "v3" if "-v3" in run else "v2"


def wilson_ci(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """与 report.RunResult.wilson_ci 同式,这里对严格口径的计数再算一次。"""
    if n == 0:
        return (0.0, 0.0)
    p = correct / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


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


def load_usability(run: str, eval_set: str) -> list[dict[str, Any]]:
    """逐题的严格可用标签(usability.jsonl);没跑过 usability 的 run 返回空表。"""
    f = RUNS_DIR / run / f"{eval_set}.usability.jsonl"
    if not f.exists():
        return []
    return [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]


def bucket_of(verdicts: dict[str, bool | None]) -> str:
    """按各 run 的对错组合给题目归类(故事桶)。

    五个 run 下三个「主角桶」互斥,所以判定顺序不影响结果:
    regression 要求 base-v2 对、decoding_fix 要求 v2 臂全不对、dpo_fix 要求 dpo 对而 base-v2 不对。
    all_correct / all_wrong 覆盖全部 run(含 v3),因此正确解码后新做对的题会从 all_wrong 迁到 decoding_fix,
    这正是今天要展示的现象。"""
    base = verdicts.get("base-v2")
    sfts = [verdicts.get("sft-v2"), verdicts.get("sft-r1-v2")]
    dpo = verdicts.get("dpo-v2")
    v3 = verdicts.get("base-v3-sample")
    v2_vals = [verdicts[k] for k in V2_RUNS if k in verdicts]
    if base is True and all(v is False for v in sfts if v is not None) and sfts.count(None) < 2:
        return "regression"          # 降智:原装会做,抄完不会了
    if v3 is True and v2_vals and all(v is not True for v in v2_vals):
        return "decoding_fix"        # 换解码就会了:同一权重,v2 协议下四臂全错,v3 采样答对
    if dpo is True and base is not True:
        return "dpo_fix"             # DPO 修复:刷题后学会了
    if all(v is True for v in verdicts.values() if v is not None):
        return "all_correct"
    if all(v is not True for v in verdicts.values()):
        return "all_wrong"
    return "mixed"


BUCKET_LABELS = {
    "regression": "抄笔记后退步",
    "decoding_fix": "换解码就会了",
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
            row: dict[str, Any] = {
                "run": run, "label": zh, "set": eval_set, "n": r.n,
                "protocol": protocol_of(run),
                # acc / ci / abstain 是宽口径(as-scored):含 LLM 兜底与从复读段刮出的分
                "acc": round(r.acc * 100, 1),
                "ci": [round(lo * 100, 1), round(hi * 100, 1)],
                "abstain": round(r.abstain_rate * 100, 1),
            }
            # 严格口径逐题重算:strict = 写完 ∧ 有结论 ∧ 答对;finished = 写出 </think> 的比例
            tags = load_usability(run, eval_set)
            if tags:
                nt = len(tags)
                n_strict = sum(1 for t in tags if t.get("strict"))
                n_fin = sum(1 for t in tags if t.get("finished"))
                s_lo, s_hi = wilson_ci(n_strict, nt)
                row["strict"] = round(n_strict / nt * 100, 1)
                row["strictCi"] = [round(s_lo * 100, 1), round(s_hi * 100, 1)]
                row["finished"] = round(n_fin / nt * 100, 1)
            summary.append(row)

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
                    redacted = eval_set in REDACTED_SETS
                    # 结论若本身超长(模型没写 </think> 时整段都算结论),保留头尾各一半,
                    # 中段省略——宁可断在中间,也不能丢掉结尾的答案声明
                    if len(conclusion) > args.max_chars:
                        half = args.max_chars // 2
                        conclusion = f"{conclusion[:half]}\n\n……（中段省略）……\n\n{conclusion[-half:]}"
                    answers[run] = {
                        "thinking": "" if redacted else thinking[: args.max_chars],
                        "thinkingChars": len(thinking),
                        "thinkingTruncated": (not redacted) and len(thinking) > args.max_chars,
                        "conclusion": "" if redacted else conclusion,
                        "chars": len(a["text"]),
                        "correct": a["correct"],
                        "method": a["method"],
                        "redacted": redacted,
                    }
                redacted = eval_set in REDACTED_SETS
                questions.append({
                    "id": sid,
                    "set": eval_set,
                    "bucket": bucket,
                    "bucketLabel": BUCKET_LABELS[bucket],
                    "question": REDACTED_NOTE if redacted else s.question,
                    "options": None if redacted else s.options,
                    "gold": "" if redacted else s.gold,
                    "meta": {} if redacted else s.meta,
                    "redacted": redacted,
                    "answers": answers,
                })

    payload = {
        "meta": {
            "protocol": "v2(temperature 0,max_tokens 8192)与 v3(官方采样 1.0/0.95/top_k 20/presence 1.5,max_tokens 32768),同一批固定种子抽样卷",
            "protocols": {
                "v2": "贪心 · temperature 0 · max_tokens 8192",
                "v3": "官方采样 · temperature 1.0 · top_p 0.95 · top_k 20 · presence_penalty 1.5 · max_tokens 32768",
            },
            "runs": [
                {"key": k, "label": zh, "desc": d, "protocol": protocol_of(k)} for k, zh, d in runs
            ],
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

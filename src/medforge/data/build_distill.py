"""蒸馏 2.0:强老师(DeepSeek V4)在去污染题池上出题解 → 本项目验证器筛 → ms-swift SFT 教材。

教师只能是 DeepSeek 这类允许蒸馏的 provider(其条款 §4.2 明确许可),**不得**换成
MEDFORGE_JUDGE_PROVIDER=claude-code:Anthropic 消费者条款禁止用 Claude 输出训练其他模型。
判卷/代理标注/参考臂可以用 Claude(那是测量,不是训练),见 docs/claude-code-provider.md。

两步 CLI(采样与构造分离:采样按次付费且要断点续采,构造要能反复重跑调闸门):

    # 1. 采样(唯一花钱的一步)
    uv run python -m medforge.data.build_distill sample \
      --endpoint https://api.deepseek.com/v1 --model deepseek-chat \
      --api-key-env MEDFORGE_TEACHER_API_KEY --n-questions 3000 --k-samples 2

    # 2. 构造(纯本地,免费,可反复重跑)
    uv run python -m medforge.data.build_distill build --accept majority --general-ratio 0.15

为什么重做一版蒸馏数据(8 月 SFT 教材失败的五个坑,逐条对应本模块的设计):
  ① 老师不在题型内 —— 旧教材是别人合成的开放题 CoT,评测考的是 CMExam 选择题。
     这里直接在 CMExam 训练集的去污染子集上出题解,题型与考卷同分布。
  ② 教材末段没有评测要求的「答案:X」格式(实测仅 4%)—— 闸门 ⑤ 硬性要求 answer 段含「答案:」字面,
     且闸门 ② 要求这一段能被 extract 抽出与 gold 相同的字母,格式对齐是收样本的前提而不是希望。
  ③ 训练 user 是裸题干、评测 user 是带格式指令的提示词 —— 本模块的 user 一律用
     `from medforge.eval.run import PROMPT_CHOICE, PROMPT_OPEN` 同一份常量渲染,
     采样与构造共用 render_prompt(),训练/评测提示词不可能再漂。
  ④ 20% 样本超过 max_length 被框架截断 —— 闸门 ④ 在数据侧硬筛长度(思考 / 答案 / 总长三条线),
     不依赖框架 truncation:被截断的教材末尾恰好是「答案:X」那一行,截掉就等于教「别写答案」。
  ⑤ 教材把英文思考换成中文 —— 基座原生英文思考(存档 CJK 占比中位 5%),DeepSeek 老师同样英文思考,
     语言分布一致;闸门 ⑤ 默认关(--zh-ratio-min 0),只在换了中文思考的老师时才该打开。

老师侧要点(与 eval/run.py 的 v3 协议参照一致,这样「老师在考卷上的成绩」与蒸馏出的样本同源):
  temperature 1.0 / top_p 0.95 / presence_penalty 1.5,extra_body {"thinking": {"type": "enabled"}},
  seed = 基础 seed + 第 k 次(k 之间必须真的不同,否则 K 采样退化成一条)。
  思考在 message.reasoning_content、答案在 message.content,两段分开落盘——
  Qwen3.5 的 chat 模板吃掉 <think> 开标签(见 docs/gpu-租卡实操笔记.md 与 eval/run.py 的 FORCE_PREFIX),
  所以存档答卷里只有 </think>;训练数据必须是完整的 <think>…</think> 对,开标签在 build 阶段补回。

成本估算(DeepSeek V4-flash off-peak 输出 $0.66/M,平均每样本约 400 输出 token):
  成本 ≈ 题数 × k × 400 / 1e6 × 0.66 美元;3000 题 × 2 样本 ≈ 2.4M token ≈ $1.58(输入侧题干短,可忽略)。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from rich import print as rprint

from medforge.data.schema import Sample
from medforge.data.sources import ROOT
from medforge.eval.run import PROMPT_CHOICE, PROMPT_OPEN, git_describe
from medforge.verify.verifier import verify_by_rule

PROCESSED = ROOT / "data" / "processed"
SEED = 42
DEFAULT_POOL = PROCESSED / "train_pool-cmexam-train.jsonl"
DEFAULT_SAMPLES = PROCESSED / "distill_samples.jsonl"
DEFAULT_OUT = PROCESSED / "sft_distill_v1.jsonl"
DEFAULT_REPORT = ROOT / "reports" / "distill-dataset.md"

MAX_FAIL_RATE = 0.02  # 与 eval/run.py 同口径:采样失败超过这个比例就退出,不让半张教材看起来正常
# 老师默认开思考;厂商对不认识的键静默忽略,所以这份 extra_body 对非 DeepSeek 端点也无害
DEFAULT_EXTRA_BODY = '{"thinking": {"type": "enabled"}}'
# 中文实测换算:Qwen3 系 BPE 上 1 token ≈ 1.6 个汉字(英文/数字/标点密的段落 token 更省,
# 所以这个估算对中文卷偏保守)。本机 venv 刻意不装 transformers(见 pyproject 注释),不引 tokenizer。
CHARS_PER_TOKEN = 1.6
# 训练侧长度上限:configs/*.yaml 的 max_length。数据侧硬筛的依据——
# 单条样本(提示词 + 思考 + 答案 + 模板开销)估算超过它就不该存在于教材里。
TRAIN_MAX_LENGTH = 8192
CHAT_TEMPLATE_OVERHEAD = 32  # <|im_start|>user … <|im_end|><|im_start|>assistant<think>\n 等标记的粗估

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
# 「答案:」字面(评测提示词要求全角冒号;半角一并放行,extract 两种都认)
_ANSWER_LITERAL_RE = re.compile(r"答案\s*[::]")

ACCEPT_MODES = ("majority", "any")

# 闸门计数键 → 报告里的中文标签。顺序 = 实际执行顺序,报告的漏斗按这个顺序做累减。
GATE_LABELS: tuple[tuple[str, str], ...] = (
    ("pool_missing", "前置:id 不在题池里(样本文件比题池旧/换了 --source)"),
    ("g1_finish", "① 结构:finish_reason ≠ stop(撞上限/被中断)"),
    ("g1_empty_reasoning", "① 结构:reasoning 为空(老师没开思考)"),
    ("g1_empty_answer", "① 结构:answer 为空"),
    ("g2_wrong", "② 答案:规则层判错"),
    ("g2_abstain", "② 答案:规则层弃权(抽不出/声明弃权),按设计不走 LLM"),
    ("g3_reject", "③ 接受条件:该题未达门槛,已判对的样本一并丢弃"),
    ("g4_think_short", "④ 长度:思考过短(< --min-think-tokens)"),
    ("g4_think_long", "④ 长度:思考过长(> --max-think-tokens)"),
    ("g4_answer_long", "④ 长度:答案段过长(> --max-answer-tokens)"),
    ("g4_total_long", "④ 长度:单条总长超训练 max_length"),
    ("g5_no_literal", "⑤ 格式:answer 段缺「答案:」字面"),
    ("g5_zh_ratio", "⑤ 语言:reasoning 的 CJK 占比 < --zh-ratio-min"),
    ("cap", "每题上限:--max-per-question 截掉(按长度取中位保留)"),
)


# ---------------------------------------------------------------- 共用


def render_prompt(s: Sample) -> str:
    """user 提示词:与评测完全同一份常量。采样与构造都走这里,训练/评测不可能再漂(坑 ③)。"""
    return (PROMPT_CHOICE if s.is_choice else PROMPT_OPEN).format(question=s.render_question())


def load_pool(pool_file: Path, source: str) -> list[Sample]:
    """题池 jsonl(每行 Sample.to_dict())→ 指定 source 的样本,保持文件序。"""
    pool: list[Sample] = []
    with pool_file.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("source") == source:
                pool.append(Sample(**row))
    return pool


def pick_questions(pool: list[Sample], n: int, seed: int = SEED) -> list[Sample]:
    """固定 seed 洗牌后取前缀(build_dpo.load_questions 同款):小规模试跑与后续扩量抽到的是
    同一批题的前缀,断点缓存全程有效——先跑 300 题看闸门通过率,再扩到 3000 题不浪费已采的。"""
    rng = random.Random(seed)
    full = rng.sample(pool, len(pool))
    return full[:n] if n > 0 else full


def est_tokens(text: str, chars_per_token: float = CHARS_PER_TOKEN) -> int:
    return math.ceil(len(text) / chars_per_token) if text else 0


def cjk_ratio(text: str) -> float:
    """CJK 汉字 / 非空白字符。纯英文思考约 0,中文卷正常在 0.6~0.9。"""
    dense = "".join(text.split())
    return len(_CJK_RE.findall(dense)) / len(dense) if dense else 0.0


def to_message_row(sample: Sample, reasoning: str, answer: str) -> dict:
    """ms-swift SFT 一行。assistant 必须是完整的 <think>…</think> 对:
    老师的 reasoning_content 与 Qwen3.5 存档答卷一样只有收尾的 </think>,开标签在这里补回。"""
    return {
        "messages": [
            {"role": "user", "content": render_prompt(sample)},
            {"role": "assistant", "content": f"<think>\n{reasoning.strip()}\n</think>\n\n{answer.strip()}"},
        ]
    }


def pick_median(rows: list[dict], k: int) -> list[dict]:
    """按 reasoning 长度排序,取中位附近的 k 条——**不取最短**。

    build_dpo.py:92-93 的「对解取最短」是 W2 的坑:短而对的解多半是蒙对或跳步,
    P2 报告 §3.2 也实测「模型在转圈之前其实已经想到了答案」——最短样本的教学信号最弱,
    却因为「短 = 干净」的直觉被系统性选中。这里改成中位窗口,只有 k ≥ 候选数时才会带上最短那条。
    """
    if len(rows) <= k:
        return rows
    ordered = sorted(rows, key=lambda r: len(r["reasoning"]))
    start = (len(ordered) - k + 1) // 2  # 向长的一侧偏半步:窗口覆盖中位数且不含最短
    return ordered[start : start + k]


def _pct(values: list[int], q: float) -> int:
    """最近秩分位数;空集返回 0。"""
    if not values:
        return 0
    xs = sorted(values)
    i = max(0, min(len(xs) - 1, math.ceil(q * len(xs)) - 1))
    return xs[i]


# ---------------------------------------------------------------- 步骤 1:采样


def sample_teacher(
    samples: list[Sample],
    out_file: Path,
    *,
    base_url: str,
    api_key: str,
    model: str,
    k_samples: int = 2,
    concurrency: int = 8,
    max_tokens: int = 8192,
    temperature: float = 1.0,
    top_p: float = 0.95,
    presence_penalty: float = 1.5,
    seed: int = SEED,
    timeout: float = 600.0,
    extra_body: dict | None = None,
) -> dict[str, int]:
    """向老师请求题解,逐条落盘。断点续采:已有的 (id, k) 跳过(采样按次付费,缓存就是钱)。

    落盘字段 {"id", "k", "reasoning", "answer", "finish_reason", "completion_tokens"}——
    思考与答案分开存,判分只看 answer 段(闸门 ②),训练时才拼成 <think> 对。
    答案为空的不落盘:多半是端点异常,落了盘断点续采就永远是空的(eval/run.py 同款处理)。
    """
    from openai import OpenAI

    out_file.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple[str, int]] = set()
    if out_file.exists():
        for line in out_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done.add((row["id"], row["k"]))
    todo = [(s, k) for s in samples for k in range(k_samples) if (s.id, k) not in done]
    rprint(f"  待采 {len(todo)} / {len(samples) * k_samples}(已有 {len(done)} 条复用)")
    if not todo:
        return {"todo": 0, "written": 0, "failed": 0, "reused": len(done)}

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=2)
    lock = threading.Lock()
    f = out_file.open("a", encoding="utf-8")

    def gen_one(s: Sample, k: int) -> dict:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": render_prompt(s)}],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            seed=seed + k,  # 每次采样换种子:同 seed 的 K 次请求会退化成同一条
            extra_body=extra_body or {},
        )
        ch = resp.choices[0]
        usage = getattr(resp, "usage", None)
        return {
            "id": s.id,
            "k": k,
            "reasoning": getattr(ch.message, "reasoning_content", None) or "",
            "answer": ch.message.content or "",
            "finish_reason": ch.finish_reason,
            "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
        }

    written = failed = n_done = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(gen_one, s, k) for s, k in todo]
        for fut in as_completed(futures):
            with lock:
                n_done += 1
                try:
                    row = fut.result()
                    if not row["answer"].strip():
                        failed += 1
                        rprint(f"  ✗ {row['id']}#{row['k']} 空答案(finish_reason={row['finish_reason']}),留待重采")
                    else:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        f.flush()
                        written += 1
                except Exception as e:  # noqa: BLE001  单条失败不废整批,重跑补缺
                    failed += 1
                    rprint(f"  ✗ {type(e).__name__}: {e}")
                if n_done % 100 == 0:
                    rprint(f"  [{n_done}/{len(todo)}]")
    f.close()
    if failed and failed / len(todo) > MAX_FAIL_RATE:
        # 已采的都落盘了,重跑只补缺;但不能带着一堆窟窿继续往下走
        raise SystemExit(f"✗ 采样失败 {failed}/{len(todo)} 超过 {MAX_FAIL_RATE:.0%}:检查端点/参数后重跑补缺")
    if failed:
        rprint(f"  [yellow]! {failed} 条采样失败,重跑可补缺[/]")
    return {"todo": len(todo), "written": written, "failed": failed, "reused": len(done)}


# ---------------------------------------------------------------- 步骤 2:五道闸门


def load_samples_file(samples_file: Path) -> dict[str, list[dict]]:
    """采样文件 → {id: [row 按 k 升序]}。同 (id,k) 重复行取最后一条(重采覆盖)。"""
    by_id: dict[str, dict[int, dict]] = defaultdict(dict)
    with samples_file.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                by_id[row["id"]][row["k"]] = row
    return {sid: [ks[k] for k in sorted(ks)] for sid, ks in by_id.items()}


def build_dataset(
    pool_file: Path,
    samples_file: Path,
    out_file: Path,
    report_file: Path,
    *,
    source: str = "cmexam-train",
    accept: str = "majority",
    max_per_question: int = 2,
    chars_per_token: float = CHARS_PER_TOKEN,
    min_think_tokens: int = 100,
    max_think_tokens: int = 4096,
    max_answer_tokens: int = 512,
    train_max_length: int = TRAIN_MAX_LENGTH,
    zh_ratio_min: float = 0.0,
    general_ratio: float = 0.15,
    general_loader=None,
) -> dict:
    """采样 → 五道闸门 → ms-swift SFT 文件 + 报告。返回统计字典(报告与测试都读它)。

    闸门顺序即 GATE_LABELS 的顺序;③ 是整题级判定,所以 ①② 先逐条判、③ 再回看整题、④⑤ 再逐条判。
    """
    if accept not in ACCEPT_MODES:
        raise ValueError(f"--accept 只能是 {ACCEPT_MODES},收到 {accept!r}")
    pool = {s.id: s for s in load_pool(pool_file, source)}
    rows_by_id = load_samples_file(samples_file)
    n_rows = sum(len(v) for v in rows_by_id.values())
    counts: Counter[str] = Counter()
    by_k: dict[int, Counter[str]] = defaultdict(Counter)
    med_rows: list[dict] = []
    think_tokens: list[int] = []
    answer_tokens: list[int] = []
    total_tokens: list[int] = []
    n_questions_kept = 0

    for qid in sorted(rows_by_id):
        rows = rows_by_id[qid]
        for r in rows:
            by_k[r["k"]]["n"] += 1
        sample = pool.get(qid)
        if sample is None:
            counts["pool_missing"] += len(rows)
            continue
        n_seen = len(rows)  # ③ 的分母 = 这道题实际采到的样本数

        # ① 结构 + ② 答案(逐条)
        correct: list[dict] = []
        for r in rows:
            if r.get("finish_reason") != "stop":
                counts["g1_finish"] += 1
                continue
            if not r.get("reasoning", "").strip():
                counts["g1_empty_reasoning"] += 1
                continue
            if not r.get("answer", "").strip():
                counts["g1_empty_answer"] += 1
                continue
            # 只看 answer 段,不看 reasoning:思考里写满了「可能是 A」不算作答(与评测口径一致)
            v = verify_by_rule(sample, r["answer"])
            if v is None or v.correct is None:
                counts["g2_abstain"] += 1  # 规则层抽不出或声明弃权:一律丢,不花钱走 LLM
                continue
            if v.correct is not True:
                counts["g2_wrong"] += 1
                continue
            by_k[r["k"]]["correct"] += 1
            correct.append(r)

        # ③ 接受条件(整题级)
        if accept == "majority" and len(correct) * 2 <= n_seen:
            counts["g3_reject"] += len(correct)  # 严格多数:k=2 时必须两条都对
            continue
        if not correct:
            continue

        # ④ 长度 + ⑤ 格式与语言(逐条)
        survivors: list[dict] = []
        for r in correct:
            ans_t = est_tokens(r["answer"], chars_per_token)
            # 思考段优先用 API 返回的真实 token 数(completion_tokens = 思考 + 答案):老师用英文思考时
            # 按中文 1.6 字符/token 估算会高估两倍多,把正常样本当「过长」砍掉(实测 451 条)
            if r.get("completion_tokens"):
                think_t = max(0, int(r["completion_tokens"]) - ans_t)
            else:
                think_t = est_tokens(r["reasoning"], chars_per_token)
            if think_t < min_think_tokens:
                counts["g4_think_short"] += 1
                continue
            if think_t > max_think_tokens:
                counts["g4_think_long"] += 1
                continue
            if ans_t > max_answer_tokens:
                counts["g4_answer_long"] += 1
                continue
            total = est_tokens(render_prompt(sample), chars_per_token) + think_t + ans_t + CHAT_TEMPLATE_OVERHEAD
            if total > train_max_length:
                # 数据侧硬筛:交给框架 truncation 就是把末尾的「答案:X」截掉,等于教模型别写答案(坑 ④)
                counts["g4_total_long"] += 1
                continue
            if not _ANSWER_LITERAL_RE.search(r["answer"]):
                # ② 已保证 answer 段能被 extract 抽出与 gold 相同的字母;这里再要求「答案:」字面,
                # 是为了让教材末段与评测提示词要求的格式逐字一致
                counts["g5_no_literal"] += 1
                continue
            if cjk_ratio(r["reasoning"]) < zh_ratio_min:
                counts["g5_zh_ratio"] += 1
                continue
            r = {**r, "_think_t": think_t, "_ans_t": ans_t, "_total_t": total}
            survivors.append(r)
        if not survivors:
            continue

        picked = pick_median(survivors, max_per_question)
        counts["cap"] += len(survivors) - len(picked)
        n_questions_kept += 1
        for r in sorted(picked, key=lambda x: x["k"]):
            by_k[r["k"]]["kept"] += 1
            think_tokens.append(r["_think_t"])
            answer_tokens.append(r["_ans_t"])
            total_tokens.append(r["_total_t"])
            med_rows.append(to_message_row(sample, r["reasoning"], r["answer"]))

    # 通用回放混料:复用 build_sft 的 mix 公式与 alpaca 加载(SFT 教材只此一份口径)
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
            raise SystemExit(f"✗ 通用回放数据缺失({e.filename});先跑 medforge.data.download 或加 --general-ratio 0") from e
        out_rows = mix(med_rows, general, general_ratio)
        n_general = len(out_rows) - len(med_rows)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {
        "n_rows": n_rows,
        "n_questions_sampled": len(rows_by_id),
        "n_questions_kept": n_questions_kept,
        "n_med": len(med_rows),
        "n_general": n_general,
        "n_total": len(out_rows),
        "counts": dict(counts),
        "by_k": {k: dict(v) for k, v in sorted(by_k.items())},
        "think_tokens": think_tokens,
        "answer_tokens": answer_tokens,
        "total_tokens": total_tokens,
        "params": {
            "source": source, "accept": accept, "max_per_question": max_per_question,
            "chars_per_token": chars_per_token, "min_think_tokens": min_think_tokens,
            "max_think_tokens": max_think_tokens, "max_answer_tokens": max_answer_tokens,
            "train_max_length": train_max_length, "zh_ratio_min": zh_ratio_min,
            "general_ratio": general_ratio,
        },
        "out_file": str(out_file),
    }
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(render_report(stats), encoding="utf-8")
    return stats


def render_report(stats: dict) -> str:
    p = stats["params"]
    counts = stats["counts"]
    lines = [
        "# 蒸馏数据集报告(sft_distill_v1)",
        "",
        f"生成:{datetime.now(UTC).astimezone().isoformat(timespec='seconds')} · git {git_describe(ROOT)}",
        "",
        ("老师:强模型带思考出题解(思考在 reasoning_content、答案在 content,两段分开落盘);"
         "user 提示词与评测共用 `eval.run.PROMPT_CHOICE/PROMPT_OPEN`,训练与评测同一份。"),
        "",
        "参数:" + " · ".join([
            f"source={p['source']}", f"accept={p['accept']}", f"max-per-question={p['max_per_question']}",
            f"think∈[{p['min_think_tokens']},{p['max_think_tokens']}]token",
            f"answer≤{p['max_answer_tokens']}token", f"总长≤{p['train_max_length']}",
            f"chars-per-token={p['chars_per_token']}", f"zh-ratio≥{p['zh_ratio_min']}",
            f"general-ratio={p['general_ratio']}",
        ]),
        "",
        "## 1. 闸门漏斗",
        "",
        f"采样条数 {stats['n_rows']} 条(覆盖 {stats['n_questions_sampled']} 道题)",
        "",
        "| 闸门 | 剔除 | 剩余 |",
        "|---|---|---|",
    ]
    remain = stats["n_rows"]
    for key, label in GATE_LABELS:
        dropped = counts.get(key, 0)
        remain -= dropped
        lines.append(f"| {label} | {dropped} | {remain} |")
    lines += [
        "",
        (f"**最终 {stats['n_med']} 条医疗样本,覆盖 {stats['n_questions_kept']} 道题"
         f"(采样题目的 {stats['n_questions_kept'] / max(stats['n_questions_sampled'], 1):.1%})**;"
         f"混入通用回放 {stats['n_general']} 条,合计 {stats['n_total']} 条 → `{stats['out_file']}`"),
        "",
        "## 2. 长度分布(估算 token,换算见模块 CHARS_PER_TOKEN)",
        "",
        "| 指标 | p50 | p90 | p99 | max |",
        "|---|---|---|---|---|",
    ]
    for label, values in (
        ("思考(reasoning)", stats["think_tokens"]),
        ("答案(answer)", stats["answer_tokens"]),
        ("单条总长(提示词+思考+答案)", stats["total_tokens"]),
    ):
        mx = max(values) if values else 0
        lines.append(f"| {label} | {_pct(values, 0.5)} | {_pct(values, 0.9)} | {_pct(values, 0.99)} | {mx} |")
    lines += [
        "",
        (f"总长 max {max(stats['total_tokens']) if stats['total_tokens'] else 0}"
         f" ≤ 训练 max_length {p['train_max_length']}:训练配置的 `max_length` 必须 ≥ 这个数,"
         "否则末段的「答案:X」会被框架截掉(8 月 SFT 的坑之一)。"),
        "",
        "## 3. 按第 k 次采样的通过率",
        "",
        "| k | 采样条数 | ② 判对 | 判对率 | 最终入选 |",
        "|---|---|---|---|---|",
    ]
    for k, c in stats["by_k"].items():
        n, ok = c.get("n", 0), c.get("correct", 0)
        lines.append(f"| {k} | {n} | {ok} | {ok / n:.1%} | {c.get('kept', 0)} |" if n else f"| {k} | 0 | 0 | — | 0 |")
    lines += [
        "",
        "## 4. 待办",
        "",
        ("- TODO(人工抽检,本轮未做):**「答案对但推理不成立」规则层查不出**——闸门 ② 只判 answer 段,"
         "选择题四选一蒙对也算对。需人工抽 50 条读 reasoning,统计跳步 / 编造文献 / 推理与答案矛盾的比例;"
         "若超过 10%,考虑加一道 LLM 过程审查闸门或提高 --accept 门槛。"),
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- CLI


def _add_common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--pool", default=str(DEFAULT_POOL), help="题池 jsonl(每行 Sample.to_dict())")
    ap.add_argument("--source", default="cmexam-train", help="只用题池里这个 source 的题")
    ap.add_argument("--samples-file", default=str(DEFAULT_SAMPLES), help="老师题解的落盘文件")


def main(argv: list[str] | None = None) -> None:
    from medforge.env import load_env

    load_env()
    ap = argparse.ArgumentParser(prog="medforge.data.build_distill", description="蒸馏 2.0:老师出题解 → 验证器筛 → SFT 教材")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample", help="向老师采样题解(唯一花钱的一步,可断点续采)")
    _add_common(s)
    s.add_argument("--endpoint", required=True, help="老师的 OpenAI 兼容 base_url")
    s.add_argument("--model", required=True)
    s.add_argument("--api-key-env", default="MEDFORGE_TEACHER_API_KEY", help="从哪个环境变量读老师的 API key")
    s.add_argument("--n-questions", type=int, default=3000, help="固定 seed 洗牌后取前缀 N 题")
    s.add_argument("--k-samples", type=int, default=2, help="每题采几次(seed = --seed + k)")
    s.add_argument("--concurrency", type=int, default=8)
    s.add_argument("--max-tokens", type=int, default=8192)
    s.add_argument("--timeout", type=float, default=600.0)
    s.add_argument("--seed", type=int, default=SEED)
    s.add_argument("--temperature", type=float, default=1.0)   # 以下三项与 eval/run.py 的 v3 协议一致
    s.add_argument("--top-p", type=float, default=0.95)
    s.add_argument("--presence-penalty", type=float, default=1.5)
    s.add_argument("--extra-body", default=DEFAULT_EXTRA_BODY, help="附加到每个请求的 JSON,默认开厂商思考开关")

    b = sub.add_parser("build", help="五道闸门 → ms-swift SFT 文件 + 报告(纯本地,可反复重跑)")
    _add_common(b)
    b.add_argument("--out", default=str(DEFAULT_OUT))
    b.add_argument("--report", default=str(DEFAULT_REPORT))
    b.add_argument("--accept", choices=ACCEPT_MODES, default="majority",
                   help="majority = 该题判对的占严格多数才收(k=2 即两条全对);any = 任一判对即收")
    b.add_argument("--max-per-question", type=int, default=2)
    b.add_argument("--chars-per-token", type=float, default=CHARS_PER_TOKEN)
    b.add_argument("--min-think-tokens", type=int, default=100)
    b.add_argument("--max-think-tokens", type=int, default=4096)
    b.add_argument("--max-answer-tokens", type=int, default=512)
    b.add_argument("--train-max-length", type=int, default=TRAIN_MAX_LENGTH, help="训练配置的 max_length,数据侧硬筛")
    b.add_argument(
        "--zh-ratio-min", type=float, default=0.0,
        help="reasoning 的 CJK 占比下限;默认 0 = 不筛。基座原生用英文思考(存档 CJK 占比中位 5%),DeepSeek 老师也是英文思考,"
             "两者一致就不存在「语言被换掉」的问题;实测设 0.5 会砍掉 99% 的样本",
    )
    b.add_argument("--general-ratio", type=float, default=0.15, help="通用回放占比(build_sft.mix 同一公式);0 = 不混")
    args = ap.parse_args(argv)

    if args.cmd == "sample":
        api_key = os.environ.get(args.api_key_env, "")
        if not api_key:
            rprint(f"[red]✗ 环境变量 {args.api_key_env} 为空(.env 已加载)[/]")
            sys.exit(2)
        pool = load_pool(Path(args.pool), args.source)
        if not pool:
            rprint(f"[red]✗ 题池 {args.pool} 里没有 source={args.source} 的题[/]")
            sys.exit(2)
        questions = pick_questions(pool, args.n_questions)
        rprint(f"[bold]▶ 采样[/] {len(questions)} 题 × {args.k_samples} 次(题池 {len(pool)} 题)")
        r = sample_teacher(
            questions, Path(args.samples_file),
            base_url=args.endpoint, api_key=api_key, model=args.model,
            k_samples=args.k_samples, concurrency=args.concurrency, max_tokens=args.max_tokens,
            temperature=args.temperature, top_p=args.top_p, presence_penalty=args.presence_penalty,
            seed=args.seed, timeout=args.timeout,
            extra_body=json.loads(args.extra_body) if args.extra_body else None,
        )
        rprint(f"[green]✓[/] 新采 {r['written']} 条(复用 {r['reused']},失败 {r['failed']})→ {args.samples_file}")
        return

    stats = build_dataset(
        Path(args.pool), Path(args.samples_file), Path(args.out), Path(args.report),
        source=args.source, accept=args.accept, max_per_question=args.max_per_question,
        chars_per_token=args.chars_per_token, min_think_tokens=args.min_think_tokens,
        max_think_tokens=args.max_think_tokens, max_answer_tokens=args.max_answer_tokens,
        train_max_length=args.train_max_length, zh_ratio_min=args.zh_ratio_min,
        general_ratio=args.general_ratio,
    )
    if not stats["n_med"]:
        rprint("[red]✗ 五道闸门后 0 条医疗样本——采样已落盘可复用,检查闸门参数与报告漏斗后重跑[/]")
        sys.exit(1)
    rprint(
        f"[green]✓[/] SFT 教材 {stats['n_total']} 条(医疗 {stats['n_med']} 覆盖 {stats['n_questions_kept']} 题"
        f" + 通用 {stats['n_general']})→ {args.out};报告 → {args.report}"
    )


if __name__ == "__main__":
    main()

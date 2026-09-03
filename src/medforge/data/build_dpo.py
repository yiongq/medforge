"""DPO 偏好对构造:可验证题自采样 → 验证器判分 → 对/错配对。

在 GPU 机上运行(需要 vLLM 起着 SFT 后的模型):
    uv run python -m medforge.data.build_dpo \
      --endpoint http://127.0.0.1:8000/v1 --model target \
      [--n-questions 8000] [--k-samples 6]

方法即 HuatuoGPT-o1 的 RL 数据构造思路的离线版(拒绝采样 + 配对):
同一道题采 K 个解,验证器判对错,对解做 chosen、错解做 rejected。
判分分层(见 classify_solution):规则层能定的直接用;声明了答案但与 gold
不符的默认交 LLM 仲裁(约 2-3 解/题 × 8K 题 ≈ 2 万次调用 ≈ 20-40 元,防同义答案
被误杀成负例;--no-llm-arbitrate 可关,转为严格判错);无声明的解丢弃。

输出 ms-swift DPO 格式:{"messages": [user, assistant(chosen)], "rejected_response": "..."}
(字段名已核实=官方 Custom-dataset 文档 DPO 示例原文,swift.readthedocs.io/en/latest/Customization/Custom-dataset.html)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich import print as rprint

from medforge.data.schema import Sample
from medforge.data.sources import ROOT
from medforge.verify.extract import extract
from medforge.verify.verifier import split_answer, verify_by_llm, verify_by_rule

PROCESSED = ROOT / "data" / "processed"
SEED = 42
PROMPT = "你是医学助手,回答下面的问题。先给出推理过程,最后一行以「最终答案:」开头给出结论。\n\n{question}"


def load_questions(n: int, offset: int = 0) -> list[Sample]:
    """从去污染题池抽可验证开放题。固定 seed:重跑抽到同一批,断点续采才成立。"""
    pool = []
    with (PROCESSED / "train_pool.jsonl").open(encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            if s["source"] == "med-o1-verifiable":
                pool.append(Sample(**s))
    # 前缀稳定抽样:先按固定 seed 抽满上限再取前 n——小规模试跑与后续扩量
    # 抽到的是同一批题的前缀,断点缓存全程有效
    rng = random.Random(SEED)
    full = rng.sample(pool, min(8000, len(pool)))
    # offset 支持多机分段采样:同一 seed 的固定洗牌序列上切片,各机不重不漏
    return full[offset : offset + n]


def classify_solution(sample: Sample, sol: str, llm_arbitrate: bool) -> str:
    """DPO 场景的分层判定 → "correct" | "wrong" | "drop"。

    评测用的规则层对开放题「宁弃权不判错」,永远产不出负例(测试抓出的真缺陷)。
    配对场景分三层:规则层能定的直接用;明确声明了答案但与 gold 不符的,
    交 LLM 仲裁(防「同义答案被误杀成负例」);连答案声明都没有的丢弃——
    没结论的解进 rejected 教的是「别写结论」,是毒信号。
    无仲裁模式下声明不符直接判错:可验证题答案短而规范,误杀率低【暂定,抽检验证】。
    截断守卫与评测共用(W2 审查后):采样解撞上 max_tokens 没写出 </think> 的一律 drop——
    规则层会从复读段刮出「最终答案:X」,与 gold 相符就进 chosen,等于拿半截思考流当教学信号。
    采样落盘时没有记 finish_reason(P7 待补),这里只能靠思考型口径的 </think> 判据。
    """
    answer, unfinished = split_answer(sol, thinking=True)
    if unfinished is not None:
        return "drop"
    v = verify_by_rule(sample, answer)
    if v is not None:
        return "correct" if v.correct else "wrong"
    ext = extract(answer, sample.is_choice, options=sample.options)
    if ext is None:
        return "drop"
    if llm_arbitrate:
        v2 = verify_by_llm(sample, answer)
        if v2.correct is True:
            return "correct"
        if v2.correct is False:
            return "wrong"
        return "drop"
    return "wrong"


def pair_from_classified(sample: Sample, solutions: list[str], classes: list[str]) -> list[dict]:
    """已分类的一题多解 → 偏好对。规则:对解×错解笛卡尔积太冗余,每题最多出 2 对
    (对解取最短的两个——短而对的推理是更好的教学信号【暂定】,错解无放回抽)。"""
    correct = [sol for sol, c in zip(solutions, classes) if c == "correct"]
    wrong = [sol for sol, c in zip(solutions, classes) if c == "wrong"]
    if not correct or not wrong:
        return []
    # crc32 而非 hash():str 的 hash 被 PYTHONHASHSEED 加盐,跨进程不稳定,
    # 「重跑产出相同偏好对」的承诺会静默失效(审查实测三种 HASHSEED 选出不同 rejected)
    rng = random.Random(SEED + zlib.crc32(sample.id.encode()) % 10000)
    correct.sort(key=len)
    chosen_list = correct[:2]
    # 无放回:错解够用时两对不共享同一条 rejected,保住负例多样性
    if len(wrong) >= len(chosen_list):
        rejected_list = rng.sample(wrong, len(chosen_list))
    else:
        rejected_list = [rng.choice(wrong) for _ in chosen_list]
    pairs = []
    for chosen, rejected in zip(chosen_list, rejected_list):
        pairs.append({
            "messages": [
                {"role": "user", "content": PROMPT.format(question=sample.render_question())},
                {"role": "assistant", "content": chosen},
            ],
            "rejected_response": rejected,
        })
    return pairs


def make_pairs(sample: Sample, solutions: list[str], llm_arbitrate: bool = False) -> list[dict]:
    """串行便捷入口(测试与小规模用);大规模并行路径见 main() 的 classify 阶段。"""
    classes = [classify_solution(sample, sol, llm_arbitrate) for sol in solutions]
    return pair_from_classified(sample, solutions, classes)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-questions", type=int, default=8000)  # 【暂定】8K 题 × K 解,GPU ~2-3 小时
    ap.add_argument("--offset", type=int, default=0, help="分段起点(多机拆采样用,如机2从1500起)")
    ap.add_argument("--k-samples", type=int, default=6)       # 【暂定】需要对错并存,6 解命中率与成本的折中
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--no-llm-arbitrate", action="store_true", help="声明不符不走 LLM 仲裁,直接判错")
    args = ap.parse_args()

    # fail-fast:开放题的负例只能来自 LLM 仲裁——judge 没配就跑采样,
    # 结局是烧完 2-3 小时 GPU 后静默产出 0 条偏好对(审查实测复现)
    if not args.no_llm_arbitrate:
        from medforge.env import load_env

        load_env()
        missing = [k for k in ("MEDFORGE_JUDGE_BASE_URL", "MEDFORGE_JUDGE_API_KEY", "MEDFORGE_JUDGE_MODEL")
                   if not os.environ.get(k)]
        if missing:
            rprint(f"[red]✗ LLM 仲裁已启用但 judge 未配置: {missing};配好 .env 或显式 --no-llm-arbitrate[/]")
            sys.exit(2)

    from openai import OpenAI

    questions = load_questions(args.n_questions, args.offset)
    raw_file = PROCESSED / "dpo_samples.jsonl"   # 原始采样落盘:断点续采 + 可审计
    done: dict[str, list[str]] = {}
    if raw_file.exists():
        for line in raw_file.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[r["id"]] = r["solutions"]
    todo = [s for s in questions if s.id not in done]
    rprint(f"采样 {len(todo)} / {len(questions)} 题(已有 {len(done)} 复用)× {args.k_samples} 解")

    client = OpenAI(base_url=args.endpoint, api_key="EMPTY", timeout=300, max_retries=2)
    lock = threading.Lock()
    f = raw_file.open("a", encoding="utf-8")

    def sample_one(s: Sample) -> tuple[str, list[str]]:
        # n=K 一次请求出 K 解:vLLM 对同 prompt 多样本有前缀缓存优势
        resp = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": PROMPT.format(question=s.render_question())}],
            temperature=1.0,   # 采样要多样性:太低全对/全错都配不成对
            max_tokens=8192,   # 6144 实测截断率 51%(半数解法没写到答案就被掐),与评测协议 v2 对齐
            n=args.k_samples,
        )
        return s.id, [c.message.content or "" for c in resp.choices]

    n_done = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(sample_one, s) for s in todo]
        for fut in as_completed(futures):
            with lock:
                n_done += 1
                try:
                    sid, sols = fut.result()
                    done[sid] = sols
                    f.write(json.dumps({"id": sid, "solutions": sols}, ensure_ascii=False) + "\n")
                    f.flush()
                except Exception as e:  # noqa: BLE001  单题失败不废整批
                    rprint(f"  ✗ {type(e).__name__}: {e}")
                if n_done % 200 == 0:
                    rprint(f"  [{n_done}/{len(todo)}]")
    f.close()

    by_id = {s.id: s for s in questions}
    # 分类阶段并行 + 标签落盘缓存:2 万次 LLM 仲裁串行要 20+ 小时,16 并发 ≈ 1-2 小时;
    # 中断重跑只补缺(仲裁按次计费,缓存就是钱)
    label_file = PROCESSED / "dpo_labels.jsonl"
    labels: dict[str, str] = {}
    if label_file.exists():
        for line in label_file.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            labels[r["key"]] = r["cls"]
    tasks = [(sid, i, sol) for sid, sols in done.items() if sid in by_id
             for i, sol in enumerate(sols) if f"{sid}#{i}" not in labels]
    rprint(f"分类 {len(tasks)} 个解(缓存命中 {len(labels)})")
    lf = label_file.open("a", encoding="utf-8")
    llock = threading.Lock()
    n_cls = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = {pool.submit(classify_solution, by_id[sid], sol, not args.no_llm_arbitrate): (sid, i)
                for sid, i, sol in tasks}
        for fut in as_completed(futs):
            sid, i = futs[fut]
            with llock:
                n_cls += 1
                try:
                    cls = fut.result()
                except Exception as e:  # noqa: BLE001  单解仲裁失败按 drop 处理,不废整批
                    rprint(f"  ✗ 仲裁失败 {sid}#{i}: {type(e).__name__}")
                    cls = "drop"
                labels[f"{sid}#{i}"] = cls
                lf.write(json.dumps({"key": f"{sid}#{i}", "cls": cls}, ensure_ascii=False) + "\n")
                lf.flush()
                if n_cls % 500 == 0:
                    rprint(f"  [{n_cls}/{len(tasks)}]")
    lf.close()

    pairs = []
    for sid, sols in done.items():
        if sid in by_id:
            classes = [labels.get(f"{sid}#{i}", "drop") for i in range(len(sols))]
            pairs.extend(pair_from_classified(by_id[sid], sols, classes))
    dst = PROCESSED / "dpo_pairs.jsonl"
    with dst.open("w", encoding="utf-8") as f2:
        for p in pairs:
            f2.write(json.dumps(p, ensure_ascii=False) + "\n")
    if not pairs:
        rprint("[red]✗ 偏好对为 0 条——采样已落盘可复用,检查判定链路后重跑配对[/]")
        sys.exit(1)
    rprint(f"[green]✓[/] 偏好对 {len(pairs)} 条(源自 {len(done)} 题)→ {dst}")


if __name__ == "__main__":
    main()

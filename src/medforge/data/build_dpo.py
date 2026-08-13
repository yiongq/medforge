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
【暂定】字段名以 GPU 日 ms-swift 文档实测为准。
"""

from __future__ import annotations

import argparse
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich import print as rprint

from medforge.data.schema import Sample
from medforge.data.sources import ROOT
from medforge.verify.extract import extract
from medforge.verify.verifier import verify_by_llm, verify_by_rule

PROCESSED = ROOT / "data" / "processed"
SEED = 42
PROMPT = "你是医学助手,回答下面的问题。先给出推理过程,最后一行以「最终答案:」开头给出结论。\n\n{question}"


def load_questions(n: int) -> list[Sample]:
    """从去污染题池抽可验证开放题。固定 seed:重跑抽到同一批,断点续采才成立。"""
    pool = []
    with (PROCESSED / "train_pool.jsonl").open(encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            if s["source"] == "med-o1-verifiable":
                pool.append(Sample(**s))
    rng = random.Random(SEED)
    return rng.sample(pool, min(n, len(pool)))


def classify_solution(sample: Sample, sol: str, llm_arbitrate: bool) -> str:
    """DPO 场景的分层判定 → "correct" | "wrong" | "drop"。

    评测用的规则层对开放题「宁弃权不判错」,永远产不出负例(测试抓出的真缺陷)。
    配对场景分三层:规则层能定的直接用;明确声明了答案但与 gold 不符的,
    交 LLM 仲裁(防「同义答案被误杀成负例」);连答案声明都没有的丢弃——
    没结论的解进 rejected 教的是「别写结论」,是毒信号。
    无仲裁模式下声明不符直接判错:可验证题答案短而规范,误杀率低【暂定,抽检验证】。
    """
    v = verify_by_rule(sample, sol)
    if v is not None:
        return "correct" if v.correct else "wrong"
    ext = extract(sol, sample.is_choice)
    if ext is None:
        return "drop"
    if llm_arbitrate:
        v2 = verify_by_llm(sample, sol)
        if v2.correct is True:
            return "correct"
        if v2.correct is False:
            return "wrong"
        return "drop"
    return "wrong"


def make_pairs(sample: Sample, solutions: list[str], llm_arbitrate: bool = False) -> list[dict]:
    """一题多解 → 偏好对。规则:对解×错解笛卡尔积太冗余,每题最多出 2 对
    (对解取最短的两个——短而对的推理是更好的教学信号【暂定】,错解随机)。"""
    correct, wrong = [], []
    for sol in solutions:
        cls = classify_solution(sample, sol, llm_arbitrate)
        if cls == "correct":
            correct.append(sol)
        elif cls == "wrong":
            wrong.append(sol)
    if not correct or not wrong:
        return []
    rng = random.Random(SEED + hash(sample.id) % 10000)
    correct.sort(key=len)
    pairs = []
    for chosen in correct[:2]:
        rejected = rng.choice(wrong)
        pairs.append({
            "messages": [
                {"role": "user", "content": PROMPT.format(question=sample.render_question())},
                {"role": "assistant", "content": chosen},
            ],
            "rejected_response": rejected,
        })
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-questions", type=int, default=8000)  # 【暂定】8K 题 × K 解,GPU ~2-3 小时
    ap.add_argument("--k-samples", type=int, default=6)       # 【暂定】需要对错并存,6 解命中率与成本的折中
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--no-llm-arbitrate", action="store_true", help="声明不符不走 LLM 仲裁,直接判错")
    args = ap.parse_args()

    from openai import OpenAI

    questions = load_questions(args.n_questions)
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
            max_tokens=2048,
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
    pairs = []
    for sid, sols in done.items():
        if sid in by_id:
            pairs.extend(make_pairs(by_id[sid], sols, llm_arbitrate=not args.no_llm_arbitrate))
    dst = PROCESSED / "dpo_pairs.jsonl"
    with dst.open("w", encoding="utf-8") as f2:
        for p in pairs:
            f2.write(json.dumps(p, ensure_ascii=False) + "\n")
    rprint(f"[green]✓[/] 偏好对 {len(pairs)} 条(源自 {len(done)} 题)→ {dst}")


if __name__ == "__main__":
    main()

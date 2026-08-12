"""校准集构造 CLI(ADR 口径 3 的前置工序)。

两步用法:
    # ① 分层抽题 → data/calibration/pending.jsonl(output 为空待生成)
    uv run python -m medforge.verify.sample_calibration

    # ② 用 OpenAI 兼容 API 生成模型作答(复用 MEDFORGE_JUDGE_* 环境变量指向的服务)
    uv run python -m medforge.verify.sample_calibration --generate

之后人工逐条填 human_correct(true/false),再跑 calibrate.py 出一致率。

抽样设计:开放题 120(验证器最难判的形态,DPO 数据构造的主战场)+
选择题 80(评测判分的主形态,从 cmexam-validation 抽——刻意避开 test,
不给「对着考卷调判卷程序」留任何口实)。seed 固定,抽样可复现。
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

from rich import print as rprint

from medforge.data.normalize import ADAPTERS
from medforge.data.schema import Sample

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "calibration" / "pending.jsonl"
SEED = 42
N_OPEN, N_CHOICE = 120, 80

_GEN_PROMPT = "你是医学助手,回答下面的问题。先给出推理过程,最后一行以「最终答案:」开头给出结论。\n\n{question}"


def sample() -> None:
    rng = random.Random(SEED)
    rows = [json.loads(line) for line in (RAW / "med-o1-verifiable.train.jsonl").open(encoding="utf-8")]
    open_qs = list(ADAPTERS["med-o1-verifiable"](rows, ""))
    rows = [json.loads(line) for line in (RAW / "cmexam.validation.jsonl").open(encoding="utf-8")]
    choice_qs = list(ADAPTERS["cmexam"](rows, "validation"))

    picked = rng.sample(open_qs, N_OPEN) + rng.sample(choice_qs, N_CHOICE)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for s in picked:
            f.write(json.dumps(
                {"sample": s.to_dict(), "output": None, "human_correct": None},
                ensure_ascii=False) + "\n")
    rprint(f"[green]✓[/] 抽样 {len(picked)} 题(开放 {N_OPEN} + 选择 {N_CHOICE})→ {OUT}")


def generate() -> None:
    from medforge.env import load_env

    load_env()
    base_url = os.environ.get("MEDFORGE_JUDGE_BASE_URL")
    api_key = os.environ.get("MEDFORGE_JUDGE_API_KEY")
    model = os.environ.get("MEDFORGE_JUDGE_MODEL")
    if not (base_url and api_key and model):
        rprint("[red]需配置 MEDFORGE_JUDGE_BASE_URL / _API_KEY / _MODEL[/]")
        sys.exit(2)
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=120, max_retries=2)
    rows = [json.loads(line) for line in OUT.read_text(encoding="utf-8").splitlines() if line.strip()]
    todo = [i for i, row in enumerate(rows) if row["output"] is None]
    rprint(f"待生成 {len(todo)} / {len(rows)} 条(已生成的跳过,中断重跑只补空缺)")
    lock = threading.Lock()
    finished = 0

    def flush() -> None:
        OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

    def gen_one(i: int) -> None:
        s = Sample(**rows[i]["sample"])
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": _GEN_PROMPT.format(question=s.render_question())}],
            temperature=0.7,  # 刻意要多样性:校准集需要覆盖对/错/格式混乱三种作答形态
            max_tokens=1024,
        )
        rows[i]["output"] = resp.choices[0].message.content or ""

    # 8 并发:串行 200 条要近 1 小时;并发上限保持克制,避免触发 API 限流
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(gen_one, i): i for i in todo}
        for fut in as_completed(futures):
            i = futures[fut]
            with lock:
                finished += 1
                try:
                    fut.result()
                    rprint(f"  [{finished}/{len(todo)}] {rows[i]['sample']['id']} ✓")
                except Exception as e:  # noqa: BLE001  单条失败不废整批,落盘后重跑补缺
                    rprint(f"  [{finished}/{len(todo)}] {rows[i]['sample']['id']} ✗ {type(e).__name__}: {e}")
                if finished % 10 == 0:
                    flush()
    flush()
    remaining = sum(1 for r in rows if r["output"] is None)
    rprint(f"[green]✓[/] 生成完毕(剩余空缺 {remaining})→ {OUT},下一步填 human_correct 后跑 calibrate")


def main() -> None:
    if "--generate" in sys.argv:
        generate()
    else:
        sample()


if __name__ == "__main__":
    main()

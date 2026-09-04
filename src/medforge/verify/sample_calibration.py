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

第三种用法(W2 审查后补):LLM 兜底层在评测里实际判的是「写完了、但规则层从作答段抽不出字母」的
**选择题**,而上面那 200 题里选择题全被规则层接住、LLM 层只判过开放题——校准分布与工作分布不重叠。
    # ③ 从存档答卷里按卷分层抽 LLM 层真正判过的题 → data/calibration/pending-mcq.jsonl
    uv run python -m medforge.verify.sample_calibration --from-runs base-v2,sft-v2,dpo-v2,base-v3-sample --n 150
    # ④ 用另一个更强的模型做代理标注(proxy_correct),人工再抽检;calibrate 用 --label-field 选标签列
    uv run python -m medforge.verify.sample_calibration --label-proxy --file data/calibration/pending-mcq.jsonl
    # ④' 同上,但用本机已登录的 Claude 订阅(不花 API 费,额度按 Max 的 5 小时窗口算)
    uv run python -m medforge.verify.sample_calibration --label-proxy --provider claude-code \
      --model claude-opus-5 --concurrency 4

后端由 --provider 选(默认取 MEDFORGE_JUDGE_PROVIDER,再默认 openai):
  openai      需要 MEDFORGE_JUDGE_BASE_URL / _API_KEY,模型默认 deepseek-v4-pro,思考模式走 extra_body;
  claude-code 只需要本机 `claude` 已登录,模型写全名(claude-opus-5),扩展思考用 --effort(默认 high)。
标注结果的字段不随后端变(proxy_correct / proxy_chosen / proxy_reason),calibrate.py 那侧无感。
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
MCQ_OUT = ROOT / "data" / "calibration" / "pending-mcq.jsonl"

# 代理标注用的提示词刻意与 verifier._JUDGE_PROMPT 不同(否则量的是判卷员和它自己的一致率):
# 给全部选项、要求先抽出考生最终选的字母再比对,并允许 null
_PROXY_PROMPT = """你是严格的医学考试判卷员。下面是一道选择题、标准答案,以及考生作答的结论段(思考过程已去掉)。
请先找出考生**最终**选定的选项字母(多选题写全部字母;若考生改口,以最后一次明确表态为准;
若只给了解释没有明确选项、或同时给出多个互相矛盾的答案,视为无法确定),再判断它与标准答案是否**完全相同**
(多选题多选或少选都算错)。

题目与选项:
{question}

标准答案:{gold}

考生作答结论段:
{answer}

只输出 JSON:{{"chosen": "字母或 null", "correct": true/false/null, "reason": "一句话"}}"""
_PROXY_SYSTEM = "你是严格的医学考试判卷员,只输出 JSON,不要输出任何其他文字。"
# claude-code 后端用结构化输出;字段与提示词逐字对齐,openai 后端仍靠正则刮 JSON
PROXY_SCHEMA = {
    "type": "object",
    "properties": {
        "chosen": {"type": ["string", "null"]},
        "correct": {"type": ["boolean", "null"]},
        "reason": {"type": "string"},
    },
    "required": ["chosen", "correct", "reason"],
}


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


def sample_from_runs(runs: list[str], n: int, out: Path = MCQ_OUT, seed: int = SEED) -> int:
    """从存档答卷里抽 LLM 层真正判过的题:method == "llm" 且作答已收尾(未收尾的现在被守卫挡在 LLM 层之前)。
    按卷分层等额抽,seed 固定。落盘字段:sample / output / run / set / machine_correct(当时的 LLM 判定)
    / human_correct / proxy_correct(均待填)。返回落盘条数。"""
    from medforge.data.sources import EVAL_SOURCES, load_source
    from medforge.verify.verifier import split_answer

    runs_dir = ROOT / "reports" / "runs"
    by_set: dict[str, list[dict]] = {}
    cache: dict[str, dict[str, Sample]] = {}
    for run in runs:
        for eval_set in EVAL_SOURCES:
            scored, outs = runs_dir / run / f"{eval_set}.scored.jsonl", runs_dir / run / f"{eval_set}.outputs.jsonl"
            if not (scored.exists() and outs.exists()):
                continue
            samples = cache.setdefault(eval_set, {x.id: x for x in load_source(eval_set)})
            outputs = {r["id"]: r["output"] for r in map(json.loads, outs.read_text(encoding="utf-8").splitlines()) if r}
            for row in map(json.loads, scored.read_text(encoding="utf-8").splitlines()):
                if row.get("method") != "llm" or row["id"] not in outputs or row["id"] not in samples:
                    continue
                _, unfinished = split_answer(outputs[row["id"]], thinking=True)
                if unfinished:
                    continue
                by_set.setdefault(eval_set, []).append({
                    "sample": samples[row["id"]].to_dict(), "output": outputs[row["id"]],
                    "run": run, "set": eval_set, "machine_correct": row.get("correct"),
                    "human_correct": None, "proxy_correct": None,
                })
    rng = random.Random(seed)
    quota = {k: n // len(by_set) for k in by_set} if by_set else {}
    picked: list[dict] = []
    for k, pool in by_set.items():
        rng.shuffle(pool)
        picked += pool[: quota[k]]
    leftover = [r for k, pool in by_set.items() for r in pool[quota[k] :]]
    rng.shuffle(leftover)
    picked += leftover[: max(0, n - len(picked))]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in picked:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    pool_sizes = {k: len(v) for k, v in by_set.items()}
    rprint(f"[green]✓[/] 从 {runs} 抽 {len(picked)} 题(候选池 {pool_sizes})→ {out}")
    return len(picked)


def label_proxy(
    path: Path, model: str, concurrency: int = 8, provider: str | None = None, effort: str | None = None,
) -> None:
    """用另一个更强的模型代理标注 proxy_correct(仍需人工抽检);中断重跑只补空缺。

    provider=None 时取 MEDFORGE_JUDGE_PROVIDER(再默认 openai)。落盘字段与后端无关。
    """
    from medforge.env import load_env
    from medforge.verify.claude_code import EFFORTS, claude_code_query, parse_json_object
    from medforge.verify.verifier import judge_provider, split_answer

    load_env()
    provider = provider or judge_provider()
    if model == os.environ.get("MEDFORGE_JUDGE_MODEL"):
        rprint(f"[red]代理标注模型不能与判卷模型相同({model}):那量的是它和自己的一致率[/]")
        sys.exit(2)
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if provider == "claude-code":
        # 扩展思考默认拉满档:代理标注是「更强的第二意见」,省这点思考预算没意义
        effort = effort or "high"
        if effort not in EFFORTS:
            rprint(f"[red]--effort 只能是 {'|'.join(EFFORTS)},收到 {effort!r}[/]")
            sys.exit(2)

        def ask(prompt: str) -> dict:
            r = claude_code_query(
                prompt, model=model, system_prompt=_PROXY_SYSTEM, json_schema=PROXY_SCHEMA,
                effort=effort, timeout=600,
            )
            return r.structured or parse_json_object(r.text) or {"reason": f"[未解析] {r.text[:200]}"}
    else:
        base_url, api_key = os.environ.get("MEDFORGE_JUDGE_BASE_URL"), os.environ.get("MEDFORGE_JUDGE_API_KEY")
        if not (base_url and api_key):
            rprint("[red]需配置 MEDFORGE_JUDGE_BASE_URL / _API_KEY(或改用 --provider claude-code)[/]")
            sys.exit(2)
        from openai import OpenAI

        client = OpenAI(base_url=base_url, api_key=api_key, timeout=300, max_retries=2)

        def ask(prompt: str) -> dict:
            resp = client.chat.completions.create(
                model=model, temperature=0.0, max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
                extra_body={"thinking": {"type": "enabled"}},
            )
            text = resp.choices[0].message.content or ""
            return parse_json_object(text) or {"reason": f"[未解析] {text[:200]}"}

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    todo = [i for i, r in enumerate(rows) if r.get("proxy_correct") is None and not r.get("proxy_reason")]
    rprint(f"代理标注 {len(todo)} / {len(rows)} 条(provider {provider} / 模型 {model}"
           + (f" / effort {effort}" if effort else "") + ")")
    lock = threading.Lock()

    def flush() -> None:
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

    def one(i: int) -> None:
        s = Sample(**rows[i]["sample"])
        answer, _ = split_answer(rows[i]["output"], thinking=True)
        data = ask(_PROXY_PROMPT.format(
            question=s.render_question()[:3000], gold=s.gold, answer=answer[-3000:]))
        c = data.get("correct")
        rows[i]["proxy_correct"] = c if isinstance(c, bool) else None
        rows[i]["proxy_chosen"] = data.get("chosen")
        rows[i]["proxy_reason"] = str(data.get("reason", ""))[:300]

    done = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = {pool.submit(one, i): i for i in todo}
        for fut in as_completed(futs):
            with lock:
                done += 1
                try:
                    fut.result()
                except Exception as e:  # noqa: BLE001
                    rprint(f"  ✗ {rows[futs[fut]]['sample']['id']}: {type(e).__name__}: {e}")
                if done % 10 == 0:
                    flush()
    flush()
    n_null = sum(1 for r in rows if r.get("proxy_correct") is None)
    rprint(f"[green]✓[/] 代理标注完成,无法判定 {n_null} 条 → {path};人工抽检后填 human_correct")


def main() -> None:
    argv = sys.argv[1:]

    def opt(name: str, default: str) -> str:
        return argv[argv.index(name) + 1] if name in argv else default

    if "--from-runs" in argv:
        sample_from_runs(opt("--from-runs", "").split(","), int(opt("--n", "150")), Path(opt("--out", str(MCQ_OUT))))
    elif "--label-proxy" in argv:
        from medforge.env import load_env
        from medforge.verify.verifier import JUDGE_PROVIDERS, judge_provider

        # 后端必须先定下来再挑默认模型:它也可能来自 .env 的 MEDFORGE_JUDGE_PROVIDER,
        # 只看命令行 flag 会在 env 切到 claude-code 时拿 deepseek-v4-pro 去调 CLI,整轮标注白跑
        load_env()
        provider = opt("--provider", "") or judge_provider()
        if provider not in JUDGE_PROVIDERS:
            rprint(f"[red]--provider 只能是 {'|'.join(JUDGE_PROVIDERS)},收到 {provider!r}[/]")
            sys.exit(2)
        # 换后端就换默认模型:claude-code 上 deepseek-v4-pro 这个名字根本不存在
        default_model = "claude-opus-5" if provider == "claude-code" else "deepseek-v4-pro"
        label_proxy(
            Path(opt("--file", str(MCQ_OUT))), opt("--model", default_model),
            concurrency=int(opt("--concurrency", "8")), provider=provider,
            effort=opt("--effort", "") or None,
        )
    elif "--generate" in argv:
        generate()
    else:
        sample()


if __name__ == "__main__":
    main()

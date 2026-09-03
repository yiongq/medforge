"""评测执行器:考卷 → OpenAI 兼容端点 → 验证器判分 → 落盘。

刻意的薄封装而非 EvalScope 深度集成(决策修正记录在 ADR-001 评测行):
四套考卷在 EvalScope 里全部要自写适配、判分必须复用已校准的 verifier.py,
框架只剩「发请求」的价值,却带来插件 API 版本耦合。这 200 行本机可完整
测试(tests/ 用假 OpenAI 服务端跑通全链路),EvalScope 保留给压测。

用法(端点通常是租卡上的 vllm serve):
    uv run python -m medforge.eval.run \
      --endpoint http://127.0.0.1:8000/v1 --model Qwen3.5-4B \
      --run-name base --sets cmexam,cmb-val,medxpertqa

产出 reports/runs/<run-name>/:
    run_meta.json         协议指纹(模型/解码参数/抽样/git):同一目录只允许一套协议
    <set>.outputs.jsonl   原始作答 + finish_reason + completion_tokens(断点续跑的依据)
    <set>.scored.jsonl    判分结果(report.load_run 的输入)
    summary.md            各集准确率表(Wilson CI + 弃权率 + 未收尾率)+ 协议抬头

协议版本:v1 = max_tokens 2048(截断思考型模型,存档保留);v2 = 8192 + temperature 0 +
固定种子抽样卷;v3(W2 审查后)= 可配解码参数 + finish_reason 落盘 + 截断守卫
(--thinking:没有恰好一个 </think> 即判未收尾)。参数默认值保持 v2,由命令行显式切 v3。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from rich import print as rprint

from medforge.data.schema import Sample
from medforge.eval.report import load_run, markdown_table
from medforge.verify.verifier import verify

# 固定模板:base/SFT/DPO 各配置必须用同一份提示词,否则对照不成立。
# 【暂定】答案声明格式与验证器抽取规则(「答案:X」)对齐
PROMPT_CHOICE = (
    "以下是一道医学选择题,请先简要推理,最后一行以「答案:X」的格式给出选项字母"
    "(多选题写出全部字母,如「答案:ABD」)。\n\n{question}"
)
PROMPT_OPEN = "你是医学助手,回答下面的问题。先给出推理过程,最后一行以「最终答案:」开头给出结论。\n\n{question}"

# 解码参数:一个 run 目录内必须完全一致(见 check_protocol)
PROTOCOL_KEYS = ("model", "max_tokens", "temperature", "top_p", "top_k", "presence_penalty", "seed", "thinking")
JUDGE_ENV = ("MEDFORGE_JUDGE_BASE_URL", "MEDFORGE_JUDGE_API_KEY", "MEDFORGE_JUDGE_MODEL")


def _gen_outputs(
    samples: list[Sample],
    out_file: Path,
    *,
    base_url: str,
    api_key: str,
    model: str,
    concurrency: int,
    max_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = -1,
    presence_penalty: float = 0.0,
    seed: int = 42,
) -> dict[str, dict]:
    """生成作答,断点续跑:已有输出的样本跳过。返回 {id: {"output", "finish_reason", "completion_tokens"}}。"""
    from openai import OpenAI

    outputs: dict[str, dict] = {}
    if out_file.exists():
        for line in out_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                outputs[row["id"]] = row
    todo = [s for s in samples if s.id not in outputs]
    rprint(f"  待生成 {len(todo)} / {len(samples)}(已有 {len(outputs)} 条复用)")
    if not todo:
        return outputs

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=300, max_retries=2)
    lock = threading.Lock()
    f = out_file.open("a", encoding="utf-8")

    def gen_one(s: Sample) -> dict:
        prompt = (PROMPT_CHOICE if s.is_choice else PROMPT_OPEN).format(question=s.render_question())
        kwargs: dict = {
            "model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature, "max_tokens": max_tokens, "seed": seed,  # seed 逐请求下发:采样也可复现
        }
        if top_p != 1.0:
            kwargs["top_p"] = top_p
        if presence_penalty:
            kwargs["presence_penalty"] = presence_penalty
        if top_k != -1:
            kwargs["extra_body"] = {"top_k": top_k}  # vLLM 私有参数,OpenAI SDK 不认
        resp = client.chat.completions.create(**kwargs)
        ch = resp.choices[0]
        usage = getattr(resp, "usage", None)
        return {
            "id": s.id,
            "output": ch.message.content or "",
            "finish_reason": ch.finish_reason,
            "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
        }

    done = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(gen_one, s) for s in todo]
        for fut in as_completed(futures):
            with lock:
                done += 1
                try:
                    row = fut.result()
                    if not row["output"]:
                        # 空作答不落盘:多半是端点异常;落了盘断点续跑就永远是空的
                        rprint(f"  ✗ {row['id']} 空作答(finish_reason={row['finish_reason']}),留待重跑")
                    else:
                        outputs[row["id"]] = row
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        f.flush()
                except Exception as e:  # noqa: BLE001  单条失败不废整卷,重跑补缺
                    rprint(f"  ✗ {type(e).__name__}: {e}")
                if done % 100 == 0:
                    rprint(f"  [{done}/{len(todo)}]")
    f.close()
    return outputs


def run_set(
    name: str,
    samples: list[Sample],
    out_dir: Path,
    *,
    base_url: str,
    api_key: str = "EMPTY",
    model: str,
    concurrency: int = 16,
    max_tokens: int = 2048,
    allow_llm_judge: bool = True,
    thinking: bool | None = None,
    gen: dict | None = None,
) -> Path:
    """跑一套考卷,返回 scored 文件路径。判分与 DPO 构造共用 verify()——口径唯一。

    thinking=True 时按思考型口径:没有恰好一个 </think> 即判未收尾;None 为自动。
    gen 是解码参数(temperature/top_p/top_k/presence_penalty/seed),缺省即 v2 贪心。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rprint(f"[bold]▶ {name}[/]({len(samples)} 题)")
    outputs = _gen_outputs(
        samples, out_dir / f"{name}.outputs.jsonl",
        base_url=base_url, api_key=api_key, model=model,
        concurrency=concurrency, max_tokens=max_tokens, **(gen or {}),
    )
    scored_file = out_dir / f"{name}.scored.jsonl"
    with scored_file.open("w", encoding="utf-8") as f:
        for s in samples:
            row = outputs.get(s.id)
            if row is None:
                v_correct, method, detail = None, "missing", ""  # 生成失败的题:弃权口径计错,不静默跳过
            else:
                v = verify(
                    s, row["output"], allow_llm=allow_llm_judge,
                    finish_reason=row.get("finish_reason"), thinking=thinking,
                )
                v_correct, method, detail = v.correct, v.method, v.detail
            f.write(json.dumps({
                "id": s.id, "correct": v_correct, "method": method,
                "finish_reason": (row or {}).get("finish_reason"),
                "completion_tokens": (row or {}).get("completion_tokens"),
                "detail": detail if method == "unfinished" else "",
            }, ensure_ascii=False) + "\n")
    r = load_run(scored_file, name)
    rprint(
        f"  [green]✓[/] 准确率 {r.acc * 100:.1f}%(n={r.n},弃权 {r.abstained},未收尾 {r.unfinished})"
    )
    return scored_file


def git_describe(root: Path) -> str:
    """短 hash + dirty 标记;不在 git 里就 unknown——报告数字要能追到提交。"""
    try:
        h = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], cwd=root, stderr=subprocess.DEVNULL) != 0
        return h + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def check_protocol(out_dir: Path, meta: dict) -> None:
    """同一 run 目录只能有一套解码协议:断点续跑时若参数变了,新旧答卷混在一起无法解读。

    首次运行写 run_meta.json;之后每次比对 PROTOCOL_KEYS,不一致直接退出。
    """
    meta_file = out_dir / "run_meta.json"
    if meta_file.exists():
        old = json.loads(meta_file.read_text(encoding="utf-8"))
        diff = {k: (old.get(k), meta.get(k)) for k in PROTOCOL_KEYS if old.get(k) != meta.get(k)}
        if diff:
            raise SystemExit(f"✗ {out_dir.name} 已按另一套协议落过盘 {diff};换 --run-name 或删掉目录")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def protocol_line(meta: dict) -> str:
    keys = ("model", "max_tokens", "temperature", "top_p", "top_k", "presence_penalty", "seed", "thinking", "git")
    parts = [f"{k}={meta.get(k)}" for k in keys]
    if meta.get("samples"):
        parts.append("抽样=" + ",".join(f"{k}={v}" for k, v in meta["samples"].items()))
    return "协议:" + " · ".join(parts)


def main() -> None:
    from medforge.data.sources import EVAL_SOURCES, ROOT, load_source
    from medforge.env import load_env

    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True, help="OpenAI 兼容 base_url,如 http://127.0.0.1:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--run-name", required=True, help="如 base / sft / sft-dpo")
    ap.add_argument("--sets", default=",".join(EVAL_SOURCES))
    ap.add_argument("--limit", type=int, default=0, help="每套只跑前 N 题(冒烟用)")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--no-llm-judge", action="store_true", help="判分禁用 LLM 兜底(纯规则,便宜)")
    # 协议 v1=2048(截断思考型模型,存档保留);v2=8192,W2 起基线与训练后统一用 v2
    ap.add_argument("--max-tokens", type=int, default=8192)
    # 解码参数默认保持 v2 贪心口径;v3 由命令行显式切(Qwen3 thinking 官方推荐 0.6/0.95/20)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=-1, help="-1 = 不限制(vLLM 语义)")
    ap.add_argument("--presence-penalty", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42, help="逐请求下发,采样协议也可复现")
    ap.add_argument(
        "--thinking", action="store_true",
        help="思考型模型口径:没有恰好一个 </think> 即判未收尾,不看能否刮出答案(评测 Qwen3.5 应开)",
    )
    # 预登记随机抽样卷(协议 v2 的一部分):种子固定 → 每个 run 考完全相同的题,
    # 对照有效;抽样量按 Wilson CI ±2pp 定。格式 "cmexam=2000,medxpertqa=1000"
    ap.add_argument("--samples", default="", help="逐集抽样量,如 cmexam=2000,medxpertqa=1000;未列出的集全量")
    args = ap.parse_args()
    sample_map = {}
    for kv in filter(None, args.samples.split(",")):
        k, _, v = kv.partition("=")
        sample_map[k.strip()] = int(v)

    # fail-fast:judge 没配时 verify_by_llm 只会静默弃权,弃权计错——整卷分数无声塌一半
    if not args.no_llm_judge:
        missing = [k for k in JUDGE_ENV if not os.environ.get(k)]
        if missing:
            rprint(f"[red]✗ LLM 兜底已启用但 judge 未配置: {missing};配好 .env 或显式 --no-llm-judge[/]")
            sys.exit(2)

    gen = {
        "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
        "presence_penalty": args.presence_penalty, "seed": args.seed,
    }
    meta = {
        "run_name": args.run_name, "model": args.model, "endpoint": args.endpoint,
        "max_tokens": args.max_tokens, **gen, "thinking": args.thinking,
        "samples": sample_map, "llm_judge": not args.no_llm_judge,
        "git": git_describe(ROOT), "created": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
    }
    out_dir = ROOT / "reports" / "runs" / args.run_name
    check_protocol(out_dir, meta)

    tables = []
    for name in args.sets.split(","):
        samples = load_source(name.strip())
        if name.strip() in sample_map:
            # 固定种子抽样:adapter 按文件序稳定产出,同 seed 必得同一批题
            samples = random.Random(42).sample(samples, min(sample_map[name.strip()], len(samples)))
        if args.limit:
            samples = samples[: args.limit]
        scored = run_set(
            name.strip(), samples, out_dir,
            base_url=args.endpoint, model=args.model,
            concurrency=args.concurrency, max_tokens=args.max_tokens,
            allow_llm_judge=not args.no_llm_judge, thinking=args.thinking or None, gen=gen,
        )
        tables.append((name.strip(), load_run(scored, args.run_name)))

    lines = [f"# {args.run_name} 评测汇总", "", protocol_line(meta), ""]
    for name, r in tables:
        lines += [f"## {name}", "", markdown_table([r], baseline="__none__"), ""]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    rprint(f"[green]✓[/] 汇总 → {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()

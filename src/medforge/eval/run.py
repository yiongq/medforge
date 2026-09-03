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
固定种子抽样卷;v3(P2 解码裁决后为默认)= Qwen3.5-4B 官方卡思考模式采样参数(1.0 / 0.95 / 20 / min_p 0 /
presence_penalty 1.5)+ 32768 预算 + finish_reason 落盘 + 截断守卫(--thinking on:没有 </think> 即判未收尾)。
复现 v2 协议:--temperature 0 --top-p 1 --top-k -1 --presence-penalty 0 --max-tokens 8192。
"""

from __future__ import annotations

import argparse
import hashlib
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
# 弃权变体(P2 对照臂 d):唯一差别是允许说「不确定」。用来量「弃权能力是不是 prompt 就能拿到的」
PROMPT_CHOICE_ABSTAIN = (
    "以下是一道医学选择题,请先简要推理,最后一行以「答案:X」的格式给出选项字母"
    "(多选题写出全部字母,如「答案:ABD」);如果没有把握,最后一行写「答案:不确定」,不要猜。\n\n{question}"
)
PROMPT_OPEN_ABSTAIN = (
    "你是医学助手,回答下面的问题。先给出推理过程,最后一行以「最终答案:」开头给出结论;"
    "如果没有把握,最后一行写「最终答案:不确定」,不要猜。\n\n{question}"
)
PROMPTS = {"default": (PROMPT_CHOICE, PROMPT_OPEN), "abstain": (PROMPT_CHOICE_ABSTAIN, PROMPT_OPEN_ABSTAIN)}

# budget forcing(P2 对照臂 c,s1 式):撞上 max_tokens 时把思考流原样接回去、强行写上收尾与答案触发词,
# 再让模型续写几十个 token。走 /v1/completions 裸 prompt,所以要自己渲染 chat 模板——
# 下面是 Qwen3.5 的 chat_template.jinja(2026-09-03 核对)对 [user] + add_generation_prompt 的渲染结果,
# 思考模式下生成提示以 <think>\n 结尾,所以模型输出里只有收尾的 </think>。换基座必须重新核对。
FORCE_PREFIX = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n"
FORCE_SUFFIX_CHOICE = "\n</think>\n\n答案:"
FORCE_SUFFIX_OPEN = "\n</think>\n\n最终答案:"
FORCE_MAX_TOKENS = 32
MODES = ("plain", "budget-forcing")

# 一个 run 目录内必须完全一致的东西(见 check_protocol):模型、解码参数、提示词模板、抽样卷、判分口径
PROTOCOL_KEYS = (
    "model", "max_tokens", "temperature", "top_p", "top_k", "min_p", "presence_penalty", "seed",
    "prompt", "prompt_sha", "mode", "samples", "limit", "thinking", "llm_judge",
)


def prompt_sha(variant: str) -> str:
    return hashlib.sha256("\n".join(PROMPTS[variant]).encode()).hexdigest()[:8]
THINKING_MODES = {"on": True, "off": False, "auto": None}
MAX_FAIL_RATE = 0.02  # 生成失败超过这个比例就不出表:一张 0% 的 summary 与一次成功评测在退出码上不可区分
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
    min_p: float = 0.0,
    presence_penalty: float = 0.0,
    seed: int = 42,
    prompt_variant: str = "default",
    mode: str = "plain",
    timeout: float = 300.0,
) -> dict[str, dict]:
    """生成作答,断点续跑:已有输出的样本跳过。返回 {id: {"output", "finish_reason", "completion_tokens", "forced"}}。"""
    from openai import OpenAI

    outputs: dict[str, dict] = {}
    if out_file.exists():
        for line in out_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                outputs[row["id"]] = row
    todo = [s for s in samples if s.id not in outputs]
    reused = len(samples) - len(todo)
    stray = len(outputs) - reused
    rprint(f"  待生成 {len(todo)} / {len(samples)}(已有 {reused} 条复用" + (f",另有 {stray} 条不属于本卷的残留)" if stray else ")"))
    if not todo:
        return outputs

    # timeout 按预算定:贪心复读到 32768 token 一条请求要十来分钟,300 秒会把整臂超时打成 missing
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=2)
    lock = threading.Lock()
    f = out_file.open("a", encoding="utf-8")

    p_choice, p_open = PROMPTS[prompt_variant]
    # 全部无条件下发:vLLM 会拿模型自带 generation_config 当默认值,不下发 ≠ 取 1.0,
    # 而 run_meta.json 记的是这里的值——记了就得真发出去,指纹才不说谎
    sampling = {"temperature": temperature, "top_p": top_p, "presence_penalty": presence_penalty, "seed": seed}
    extra = {"top_k": top_k, "min_p": min_p}  # vLLM 私有参数(top_k -1 = 不限),OpenAI SDK 不认所以走 extra_body

    def gen_one(s: Sample) -> dict:
        prompt = (p_choice if s.is_choice else p_open).format(question=s.render_question())
        resp = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, extra_body=extra, **sampling,
        )
        ch = resp.choices[0]
        usage = getattr(resp, "usage", None)
        row = {
            "id": s.id,
            "output": ch.message.content or "",
            "finish_reason": ch.finish_reason,
            "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
            "forced": False,
        }
        if mode == "budget-forcing" and row["output"] and ch.finish_reason == "length":
            # 强制收尾:思考流原样接回,写上 </think> 与答案触发词,续写几十个 token
            suffix = FORCE_SUFFIX_CHOICE if s.is_choice else FORCE_SUFFIX_OPEN
            raw = FORCE_PREFIX.format(prompt=prompt) + row["output"].rstrip("\n") + suffix
            cont = client.completions.create(model=model, prompt=raw, max_tokens=FORCE_MAX_TOKENS, extra_body=extra, **sampling)
            c0 = cont.choices[0]
            c_usage = getattr(cont, "usage", None)
            row["output"] = row["output"].rstrip("\n") + suffix + (c0.text or "")
            # 续写的 finish_reason 只描述那 32 个 token 是否写满,与「答案有没有写出来」无关——
            # 答案触发词是我们强写的,守卫不该再按 length 判未收尾;保留原值供审计,加前缀区分
            row["finish_reason"] = f"forced-{c0.finish_reason}"
            row["completion_tokens"] = (row["completion_tokens"] or 0) + (getattr(c_usage, "completion_tokens", 0) or 0)
            row["forced"] = True
        return row

    done = failed = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(gen_one, s) for s in todo]
        for fut in as_completed(futures):
            with lock:
                done += 1
                try:
                    row = fut.result()
                    if not row["output"]:
                        # 空作答不落盘:多半是端点异常;落了盘断点续跑就永远是空的
                        failed += 1
                        rprint(f"  ✗ {row['id']} 空作答(finish_reason={row['finish_reason']}),留待重跑")
                    else:
                        outputs[row["id"]] = row
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        f.flush()
                except Exception as e:  # noqa: BLE001  单条失败不废整卷,重跑补缺
                    failed += 1
                    rprint(f"  ✗ {type(e).__name__}: {e}")
                if done % 100 == 0:
                    rprint(f"  [{done}/{len(todo)}]")
    f.close()
    if failed and failed / len(todo) > MAX_FAIL_RATE:
        # 已生成的都落盘了,重跑只补缺;但不能带着一堆 missing 出一张看起来正常的表
        raise SystemExit(f"✗ 生成失败 {failed}/{len(todo)} 超过 {MAX_FAIL_RATE:.0%}:检查端点/参数后重跑补缺")
    if failed:
        rprint(f"  [yellow]! {failed} 条生成失败,已按 missing 计错;重跑可补缺[/]")
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

    thinking=True 时按思考型口径:没有 </think> 即判未收尾;None 为自动(语义见 verifier.split_answer)。
    gen 是解码参数(temperature/top_p/top_k/presence_penalty/seed/prompt_variant/mode),缺省即 v2 贪心。
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
                finish_reason = row.get("finish_reason")
                if row.get("forced") and finish_reason == "length":
                    finish_reason = "forced-length"  # 修复前落盘的 forced 行:续写撞 32 token 上限不等于未收尾
                v = verify(
                    s, row["output"], allow_llm=allow_llm_judge, finish_reason=finish_reason, thinking=thinking,
                )
                v_correct, method, detail = v.correct, v.method, v.detail
            f.write(json.dumps({
                "id": s.id, "correct": v_correct, "method": method,
                "finish_reason": (row or {}).get("finish_reason"),
                "completion_tokens": (row or {}).get("completion_tokens"),
                "forced": bool((row or {}).get("forced")),
                "detail": detail if method in ("unfinished", "abstain") else "",
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
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True, stderr=subprocess.DEVNULL,
        )
        return h + ("-dirty" if status.strip() else "")  # 已暂存与未暂存都算脏;未跟踪文件不算
    except Exception:  # noqa: BLE001
        return "unknown"


def check_protocol(out_dir: Path, meta: dict, *, adopt_legacy: bool = False) -> None:
    """同一 run 目录只能有一套协议:断点续跑时若参数变了,新旧答卷混在一起无法解读。

    首次运行写 run_meta.json;之后每次比对 PROTOCOL_KEYS,不一致直接退出。
    目录里已有 *.outputs.jsonl 却没有 run_meta.json 的是 W2 之前的存档,不知道它们按什么协议生成,
    默认拒绝——否则一次 v3 续跑会给一批 v2 贪心答卷盖上 v3 指纹,还会以 "w" 模式重写存档 scored。
    显式 --adopt-legacy 才补写一份标了 legacy 的指纹。每次运行追加一条 history(git/时间),
    因为续跑常常跨天跨提交,首跑的 git 不能代表整份答卷。
    """
    meta_file = out_dir / "run_meta.json"
    stamp = {"git": meta.get("git"), "created": meta.get("created")}
    if meta_file.exists():
        old = json.loads(meta_file.read_text(encoding="utf-8"))
        diff = {k: (old.get(k), meta.get(k)) for k in PROTOCOL_KEYS if old.get(k) != meta.get(k)}
        if diff:
            raise SystemExit(f"✗ {out_dir.name} 已按另一套协议落过盘 {diff};换 --run-name 或删掉目录")
        old.setdefault("history", []).append(stamp)
        meta_file.write_text(json.dumps(old, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    if any(out_dir.glob("*.outputs.jsonl")) and not adopt_legacy:
        raise SystemExit(
            f"✗ {out_dir.name} 已有存档答卷但没有 run_meta.json(W2 之前的目录):"
            "不知道它们按什么协议生成,拒绝续跑;确认协议一致再加 --adopt-legacy,或换 --run-name"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_file.write_text(
        json.dumps({**meta, "legacy": adopt_legacy, "history": [stamp]}, ensure_ascii=False, indent=2), encoding="utf-8",
    )


def protocol_line(meta: dict) -> str:
    keys = (
        "model", "max_tokens", "temperature", "top_p", "top_k", "min_p", "presence_penalty", "seed",
        "prompt", "prompt_sha", "mode", "thinking", "llm_judge", "git",
    )
    parts = [f"{k}={meta.get(k)}" for k in keys]
    parts.append("抽样=" + (",".join(f"{k}={v}" for k, v in meta["samples"].items()) if meta.get("samples") else "全量"))
    if meta.get("limit"):
        parts.append(f"limit={meta['limit']}(冒烟)")
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
    ap.add_argument("--timeout", type=float, default=300.0, help="单条请求超时秒数;32768 预算的贪心臂要给 3600")
    ap.add_argument("--no-llm-judge", action="store_true", help="判分禁用 LLM 兜底(纯规则,便宜)")
    # 协议 v1=2048、v2=8192 均为存档协议;v3=32768(官方卡建议,P2 实测三卷无一撞顶,最长 21k)
    ap.add_argument("--max-tokens", type=int, default=32768)
    # 解码默认 = Qwen3.5-4B 官方卡(2026-09-03 核对)思考模式通用任务参数;官方明令禁止贪心(会无尽复读),
    # P2 实测贪心让基座三卷 26~67% 的题转圈交不了卷(reports/p2-decoding-arms.md)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20, help="-1 = 不限制(vLLM 语义)")
    ap.add_argument("--min-p", type=float, default=0.0)
    ap.add_argument("--presence-penalty", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=42, help="逐请求下发,采样协议也可复现")
    ap.add_argument(
        "--thinking", choices=sorted(THINKING_MODES), default="on",
        help="截断守卫口径:on = 没有 </think> 即判未收尾(默认,思考型模型);off = 全文判分;auto = 含 </think> 才按思考型",
    )
    ap.add_argument("--adopt-legacy", action="store_true", help="给 W2 之前没有 run_meta.json 的存档目录补写协议指纹")
    ap.add_argument("--prompt", choices=sorted(PROMPTS), default="default", help="提示词变体:abstain = 允许写「答案:不确定」")
    ap.add_argument(
        "--mode", choices=MODES, default="plain",
        help="budget-forcing = 撞上 max_tokens 时接回思考流、强写 </think> 与答案触发词再续写 32 token(s1 式)",
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
        "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k, "min_p": args.min_p,
        "presence_penalty": args.presence_penalty, "seed": args.seed,
        "prompt_variant": args.prompt, "mode": args.mode, "timeout": args.timeout,
    }
    meta = {
        "run_name": args.run_name, "model": args.model, "endpoint": args.endpoint,
        "max_tokens": args.max_tokens, **{k: v for k, v in gen.items() if k not in ("prompt_variant", "timeout")},
        "prompt": args.prompt, "prompt_sha": prompt_sha(args.prompt),
        "samples": sample_map, "limit": args.limit,
        "thinking": args.thinking, "llm_judge": not args.no_llm_judge,
        "git": git_describe(ROOT), "created": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
    }
    out_dir = ROOT / "reports" / "runs" / args.run_name
    check_protocol(out_dir, meta, adopt_legacy=args.adopt_legacy)

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
            allow_llm_judge=not args.no_llm_judge, thinking=THINKING_MODES[args.thinking], gen=gen,
        )
        tables.append((name.strip(), load_run(scored, args.run_name)))

    lines = [f"# {args.run_name} 评测汇总", "", protocol_line(meta), ""]
    for name, r in tables:
        lines += [f"## {name}", "", markdown_table([r], baseline="__none__"), ""]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    rprint(f"[green]✓[/] 汇总 → {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()

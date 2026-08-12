"""评测执行器:考卷 → OpenAI 兼容端点 → 验证器判分 → 落盘。

刻意的薄封装而非 EvalScope 深度集成(决策修正记录在 ADR-001 评测行):
四套考卷在 EvalScope 里全部要自写适配、判分必须复用已校准的 verifier.py,
框架只剩「发请求」的价值,却带来插件 API 版本耦合。这 150 行本机可完整
测试(tests/ 用假 OpenAI 服务端跑通全链路),EvalScope 保留给压测。

用法(端点通常是租卡上的 vllm serve):
    uv run python -m medforge.eval.run \
      --endpoint http://127.0.0.1:8000/v1 --model Qwen3.5-4B \
      --run-name base --sets cmexam,cmb-val,medxpertqa

产出 reports/runs/<run-name>/:
    <set>.outputs.jsonl   原始作答(断点续跑的依据,重跑只补空缺)
    <set>.scored.jsonl    判分结果(report.load_run 的输入)
    summary.md            各集准确率表(Wilson CI + 弃权率)
"""

from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _gen_outputs(
    samples: list[Sample],
    out_file: Path,
    *,
    base_url: str,
    api_key: str,
    model: str,
    concurrency: int,
    max_tokens: int,
) -> dict[str, str]:
    """生成作答,断点续跑:已有输出的样本跳过。返回 {id: output}。"""
    from openai import OpenAI

    outputs: dict[str, str] = {}
    if out_file.exists():
        for line in out_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                outputs[row["id"]] = row["output"]
    todo = [s for s in samples if s.id not in outputs]
    rprint(f"  待生成 {len(todo)} / {len(samples)}(已有 {len(outputs)} 条复用)")
    if not todo:
        return outputs

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=300, max_retries=2)
    lock = threading.Lock()
    f = out_file.open("a", encoding="utf-8")

    def gen_one(s: Sample) -> tuple[str, str]:
        prompt = (PROMPT_CHOICE if s.is_choice else PROMPT_OPEN).format(question=s.render_question())
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,  # 评测口径:贪心解码,可复现
            max_tokens=max_tokens,
        )
        return s.id, resp.choices[0].message.content or ""

    done = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(gen_one, s) for s in todo]
        for fut in as_completed(futures):
            with lock:
                done += 1
                try:
                    sid, text = fut.result()
                    outputs[sid] = text
                    f.write(json.dumps({"id": sid, "output": text}, ensure_ascii=False) + "\n")
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
) -> Path:
    """跑一套考卷,返回 scored 文件路径。判分与 DPO 构造共用 verify()——口径唯一。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    rprint(f"[bold]▶ {name}[/]({len(samples)} 题)")
    outputs = _gen_outputs(
        samples, out_dir / f"{name}.outputs.jsonl",
        base_url=base_url, api_key=api_key, model=model,
        concurrency=concurrency, max_tokens=max_tokens,
    )
    scored_file = out_dir / f"{name}.scored.jsonl"
    with scored_file.open("w", encoding="utf-8") as f:
        for s in samples:
            out = outputs.get(s.id)
            if out is None:
                v_correct, method = None, "missing"  # 生成失败的题:弃权口径计错,不静默跳过
            else:
                v = verify(s, out, allow_llm=allow_llm_judge)
                v_correct, method = v.correct, v.method
            f.write(json.dumps({"id": s.id, "correct": v_correct, "method": method}, ensure_ascii=False) + "\n")
    r = load_run(scored_file, name)
    rprint(f"  [green]✓[/] 准确率 {r.acc * 100:.1f}%(n={r.n},弃权 {r.abstained})")
    return scored_file


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
    args = ap.parse_args()

    out_dir = ROOT / "reports" / "runs" / args.run_name
    tables = []
    for name in args.sets.split(","):
        samples = load_source(name.strip())
        if args.limit:
            samples = samples[: args.limit]
        scored = run_set(
            name.strip(), samples, out_dir,
            base_url=args.endpoint, model=args.model,
            concurrency=args.concurrency, allow_llm_judge=not args.no_llm_judge,
        )
        tables.append((name.strip(), load_run(scored, args.run_name)))

    lines = [f"# {args.run_name} 评测汇总", ""]
    for name, r in tables:
        lines += [f"## {name}", "", markdown_table([r], baseline="__none__"), ""]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    rprint(f"[green]✓[/] 汇总 → {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()

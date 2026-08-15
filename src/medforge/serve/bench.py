"""推理服务压测:并发扫描 → TTFT / TPOT / 吞吐 曲线。

用法(在起着 vLLM 的机器上):
    uv run python -m medforge.serve.bench --endpoint http://127.0.0.1:8000/v1 \
        --model target --label bf16 [--concurrency 1,2,4,8,16,32,64]

为什么自写而不是直接用 vllm bench serve:
- 我们要的是「同一批真实医学题」在不同并发下的曲线,并且要和评测用的题源一致——
  合成随机 token 测出来的数字好看但不代表本项目的负载
- 产物要能直接喂给前台图表(bench.json),官方工具的输出格式还得再转一道

口径(与业界压测报告一致):
- TTFT = 首 token 到达时间;TPOT = (总时长 − TTFT) / (输出 token 数 − 1)
- 固定输出长度 + ignore_eos:让不同配置的 TPOT 可比,否则短答案会虚高吞吐
- 每档并发前先跑 3 条预热请求(排除 CUDA graph 捕获、首批 KV 分配的冷启动)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import asdict, dataclass

from rich import print as rprint

from medforge.data.sources import ROOT, load_source

REPORTS = ROOT / "reports"


@dataclass
class ReqResult:
    ttft_ms: float
    total_ms: float
    out_tokens: int
    ok: bool

    @property
    def tpot_ms(self) -> float:
        """每输出 token 的平均间隔(不含首 token)。"""
        return (self.total_ms - self.ttft_ms) / max(1, self.out_tokens - 1)


async def one_request(client, model: str, prompt: str, max_tokens: int) -> ReqResult:
    t0 = time.perf_counter()
    ttft = None
    n = 0
    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7,
            stream=True,
            extra_body={"ignore_eos": True},  # 固定输出长度,让 TPOT 可比
        )
        async for chunk in stream:
            if not chunk.choices or not chunk.choices[0].delta.content:
                continue
            if ttft is None:
                ttft = (time.perf_counter() - t0) * 1000
            n += 1
    except Exception:  # noqa: BLE001  单条失败不废整档,计入失败率
        return ReqResult(0, (time.perf_counter() - t0) * 1000, 0, False)
    total = (time.perf_counter() - t0) * 1000
    return ReqResult(ttft or total, total, n, True)


async def run_level(client, model: str, prompts: list[str], conc: int, max_tokens: int) -> dict:
    """一档并发:发满 conc 路并行,共 n_req 条,统计分位数与聚合吞吐。"""
    n_req = max(conc * 3, 12)
    sem = asyncio.Semaphore(conc)

    async def guarded(p: str) -> ReqResult:
        async with sem:
            return await one_request(client, model, p, max_tokens)

    t0 = time.perf_counter()
    results = await asyncio.gather(*(guarded(prompts[i % len(prompts)]) for i in range(n_req)))
    wall = time.perf_counter() - t0

    ok = [r for r in results if r.ok]
    if not ok:
        return {"concurrency": conc, "failed": n_req, "requests": 0}
    ttfts = sorted(r.ttft_ms for r in ok)
    tpots = sorted(r.tpot_ms for r in ok)
    out_total = sum(r.out_tokens for r in ok)

    def pct(xs: list[float], q: float) -> float:
        return round(xs[min(len(xs) - 1, int(len(xs) * q))], 1)

    return {
        "concurrency": conc,
        "requests": len(ok),
        "failed": n_req - len(ok),
        "ttft_p50": pct(ttfts, 0.5), "ttft_p95": pct(ttfts, 0.95),
        "tpot_p50": pct(tpots, 0.5), "tpot_p95": pct(tpots, 0.95),
        "output_tok_s": round(out_total / wall, 1),
        "req_per_s": round(len(ok) / wall, 3),
        "wall_s": round(wall, 1),
    }


async def main_async(args) -> None:
    from openai import AsyncOpenAI

    # 真实负载:从评测题源取题面,而不是随机 token
    samples = load_source("cmexam")[: args.num_prompts]
    prompts = [s.render_question() for s in samples]
    rprint(f"负载:{len(prompts)} 道 CMExam 真题,题面均长 {statistics.mean(len(p) for p in prompts):.0f} 字")

    client = AsyncOpenAI(base_url=args.endpoint, api_key="EMPTY", timeout=600, max_retries=0)
    rprint("预热 3 条…")
    await asyncio.gather(*(one_request(client, args.model, prompts[i], 64) for i in range(3)))

    levels = []
    for conc in [int(c) for c in args.concurrency.split(",")]:
        rprint(f"▶ 并发 {conc}")
        lv = await run_level(client, args.model, prompts, conc, args.max_tokens)
        rprint(
            f"  TTFT p50 {lv.get('ttft_p50')}ms / p95 {lv.get('ttft_p95')}ms · "
            f"TPOT p50 {lv.get('tpot_p50')}ms · 吞吐 {lv.get('output_tok_s')} tok/s"
        )
        levels.append(lv)

    metrics = None
    try:  # vLLM 自带的 Prometheus 端点:留存一份服务侧口径,便于与客户端测量交叉核对
        import httpx

        base = args.endpoint.rstrip("/").removesuffix("/v1")
        async with httpx.AsyncClient(timeout=10) as hc:
            r = await hc.get(f"{base}/metrics")
        metrics = r.text if r.status_code == 200 else None
    except Exception:  # noqa: BLE001  指标端点缺失不影响压测结论
        metrics = None

    out = {
        "label": args.label,
        "model": args.model,
        "gpu": args.gpu,
        "max_tokens": args.max_tokens,
        "levels": levels,
    }
    REPORTS.mkdir(exist_ok=True)
    dst = REPORTS / f"bench-{args.label}.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    rprint(f"[green]✓[/] {dst}")
    if metrics:
        (REPORTS / f"bench-{args.label}.metrics.txt").write_text(metrics, encoding="utf-8")
        rprint("[green]✓[/] 服务端 /metrics 快照已存档")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True, help="配置名,如 bf16 / fp8")
    ap.add_argument("--gpu", default="", help="卡型,写进报告便于复现")
    ap.add_argument("--concurrency", default="1,2,4,8,16,32,64")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--num-prompts", type=int, default=64)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()


__all__ = ["ReqResult", "asdict", "one_request", "run_level"]

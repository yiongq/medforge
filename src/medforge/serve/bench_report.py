"""压测 JSON → 部署报告(markdown)+ 前台图表数据(web/public/bench.json)。

报告只陈述测得的事实与口径,不做「比 X 快 N 倍」这类跨硬件外推——
压测数字离开卡型、上下文长度、输出长度就没有意义,这三项一律随数字同行。
"""

from __future__ import annotations

import json

from rich import print as rprint

from medforge.data.sources import ROOT

REPORTS = ROOT / "reports"
WEB_PUBLIC = ROOT / "web" / "public"


def load_benches() -> list[dict]:
    out = []
    for f in sorted(REPORTS.glob("bench-*.json")):
        if f.name.endswith(".metrics.txt"):
            continue
        out.append(json.loads(f.read_text(encoding="utf-8")))
    return out


def main() -> None:
    benches = load_benches()
    if not benches:
        rprint("[red]没有 bench-*.json,先跑 scripts/serve_bench.sh[/]")
        raise SystemExit(1)

    lines = [
        "# 部署与压测报告",
        "",
        (f"模型:`{benches[0]['model']}` · 卡型:{benches[0].get('gpu') or '未记录'} · "
         f"引擎:vLLM · 固定输出 {benches[0]['max_tokens']} token(ignore_eos)"),
        "",
        ("负载为 CMExam 真实题面(非合成 token);TTFT=首 token 到达时间,"
         "TPOT=后续每 token 平均间隔,吞吐为聚合输出速率。每档并发前有 3 条预热。"),
        "",
    ]
    for b in benches:
        lines += [
            f"## {b['label'].upper()}",
            "",
            "| 并发 | TTFT p50 | TTFT p95 | TPOT p50 | 输出吞吐 | 请求吞吐 | 失败 |",
            "|---|---|---|---|---|---|---|",
        ]
        for lv in b["levels"]:
            if not lv.get("requests"):
                lines.append(f"| {lv['concurrency']} | — | — | — | — | — | 全部失败 |")
                continue
            lines.append(
                f"| {lv['concurrency']} | {lv['ttft_p50']} ms | {lv['ttft_p95']} ms | "
                f"{lv['tpot_p50']} ms | {lv['output_tok_s']} tok/s | {lv['req_per_s']} req/s | {lv['failed']} |"
            )
        lines.append("")

    if len(benches) >= 2:
        a, b2 = benches[0], benches[1]
        def peak(x: dict) -> float:
            return max((lv.get("output_tok_s", 0) for lv in x["levels"]), default=0)

        lines += [
            "## 配置对比",
            "",
            f"峰值输出吞吐:{a['label'].upper()} {peak(a)} tok/s vs {b2['label'].upper()} {peak(b2)} tok/s。",
            "",
            "口径提示:两档用同一份权重,仅数值精度不同;数字仅对本卡型与本负载成立。",
            "",
        ]

    (REPORTS / "deployment.md").write_text("\n".join(lines), encoding="utf-8")
    WEB_PUBLIC.mkdir(parents=True, exist_ok=True)
    (WEB_PUBLIC / "bench.json").write_text(json.dumps(benches, ensure_ascii=False), encoding="utf-8")
    rprint(f"[green]✓[/] reports/deployment.md · web/public/bench.json({len(benches)} 组配置)")


if __name__ == "__main__":
    main()

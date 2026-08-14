"""SFT 教材构造:带 CoT 的医疗数据 + 通用指令混料 → ms-swift 训练文件。

用法:
    uv run python -m medforge.data.build_sft [--general-ratio 0.15]

数据设计(ADR-001 数据行,与 HuatuoGPT-o1 原谱的偏离及理由):
- 原谱 SFT 数据 = 调 API 对 4 万道可验证题做验证器引导搜索合成 CoT,当前 API 价格约 600+ 元;
  官方已开源 2 万条同管线产出的中文 CoT 数据(med-o1-sft-zh),免费同质——直接用它
- 可验证题(4 万道)不进 SFT,留给 DPO/GRPO 的自采样(build_dpo,GPU 时费十几元)
- 混入通用指令 replay 防灾难性遗忘,【暂定】比例 0.15(文献典型区间 5%-20% 取中偏上,
  W2 若医疗涨分/通用退化失衡再调)

思考格式:assistant 内容为 <think>…</think> 包裹 CoT 后接正文——
已核实为 ms-swift 官方 thinking SFT 格式(Custom-dataset 文档示例 + Qwen3.5 Best Practice)。
"""

from __future__ import annotations

import argparse
import json
import random

from rich import print as rprint

from medforge.data.sources import RAW, ROOT

PROCESSED = ROOT / "data" / "processed"
SEED = 42


def med_rows(source: str) -> list[dict]:
    """任一带 CoT 的训练源 → 消息格式。以去污染后的题池为准(train_pool 已剔撞题样本)。"""
    from medforge.data.sources import load_source

    pool_ids = set()
    with (PROCESSED / "train_pool.jsonl").open(encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            if s["source"] == source:
                pool_ids.add(s["id"])
    rows = []
    seen: set[tuple[str, str]] = set()
    n_dup = 0
    for smp in load_source(source):
        if smp.id not in pool_ids or not smp.cot:
            continue  # 去污染剔除,或无 CoT 的样本
        assistant = f"<think>\n{smp.cot}\n</think>\n\n{smp.gold}"
        # 上游数据集实测自带成组逐字节重复(med-o1-zh 占 30%):不去重等于双倍加权
        key = (smp.question, assistant)
        if key in seen:
            n_dup += 1
            continue
        seen.add(key)
        rows.append({"messages": [
            {"role": "user", "content": smp.question},
            {"role": "assistant", "content": assistant},
        ]})
    if n_dup:
        print(f"[build_sft] {source} 内容去重:剔 {n_dup} 条重复")
    return rows


def general_rows() -> list[dict]:
    """alpaca-gpt4-chinese → 消息格式。只取单轮(多轮在该集占比极小,不值得引入复杂度)。"""
    rows = []
    with (RAW / "alpaca-zh.train.jsonl").open(encoding="utf-8") as f:
        for line in f:
            conv = json.loads(line)["conversations"]
            if len(conv) != 2 or conv[0]["from"] != "human" or conv[1]["from"] != "gpt":
                continue
            rows.append({"messages": [
                {"role": "user", "content": conv[0]["value"].strip()},
                {"role": "assistant", "content": conv[1]["value"].strip()},
            ]})
    return rows


def mix(med: list[dict], general: list[dict], ratio: float, seed: int = SEED) -> list[dict]:
    """按目标占比抽取通用数据并整体打散。ratio = 通用条数 / 总条数。"""
    rng = random.Random(seed)
    n_general = int(len(med) * ratio / (1 - ratio))
    if n_general > len(general):
        raise ValueError(f"通用数据不足:需 {n_general},仅有 {len(general)}")
    picked = rng.sample(general, n_general)
    out = med + picked
    rng.shuffle(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--general-ratio", type=float, default=0.15)
    ap.add_argument("--source", default="med-o1-sft-zh", help="训练源(需已在 TRAIN_SOURCES 注册并跑过 build 去污染)")
    ap.add_argument("--out", default="sft_train.jsonl")
    args = ap.parse_args()

    med = med_rows(args.source)
    general = general_rows()
    out = mix(med, general, args.general_ratio)
    dst = PROCESSED / args.out
    with dst.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rprint(
        f"[green]✓[/] SFT 教材 {len(out)} 条(医疗 {len(med)} + 通用 {len(out) - len(med)},"
        f"通用占比 {(len(out) - len(med)) / len(out):.1%})→ {dst}"
    )


if __name__ == "__main__":
    main()

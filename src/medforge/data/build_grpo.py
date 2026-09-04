"""GRPO(第三阶段)题单构造:从去污染题池切一段与前两阶段不重叠的选择题,渲染成 ms-swift GRPO 行。

GRPO 不需要教材,只需要「题 + 标准答案」——奖励由 medforge 自己的验证器现算
(src/medforge/train/grpo_reward.py 的 medforge_selective 插件),所以这一步纯本地、免费、可反复重跑。

    uv run python -m medforge.data.build_grpo \
      --pool data/processed/abstain_pool.jsonl --skip 4000 --n 6000 \
      --out data/processed/grpo_train.jsonl --eval-n 200

产出 data/processed/grpo_train.jsonl(以及 --eval-n > 0 时的 grpo_eval.jsonl),每行:

    {"messages": [{"role": "user", "content": <与评测完全同一份提示词>}],
     "solution": "AC", "options": {"A": "...", ...}, "id": "cmexam-train-123"}

三条设计约束:

  ① 提示词必须与评测逐字节相同 —— 复用 build_distill.render_prompt(它又复用 eval/run.py 的
     PROMPT_CHOICE/PROMPT_OPEN)。训练时优化的分布若与评测时的提示词不同,涨的分在考卷上兑不出来
     (这是 W2 蒸馏教材的坑 ③,已经付过一次学费)。

  ② 与弃权 SFT(第二阶段)的题不重叠 —— 两阶段用同一份题池、同一个 seed 洗牌:
     弃权阶段取 pick_questions(pool, 4000) 也就是洗牌后的前 4000,GRPO 取 [skip, skip+n)。
     同一批题先被 SFT 教「怎么答」再被 GRPO 按同一答案打分,等于让 RL 在已经背下来的题上刷奖励,
     测出来的提升是记忆不是策略。--eval-n 的评估题再往后顺延,与训练段同样不重叠。

  ③ 只收选择题 —— 奖励是「抽出的字母集合 == gold 字母集合」,开放题在规则层判不了
     (verify_by_rule 对开放题只认归一化精确相等,大量弃权 → 奖励信号全是噪声)。
     abstain_pool.jsonl 是 CMExam-train,本就全是选择题;非选择题按计数丢弃并在末尾报数。

额外列 solution / options / id 的命名依据(ms-swift 4.5.2 实测路径):
  GRPO 会强制 remove_unused_columns=False(swift/arguments/rlhf_args.py:348),
  数据集里的非标准列进入 GRPOSample.extra(swift/rl_core/data.py from_row),
  再由 to_reward_row() 摊平、RowPreprocessor.rows_to_batched() 转成「列名 → 按样本对齐的 list」,
  最后以 **kwargs 传给奖励函数(swift/rl_core/grpo_algorithm.py compute_rewards_per_func)。
  所以列名就是奖励函数的形参名;solution 沿用官方 accuracy/math ORM 的既有约定。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich import print as rprint

from medforge.data.build_distill import SEED, load_pool, pick_questions, render_prompt
from medforge.data.schema import Sample
from medforge.data.sources import ROOT

PROCESSED = ROOT / "data" / "processed"
DEFAULT_POOL = PROCESSED / "abstain_pool.jsonl"
DEFAULT_OUT = PROCESSED / "grpo_train.jsonl"
DEFAULT_EVAL_OUT = PROCESSED / "grpo_eval.jsonl"
# 弃权 SFT(第二阶段)吃掉的题数:洗牌后的前 4000。GRPO 必须从这之后开始切。
DEFAULT_SKIP = 4000
DEFAULT_N = 6000


def gold_letters(sample: Sample) -> str:
    """标准答案规范化成升序去重的字母串:多选题 "CA" / "A、C" 一律变成 "AC"。

    奖励侧比的是集合,这里先规范化只是为了让落盘的 solution 一眼能读、也能直接做字面比较。
    """
    return "".join(sorted({c for c in sample.gold.upper() if c.isalpha()}))


def pick_slice(pool: list[Sample], skip: int, n: int, seed: int = SEED) -> list[Sample]:
    """同一 seed 洗牌后取 [skip, skip+n) 这一段。

    刻意复用 build_distill.pick_questions 的洗牌方式(random.Random(seed).sample(pool, len(pool))),
    而不是自己再洗一遍:两阶段的「不重叠」只有在洗牌结果逐元素相同时才成立。
    n <= 0 表示「从 skip 开始全都要」。
    """
    full = pick_questions(pool, 0, seed)
    return full[skip:] if n <= 0 else full[skip : skip + n]


def to_grpo_row(sample: Sample) -> dict:
    """ms-swift GRPO 一行。messages 只有 user——completion 由 rollout 现场生成,不存在 assistant 侧。"""
    return {
        "messages": [{"role": "user", "content": render_prompt(sample)}],
        "solution": gold_letters(sample),
        "options": dict(sample.options or {}),
        "id": sample.id,
    }


def build_rows(samples: list[Sample]) -> tuple[list[dict], int]:
    """(可用行, 被丢弃的非选择题/无金标题数)。"""
    rows, dropped = [], 0
    for s in samples:
        if not s.is_choice or not gold_letters(s):
            dropped += 1
            continue
        rows.append(to_grpo_row(s))
    return rows, dropped


def write_rows(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GRPO 题单构造(第三阶段)")
    ap.add_argument("--pool", type=Path, default=DEFAULT_POOL, help="题池 jsonl(每行 Sample.to_dict())")
    ap.add_argument("--source", default="cmexam-train", help="只用题池里这个 source 的题")
    ap.add_argument("--skip", type=int, default=DEFAULT_SKIP, help="跳过洗牌后的前若干题(留给弃权 SFT)")
    ap.add_argument("--n", type=int, default=DEFAULT_N, help="训练题数;<= 0 表示取到题池末尾")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--eval-n", type=int, default=0, help="> 0 时再往后顺延切出这么多题作验证集")
    ap.add_argument("--eval-out", type=Path, default=DEFAULT_EVAL_OUT)
    ap.add_argument("--seed", type=int, default=SEED, help="洗牌 seed,必须与弃权阶段一致")
    args = ap.parse_args(argv)

    if not args.pool.exists():
        rprint(f"[red]✗ 题池不存在: {args.pool}[/red]")
        return 2
    pool = load_pool(args.pool, args.source)
    if not pool:
        rprint(f"[red]✗ 题池里没有 source={args.source} 的题: {args.pool}[/red]")
        return 2
    rprint(f"题池 {args.pool}(source={args.source}): {len(pool)} 题")

    picked = pick_slice(pool, args.skip, args.n, args.seed)
    rows, dropped = build_rows(picked)
    write_rows(rows, args.out)
    rprint(
        f"[green]✓[/green] 训练题单 {args.out}: {len(rows)} 行(切片 [{args.skip}, {args.skip + len(picked)}),丢弃非选择题 {dropped})"
    )

    if args.eval_n > 0:
        # 验证集紧接训练段之后:与弃权 SFT 的 4000 题、GRPO 训练段都不重叠
        eval_start = args.skip + len(picked)
        eval_picked = pick_slice(pool, eval_start, args.eval_n, args.seed)
        eval_rows, eval_dropped = build_rows(eval_picked)
        write_rows(eval_rows, args.eval_out)
        rprint(
            f"[green]✓[/green] 验证题单 {args.eval_out}: {len(eval_rows)} 行(切片 [{eval_start}, {eval_start + len(eval_picked)}),丢弃 {eval_dropped})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

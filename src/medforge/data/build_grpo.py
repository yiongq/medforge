"""GRPO(第三阶段)题单构造:挑出「模型半会不会」的选择题,渲染成 ms-swift GRPO 行。

GRPO 不需要教材,只需要「题 + 标准答案」——奖励由 medforge 自己的验证器现算
(src/medforge/train/grpo_reward.py 的 medforge_selective 插件),所以这一步纯本地、免费、可反复重跑。

两种取题模式:

  ① 切片(--skip/--n):同 seed 洗牌后切一段与前两阶段不重叠的题。第一次上机用的就是它。

        uv run python -m medforge.data.build_grpo \
          --pool data/processed/abstain_pool.jsonl --skip 4000 --n 6000 \
          --out data/processed/grpo_train.jsonl --eval-n 200

  ② 难度筛选(--samples):只留「4 次采样里对 1~3 次」的题。**推荐**。

        uv run python -m medforge.data.build_grpo \
          --samples data/processed/abstain_samples.jsonl --pool data/processed/abstain_pool.jsonl \
          --k 4 --min-correct 1 --max-correct 3 --eval-n 100 \
          --out data/processed/grpo_train_unstable.jsonl \
          --eval-out data/processed/grpo_eval_unstable.jsonl

为什么要有 ②(2026-09 单卡实测):用 ① 随机切 6000 题上机,ms-swift 自己的
`frac_reward_zero_std` 报到 **0.60~0.69**——60%~69% 的 prompt 组里 8 个 rollout 拿到完全相同的奖励
(模型要么全对要么全错),组内优势恒为 0,这些 rollout 对梯度零贡献。GRPO 的梯度只来自**组内奖励方差**,
题太简单或太难都没有方差。而单卡一个优化步(128 条 completion)约 6 分钟,等于每步有 4 分钟在烧废题。

② 的题从哪来:第二阶段弃权 SFT 的自采样结果(data/processed/abstain_samples.jsonl,4000 题 × K=4,
由 build_distill.sample_teacher 对蒸馏模型采出)。build_abstain 只用 known(K 次全对)和 unknown(0 次对)
两类,「对 1~3 次」的 unstable 全部丢弃——它们既是天然的高方差题,又没被任何一个阶段训过,拿来做
GRPO 既不重叠也不浪费。判对与 build_abstain 同一把尺(严格可用协议 v3 的规则层,见 count_correct)。

产出 data/processed/grpo_train.jsonl(以及 --eval-n > 0 时的 grpo_eval.jsonl),每行:

    {"messages": [{"role": "user", "content": <与评测完全同一份提示词>}],
     "solution": "AC", "options": "{\\"A\\": \\"...\\", ...}", "id": "cmexam-train-123"}

  (options 落盘成 **JSON 字符串**而不是 JSON 对象,理由见 to_grpo_row 的注释:HF datasets 的 json
   builder 跨 block 不合并 struct 字段,选项数不齐的题会让整份数据集在加载期直接 cast 失败。)

三条设计约束:

  ① 提示词必须与评测逐字节相同 —— 复用 build_distill.render_prompt(它又复用 eval/run.py 的
     PROMPT_CHOICE/PROMPT_OPEN)。训练时优化的分布若与评测时的提示词不同,涨的分在考卷上兑不出来
     (这是 W2 蒸馏教材的坑 ③,已经付过一次学费)。

  ② 与弃权 SFT(第二阶段)的题不重叠 —— 两阶段用同一份题池、同一个 seed 洗牌:
     弃权阶段取 pick_questions(pool, 4000) 也就是洗牌后的前 4000,GRPO 取 [skip, skip+n)。
     同一批题先被 SFT 教「怎么答」再被 GRPO 按同一答案打分,等于让 RL 在已经背下来的题上刷奖励,
     测出来的提升是记忆不是策略。--eval-n 的评估题再往后顺延,与训练段同样不重叠。
     模式 ② 的不重叠来自另一条更强的理由:unstable 题被 build_abstain 整类丢弃,一行都没进教材。

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
import random
import sys
from collections import Counter
from pathlib import Path

from rich import print as rprint

from medforge.data.build_distill import (
    SEED,
    load_pool,
    load_samples_file,
    pick_questions,
    render_prompt,
)
from medforge.data.schema import Sample
from medforge.data.sources import ROOT
from medforge.verify.verifier import split_answer, verify_by_rule

PROCESSED = ROOT / "data" / "processed"
DEFAULT_POOL = PROCESSED / "abstain_pool.jsonl"
DEFAULT_OUT = PROCESSED / "grpo_train.jsonl"
DEFAULT_EVAL_OUT = PROCESSED / "grpo_eval.jsonl"
# 弃权 SFT(第二阶段)吃掉的题数:洗牌后的前 4000。GRPO 必须从这之后开始切。
DEFAULT_SKIP = 4000
DEFAULT_N = 6000
# 自采样的 K(build_distill.sample_teacher 的 --k):条数不足 K 的残缺题直接丢,
# 否则「2 条里对 1 条」会和「4 条里对 2 条」以同样的证据强度混进题单
DEFAULT_K = 4


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


def count_correct(sample: Sample, rows: list[dict]) -> int:
    """一道题的自采样里,规则层判对的条数。

    与 build_abstain 共用同一把尺的**收尾判定 + 规则层**(严格可用协议 v3),不另起一套判分;
    但**不套用**它那条「答卷里出现多个 </think> 就整条丢」(SAMPLE_GATES 的 double_close)——
    那道闸是为教材行的标签成对性写的(教材要把 <think>…</think> 原样喂给 SFT),难度筛选只数判对数,
    split_answer 取最后一个 </think> 之后的段落已经够用。实测全库 4000 题 × 4 条里仅 1 条命中这条闸,
    影响面是一道题:两边口径在此之外逐条一致。

    严格可用协议 v3 的规则层:先 split_answer(thinking=True)切出最后一个 </think> 之后的作答段;
    未收尾(finish_reason=length,或思考型输出里根本没有 </think>)直接不算对——**不进规则层**。
    这一条不是洁癖:截断答卷的末段多半是复读循环,extract 能从循环里刮出一个「答案:X」判成硬分
    (verifier 的截断守卫就是为它写的),放进来会把「没写完」的题冒充成「会做一半」的高方差题。

    verify_by_rule 返回 None 是**弃权不是判错**,这里只数判对,弃权与判错一起落在「不对」侧:
    难度筛选要的只是「组内会不会出现分歧」,弃权与答错在奖励表里本就是两个分数,分歧照样存在。
    """
    n = 0
    for r in rows:
        seg, unfinished = split_answer(
            r.get("answer", ""), finish_reason=r.get("finish_reason"), thinking=True
        )
        if unfinished is not None:
            continue
        v = verify_by_rule(sample, seg)
        if v is not None and v.correct is True:
            n += 1
    return n


def select_unstable(
    pool: list[Sample],
    rows_by_id: dict[str, list[dict]],
    *,
    k: int = DEFAULT_K,
    min_correct: int = 1,
    max_correct: int | None = None,
) -> tuple[list[Sample], Counter, dict[str, int]]:
    """自采样结果 → 「模型半会不会」的题。返回 (题, 判对数直方图, 漏斗计数)。

    max_correct 缺省 = k - 1,也就是「不全对」。min_correct=1 即「不全错」。两头都排掉的正是
    组内零方差的那批题(见模块 docstring 的 frac_reward_zero_std 0.60~0.69)。

    只收**恰好 k 条**的题:sample_teacher 对空答案不落盘、单条异常只计 failed,采样文件天然带残缺题;
    3 条里对 1 条与 4 条里对 1 条的难度不是一回事,混进来会让「1~3 次对」这个刻度失真。
    题的顺序是**采样文件里 id 的首次出现序**——落盘顺序是并发采样的产物,与题池序无关,
    但对同一份输入是固定的,后面的洗牌因此可复现。
    """
    if max_correct is None:
        max_correct = k - 1
    by_id = {s.id: s for s in pool}
    hist: Counter = Counter()
    funnel = {"seen": 0, "in_pool": 0, "with_k": 0, "unstable": 0}
    picked: list[Sample] = []
    for qid, rows in rows_by_id.items():
        funnel["seen"] += 1
        sample = by_id.get(qid)
        if sample is None:
            continue  # 采样文件比题池旧 / 换了 --source
        funnel["in_pool"] += 1
        if len(rows) != k:
            continue
        funnel["with_k"] += 1
        n_correct = count_correct(sample, rows)
        if not min_correct <= n_correct <= max_correct:
            continue
        hist[n_correct] += 1
        funnel["unstable"] += 1
        picked.append(sample)
    return picked, hist, funnel


def split_train_eval(
    samples: list[Sample], eval_n: int, seed: int = SEED
) -> tuple[list[Sample], list[Sample]]:
    """洗牌后前 eval_n 道作验证集,其余作训练集。

    洗的是 select_unstable 给出的顺序(采样文件首次出现序),用 random.Random(seed).shuffle 就地洗——
    换成 sample/sorted 会得到另一种排列,同一份输入就复现不出已经上过机的那份题单了。
    验证集取在洗牌之后而不是文件头部:采样是按题池顺序发的,文件头部那批题在难度上不是随机子集。
    """
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    return shuffled[eval_n:], shuffled[:eval_n]


def to_grpo_row(sample: Sample) -> dict:
    """ms-swift GRPO 一行。messages 只有 user——completion 由 rollout 现场生成,不存在 assistant 侧。

    options 存成 JSON 字符串是必须的,不是风格选择:ms-swift 用 hf_load_dataset("json", ...) 读
    (swift/dataset/loader.py:58),HF datasets 的 json builder 按约 10 MB 分块读,后一块出现前一块没有的
    struct 字段时不做 union 而是直接 `TypeError: Couldn't cast ... struct<A..F>`。CMExam 的选项键集合并不固定
    (normalize.py 会丢掉空选项,cmb-val 里实测有六选项题),6000 题的 jsonl 又正好落在这个体积带上——
    存成对象的话,炸不炸取决于洗牌后哪道多选项题排在第几行,本地永远测不出来,只在租卡机启动时随机翻车。
    存成字符串则列类型恒为 string,与选项数无关。奖励侧无需改动:grpo_reward._clean_options 接受 JSON 字符串。
    """
    return {
        "messages": [{"role": "user", "content": render_prompt(sample)}],
        "solution": gold_letters(sample),
        "options": json.dumps(dict(sample.options or {}), ensure_ascii=False),
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


def _run_unstable(args, pool: list[Sample]) -> int:
    """难度筛选模式:自采样 → 判对数 1~K-1 的题 → 洗牌 → 验证集 / 训练集。"""
    if not args.samples.exists():
        rprint(f"[red]✗ 自采样文件不存在: {args.samples}[/red]")
        return 2
    rows_by_id = load_samples_file(args.samples)
    max_correct = args.max_correct if args.max_correct is not None else args.k - 1
    picked, hist, funnel = select_unstable(
        pool, rows_by_id, k=args.k, min_correct=args.min_correct, max_correct=max_correct
    )
    if not picked:
        rprint(
            f"[red]✗ 没有一道题的判对数落在 [{args.min_correct}, {max_correct}]:"
            f"确认 --k 与采样时的 K 一致、题池 --source 与采样文件对得上[/red]"
        )
        return 2

    # eval_n 不设闸的话:负数会把训练集切成 shuffled[-3:] 这种残段、又因为 `eval_n > 0` 不落 eval 文件,
    # 超量则写出空训练题单——两种都返回 0、漏斗还打得挺像样,要到租卡机上 ms-swift 加载数据集才炸。
    if not 0 <= args.eval_n < len(picked):
        rprint(f"[red]✗ --eval-n 必须落在 [0, {len(picked)}) 内(unstable 题数),收到 {args.eval_n}[/red]")
        return 2

    train, ev = split_train_eval(picked, args.eval_n, args.seed)
    rows, dropped = build_rows(train)
    write_rows(rows, args.out)
    eval_rows, eval_dropped = build_rows(ev)
    if args.eval_n > 0:
        write_rows(eval_rows, args.eval_out)

    rprint(
        f"漏斗:采样 {funnel['seen']} 题 → 题池内 {funnel['in_pool']} → 恰好 {args.k} 条 "
        f"{funnel['with_k']} → 判对 {args.min_correct}~{max_correct} 次(unstable) {funnel['unstable']}"
        f" → 训练 {len(train)} / 验证 {len(ev)}"
    )
    rprint("判对数直方图:" + " · ".join(f"{c} 次对 {hist[c]} 题" for c in sorted(hist)))
    rprint(f"[green]✓[/green] 训练题单 {args.out}: {len(rows)} 行(丢弃非选择题 {dropped})")
    if args.eval_n > 0:
        rprint(f"[green]✓[/green] 验证题单 {args.eval_out}: {len(eval_rows)} 行(丢弃 {eval_dropped})")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GRPO 题单构造(第三阶段)")
    ap.add_argument("--pool", type=Path, default=DEFAULT_POOL, help="题池 jsonl(每行 Sample.to_dict())")
    ap.add_argument("--source", default="cmexam-train", help="只用题池里这个 source 的题")
    ap.add_argument(
        "--samples",
        type=Path,
        default=None,
        help="自采样文件(build_distill.sample_teacher 的形状):给了就走难度筛选模式,忽略 --skip/--n",
    )
    ap.add_argument("--k", type=int, default=DEFAULT_K, help="难度筛选:每题的采样条数,必须等于采样时的 K")
    ap.add_argument("--min-correct", type=int, default=1, help="难度筛选:保留的最小判对数(1 = 不全错)")
    ap.add_argument(
        "--max-correct", type=int, default=None, help="难度筛选:保留的最大判对数;缺省 = K-1(不全对)"
    )
    ap.add_argument("--skip", type=int, default=DEFAULT_SKIP, help="跳过洗牌后的前若干题(留给弃权 SFT)")
    ap.add_argument("--n", type=int, default=DEFAULT_N, help="训练题数;<= 0 表示取到题池末尾")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--eval-n",
        type=int,
        default=0,
        help="> 0 时切出这么多题作验证集(切片模式往训练段之后顺延,难度筛选模式取洗牌后的前若干道)",
    )
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

    if args.samples is not None:
        return _run_unstable(args, pool)

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

"""GRPO 题单构造测试:提示词与评测同源 · 与弃权阶段的 4000 题不重叠 · 同 seed 可复现。

这三条各自对应一个已经付过学费或者会直接毁掉实验的失效模式,不是「顺手测一下」。
"""

from __future__ import annotations

import json

from medforge.data import build_grpo as bg
from medforge.data.build_distill import pick_questions, render_prompt
from medforge.data.schema import Sample
from medforge.eval.run import PROMPT_CHOICE
from medforge.train import grpo_reward as gr

POOL_SIZE = 60
SKIP = 20
N = 25


def _pool(n: int = POOL_SIZE) -> list[Sample]:
    return [
        Sample(
            id=f"cmexam-train-{i}",
            source="cmexam-train",
            question=f"题号q{i}:下列哪项最可能?",
            gold="B" if i % 3 else "AC",
            options={"A": "甲", "B": "乙", "C": "丙", "D": "丁"},
        )
        for i in range(n)
    ]


def _write_pool(tmp_path, samples: list[Sample], extra: list[dict] | None = None):
    f = tmp_path / "pool.jsonl"
    rows = [s.to_dict() for s in samples] + (extra or [])
    f.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    return f


# ---------------------------------------------------------------- 提示词身份


def test_prompt_is_byte_identical_to_eval_prompt():
    """训练 user 必须与评测 user 逐字节相同——W2 坑 ③,训练/评测提示词一漂,涨的分兑不出来。"""
    s = _pool(1)[0]
    row = bg.to_grpo_row(s)
    assert row["messages"] == [{"role": "user", "content": render_prompt(s)}]
    assert row["messages"][0]["content"] == PROMPT_CHOICE.format(question=s.render_question())


def test_row_columns_match_reward_kwargs():
    """列名就是奖励函数的形参名(ms-swift 把额外列摊平成 kwargs 传进去)。"""
    s = _pool(2)[1]  # i=1 → 单选 gold "B";i=0 是多选 "AC"
    row = bg.to_grpo_row(s)
    assert set(row) == {"messages", "solution", "options", "id"}
    assert row["solution"] == "B"
    assert row["id"] == "cmexam-train-1"


def test_options_is_a_json_string_not_a_struct():
    """必须是字符串:HF datasets 的 json builder 跨 block 不合并 struct 字段,选项数不齐的题
    会让整份数据集在加载期 `Couldn't cast ... struct<A..F>` 直接失败(见 to_grpo_row 注释)。
    奖励侧 grpo_reward._clean_options 本来就吃 JSON 字符串。"""
    row = bg.to_grpo_row(_pool(2)[1])
    assert isinstance(row["options"], str)
    assert json.loads(row["options"]) == {"A": "甲", "B": "乙", "C": "丙", "D": "丁"}
    assert gr._clean_options(row["options"]) == {"A": "甲", "B": "乙", "C": "丙", "D": "丁"}


def test_gold_letters_normalizes_multi_answer():
    s = Sample(id="x", source="cmexam-train", question="q", gold="ca", options={"A": "1", "C": "3"})
    assert bg.gold_letters(s) == "AC"


# ---------------------------------------------------------------- 不重叠 / 可复现


def test_slice_is_disjoint_from_abstain_stage():
    """弃权 SFT 吃掉洗牌后的前 4000(此处按比例缩成 SKIP),GRPO 的切片不能碰到它们。"""
    pool = _pool()
    abstain = pick_questions(pool, SKIP)
    grpo = bg.pick_slice(pool, SKIP, N)
    assert len(abstain) == SKIP and len(grpo) == N
    assert not ({s.id for s in abstain} & {s.id for s in grpo})


def test_eval_slice_is_disjoint_from_train_slice():
    pool = _pool()
    train = bg.pick_slice(pool, SKIP, N)
    ev = bg.pick_slice(pool, SKIP + N, 10)
    assert len(ev) == 10
    assert not ({s.id for s in train} & {s.id for s in ev})
    assert not ({s.id for s in pick_questions(pool, SKIP)} & {s.id for s in ev})


def test_slice_is_deterministic_across_calls():
    pool = _pool()
    assert [s.id for s in bg.pick_slice(pool, SKIP, N)] == [s.id for s in bg.pick_slice(pool, SKIP, N)]
    # 换 seed 必须真的换一批题,否则「同 seed 才不重叠」这条约束是空的
    assert [s.id for s in bg.pick_slice(pool, SKIP, N, seed=7)] != [
        s.id for s in bg.pick_slice(pool, SKIP, N)
    ]


def test_slice_is_prefix_extensible():
    """先切 10 题试跑、再扩到 25 题,前 10 题必须还是同一批(与 pick_questions 的前缀性质一致)。"""
    pool = _pool()
    assert [s.id for s in bg.pick_slice(pool, SKIP, 10)] == [s.id for s in bg.pick_slice(pool, SKIP, N)][:10]


def test_slice_n_le_zero_takes_the_rest():
    pool = _pool()
    assert len(bg.pick_slice(pool, SKIP, 0)) == POOL_SIZE - SKIP


# ---------------------------------------------------------------- 过滤 / CLI


def test_build_rows_drops_non_choice_and_goldless():
    samples = _pool(2) + [
        Sample(id="open-1", source="cmexam-train", question="q", gold="肺栓塞"),  # 开放题
        Sample(id="bad-1", source="cmexam-train", question="q", gold="", options={"A": "1"}),  # 无金标
    ]
    rows, dropped = bg.build_rows(samples)
    assert dropped == 2
    assert [r["id"] for r in rows] == ["cmexam-train-0", "cmexam-train-1"]


def test_cli_writes_train_and_eval_files(tmp_path):
    pool_file = _write_pool(tmp_path, _pool())
    out = tmp_path / "grpo_train.jsonl"
    ev = tmp_path / "grpo_eval.jsonl"
    rc = bg.main(
        [
            "--pool",
            str(pool_file),
            "--skip",
            str(SKIP),
            "--n",
            str(N),
            "--out",
            str(out),
            "--eval-n",
            "10",
            "--eval-out",
            str(ev),
        ]
    )
    assert rc == 0
    train_rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    eval_rows = [json.loads(line) for line in ev.read_text(encoding="utf-8").splitlines()]
    assert len(train_rows) == N and len(eval_rows) == 10
    assert not ({r["id"] for r in train_rows} & {r["id"] for r in eval_rows})
    assert all(r["messages"][0]["role"] == "user" for r in train_rows)


def test_cli_filters_by_source(tmp_path):
    """题池里混进别的 source(评测集)时必须被挡在外面——不然直接污染。"""
    other = Sample(id="cmexam-1", source="cmexam", question="q", gold="A", options={"A": "1"}).to_dict()
    pool_file = _write_pool(tmp_path, _pool(30), extra=[other])
    out = tmp_path / "t.jsonl"
    assert bg.main(["--pool", str(pool_file), "--skip", "0", "--n", "0", "--out", str(out)]) == 0
    ids = {json.loads(line)["id"] for line in out.read_text(encoding="utf-8").splitlines()}
    assert "cmexam-1" not in ids and len(ids) == 30


def test_cli_missing_pool_returns_2(tmp_path):
    assert bg.main(["--pool", str(tmp_path / "nope.jsonl"), "--out", str(tmp_path / "o.jsonl")]) == 2

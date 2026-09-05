"""GRPO 题单构造测试:提示词与评测同源 · 与弃权阶段的 4000 题不重叠 · 同 seed 可复现。

这三条各自对应一个已经付过学费或者会直接毁掉实验的失效模式,不是「顺手测一下」。
"""

from __future__ import annotations

import json

import pytest

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


# ---------------------------------------------------------------- 难度筛选(unstable 池)


def _hard_pool(n: int = 6) -> list[Sample]:
    """全是单选、gold 恒为 B 的题:难度筛选只关心「对了几次」,不想被多选归一化干扰。"""
    return [
        Sample(
            id=f"cmexam-train-{i}",
            source="cmexam-train",
            question=f"题号q{i}:下列哪项最可能?",
            gold="B",
            options={"A": "甲", "B": "乙", "C": "丙", "D": "丁"},
        )
        for i in range(n)
    ]


def _row(qid: str, k: int, letter: str, *, finish_reason: str = "stop", closed: bool = True) -> dict:
    """一条自采样(build_distill.sample_teacher 的形状)。closed=False 模拟「思考没收尾」。"""
    think = f"<think>\n第 {k} 次推理"
    answer = f"{think}\n</think>\n\n答案:{letter}" if closed else f"{think}(写到一半就断了)"
    return {
        "id": qid,
        "k": k,
        "reasoning": "",
        "answer": answer,
        "finish_reason": finish_reason,
        "completion_tokens": 120,
    }


def _rows_for(qid: str, letters: list[str], **kw) -> list[dict]:
    return [_row(qid, k, letter, **kw) for k, letter in enumerate(letters)]


def _write_samples(tmp_path, rows: list[dict]):
    f = tmp_path / "samples.jsonl"
    f.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    return f


def test_unstable_keeps_only_partially_correct_questions():
    """全对与全错的题正是 frac_reward_zero_std 里那 60~69%:组内 8 个 rollout 同分、优势恒 0。"""
    pool = _hard_pool(4)
    rows_by_id = {
        "cmexam-train-0": _rows_for("cmexam-train-0", ["B", "B", "B", "B"]),  # 4/4 全对
        "cmexam-train-1": _rows_for("cmexam-train-1", ["A", "C", "D", "A"]),  # 0/4 全错
        "cmexam-train-2": _rows_for("cmexam-train-2", ["B", "A", "B", "C"]),  # 2/4 半对
        "cmexam-train-3": _rows_for("cmexam-train-3", ["B", "A", "A", "A"]),  # 1/4
    }
    picked, hist, funnel = bg.select_unstable(pool, rows_by_id, k=4)
    assert [s.id for s in picked] == ["cmexam-train-2", "cmexam-train-3"]
    assert dict(hist) == {2: 1, 1: 1}
    assert funnel == {"seen": 4, "in_pool": 4, "with_k": 4, "unstable": 2}


def test_unstable_drops_questions_short_of_k():
    """采样条数不足 K 的残缺题必须丢:「2 条里对 1 条」与「4 条里对 1 条」不是同一个难度刻度。"""
    pool = _hard_pool(2)
    rows_by_id = {
        "cmexam-train-0": _rows_for("cmexam-train-0", ["B", "A"]),  # 只落了 2 条
        "cmexam-train-1": _rows_for("cmexam-train-1", ["B", "A", "A", "A"]),
    }
    picked, _, funnel = bg.select_unstable(pool, rows_by_id, k=4)
    assert [s.id for s in picked] == ["cmexam-train-1"]
    assert funnel["in_pool"] == 2 and funnel["with_k"] == 1


def test_unstable_ignores_ids_missing_from_pool():
    pool = _hard_pool(1)
    rows_by_id = {
        "cmexam-train-0": _rows_for("cmexam-train-0", ["B", "A", "A", "A"]),
        "cmexam-train-999": _rows_for("cmexam-train-999", ["B", "A", "A", "A"]),  # 采样比题池旧
    }
    picked, _, funnel = bg.select_unstable(pool, rows_by_id, k=4)
    assert [s.id for s in picked] == ["cmexam-train-0"]
    assert funnel == {"seen": 2, "in_pool": 1, "with_k": 1, "unstable": 1}


def test_truncated_samples_never_count_as_correct():
    """finish_reason=length 的答卷不进规则层:末段的「答案:B」可能是复读循环里刮出来的硬分,
    把「没写完」冒充成「会做一半」会直接污染这个池子。"""
    s = _hard_pool(1)[0]
    assert bg.count_correct(s, _rows_for(s.id, ["B", "B", "B", "B"], finish_reason="length")) == 0
    mixed = _rows_for(s.id, ["B", "B"]) + _rows_for(s.id, ["B", "B"], finish_reason="length")
    assert bg.count_correct(s, mixed) == 2


def test_unclosed_think_counts_as_unfinished():
    """思考型口径:没有 </think> 就是没收尾,同样不算对(与 build_abstain / 评测同一把尺)。"""
    s = _hard_pool(1)[0]
    assert bg.count_correct(s, _rows_for(s.id, ["B", "B"], closed=False)) == 0


def test_all_truncated_question_is_not_unstable():
    """4 条全截断 → 判对 0 次 → 归「全错」被排掉,而不是变成一道假的高方差题。"""
    pool = _hard_pool(1)
    rows_by_id = {"cmexam-train-0": _rows_for("cmexam-train-0", ["B"] * 4, finish_reason="length")}
    picked, _, funnel = bg.select_unstable(pool, rows_by_id, k=4)
    assert picked == [] and funnel["unstable"] == 0


def test_correct_bounds_are_inclusive():
    """--min-correct / --max-correct 是闭区间:2/4 这道题在 [2,2] 里要留下,在 [3,3] 里要排掉。"""
    pool = _hard_pool(1)
    rows_by_id = {"cmexam-train-0": _rows_for("cmexam-train-0", ["B", "B", "A", "A"])}
    assert bg.select_unstable(pool, rows_by_id, k=4, min_correct=2, max_correct=2)[0]
    assert not bg.select_unstable(pool, rows_by_id, k=4, min_correct=3, max_correct=3)[0]


def test_split_train_eval_is_deterministic_and_disjoint():
    picked = _hard_pool(20)
    train, ev = bg.split_train_eval(picked, 5, seed=42)
    again_train, again_eval = bg.split_train_eval(picked, 5, seed=42)
    assert [s.id for s in train] == [s.id for s in again_train]
    assert [s.id for s in ev] == [s.id for s in again_eval]
    assert len(ev) == 5 and len(train) == 15
    assert not ({s.id for s in train} & {s.id for s in ev})
    assert {s.id for s in train} | {s.id for s in ev} == {s.id for s in picked}


def test_split_train_eval_actually_shuffles():
    """验证集不能是文件头部的前 5 题(采样按题池顺序发,头部那批在难度上不是随机子集);
    换 seed 也必须真的换一批,否则 seed 参数是摆设。"""
    picked = _hard_pool(20)
    _, ev = bg.split_train_eval(picked, 5, seed=42)
    assert [s.id for s in ev] != [s.id for s in picked[:5]]
    assert [s.id for s in ev] != [s.id for s in bg.split_train_eval(picked, 5, seed=7)[1]]


def test_cli_unstable_mode_writes_train_and_eval(tmp_path):
    pool = _hard_pool(8)
    pool_file = _write_pool(tmp_path, pool)
    rows: list[dict] = []
    for i in range(4):  # 0~3 号半对半错:只有它们该进题单
        rows += _rows_for(f"cmexam-train-{i}", ["B", "A", "B", "A"])
    for i in (4, 5):
        rows += _rows_for(f"cmexam-train-{i}", ["B"] * 4)
    for i in (6, 7):
        rows += _rows_for(f"cmexam-train-{i}", ["A"] * 4)
    out, ev = tmp_path / "t.jsonl", tmp_path / "e.jsonl"
    rc = bg.main(
        [
            "--pool",
            str(pool_file),
            "--samples",
            str(_write_samples(tmp_path, rows)),
            "--k",
            "4",
            "--min-correct",
            "1",
            "--max-correct",
            "3",
            "--eval-n",
            "1",
            "--seed",
            "42",
            "--out",
            str(out),
            "--eval-out",
            str(ev),
        ]
    )
    assert rc == 0
    train_rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    eval_rows = [json.loads(x) for x in ev.read_text(encoding="utf-8").splitlines()]
    assert len(train_rows) == 3 and len(eval_rows) == 1
    assert {r["id"] for r in train_rows} | {r["id"] for r in eval_rows} == {
        f"cmexam-train-{i}" for i in range(4)
    }
    assert not ({r["id"] for r in train_rows} & {r["id"] for r in eval_rows})


def test_cli_unstable_mode_row_format_matches_slice_mode(tmp_path):
    """两种取题模式只换「取哪些题」,行的形状必须一模一样——奖励插件按列名收 kwargs。"""
    pool = _hard_pool(2)
    pool_file = _write_pool(tmp_path, pool)
    rows = [r for i in range(2) for r in _rows_for(f"cmexam-train-{i}", ["B", "A", "B", "A"])]
    out = tmp_path / "t.jsonl"
    rc = bg.main(
        ["--pool", str(pool_file), "--samples", str(_write_samples(tmp_path, rows)), "--out", str(out)]
    )
    assert rc == 0
    written = {json.loads(x)["id"]: json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()}
    assert written == {s.id: bg.to_grpo_row(s) for s in pool}


def test_cli_unstable_mode_is_reproducible(tmp_path):
    """同一份输入 + 同一个 seed 必须逐字节复现:题单换了,GRPO 的曲线就没法跨轮次比。"""
    pool_file = _write_pool(tmp_path, _hard_pool(12))
    rows = [r for i in range(12) for r in _rows_for(f"cmexam-train-{i}", ["B", "A", "B", "A"])]
    samples_file = _write_samples(tmp_path, rows)
    written = []
    for tag in ("a", "b"):
        out, ev = tmp_path / f"t{tag}.jsonl", tmp_path / f"e{tag}.jsonl"
        rc = bg.main(
            [
                "--pool",
                str(pool_file),
                "--samples",
                str(samples_file),
                "--eval-n",
                "3",
                "--out",
                str(out),
                "--eval-out",
                str(ev),
            ]
        )
        assert rc == 0
        written.append((out.read_text(encoding="utf-8"), ev.read_text(encoding="utf-8")))
    assert written[0] == written[1]


def test_cli_unstable_mode_missing_samples_returns_2(tmp_path):
    pool_file = _write_pool(tmp_path, _hard_pool(2))
    rc = bg.main(
        [
            "--pool",
            str(pool_file),
            "--samples",
            str(tmp_path / "nope.jsonl"),
            "--out",
            str(tmp_path / "o.jsonl"),
        ]
    )
    assert rc == 2


def test_cli_unstable_mode_empty_selection_returns_2(tmp_path):
    """一道题都没选出来必须报错退出:静默产出空题单,要到 GPU 上 ms-swift 加载数据集才炸。"""
    pool_file = _write_pool(tmp_path, _hard_pool(2))
    rows = [r for i in range(2) for r in _rows_for(f"cmexam-train-{i}", ["B"] * 4)]
    rc = bg.main(
        [
            "--pool",
            str(pool_file),
            "--samples",
            str(_write_samples(tmp_path, rows)),
            "--out",
            str(tmp_path / "o.jsonl"),
        ]
    )
    assert rc == 2


def test_split_shuffle_matches_the_permutation_that_shipped():
    """把排列本身钉死:换成 random.Random(seed).sample(...) 或把 eval 取到尾部,都能过掉
    「跑两次结果一样」那类自比测试,却复现不出已经上过机的 812/100 那份题单。
    期望值是 random.Random(42).shuffle 在 10 个元素上的真实结果,手算于测试之外。"""
    picked = _hard_pool(10)
    train, ev = bg.split_train_eval(picked, 3, seed=42)
    assert [s.id for s in ev] == ["cmexam-train-7", "cmexam-train-3", "cmexam-train-2"]
    assert [s.id for s in train] == [
        "cmexam-train-8",
        "cmexam-train-5",
        "cmexam-train-6",
        "cmexam-train-9",
        "cmexam-train-4",
        "cmexam-train-0",
        "cmexam-train-1",
    ]


def test_unstable_order_is_samples_file_first_seen_order():
    """洗牌的输入序也要钉死:采样文件是并发落盘的,id 顺序既不是题池序也不是字典序。
    这里把三者错开——题池 0,1,2,采样文件 2,0,1——换成 sorted() 或按题池遍历就会当场露馅,
    而洗牌吃的是这个顺序,顺序一变,train/eval 的切分跟着变。"""
    pool = _hard_pool(3)
    rows: list[dict] = []
    for i in (2, 0, 1):
        rows += _rows_for(f"cmexam-train-{i}", ["B", "A", "B", "A"])
    rows_by_id: dict[str, list[dict]] = {}
    for r in rows:
        rows_by_id.setdefault(r["id"], []).append(r)
    picked, _, _ = bg.select_unstable(pool, rows_by_id, k=4)
    assert [s.id for s in picked] == ["cmexam-train-2", "cmexam-train-0", "cmexam-train-1"]


def _unstable_argv(tmp_path, pool_file, samples_file, eval_n: str) -> list[str]:
    return [
        "--pool",
        str(pool_file),
        "--samples",
        str(samples_file),
        "--eval-n",
        eval_n,
        "--out",
        str(tmp_path / "o.jsonl"),
        "--eval-out",
        str(tmp_path / "e.jsonl"),
    ]


@pytest.mark.parametrize("eval_n", ["-3", "4"])
def test_cli_unstable_mode_rejects_out_of_range_eval_n(tmp_path, eval_n):
    """负数会把训练集切成残段还不落 eval 文件,超量会写出空训练题单——两种都得当场退 2,
    否则要到租卡机上 ms-swift 加载数据集时才发现题单是空的/只剩几行。"""
    pool_file = _write_pool(tmp_path, _hard_pool(4))
    rows = [r for i in range(4) for r in _rows_for(f"cmexam-train-{i}", ["B", "A", "B", "A"])]
    rc = bg.main(_unstable_argv(tmp_path, pool_file, _write_samples(tmp_path, rows), eval_n))
    assert rc == 2
    assert not (tmp_path / "o.jsonl").exists()

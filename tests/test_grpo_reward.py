"""GRPO 选择性预测奖励插件测试:四个分支各自可复现,阈值语义(弃权 0 vs 答错 -1)可算。

判分链与评测共用 medforge.verify,所以这里重点测的是「分支怎么映射到数值」和
「ms-swift 传进来的批式 kwargs 怎么摊开」,不是重测一遍抽取器(那是 tests/test_extract.py 的活)。
"""

from __future__ import annotations

import pytest

from medforge.train import grpo_reward as gr

OPTS = {"A": "甲", "B": "乙", "C": "丙", "D": "丁"}


def done(body: str) -> str:
    """带 </think> 收尾的完整答卷:思考型口径下没有它就算未收尾。"""
    return f"先逐项分析。\n</think>\n\n{body}"


# ---------------------------------------------------------------- 四个分支


def test_correct_answer_gets_plus_one():
    assert gr.score_one(done("答案:B"), "B", OPTS) == 1.0


def test_wrong_answer_gets_minus_one():
    assert gr.score_one(done("答案:A"), "B", OPTS) == -1.0


def test_declared_abstain_gets_zero():
    assert gr.score_one(done("答案:不确定"), "B", OPTS) == 0.0


def test_unfinished_completion_gets_minus_one():
    """没有 </think>:思考流没收尾,不许从里面刮字母判分(验证器的截断守卫,与评测同口径)。"""
    assert gr.score_one("我再想想,答案:B", "B", OPTS) == -1.0


def test_finish_reason_length_overrides_a_declared_answer():
    """端点报告撞上 max_tokens 时无条件判未收尾,哪怕末段能刮出「答案:B」。"""
    assert gr.score_one(done("答案:B"), "B", OPTS, "length") == -1.0
    assert gr.score_one(done("答案:B"), "B", OPTS, "stop") == 1.0


def test_finished_but_undeclared_gets_unfinished_reward():
    """写完了却没做出可抽取的声明:与没写完同罚,不给「含糊其辞」留生态位。"""
    assert gr.score_one(done("我倾向于乙,但也说不好。"), "B", OPTS) == -1.0


def test_empty_completion_gets_unfinished_reward():
    assert gr.score_one("", "B", OPTS) == -1.0
    assert gr.score_one(None, "B", OPTS) == -1.0


def test_missing_gold_gets_unfinished_reward():
    assert gr.score_one(done("答案:B"), "", OPTS) == -1.0
    assert gr.score_one(done("答案:B"), None, OPTS) == -1.0


# ---------------------------------------------------------------- 多选 / 选项


def test_multi_answer_compares_letter_sets():
    assert gr.score_one(done("答案:ACD"), "ACD", {"A": "1", "B": "2", "C": "3", "D": "4"}) == 1.0
    assert gr.score_one(done("答案:ACD"), "DCA", {"A": "1", "B": "2", "C": "3", "D": "4"}) == 1.0  # 顺序无关
    assert (
        gr.score_one(done("答案:AC"), "ACD", {"A": "1", "B": "2", "C": "3", "D": "4"}) == -1.0
    )  # 少一个不算对


def test_options_with_none_values_are_dropped():
    """Arrow 往返会把选项数不齐的题补成 {"E": None};带 None 的键会放宽连写多选的合法性判定。"""
    assert gr._clean_options({"A": "甲", "B": "乙", "C": None}) == {"A": "甲", "B": "乙"}
    assert gr._clean_options({}) is None
    assert gr._clean_options(None) is None


def test_options_accept_json_string():
    assert gr._clean_options('{"A": "\\u7532", "B": "\\u4e59"}') == {"A": "甲", "B": "乙"}
    assert gr._clean_options("not json") is None


# ---------------------------------------------------------------- 阈值语义与环境变量


def test_abstain_is_optimal_exactly_below_half():
    """奖励表的全部意义:期望 2p-1 与弃权的 0 在 p=0.5 相交。数值一改,这条断言就会先响。"""

    def expected(p: float) -> float:
        return p * gr.CORRECT_REWARD + (1 - p) * gr.WRONG_REWARD

    assert expected(0.5) == gr.ABSTAIN_REWARD
    assert expected(0.4) < gr.ABSTAIN_REWARD < expected(0.6)


def test_env_overrides_are_read_at_construction(monkeypatch):
    monkeypatch.setenv("MEDFORGE_GRPO_ABSTAIN_REWARD", "0.2")
    monkeypatch.setenv("MEDFORGE_GRPO_WRONG_REWARD", "-3")
    monkeypatch.setenv("MEDFORGE_GRPO_UNFINISHED_REWARD", "-2")
    monkeypatch.setenv("MEDFORGE_GRPO_CORRECT_REWARD", "2")
    orm = gr.MedforgeSelectiveReward()
    assert (orm.correct, orm.abstain, orm.wrong, orm.unfinished) == (2.0, 0.2, -3.0, -2.0)


def test_malformed_env_falls_back_to_default(monkeypatch):
    """训练跑到一半因为一个 typo 崩掉最贵:写坏了退回默认值,不抛。"""
    monkeypatch.setenv("MEDFORGE_GRPO_WRONG_REWARD", "很负")
    assert gr.MedforgeSelectiveReward().wrong == gr.WRONG_REWARD


# ---------------------------------------------------------------- ms-swift 调用约定


def test_orm_call_maps_batched_kwargs():
    """ms-swift 传的是 reward_func(completions, **{列名: 与 completions 等长的 list})。"""
    orm = gr.MedforgeSelectiveReward()
    rewards = orm(
        [done("答案:B"), done("答案:A"), done("答案:不确定"), "还没想完"],
        solution=["B", "B", "B", "B"],
        options=[OPTS] * 4,
        finish_reason=["stop", "stop", "stop", "length"],
        id=["q1", "q2", "q3", "q4"],
        trainer_state=object(),  # 非列的运行期 kwarg,必须被 **kwargs 吞掉
    )
    assert rewards == [1.0, -1.0, 0.0, -1.0]


def test_orm_call_tolerates_missing_and_short_columns():
    orm = gr.MedforgeSelectiveReward()
    assert orm([done("答案:B")], solution=["B"]) == [1.0]  # 没有 options 列
    assert orm([done("答案:B")] * 2, solution=["B"]) == [1.0, -1.0]  # 列短了:缺的补 None → 无金标
    assert orm([done("答案:B")], solution="B") == [1.0]  # 标量广播


def test_registry_name_is_stable():
    """配置里 reward_funcs: [medforge_selective] 靠的就是这个名字。"""
    assert gr.REWARD_NAME == "medforge_selective"


@pytest.mark.parametrize(
    "body,expected",
    [
        ("答案:B", 1.0),
        ("最终答案:B", 1.0),
        ("答案:不确定", 0.0),
        ("答案:C", -1.0),
    ],
)
def test_declaration_variants(body, expected):
    assert gr.score_one(done(body), "B", OPTS) == expected

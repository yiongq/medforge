"""GRPO 选择性预测奖励插件测试:四个分支各自可复现,阈值语义(弃权 0 vs 答错 -1)可算。

判分链与评测共用 medforge.verify,所以这里重点测的是「分支怎么映射到数值」和
「ms-swift 传进来的批式 kwargs 怎么摊开」,不是重测一遍抽取器(那是 tests/test_extract.py 的活)。
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import warnings
from pathlib import Path

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


def test_missing_solution_column_raises_instead_of_silently_zeroing_the_signal():
    """整列缺失若按「无金标」处理,每条都是 -1、组内零方差、优势恒 0——300 步烧完卡还一句错都没有。
    单行缺金标仍然容忍(那只是一道题的事)。"""
    orm = gr.MedforgeSelectiveReward()
    with pytest.raises(ValueError, match="solution"):
        orm([done("答案:B")], options=[OPTS])
    # 单行缺金标:照常按「这条题给不出信号」记 -1,不影响同批其它样本
    assert orm([done("答案:B")] * 2, solution=[None, "B"], options=[OPTS] * 2) == [-1.0, 1.0]


def test_malformed_column_does_not_kill_the_run():
    """奖励函数抛一次异常 = 整场已付费的训练结束(compute_rewards_per_func 不捕获),
    所以畸形列只能赔一条样本的分。"""

    class Exploding:
        def __str__(self):  # _clean_options 里 str(v) 会踩到
            raise RuntimeError("boom")

    orm = gr.MedforgeSelectiveReward()
    with pytest.warns(UserWarning):
        assert orm([done("答案:B")], solution=["B"], options=[{"A": Exploding()}]) == [-1.0]


def test_batch_without_any_think_close_warns_about_enable_thinking():
    """最典型的口径级故障(模板退回非思考模式 → completion 里没有 </think>)不报错也不掉指标,
    只是优势恒 0。必须在前几步日志里能看见。"""
    orm = gr.MedforgeSelectiveReward()
    with pytest.warns(UserWarning, match="enable_thinking"):
        assert orm(["没有思考收尾", "也没有"], solution=["B", "B"]) == [-1.0, -1.0]


def test_all_wrong_batch_does_not_cry_wolf_about_the_template():
    """答错与未收尾默认同为 -1,所以判据只能是「一条 </think> 都没有」而不是「奖励全 -1」——
    否则模型一批全答错也会被误报成模板故障,喊多了这条告警就没人看了。"""
    orm = gr.MedforgeSelectiveReward()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # 任何告警都会让这条测试失败
        assert orm([done("答案:A")] * 2, solution=["B", "B"], options=[OPTS] * 2) == [-1.0, -1.0]


def test_registry_name_is_stable():
    """配置里 reward_funcs: [medforge_selective] 靠的就是这个名字。"""
    assert gr.REWARD_NAME == "medforge_selective"


# ---------------------------------------------------------------- ms-swift 的加载方式


def test_plugin_imports_as_a_toplevel_module_without_medforge_on_path(tmp_path):
    """ms-swift 的 import_external_file 是 `sys.path.insert(0, 本文件目录)` + 顶层名导入
    (swift/utils/utils.py:401),不是 `import medforge.train.grpo_reward`。
    所以 __package__ == "" 时的 sys.path 自举(parents[2] → src/)是 GPU 机上唯一的生路,
    写错一位就是启动即 ImportError——本机唯一能守住它的办法就是照着模拟一遍。"""
    plugin_dir = Path(gr.__file__).parent
    code = (
        "import importlib, sys;"
        f"sys.path.insert(0, {str(plugin_dir)!r});"
        "m = importlib.import_module('grpo_reward');"
        "assert m.__package__ in (None, ''), m.__package__;"
        "nl = chr(10);"  # -c 传的是单行,换行只能这样拼
        "assert m.score_one('想完了' + nl + '</think>' + nl * 2 + '答案:B', 'B', {'A':'1','B':'2'}) == 1.0"
    )
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    # check=False:下一行就在断言 returncode,check=True 只会把 stderr 吞掉、看不到失败原因
    r = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True, check=False
    )
    assert r.returncode == 0, r.stderr


def test_import_registers_into_the_swift_orms_registry(monkeypatch):
    """注册是 import 副作用:reward_funcs: [medforge_selective] 全靠这一步。
    本机没有 ms-swift,塞一个假的 swift.rewards 进 sys.modules 再 reload 即可验证这条线。"""
    import types

    orms: dict = {}
    fake = types.ModuleType("swift.rewards")
    fake.orms = orms
    fake.ORM = type("ORM", (), {"__init__": lambda self, args=None, **kw: setattr(self, "args", args)})
    monkeypatch.setitem(sys.modules, "swift", types.ModuleType("swift"))
    monkeypatch.setitem(sys.modules, "swift.rewards", fake)
    reloaded = importlib.reload(gr)
    try:
        assert orms[reloaded.REWARD_NAME] is reloaded.MedforgeSelectiveReward
    finally:
        monkeypatch.undo()
        importlib.reload(gr)  # 还原成不带假 swift 的版本,免得污染同进程里的别的测试


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

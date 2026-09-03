"""严格可用口径重算的单测:标签逻辑、退化指标、配对检验、落盘与回读。"""

import json

from medforge.data.schema import Sample
from medforge.eval.report import mcnemar_exact
from medforge.eval.usability import (
    REP_THRESHOLD,
    SetStats,
    load_tags,
    paired,
    tag_output,
    tag_run,
    tail_repetition,
)


def choice(qid: str, gold: str = "B") -> Sample:
    return Sample(id=qid, source="t", question="题干", gold=gold, options={"A": "甲", "B": "乙", "C": "丙"})


class TestTag:
    def test_finished_declared_correct_is_strict(self):
        t = tag_output(choice("q"), "推理推理</think>综上,答案:B", {"correct": True, "method": "rule"})
        assert (t.finished, t.declared, t.strict, t.wide) == (True, True, True, True)

    def test_unfinished_even_if_answer_scrapeable(self):
        # 没有 </think>,末段能刮出「答案:B」:原口径判对(wide),严格口径必须是未收尾
        out = "推理…候选是 B…答案:B 等等再想想…答案:B 等等再想想…" * 3
        t = tag_output(choice("q"), out, {"correct": True, "method": "rule"})
        assert t.wide is True and t.finished is False and t.strict is False
        assert t.rule_full is True  # 不设守卫的规则层会刮出来:这就是原口径多给的分

    def test_answer_only_in_thinking_is_not_declared(self):
        t = tag_output(choice("q"), "答案:B 但我再想想</think>各选项都有道理。", {"correct": True, "method": "llm"})
        assert t.finished is True and t.declared is False and t.strict is False

    def test_wrong_answer_declared_not_strict(self):
        t = tag_output(choice("q"), "</think>答案:C", {"correct": False, "method": "rule"})
        assert t.declared is True and t.strict is False

    def test_missing_output(self):
        t = tag_output(choice("q"), None, {"correct": None, "method": "missing"})
        assert not t.finished and not t.strict and t.chars == 0


class TestRepetition:
    def test_loop_is_degenerate(self):
        loop = "但根据问题中的选项,可能正确答案是A和G。" * 200
        assert tail_repetition(loop) >= REP_THRESHOLD

    def test_normal_prose_is_not(self):
        prose = "".join(f"第{i}点:患者出现{i * 7 % 13}项体征,需鉴别{i * 3 % 11}种病因。" for i in range(300))
        assert tail_repetition(prose) < REP_THRESHOLD

    def test_short_text_is_zero(self):
        assert tail_repetition("答案:B") == 0.0


def test_mcnemar_reference():
    # b=5, c=1:双侧精确 p = 2 * P(X<=1 | n=6, 0.5) = 2 * 7/64
    assert abs(mcnemar_exact(5, 1) - 14 / 64) < 1e-9
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(10, 10) == 1.0
    assert mcnemar_exact(54, 114) < 1e-5


def test_tag_run_roundtrip_and_paired(tmp_path):
    samples = {f"q{i}": choice(f"q{i}") for i in range(4)}
    for run, outs, scored in [
        ("a", ["</think>答案:B", "</think>答案:B", "答案:B 循环 " * 50, "</think>答案:C"],
         [True, True, True, False]),
        ("b", ["</think>答案:B", "</think>答案:C", "</think>答案:B", "</think>答案:B"],
         [True, False, True, True]),
    ]:
        d = tmp_path / run
        d.mkdir()
        (d / "syn.outputs.jsonl").write_text(
            "\n".join(json.dumps({"id": f"q{i}", "output": o}, ensure_ascii=False) for i, o in enumerate(outs)), "utf-8")
        (d / "syn.scored.jsonl").write_text(
            "\n".join(json.dumps({"id": f"q{i}", "correct": c, "method": "rule"}) for i, c in enumerate(scored)), "utf-8")
    ta = tag_run(tmp_path / "a", "syn", samples)
    tb = tag_run(tmp_path / "b", "syn", samples)
    sa, sb = SetStats.of("a", ta), SetStats.of("b", tb)
    assert (sa.wide, sa.strict) == (0.75, 0.5)   # q2 原口径刮出硬分,严格口径不算
    assert (sb.wide, sb.strict) == (0.75, 0.75)
    p = paired(ta, tb, "strict")
    assert (p.n, p.a_only, p.b_only, p.both) == (4, 1, 2, 1)  # q1 只有 a 对;q2、q3 只有 b 对
    # 落盘回读与现算一致
    assert [t.strict for t in load_tags(tmp_path / "a", "syn")] == [t.strict for t in ta]

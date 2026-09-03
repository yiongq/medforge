"""验证器规则层单测(不触网:只测 verify_by_rule)。

[review] 回归锁:开放题包含匹配的三类假阳性(近似药名/否定包裹/排除表述)
曾被判 True 并将进入 DPO 正例——现在规则层对非精确匹配一律弃权。
"""

from medforge.data.schema import Sample
from medforge.verify.verifier import split_answer, verify, verify_by_rule


def choice(gold: str) -> Sample:
    return Sample(
        id="t", source="t", question="题干",
        gold=gold, options={"A": "甲", "B": "乙", "C": "丙", "D": "丁"},
    )


def open_q(gold: str) -> Sample:
    return Sample(id="t", source="t", question="题干", gold=gold)


class TestChoiceRule:
    def test_correct(self):
        v = verify_by_rule(choice("C"), "综合分析,答案是 C。")
        assert v.correct is True and v.method == "rule"

    def test_incorrect(self):
        v = verify_by_rule(choice("C"), "最终答案:B")
        assert v.correct is False

    def test_multi_select_order_insensitive(self):
        assert verify_by_rule(choice("ACD"), "答案为 D、A、C").correct is True

    def test_abstain_when_no_declaration(self):
        assert verify_by_rule(choice("C"), "这道题很难,需要综合判断。") is None


class TestOpenRule:
    def test_exact_match(self):
        assert verify_by_rule(open_q("阿莫西林"), "最终答案:阿莫西林。").correct is True

    def test_exact_match_ignores_punct(self):
        assert verify_by_rule(open_q("急性心肌梗死"), r"\boxed{急性 心肌梗死}").correct is True

    def test_review_trailing_comment_still_correct(self):
        # 声明后带尾随评论:首子句截断候选应判对(否则 DPO 无仲裁模式会把对解打成负例)
        v = verify_by_rule(open_q("阿司匹林"), "最终答案:阿司匹林,不过我不确定这是否正确")
        assert v is not None and v.correct is True

    def test_review_superstring_drug_abstains(self):
        # 阿莫西林克拉维酸钾是另一种药:不能因字面包含判对
        assert verify_by_rule(open_q("阿莫西林"), "最终答案:阿莫西林克拉维酸钾") is None

    def test_review_superstring_diagnosis_abstains(self):
        assert verify_by_rule(open_q("甲状腺功能亢进"), "答案:甲状腺功能亢进危象") is None

    def test_review_negation_wrap_abstains(self):
        assert verify_by_rule(open_q("手术治疗"), "最终答案:无需手术治疗,建议保守观察") is None

    def test_review_exclusion_abstains(self):
        assert verify_by_rule(open_q("急性心肌梗死"), "答案:可排除急性心肌梗死,考虑主动脉夹层") is None


class TestTruncationGuard:
    """[review] 截断守卫:没写完的答卷不得进规则层——规则层会从复读循环末尾刮出「答案:X」。"""

    def test_length_finish_reason_is_unfinished_even_if_scrapeable(self):
        v = verify(choice("C"), "推理…候选 C…答案:C 等等…答案:C", allow_llm=False, finish_reason="length")
        assert v.correct is None and v.method == "unfinished"

    def test_thinking_mode_requires_close_tag(self):
        v = verify(choice("C"), "推理…答案:C", allow_llm=False, thinking=True)
        assert v.method == "unfinished"

    def test_rule_layer_only_sees_answer_segment(self):
        # 思考段里的「答案:B」是草稿,作答段的 C 才算
        v = verify(choice("C"), "先猜 答案:B,再核对…</think>答案:C", allow_llm=False)
        assert v.correct is True and v.method == "rule"

    def test_answer_only_in_thinking_abstains(self):
        v = verify(choice("C"), "答案:C 但我不确定</think>各选项都有道理。", allow_llm=False)
        assert v.correct is None and v.method == "abstain"

    def test_non_thinking_output_unchanged(self):
        assert verify(choice("C"), "综合分析,答案是 C。", allow_llm=False).correct is True

    def test_split_answer(self):
        assert split_answer("a</think>b") == ("b", None)
        assert split_answer("ab")[1] is None
        assert split_answer("ab", thinking=True)[1] == "no-think-close"
        assert split_answer("a</think>b</think>c")[1] == "think-close×2"

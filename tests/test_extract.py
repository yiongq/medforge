"""答案抽取规则层单测。

用例原则:每条规则配「该抽出」和「该弃权」两面——
抽取层的契约是「宁可弃权不可抽错」,弃权面和命中面同等重要。
带 [review] 标记的用例来自 W1a 双路审查的实测复现,是回归锁。
"""

from medforge.verify.extract import extract_choice, extract_text


class TestChoice:
    def test_chinese_answer_line(self):
        assert extract_choice("……综合分析,答案是 C。").value == "C"

    def test_answer_colon(self):
        assert extract_choice("最终答案:B").value == "B"

    def test_guxuan(self):
        assert extract_choice("此处应排除甲状腺功能亢进,故选 D。").value == "D"

    def test_boxed(self):
        assert extract_choice(r"所以 \boxed{A}").value == "A"

    def test_multi_select_sorted(self):
        assert extract_choice("正确答案为 D、A、C").value == "ACD"

    def test_english(self):
        assert extract_choice("Therefore, the answer is (b).").value == "B"

    def test_last_statement_wins(self):
        out = "初步判断答案是 A。但复查心电图特征后修正——最终答案:C"
        assert extract_choice(out).value == "C"

    def test_option_f_supported(self):
        # CMB 实测存在 6 选项题
        assert extract_choice("答案是 F").value == "F"

    def test_abstain_on_no_declaration(self):
        assert extract_choice("A 选项描述了缺铁性贫血,B 选项则是巨幼细胞贫血的表现。") is None

    def test_abstain_on_boxed_with_prose(self):
        assert extract_choice(r"\boxed{肺栓塞}") is None

    # ---- [review] 回归锁 ----

    def test_review_negated_declaration_not_extracted(self):
        # 「不应选 C」不是答案声明;真声明 B 在前也不能被它压过
        assert extract_choice("正确答案为 B,不应选 C。").value == "B"

    def test_review_xuanze_as_verb_not_trigger(self):
        # 「选择」是普通动词:后半句不是声明,不能覆盖前面的 B
        assert extract_choice("最终答案:B。选择 A 的考生忽略了颈静脉怒张。").value == "B"

    def test_review_negation_after_answer(self):
        assert extract_choice("综合分析答案是 D。注意不应选 A。").value == "D"

    def test_review_b_chao_not_letter(self):
        # 「B超」是医学名词不是选项 B → 弃权
        assert extract_choice("答案是 B超检查。") is None

    def test_review_ct_not_letter(self):
        assert extract_choice("答案:CT平扫即可确诊。") is None

    def test_review_english_continuation_not_merged(self):
        # 「C, as ...」的 a 不是多选延续
        assert extract_choice("答案是 C, as the ECG shows ST elevation.").value == "C"

    def test_review_boxed_text_wrapper(self):
        assert extract_choice(r"所以 \boxed{\text{C}}").value == "C"


class TestText:
    def test_boxed_text(self):
        assert extract_text(r"综上,\boxed{急性心肌梗死}").value == "急性心肌梗死"

    def test_answer_line(self):
        assert extract_text("最终答案:苯妥英钠。").value == "苯妥英钠"

    def test_abstain_on_free_ending(self):
        assert extract_text("因此患者最可能患有肺结核,建议进一步查痰。") is None

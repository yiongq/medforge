"""归一层单测:[review] 回归锁——全角分隔符、乱序多选、动态选项集。"""

from medforge.data.normalize import norm_cmb_val, norm_cmexam


def cmexam_row(answer: str) -> dict:
    return {
        "Question": "题干",
        "Answer": answer,
        "Options": [{"key": k, "value": f"选项{k}"} for k in "ABCDE"],
    }


class TestCmexam:
    def test_fullwidth_comma_multi_select_kept(self):
        # 'B，D'(全角逗号)曾被静默整行丢弃
        s = next(norm_cmexam([cmexam_row("B，D")], "test"))
        assert s.gold == "BD"

    def test_semicolon_variants(self):
        assert next(norm_cmexam([cmexam_row("B;D")], "test")).gold == "BD"

    def test_unordered_answer_sorted(self):
        # 'DA' 曾原样保留,违反 schema「多选按字典序连写」契约
        assert next(norm_cmexam([cmexam_row("DA")], "test")).gold == "AD"

    def test_invalid_answer_dropped(self):
        assert list(norm_cmexam([cmexam_row("X")], "test")) == []


class TestCmbVal:
    def test_six_option_question_kept(self):
        # CMB 实测存在 6 选项题(answer='BEF'),曾被 ABCDE 硬编码丢弃
        row = {
            "question": "题干",
            "answer": "BEF",
            "option": {k: f"选项{k}" for k in "ABCDEF"},
        }
        s = next(norm_cmb_val([row]))
        assert s.gold == "BEF" and "F" in s.options

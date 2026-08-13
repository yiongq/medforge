"""SFT/DPO 数据构造的纯逻辑单测(不依赖 data/raw,CI 可跑)。"""

import pytest

from medforge.data.build_dpo import make_pairs
from medforge.data.build_sft import mix
from medforge.data.schema import Sample


def _msg(i: int, tag: str) -> dict:
    return {"messages": [{"role": "user", "content": f"{tag}-{i}"},
                         {"role": "assistant", "content": "答"}]}


class TestMix:
    def test_ratio_and_determinism(self):
        med = [_msg(i, "med") for i in range(850)]
        gen = [_msg(i, "gen") for i in range(500)]
        out1 = mix(med, gen, 0.15)
        out2 = mix(med, gen, 0.15)
        n_gen = sum(1 for r in out1 if r["messages"][0]["content"].startswith("gen"))
        assert abs(n_gen / len(out1) - 0.15) < 0.01   # 占比命中目标
        assert out1 == out2                            # seed 固定,可复现
        assert len(out1) == 850 + n_gen

    def test_insufficient_general_raises(self):
        with pytest.raises(ValueError):
            mix([_msg(i, "m") for i in range(1000)], [_msg(0, "g")], 0.5)


class TestMakePairs:
    def sample(self) -> Sample:
        return Sample(id="q1", source="med-o1-verifiable", question="题", gold="阿司匹林")

    def test_pairs_from_mixed_solutions(self):
        sols = [
            "推理……最终答案:阿司匹林",           # 对
            "更长的推理……所以最终答案:阿司匹林",   # 对(更长)
            "推理……最终答案:氯吡格雷",           # 错
            "没有明确结论的解。",                  # 弃权 → 丢弃
        ]
        pairs = make_pairs(self.sample(), sols)
        assert len(pairs) == 2
        for p in pairs:
            assert "阿司匹林" in p["messages"][1]["content"]      # chosen 必须是对解
            assert "氯吡格雷" in p["rejected_response"]           # rejected 必须是错解
        # 对解按长度升序:第一对的 chosen 是最短对解
        assert pairs[0]["messages"][1]["content"] == "推理……最终答案:阿司匹林"

    def test_no_pair_when_all_correct(self):
        assert make_pairs(self.sample(), ["最终答案:阿司匹林"] * 3) == []

    def test_no_pair_when_all_wrong(self):
        assert make_pairs(self.sample(), ["最终答案:布洛芬"] * 3) == []

    def test_abstain_not_treated_as_wrong(self):
        # 弃权解不得进 rejected——判不准的解配进偏好对就是毒数据
        sols = ["最终答案:阿司匹林", "含糊其辞没有结论的长篇推理。"]
        assert make_pairs(self.sample(), sols) == []

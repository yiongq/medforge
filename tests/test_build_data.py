"""SFT/DPO 数据构造的纯逻辑单测(不依赖 data/raw,CI 可跑)。"""

import os
import subprocess
import sys

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
        # 采样解来自思考型模型:作答段在 </think> 之后(截断守卫与评测共用,见 classify_solution)
        sols = [
            "推理……</think>最终答案:阿司匹林",           # 对
            "更长的推理……</think>所以最终答案:阿司匹林",   # 对(更长)
            "推理……</think>最终答案:氯吡格雷",           # 错
            "</think>没有明确结论的解。",                  # 弃权 → 丢弃
        ]
        pairs = make_pairs(self.sample(), sols)
        assert len(pairs) == 2
        for p in pairs:
            assert "阿司匹林" in p["messages"][1]["content"]      # chosen 必须是对解
            assert "氯吡格雷" in p["rejected_response"]           # rejected 必须是错解
        # 对解按长度升序:第一对的 chosen 是最短对解
        assert pairs[0]["messages"][1]["content"] == "推理……</think>最终答案:阿司匹林"

    def test_no_pair_when_all_correct(self):
        assert make_pairs(self.sample(), ["</think>最终答案:阿司匹林"] * 3) == []

    def test_no_pair_when_all_wrong(self):
        assert make_pairs(self.sample(), ["</think>最终答案:布洛芬"] * 3) == []

    def test_review_truncated_solution_dropped(self):
        # [review W2] 撞上 max_tokens 没写出 </think> 的解:末段刮得出「最终答案」也不得进 chosen/rejected——
        # 拿半截思考流当教学信号正是 DPO 学到「写更长」的一个来源
        truncated = "候选是阿司匹林……最终答案:阿司匹林 等等再想想……最终答案:阿司匹林 等等"
        pairs = make_pairs(self.sample(), [truncated, "</think>最终答案:阿司匹林", "</think>最终答案:氯吡格雷"])
        assert len(pairs) == 1 and pairs[0]["messages"][1]["content"] == "</think>最终答案:阿司匹林"
        assert make_pairs(self.sample(), [truncated, "候选是氯吡格雷……最终答案:氯吡格雷 等等"]) == []

    def test_distinct_rejected_when_available(self):
        # [review] rng.choice 放回抽样曾让两对共享同一条 rejected,损失负例多样性
        sols = [
            "</think>最终答案:阿司匹林",
            "长一点的推理……</think>最终答案:阿司匹林",
            "</think>最终答案:氯吡格雷",
            "</think>最终答案:替格瑞洛",
        ]
        pairs = make_pairs(self.sample(), sols)
        assert len(pairs) == 2
        assert pairs[0]["rejected_response"] != pairs[1]["rejected_response"]

    def test_rejected_stable_across_hash_seeds(self):
        # [review] hash(str) 被 PYTHONHASHSEED 加盐,跨进程不稳定曾让「可复现」承诺失效
        code = (
            "from medforge.data.build_dpo import make_pairs\n"
            "from medforge.data.schema import Sample\n"
            "s = Sample(id='q1', source='t', question='题', gold='阿司匹林')\n"
            "sols = ['</think>最终答案:阿司匹林', '</think>最终答案:氯吡格雷', '</think>最终答案:替格瑞洛']\n"
            "print(make_pairs(s, sols)[0]['rejected_response'])\n"
        )
        outs = set()
        for seed in ("1", "42"):
            r = subprocess.run(
                [sys.executable, "-c", code],
                env={**os.environ, "PYTHONHASHSEED": seed},
                capture_output=True, text=True, check=True,
            )
            outs.add(r.stdout.strip())
        assert len(outs) == 1

    def test_abstain_not_treated_as_wrong(self):
        # 弃权解不得进 rejected——判不准的解配进偏好对就是毒数据
        sols = ["最终答案:阿司匹林", "含糊其辞没有结论的长篇推理。"]
        assert make_pairs(self.sample(), sols) == []

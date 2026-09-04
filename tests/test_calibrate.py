"""校准报告按判分路径分层 + Cohen's κ + false-accept/reject 方向。不触网(allow_llm=False)。"""

from medforge.verify.calibrate import Bucket, evaluate


def row(qid: str, gold: str, output: str, human: bool | None):
    return {
        "sample": {"id": qid, "source": "t", "question": "题", "gold": gold, "options": {"A": "", "B": "", "C": ""}},
        "output": output, "human_correct": human,
    }


def test_evaluate_buckets_by_method_and_directions():
    rows = [
        row("q1", "B", "</think>答案:B", True),      # 一致(机对/人对)
        row("q2", "B", "</think>答案:A", False),     # 一致(机错/人错)
        row("q3", "B", "</think>答案:B", False),     # false-accept:人判错、机判对
        row("q4", "B", "</think>答案:A", True),      # false-reject
        row("q5", "B", "</think>各选项都有道理", True),  # 规则层抽不出 → 弃权(不进分母)
        row("q6", "B", "</think>答案:B", None),      # 无标签 → 跳过
    ]
    buckets, abstain, skipped = evaluate(rows, allow_llm=False)
    assert set(buckets) == {"rule"} and abstain == 1 and skipped == 1
    b = buckets["rule"]
    assert (b.judged, b.agree, b.false_accept, b.false_reject) == (4, 2, 1, 1)
    assert abs(b.rate - 0.5) < 1e-9 and len(b.mismatches) == 2


def test_kappa_reference():
    # 全一致且两类都有 → κ=1;完全随机(一致率 50%,边际各半)→ κ=0
    assert abs(Bucket(agree=10, tp=5, tn=5).kappa - 1.0) < 1e-9
    assert abs(Bucket(agree=2, false_accept=1, false_reject=1, tp=1, tn=1).kappa - 0.0) < 1e-9
    # 标签不均衡时原始一致率虚高、κ 低:9 个都判对且人判对,1 个机判对人判错
    b = Bucket(agree=9, false_accept=1, tp=9, tn=0)
    assert b.rate == 0.9 and b.kappa == 0.0

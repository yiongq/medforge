"""报告层单测:Wilson CI 数学正确性 + [review] 弃权率透明化回归锁。"""

import json

from medforge.eval.report import (
    Paired,
    RunResult,
    benjamini_hochberg,
    holm,
    load_run,
    markdown_table,
    paired_counts,
)


def test_wilson_ci_reference_values():
    # 与 Brown et al. 参考值对照(审查手算确认过的三组)
    lo, hi = RunResult("r", 10, 5).wilson_ci()
    assert abs(lo - 0.2366) < 0.001 and abs(hi - 0.7634) < 0.001
    lo, hi = RunResult("r", 10, 0).wilson_ci()
    assert lo == 0.0 and abs(hi - 0.2775) < 0.001


def test_review_abstain_counted_and_visible(tmp_path):
    # 5 对 / 4 弃权 / 1 错:弃权计错进分母(acc=50%),但必须单独可见
    rows = [{"id": i, "correct": v} for i, v in enumerate([True] * 5 + [None] * 4 + [False])]
    p = tmp_path / "run.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    r = load_run(p, "base")
    assert r.n == 10 and r.correct == 5 and r.abstained == 4
    table = markdown_table([r])
    assert "弃权率" in table and "40.0%" in table


def test_review_unfinished_split_from_abstain(tmp_path):
    # 4 对 / 2 弃权 / 3 未收尾 / 1 缺失:全部计错进分母,但弃权 / 未收尾 / 缺失三列必须分开
    rows = (
        [{"id": i, "correct": True, "method": "rule"} for i in range(4)]
        + [{"id": 4 + i, "correct": None, "method": "abstain"} for i in range(2)]
        + [{"id": 6 + i, "correct": None, "method": "unfinished"} for i in range(3)]
        + [{"id": 9, "correct": None, "method": "missing"}]
    )
    p = tmp_path / "run.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    r = load_run(p, "base")
    assert (r.n, r.correct, r.abstained, r.unfinished, r.missing) == (10, 4, 2, 3, 1)
    table = markdown_table([r])
    assert "未收尾率" in table and "缺失率" in table and "30.0%" in table and "20.0%" in table and "10.0%" in table


def test_declared_abstain_is_subset_of_abstained(tmp_path):
    rows = [
        {"id": 0, "correct": True, "method": "rule"},
        {"id": 1, "correct": None, "method": "abstain", "detail": "declared"},
        {"id": 2, "correct": None, "method": "abstain", "detail": ""},
    ]
    p = tmp_path / "run.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    r = load_run(p, "x")
    assert (r.abstained, r.declared) == (2, 1)
    assert "主动 33.3%" in markdown_table([r])


def test_paired_se_ci_phi_mde():
    # 2000 题:双方都对 1600 / 只 A 对 100 / 只 B 对 150 / 都错 150
    p = Paired(n=2000, a_only=100, b_only=150, both=1600, neither=150)
    assert abs(p.delta - 0.025) < 1e-9
    # Σd = 50, Σd² = 250 → var = (250 − 50²/2000)/1999 = 0.124437…, se = √(var/2000) ≈ 0.00789
    assert abs(p.se - 0.00789) < 1e-4
    lo, hi = p.ci()
    assert lo < 0.025 < hi and abs((hi - lo) / 2 - 1.96 * p.se) < 1e-9
    assert 0 < p.phi < 1  # 错在同一批题上:正相关
    assert abs(p.mde() - 2.8 * (0.125 / 2000) ** 0.5) < 1e-9  # 不一致率 250/2000
    assert Paired(n=0, a_only=0, b_only=0, both=0, neither=0).mde() == 0.0


def test_paired_counts_and_correlation():
    a = {"q1": True, "q2": True, "q3": False, "q4": False}
    b = {"q1": True, "q2": False, "q3": True, "q4": False, "q5": True}  # q5 不在交集
    p = paired_counts(a, b)
    assert (p.n, p.a_only, p.b_only, p.both, p.neither) == (4, 1, 1, 1, 1)
    assert p.phi == 0.0


def test_holm_and_bh_reference():
    # 经典例子:p = [0.01, 0.02, 0.03, 0.04, 0.05],m=5
    ps = [0.01, 0.02, 0.03, 0.04, 0.05]
    # Holm:0.01 ≤ 0.05/5 ✓;0.02 ≤ 0.05/4=0.0125 ✗ → 后面全不过
    assert holm(ps) == [True, False, False, False, False]
    # BH:p_(k) ≤ k·0.05/5 → 0.01≤0.01 ✓ 0.02≤0.02 ✓ 0.03≤0.03 ✓ 0.04≤0.04 ✓ 0.05≤0.05 ✓ → 全过
    assert benjamini_hochberg(ps) == [True] * 5
    # 顺序无关:乱序输入返回按原位置的结论(m=2:0.01 ≤ 0.025 过,0.06 > 0.05 不过)
    assert holm([0.06, 0.01]) == [False, True]
    assert holm([]) == [] and benjamini_hochberg([]) == []

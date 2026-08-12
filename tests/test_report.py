"""报告层单测:Wilson CI 数学正确性 + [review] 弃权率透明化回归锁。"""

import json

from medforge.eval.report import RunResult, load_run, markdown_table


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

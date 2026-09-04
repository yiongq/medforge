"""蒸馏 2.0 数据构造测试:假 OpenAI 服务端 → 断点续采 → 五道闸门 → SFT 教材 + 报告。

全程不触网:老师是本机 http.server(参考 tests/test_eval_run.py 的 _Handler),
返回带 reasoning_content 的 message,模拟 DeepSeek 的「思考与答案分家」。
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import ClassVar

import pytest

from medforge.data import build_distill as bd
from medforge.data.schema import Sample
from medforge.eval.run import PROMPT_CHOICE

# 老师剧本:q1/q2 答对(gold B),q3 答错,qempty 返回空 content(模拟端点异常 → 采样失败)
LONG_ZH = "先逐项分析选项的临床意义。" * 20   # 260 字符 ≈ 162 token,过得了 --min-think-tokens
SCRIPT = {
    "q1": (LONG_ZH, "综合以上分析。\n答案:B"),
    "q2": (LONG_ZH, "综合以上分析。\n答案:B"),
    "q3": (LONG_ZH, "综合以上分析。\n答案:A"),
    "qempty": (LONG_ZH, ""),
}


class _Handler(BaseHTTPRequestHandler):
    seen_bodies: ClassVar[list[dict]] = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Handler.seen_bodies.append(body)
        prompt = body["messages"][0]["content"]
        qid = next((k for k in SCRIPT if f"题号{k}:" in prompt), "q1")
        reasoning, answer = SCRIPT[qid]
        # 每次采样的 seed 不同 → 思考略有差异,K 采样才不是同一条
        reasoning = f"{reasoning}(seed={body.get('seed')})"
        resp = {
            "choices": [{
                "message": {"role": "assistant", "content": answer, "reasoning_content": reasoning},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 300},
        }
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):  # 静音
        pass


@pytest.fixture()
def mock_server():
    _Handler.seen_bodies.clear()
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/v1"
    server.shutdown()


def _sample(qid: str, gold: str = "B") -> Sample:
    return Sample(
        id=qid, source="cmexam-train", question=f"题号{qid}:下列哪项正确?",
        gold=gold, options={"A": "甲", "B": "乙", "C": "丙"},
    )


def _write_pool(tmp_path, samples, source: str = "cmexam-train"):
    """题池文件(另一位 agent 的 build 产出的形状):每行 Sample.to_dict(),含别的 source 混在里面。"""
    pool = tmp_path / "pool.jsonl"
    with pool.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
        other = _sample("noise")
        other.source = "med-o1-verifiable"   # 不是 --source 的题,必须被过滤掉
        f.write(json.dumps(other.to_dict(), ensure_ascii=False) + "\n")
    return pool


def _row(qid: str, k: int = 0, *, reasoning: str = LONG_ZH, answer: str = "答案:B", finish: str = "stop") -> dict:
    return {"id": qid, "k": k, "reasoning": reasoning, "answer": answer,
            "finish_reason": finish, "completion_tokens": 300}


def _write_samples(tmp_path, rows):
    f = tmp_path / "distill_samples.jsonl"
    with f.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return f


def _build(tmp_path, samples, rows, **kw):
    """跑一遍 build:默认不混通用数据(通用数据要 data/raw,CI 没有)。返回 (stats, 产出行, 报告文本)。"""
    pool = _write_pool(tmp_path, samples)
    sfile = _write_samples(tmp_path, rows)
    out, report = tmp_path / "sft.jsonl", tmp_path / "report.md"
    kw.setdefault("general_ratio", 0.0)
    stats = bd.build_dataset(pool, sfile, out, report, **kw)
    lines = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    return stats, lines, report.read_text(encoding="utf-8")


# ------------------------------------------------------------------ 步骤 1:采样


def test_sample_separates_reasoning_and_resumes(mock_server, tmp_path):
    """思考与答案分开落盘;seed 逐次递增;断点续采只补缺的 (id, k)。"""
    samples = [_sample("q1"), _sample("q2")]
    out = tmp_path / "s.jsonl"
    r = bd.sample_teacher(samples, out, base_url=mock_server, api_key="k", model="teacher",
                          k_samples=2, concurrency=2, extra_body={"thinking": {"type": "enabled"}})
    assert (r["written"], r["failed"]) == (4, 0)
    rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert {(x["id"], x["k"]) for x in rows} == {("q1", 0), ("q1", 1), ("q2", 0), ("q2", 1)}
    one = rows[0]
    assert one["reasoning"].startswith("先逐项分析") and "答案:B" in one["answer"]
    assert "</think>" not in one["reasoning"] and one["finish_reason"] == "stop" and one["completion_tokens"] == 300
    # 老师参数与评测 v3 协议一致,思考开关走 extra_body,seed = 基础 seed + k
    b = _Handler.seen_bodies[0]
    assert (b["temperature"], b["top_p"], b["presence_penalty"]) == (1.0, 0.95, 1.5)
    assert b["thinking"] == {"type": "enabled"} and b["max_tokens"] == 8192
    assert b["messages"][0]["content"] == PROMPT_CHOICE.format(question=samples[0].render_question())
    assert {x["seed"] for x in _Handler.seen_bodies} == {bd.SEED, bd.SEED + 1}

    # 续采:全部命中缓存,一个请求都不发
    _Handler.seen_bodies.clear()
    r2 = bd.sample_teacher(samples, out, base_url=mock_server, api_key="k", model="teacher", k_samples=2)
    assert (r2["todo"], r2["reused"]) == (0, 4) and _Handler.seen_bodies == []

    # 删掉一条:只补这一条
    keep = [x for x in rows if (x["id"], x["k"]) != ("q2", 1)]
    out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in keep), encoding="utf-8")
    r3 = bd.sample_teacher(samples, out, base_url=mock_server, api_key="k", model="teacher", k_samples=2)
    assert (r3["todo"], r3["written"]) == (1, 1) and len(_Handler.seen_bodies) == 1


def test_sample_exits_when_failure_rate_exceeds(mock_server, tmp_path):
    """空答案不落盘(落了盘断点续采就永远是空的),失败率超 2% 直接退出。"""
    out = tmp_path / "s.jsonl"
    with pytest.raises(SystemExit):
        bd.sample_teacher([_sample("qempty")], out, base_url=mock_server, api_key="k", model="teacher", k_samples=1)
    assert not out.exists() or out.read_text(encoding="utf-8").strip() == ""

    # 1/60 = 1.7% ≤ 2%:容忍,失败的那条留待重采
    many = [_sample(f"q1{i}") for i in range(59)] + [_sample("qempty")]
    r = bd.sample_teacher(many, tmp_path / "s2.jsonl", base_url=mock_server, api_key="k",
                          model="teacher", k_samples=1, concurrency=4)
    assert (r["written"], r["failed"]) == (59, 1)


# ------------------------------------------------------------------ 闸门 ①~⑤(各一个正反例)


def test_gate1_structure(tmp_path):
    """① finish_reason 必须是 stop,reasoning/answer 非空。"""
    rows = [_row("q1", 0), _row("q1", 1, finish="length"), _row("q2", 0, reasoning=""), _row("q3", 0, answer="")]
    stats, lines, _ = _build(tmp_path, [_sample(q) for q in ("q1", "q2", "q3")], rows, accept="any")
    c = stats["counts"]
    assert (c["g1_finish"], c["g1_empty_reasoning"], c["g1_empty_answer"]) == (1, 1, 1)
    assert stats["n_med"] == 1 and len(lines) == 1   # 只有正例 q1#0 活下来


def test_gate2_answer_only_rule_layer(tmp_path):
    """② 只看 answer 段判分:判错剔除,规则层抽不出的一律弃权丢弃(不走 LLM),思考里的答案不算数。"""
    rows = [
        _row("q1", 0),                                    # 正例:答对
        _row("q2", 0, answer="综合判断。答案:A"),           # 反例:判错
        _row("q3", 0, answer="这题几个选项都有道理。"),      # 反例:抽不出 → 弃权丢弃
        # 反例:思考里写了正确答案,但 answer 段抽不出——判分只看 answer 段
        _row("q4", 0, reasoning=LONG_ZH + "答案:B", answer="综上所述,请自行判断。"),
    ]
    stats, lines, _ = _build(tmp_path, [_sample(f"q{i}") for i in (1, 2, 3, 4)], rows, accept="any")
    assert (stats["counts"]["g2_wrong"], stats["counts"]["g2_abstain"]) == (1, 2)
    assert stats["n_med"] == 1 and lines[0]["messages"][1]["content"].endswith("答案:B")


def test_gate3_majority_vs_any(tmp_path):
    """③ majority(默认)= 判对占严格多数才收,k=2 时必须两条全对;any = 任一判对即收。收的只有判对的那几条。"""
    rows = [_row("q1", 0), _row("q1", 1),                            # 两条全对
            _row("q2", 0), _row("q2", 1, answer="答案:A")]           # 一对一错
    samples = [_sample("q1"), _sample("q2")]
    maj, _maj_lines, _ = _build(tmp_path, samples, rows, accept="majority")
    assert maj["n_med"] == 2 and maj["n_questions_kept"] == 1 and maj["counts"]["g3_reject"] == 1

    anyv, any_lines, _ = _build(tmp_path, samples, rows, accept="any")
    assert anyv["n_med"] == 3 and anyv["n_questions_kept"] == 2 and "g3_reject" not in anyv["counts"]
    # any 模式收进来的仍然只有判对的那条:q2 的错解没有进教材
    assert all(m["messages"][1]["content"].rstrip().endswith("答案:B") for m in any_lines)


def test_gate4_length(tmp_path):
    """④ 长度在数据侧硬筛:思考过短/过长、答案过长、单条总长超训练 max_length,都不进教材。"""
    rows = [
        _row("q1", 0),                                                  # 正例
        _row("q2", 0, reasoning="太短了。"),                             # 反例:思考 < 100 token
        _row("q3", 0, reasoning="超长思考。" * 2000),                    # 反例:思考 > 4096 token
        _row("q4", 0, answer="废话。" * 400 + "答案:B"),                 # 反例:答案 > 512 token
    ]
    stats, _, _ = _build(tmp_path, [_sample(f"q{i}") for i in (1, 2, 3, 4)], rows, accept="any")
    c = stats["counts"]
    assert (c["g4_think_short"], c["g4_think_long"], c["g4_answer_long"]) == (1, 1, 1) and stats["n_med"] == 1

    # 总长闸门:思考/答案各自合规,但加起来超过训练 max_length —— 不依赖框架 truncation
    tight, _, _ = _build(tmp_path, [_sample("q1")], [_row("q1", 0)], accept="any", train_max_length=100)
    assert tight["counts"]["g4_total_long"] == 1 and tight["n_med"] == 0


def test_gate5_format_and_language(tmp_path):
    """⑤ answer 段必须含「答案:」字面(哪怕 boxed 也能判对),reasoning 的 CJK 占比要够。"""
    rows = [
        _row("q1", 0),                                                   # 正例
        _row("q2", 0, answer="综上,\\boxed{B}"),                         # 反例:判对但没有「答案:」字面
        _row("q3", 0, reasoning="Let me analyze each option carefully. " * 20),  # 反例:英文思考
    ]
    stats, _, _ = _build(tmp_path, [_sample(f"q{i}") for i in (1, 2, 3)], rows, accept="any")
    assert (stats["counts"]["g5_no_literal"], stats["counts"]["g5_zh_ratio"]) == (1, 1) and stats["n_med"] == 1


# ------------------------------------------------------------------ 成品形状与选样策略


def test_think_open_tag_and_prompt_match_eval(tmp_path):
    """成品必须是完整的 <think>…</think> 对(老师答卷里只有收尾标签,开标签要补);
    user 必须与评测同一份提示词——8 月 SFT「训练裸题干 / 评测带格式指令」的坑。"""
    s = _sample("q1")
    _, lines, _ = _build(tmp_path, [s], [_row("q1", 0)], accept="any")
    msg = lines[0]["messages"]
    assert msg[0]["content"] == PROMPT_CHOICE.format(question=s.render_question())
    assert "答案:X" in msg[0]["content"] and s.render_question() in msg[0]["content"]   # 60 字格式指令在里面
    a = msg[1]["content"]
    assert a.startswith("<think>\n") and a.count("<think>") == 1 and a.count("</think>") == 1
    assert a.split("\n</think>")[0].endswith(LONG_ZH) and a.endswith("\n</think>\n\n答案:B")


def test_max_per_question_takes_median_not_shortest(tmp_path):
    """每题保留 --max-per-question 条,按思考长度取中位附近——**不取最短**(build_dpo 的坑)。"""
    lens = {0: 200, 1: 400, 2: 800}   # 三条都判对,长度递增
    rows = [_row("q1", k, reasoning="中位测试用的中文思考。" * (n // 10)) for k, n in lens.items()]
    stats, lines, _ = _build(tmp_path, [_sample("q1")], rows, accept="any", max_per_question=1)
    assert stats["n_med"] == 1 and stats["counts"]["cap"] == 2
    kept = lines[0]["messages"][1]["content"]
    mid = "中位测试用的中文思考。" * 40
    assert mid in kept                                       # 取到了中位那条
    assert len(kept) > len("中位测试用的中文思考。" * 20) + 40  # 不是最短那条

    # 纯函数层面再钉一次:候选比上限多时,最短的永远不在结果里
    cand = [{"reasoning": "x" * n, "k": i} for i, n in enumerate((10, 20, 30, 40))]
    picked = bd.pick_median(cand, 2)
    assert [len(r["reasoning"]) for r in picked] == [20, 30]
    assert bd.pick_median(cand[:2], 2) == cand[:2]   # 候选不够时原样返回


def test_report_matches_dataset(tmp_path):
    """报告的漏斗、条数、覆盖题数、按 k 的通过率必须与产出文件对得上。"""
    rows = [_row("q1", 0), _row("q1", 1), _row("q2", 0), _row("q2", 1, answer="答案:A"),
            _row("q3", 0, finish="length"), _row("q3", 1, reasoning="短。"),
            _row("qX", 0)]   # 题池里没有的 id
    stats, lines, report = _build(tmp_path, [_sample(f"q{i}") for i in (1, 2, 3)], rows, accept="any")
    assert stats["n_rows"] == 7 and stats["n_med"] == len(lines) == 3 and stats["n_questions_kept"] == 2
    assert stats["counts"]["pool_missing"] == 1

    # 漏斗:逐行累减到最后一行的「剩余」= 医疗样本条数
    table = [ln for ln in report.splitlines() if ln.startswith("| ") and ln.count("|") == 4]
    remain = [int(ln.split("|")[3]) for ln in table[1:]]   # 跳过表头分隔后的首行是「剔除/剩余」数据行
    assert remain[-1] == stats["n_med"]
    dropped = sum(int(ln.split("|")[2]) for ln in table[1:])
    assert stats["n_rows"] - dropped == stats["n_med"]

    assert f"最终 {stats['n_med']} 条医疗样本,覆盖 {stats['n_questions_kept']} 道题" in report
    assert "| 0 | 4 | 2 | 50.0% | 2 |" in report      # k=0:采 4 条(含题池外的 qX)、判对 2、入选 2
    assert "| 1 | 3 | 2 | 66.7% | 1 |" in report      # k=1:采 3 条、判对 2(其中一条被长度闸门拦下)、入选 1
    assert "p50" in report and "TODO(人工抽检" in report and "答案对但推理不成立" in report
    # 长度分布与实际样本一致
    assert f"| 思考(reasoning) | {bd._pct(stats['think_tokens'], 0.5)} |" in report


def test_general_replay_mix(tmp_path):
    """--general-ratio 复用 build_sft.mix 的公式:通用条数 = 医疗 × r/(1-r),整体打散。"""
    rows = [_row(f"q{i}", 0) for i in range(1, 5)]
    general = [{"messages": [{"role": "user", "content": f"通用{i}"}, {"role": "assistant", "content": "好"}]}
               for i in range(50)]
    stats, lines, report = _build(tmp_path, [_sample(f"q{i}") for i in range(1, 5)], rows,
                                  accept="any", general_ratio=0.5, general_loader=lambda: general)
    assert (stats["n_med"], stats["n_general"], stats["n_total"]) == (4, 4, 8) and len(lines) == 8
    assert "混入通用回放 4 条,合计 8 条" in report


# ------------------------------------------------------------------ CLI 串联


def test_cli_sample_then_build(mock_server, tmp_path, monkeypatch):
    """两步 CLI 串起来:采样 → 构造,产物与报告都在,教材条数与报告一致。"""
    samples = [_sample("q1"), _sample("q2"), _sample("q3")]   # q3 答错,majority 下整题落选
    pool = _write_pool(tmp_path, samples)
    sfile, out, report = tmp_path / "s.jsonl", tmp_path / "sft.jsonl", tmp_path / "r.md"
    monkeypatch.setenv("FAKE_TEACHER_KEY", "sk-test")
    bd.main(["sample", "--endpoint", mock_server, "--model", "teacher", "--api-key-env", "FAKE_TEACHER_KEY",
             "--pool", str(pool), "--samples-file", str(sfile), "--n-questions", "3", "--k-samples", "2",
             "--concurrency", "2"])
    assert len(sfile.read_text(encoding="utf-8").splitlines()) == 6
    bd.main(["build", "--pool", str(pool), "--samples-file", str(sfile), "--out", str(out),
             "--report", str(report), "--general-ratio", "0"])
    lines = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 4 and "最终 4 条医疗样本,覆盖 2 道题" in report.read_text(encoding="utf-8")
    assert all(m["messages"][1]["content"].startswith("<think>\n") for m in lines)
    # 题池里 source 不符的题从来没被采样过
    assert all("题号noise" not in b["messages"][0]["content"] for b in _Handler.seen_bodies)


def test_cli_sample_requires_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("MISSING_KEY", raising=False)
    pool = _write_pool(tmp_path, [_sample("q1")])
    with pytest.raises(SystemExit):
        bd.main(["sample", "--endpoint", "http://127.0.0.1:1/v1", "--model", "t",
                 "--api-key-env", "MISSING_KEY", "--pool", str(pool)])

"""评测执行器全链路集成测试:假 OpenAI 服务端 → 生成 → 判分 → 报告。

不依赖 data/raw(CI 无数据),样本在测试内合成;服务端按题号回答,
刻意混入答对/答错/无声明三种形态,验证判分与弃权口径落到文件。
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import ClassVar

import pytest

from medforge.data.schema import Sample
from medforge.eval.report import load_run
from medforge.eval.run import run_set

# 服务端剧本:q1 答对 / q2 答错 / q3 无答案声明(→弃权计错)/ q4 撞上限(finish_reason=length → 未收尾)
SCRIPT = {
    "q1": ("分析各选项后可以确定。答案:B", "stop"),
    "q2": ("综合判断。答案:A", "stop"),
    "q3": ("这道题比较复杂,各选项都有道理。", "stop"),
    "q4": ("候选是 B…等等再想想…答案:B 等等再想想…答案:B", "length"),
}


class _Handler(BaseHTTPRequestHandler):
    seen_prompts: ClassVar[list[str]] = []  # 记录收到的提示词,测试提示词变体与 budget forcing 的裸 prompt

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path.endswith("/completions") and "messages" not in body:
            # budget forcing 的续写请求:裸 prompt 以「答案:」结尾,续写一个字母
            _Handler.seen_prompts.append(body["prompt"])
            # 续写常常写完字母还接着解释,32 token 上限一到 finish_reason 就是 length——这不等于未收尾
            resp = {"choices": [{"text": " B\n\n**推理过程**:1. 分析", "finish_reason": "length"}], "usage": {"completion_tokens": 32}}
        else:
            prompt = body["messages"][0]["content"]
            _Handler.seen_prompts.append(prompt)
            answer, reason = next((v for k, v in SCRIPT.items() if f"题号{k}" in prompt), ("答案:E", "stop"))
            resp = {
                "choices": [{"message": {"role": "assistant", "content": answer}, "finish_reason": reason}],
                "usage": {"prompt_tokens": 10, "completion_tokens": len(answer)},
            }
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):  # 静音,别刷测试输出
        pass


@pytest.fixture()
def mock_server():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{server.server_port}/v1"
    server.shutdown()


def _sample(qid: str, gold: str) -> Sample:
    return Sample(
        id=qid, source="synthetic", question=f"题号{qid}:下列哪项正确?",
        gold=gold, options={"A": "甲", "B": "乙", "C": "丙"},
    )


def test_run_set_end_to_end(mock_server, tmp_path):
    samples = [_sample("q1", "B"), _sample("q2", "B"), _sample("q3", "B"), _sample("q4", "B")]
    scored = run_set(
        "synthetic", samples, tmp_path,
        base_url=mock_server, model="mock", concurrency=2,
        allow_llm_judge=False,  # 测试不触真网:规则层弃权就落 null
    )
    r = load_run(scored, "test")
    # q1 对 / q2 错 / q3 弃权计错 / q4 撞上限:末段能刮出「答案:B」也不给分,且与弃权分列
    assert (r.n, r.correct, r.abstained, r.unfinished) == (4, 1, 1, 1)

    rows = {json.loads(line)["id"]: json.loads(line) for line in scored.read_text().splitlines()}
    assert rows["q1"]["correct"] is True and rows["q3"]["correct"] is None
    assert rows["q4"]["method"] == "unfinished" and rows["q4"]["finish_reason"] == "length"
    # outputs.jsonl 落了 finish_reason 与 completion_tokens,第三方不必再猜「未收尾率」
    out = {json.loads(line)["id"]: json.loads(line) for line in (tmp_path / "synthetic.outputs.jsonl").read_text().splitlines()}
    assert out["q4"]["finish_reason"] == "length" and out["q1"]["completion_tokens"] > 0


def test_resume_skips_generated(mock_server, tmp_path):
    samples = [_sample("q1", "B")]
    run_set("s", samples, tmp_path, base_url=mock_server, model="mock", allow_llm_judge=False)
    # 篡改已落盘的输出,重跑:应复用而不是重新生成(值保持篡改后的)
    out_file = tmp_path / "s.outputs.jsonl"
    out_file.write_text(json.dumps({"id": "q1", "output": "答案:C"}, ensure_ascii=False) + "\n", encoding="utf-8")
    scored = run_set("s", samples, tmp_path, base_url=mock_server, model="mock", allow_llm_judge=False)
    row = json.loads(scored.read_text().splitlines()[0])
    assert row["correct"] is False  # 复用了篡改后的 C(≠gold B),证明没有重新生成

def test_protocol_fingerprint_rejects_mixed_decoding(tmp_path):
    from medforge.eval.run import check_protocol

    meta = {"model": "m", "max_tokens": 8192, "temperature": 0.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0,
            "presence_penalty": 0.0, "seed": 42, "prompt_sha": "abcd1234", "samples": {"cmexam": 2000},
            "limit": 0, "thinking": "on", "llm_judge": True, "git": "aaa", "created": "t1"}
    check_protocol(tmp_path, meta)  # 首次:写 run_meta.json
    check_protocol(tmp_path, {**meta, "git": "bbb", "created": "t2"})  # 同协议、换了提交:放行并追加 history
    saved = json.loads((tmp_path / "run_meta.json").read_text())
    assert [h["git"] for h in saved["history"]] == ["aaa", "bbb"] and saved["legacy"] is False
    for bad in ({"temperature": 0.6}, {"samples": {"cmexam": 500}}, {"thinking": "off"}, {"prompt_sha": "ffff0000"}):
        with pytest.raises(SystemExit):
            check_protocol(tmp_path, {**meta, **bad})  # 换了协议还往同一目录写:拒绝


def test_protocol_refuses_legacy_archive_without_meta(tmp_path):
    # W2 之前的存档目录:有答卷、没指纹——默认拒绝,--adopt-legacy 才补写并标记 legacy
    from medforge.eval.run import check_protocol

    (tmp_path / "cmexam.outputs.jsonl").write_text('{"id": "q1", "output": "答案:B"}\n', encoding="utf-8")
    meta = {"model": "m", "git": "aaa", "created": "t1"}
    with pytest.raises(SystemExit):
        check_protocol(tmp_path, meta)
    check_protocol(tmp_path, meta, adopt_legacy=True)
    assert json.loads((tmp_path / "run_meta.json").read_text())["legacy"] is True


def test_thinking_mode_changes_archive_scoring(mock_server, tmp_path):
    # 旧格式答卷(无 finish_reason)+ 未收尾复读流:thinking=True 判未收尾,auto 会放过(见 verifier)
    samples = [_sample("q1", "B")]
    loop = "候选是 B…等等再想想…答案:B 等等再想想…答案:B"
    (tmp_path / "s.outputs.jsonl").write_text(json.dumps({"id": "q1", "output": loop}, ensure_ascii=False) + "\n", "utf-8")
    r_on = load_run(run_set("s", samples, tmp_path, base_url=mock_server, model="mock", allow_llm_judge=False, thinking=True), "on")
    r_auto = load_run(run_set("s", samples, tmp_path, base_url=mock_server, model="mock", allow_llm_judge=False), "auto")
    assert (r_on.correct, r_on.unfinished) == (0, 1) and (r_auto.correct, r_auto.unfinished) == (1, 0)


def test_budget_forcing_rescues_truncated_answer(mock_server, tmp_path):
    # q4 撞上限(finish_reason=length):plain 模式判未收尾;budget-forcing 接回思考流强写「</think>\n\n答案:」再续写
    from medforge.eval.run import FORCE_PREFIX

    _Handler.seen_prompts.clear()
    samples = [_sample("q4", "B")]
    scored = run_set("f", samples, tmp_path, base_url=mock_server, model="mock", allow_llm_judge=False,
                     thinking=True, gen={"mode": "budget-forcing"})
    row = json.loads(scored.read_text().splitlines()[0])
    assert row["correct"] is True and row["forced"] is True and row["finish_reason"] == "forced-length"
    out = json.loads((tmp_path / "f.outputs.jsonl").read_text().splitlines()[0])
    assert "</think>\n\n答案: B" in out["output"] and out["forced"] is True
    # 修复前落盘的 forced 行(finish_reason 还是裸 length)重判时同样不得判未收尾
    out["finish_reason"] = "length"
    (tmp_path / "f.outputs.jsonl").write_text(json.dumps(out, ensure_ascii=False) + "\n", "utf-8")
    scored = run_set("f", samples, tmp_path, base_url=mock_server, model="mock", allow_llm_judge=False,
                     thinking=True, gen={"mode": "budget-forcing"})
    assert json.loads(scored.read_text().splitlines()[0])["correct"] is True
    raw = [p for p in _Handler.seen_prompts if p.startswith("<|im_start|>user")]
    assert len(raw) == 1 and raw[0].startswith(FORCE_PREFIX.split("{prompt}")[0]) and raw[0].endswith("\n</think>\n\n答案:")


def test_abstain_prompt_variant(mock_server, tmp_path):
    _Handler.seen_prompts.clear()
    run_set("a", [_sample("q1", "B")], tmp_path, base_url=mock_server, model="mock", allow_llm_judge=False,
            gen={"prompt_variant": "abstain"})
    assert any("答案:不确定" in p for p in _Handler.seen_prompts)


def test_main_wires_mode_prompt_and_fingerprint(mock_server, tmp_path, monkeypatch):
    """[review] --mode / --prompt / --min-p 曾只挂在 argparse 上、没传进 gen 与 meta:forcing 与 abstain 臂会静默按普通模式跑。
    通过 argparse 驱动 main(),断言指纹与实际请求都带上了这些参数。"""
    import sys

    from medforge.data import sources
    from medforge.eval import run as run_mod

    monkeypatch.setattr(sources, "ROOT", tmp_path)
    monkeypatch.setattr(sources, "EVAL_SOURCES", {"syn": ("x", "y", "z")})
    monkeypatch.setattr(sources, "load_source", lambda name: [_sample("q4", "B")])
    _Handler.seen_prompts.clear()
    argv = ["run", "--endpoint", mock_server, "--model", "mock", "--run-name", "wire", "--sets", "syn",
            "--no-llm-judge", "--mode", "budget-forcing", "--prompt", "abstain", "--min-p", "0.05", "--limit", "1",
            "--timeout", "42"]
    monkeypatch.setattr(sys, "argv", argv)
    run_mod.main()
    meta = json.loads((tmp_path / "reports" / "runs" / "wire" / "run_meta.json").read_text())
    assert (meta["mode"], meta["prompt"], meta["min_p"], meta["limit"]) == ("budget-forcing", "abstain", 0.05, 1)
    assert meta["prompt_sha"] == run_mod.prompt_sha("abstain") and "prompt_variant" not in meta and "timeout" not in meta
    assert any("答案:不确定" in p for p in _Handler.seen_prompts)            # 弃权变体真的发出去了
    assert any(p.startswith("<|im_start|>user") for p in _Handler.seen_prompts)  # forcing 的裸 prompt 真的发出去了
    summary = (tmp_path / "reports" / "runs" / "wire" / "summary.md").read_text()
    assert "mode=budget-forcing" in summary and "prompt=abstain" in summary

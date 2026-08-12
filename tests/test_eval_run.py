"""评测执行器全链路集成测试:假 OpenAI 服务端 → 生成 → 判分 → 报告。

不依赖 data/raw(CI 无数据),样本在测试内合成;服务端按题号回答,
刻意混入答对/答错/无声明三种形态,验证判分与弃权口径落到文件。
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from medforge.data.schema import Sample
from medforge.eval.report import load_run
from medforge.eval.run import run_set

# 服务端剧本:q1 答对 / q2 答错 / q3 无答案声明(→弃权计错)
SCRIPT = {
    "q1": "分析各选项后可以确定。答案:B",
    "q2": "综合判断。答案:A",
    "q3": "这道题比较复杂,各选项都有道理。",
}


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = body["messages"][0]["content"]
        answer = next((v for k, v in SCRIPT.items() if f"题号{k}" in prompt), "答案:E")
        resp = {"choices": [{"message": {"role": "assistant", "content": answer}}]}
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
    samples = [_sample("q1", "B"), _sample("q2", "B"), _sample("q3", "B")]
    scored = run_set(
        "synthetic", samples, tmp_path,
        base_url=mock_server, model="mock", concurrency=2,
        allow_llm_judge=False,  # 测试不触真网:规则层弃权就落 null
    )
    r = load_run(scored, "test")
    assert (r.n, r.correct, r.abstained) == (3, 1, 1)  # q1 对 / q2 错 / q3 弃权计错

    # 断点续跑:outputs 已存在时不再触网(服务端关掉也能重判)
    rows = {json.loads(line)["id"]: json.loads(line) for line in scored.read_text().splitlines()}
    assert rows["q1"]["correct"] is True and rows["q3"]["correct"] is None


def test_resume_skips_generated(mock_server, tmp_path):
    samples = [_sample("q1", "B")]
    run_set("s", samples, tmp_path, base_url=mock_server, model="mock", allow_llm_judge=False)
    # 篡改已落盘的输出,重跑:应复用而不是重新生成(值保持篡改后的)
    out_file = tmp_path / "s.outputs.jsonl"
    out_file.write_text(json.dumps({"id": "q1", "output": "答案:C"}, ensure_ascii=False) + "\n", encoding="utf-8")
    scored = run_set("s", samples, tmp_path, base_url=mock_server, model="mock", allow_llm_judge=False)
    row = json.loads(scored.read_text().splitlines()[0])
    assert row["correct"] is False  # 复用了篡改后的 C(≠gold B),证明没有重新生成
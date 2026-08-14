"""数据集拉取 CLI:HF Hub → data/raw/ 的 jsonl 落盘。

用法:
    uv run python -m medforge.data.download            # 全部
    uv run python -m medforge.data.download med-o1-verifiable cmexam

国内网络失败时自动退 hf-mirror 镜像;所有落盘文件带行数打印,
「拉下来多少条」必须肉眼可见,防止半截下载被当成功。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from rich import print as rprint

RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"

# 数据源清单:name -> (HF repo, 取哪些 split)
DATASETS: dict[str, tuple[str, list[str]]] = {
    "med-o1-verifiable": ("FreedomIntelligence/medical-o1-verifiable-problem", ["train"]),
    "med-o1-sft-zh": ("FreedomIntelligence/medical-o1-reasoning-SFT", ["train"]),  # config=zh,见下
    "cmexam": ("fzkuji/CMExam", ["train", "validation", "test"]),
    "medxpertqa": ("TsinghuaC3I/MedXpertQA", ["test"]),  # config=Text;防污染设计的困难集(英文)
    # 幻觉评测源:官方版(UTAustin),pqa_labeled 是人工标注的高质量子集;
    # 任务形态是「判断给定回答是否幻觉」,与 QA 考卷不同,评测协议在 EvalScope 适配层单独设计
    "medhallu": ("UTAustin-AIHealth/MedHallu", ["train"]),
    # 通用中文指令(防灾难性遗忘的 replay 混料,实测字段 conversations[{from,value}]):
    # 与医疗数据同发布方,5 万条 GPT-4 生成
    "alpaca-zh": ("FreedomIntelligence/alpaca-gpt4-chinese", ["train"]),
    # 新教材(2025,R1 蒸馏):与 med-o1-sft-zh 同发布方同类题,唯一变量=老师
    # 从 GPT-4o(2024)换成 DeepSeek-R1——老教材降智假设的控制变量实验
    "med-r1-zh": ("FreedomIntelligence/Medical-R1-Distill-Data-Chinese", ["train"]),
}

# 需要指定 config 的数据源
CONFIGS = {"med-o1-sft-zh": "zh", "medxpertqa": "Text", "medhallu": "pqa_labeled"}


def fetch_cmb() -> None:
    """CMB 特例:官方加载脚本已坏,且 test 集刻意不放答案(留给官方榜单)。

    直接拉原始 JSON:val(280 题,带答案)→ 本地评测辅助集;
    test(11200 题,无答案)→ 仅留作将来投官方榜单,不参与本地判分。
    """
    from huggingface_hub import hf_hub_download

    rprint("[bold]▶ cmb[/] ← FreedomIntelligence/CMB(原始 JSON)")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    files = {
        "cmb.val.jsonl": "CMB-Exam/CMB-val/CMB-val-merge.json",
        "cmb.test-noanswer.jsonl": "CMB-Exam/CMB-test/CMB-test-choice-question-merge.json",
    }
    for out_name, repo_path in files.items():
        p = hf_hub_download("FreedomIntelligence/CMB", repo_path, repo_type="dataset")
        rows = json.loads(Path(p).read_text(encoding="utf-8"))
        out = RAW_DIR / out_name
        with out.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        rprint(f"  [green]✓[/] {out.name}: {len(rows)} 条")


def fetch(name: str) -> None:
    from datasets import load_dataset

    repo, splits = DATASETS[name]
    kwargs = {"name": CONFIGS[name]} if name in CONFIGS else {}
    rprint(f"[bold]▶ {name}[/] ← {repo} {kwargs or ''}")
    ds = load_dataset(repo, **kwargs)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for split in splits:
        if split not in ds:
            rprint(f"  [yellow]⚠ split {split} 不存在,实际: {list(ds)}[/]")
            continue
        out = RAW_DIR / f"{name}.{split}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for row in ds[split]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        rprint(f"  [green]✓[/] {out.name}: {len(ds[split])} 条")


def main() -> None:
    all_names = [*DATASETS, "cmb"]
    names = sys.argv[1:] or all_names
    unknown = set(names) - set(all_names)
    if unknown:
        rprint(f"[red]未知数据源: {unknown},可选: {all_names}[/]")
        sys.exit(2)
    failures = []
    for name in names:
        try:
            fetch_cmb() if name == "cmb" else fetch(name)
        except Exception as e:  # noqa: BLE001  逐源隔离:一个源挂了不影响其他;错误可见且最终非零退出,不是吞错
            rprint(f"  [red]✗ {name}: {type(e).__name__}: {e}[/]")
            failures.append(name)
    if failures and "HF_ENDPOINT" not in os.environ:
        rprint("\n[yellow]提示:国内网络可试 HF_ENDPOINT=https://hf-mirror.com 重跑失败源[/]")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

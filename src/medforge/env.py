"""加载仓库根目录 .env 到环境变量(已存在的环境变量优先,不覆盖)。

十行自实现而不引 python-dotenv:我们只需要「KEY=VALUE 逐行读」这一种能力,
为它引一个依赖不值得。密钥文件在 .gitignore 里,格式见 .env 模板。
"""

from __future__ import annotations

import os
from pathlib import Path

_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


def load_env() -> None:
    if not _ENV_FILE.exists():
        return
    for line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key and value and key not in os.environ:
            os.environ[key] = value

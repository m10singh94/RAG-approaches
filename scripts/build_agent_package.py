"""Build build/agent_package.zip for AgentCore Runtime direct code deployment.

AgentCore runs on Linux ARM64, so dependencies are resolved for aarch64-manylinux wheels (works from Windows
or macOS without Docker). Run before `terraform apply`; Terraform uploads the zip and points the runtime at it.

    uv run python scripts/build_agent_package.py
"""

from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build" / "agent"
ZIP = ROOT / "build" / "agent_package.zip"
PYTHON_VERSION = "3.12"  # keep in sync with `runtime` in infra/terraform/agentcore.tf


def main() -> None:
    shutil.rmtree(BUILD, ignore_errors=True)
    BUILD.mkdir(parents=True)
    subprocess.run(
        ["uv", "pip", "install", "--quiet", "--target", str(BUILD),
         "--python-platform", "aarch64-manylinux2014", "--python-version", PYTHON_VERSION,
         "--only-binary", ":all:", "-r", str(ROOT / "agent" / "requirements.txt")],
        check=True,
    )
    shutil.copytree(ROOT / "src" / "rag_approaches", BUILD / "rag_approaches",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(ROOT / "agent" / "main.py", BUILD / "main.py")

    ZIP.unlink(missing_ok=True)
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(BUILD.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                zf.write(path, path.relative_to(BUILD).as_posix())
    print(f"Wrote {ZIP.relative_to(ROOT)} ({ZIP.stat().st_size / 1_048_576:.1f} MiB)")


if __name__ == "__main__":
    main()

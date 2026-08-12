"""Dependency-free integrity checks for the released repository tree."""

from __future__ import annotations

import json
from pathlib import Path
import py_compile
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
MAX_FILE_BYTES = 50 * 1024 * 1024
TEXT_SUFFIXES = {".cff", ".csv", ".json", ".log", ".md", ".m", ".py", ".txt", ".yaml", ".yml"}
REQUIRED_PATHS = {
    Path("README.md"),
    Path("REPRODUCIBILITY.md"),
    Path("CITATION.cff"),
    Path("LICENSE"),
    Path("requirements.txt"),
    Path("train_ofdm_scma.py"),
    Path("evaluation_utils.py"),
    Path("validate_phase1_artifacts.py"),
    Path("cb1_kmv.pt"),
}
FORBIDDEN_LOCAL_PATHS = (
    re.compile(r"[A-Za-z]:\\(?:Users|agent_workplace)\\", re.IGNORECASE),
    re.compile("/" + r"home/[^/\s]+/"),
)


def repository_files() -> list[Path]:
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={ROOT.as_posix()}",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [ROOT / Path(item.decode("utf-8")) for item in result.stdout.split(b"\0") if item]


def main() -> int:
    failures: list[str] = []
    files = repository_files()
    relative_files = {path.relative_to(ROOT) for path in files}

    missing = sorted(REQUIRED_PATHS - relative_files)
    failures.extend(f"missing required file: {path.as_posix()}" for path in missing)

    for path in files:
        relative = path.relative_to(ROOT)
        if path.stat().st_size > MAX_FILE_BYTES:
            failures.append(f"file exceeds 50 MiB: {relative.as_posix()}")

        if path.suffix == ".py":
            try:
                py_compile.compile(path, doraise=True)
            except py_compile.PyCompileError as exc:
                failures.append(f"Python syntax error in {relative.as_posix()}: {exc.msg}")

        if path.suffix == ".json":
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                failures.append(f"invalid JSON in {relative.as_posix()}: {exc}")

        if path.suffix in TEXT_SUFFIXES:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                failures.append(f"non-UTF-8 text file {relative.as_posix()}: {exc}")
                continue
            for pattern in FORBIDDEN_LOCAL_PATHS:
                if pattern.search(text):
                    failures.append(f"workstation-specific path in {relative.as_posix()}")
                    break

    if failures:
        print("[FAIL] Repository audit found problems:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print(
        f"[OK] Repository audit passed: {len(files)} files, "
        f"{sum(path.suffix == '.py' for path in files)} Python modules."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

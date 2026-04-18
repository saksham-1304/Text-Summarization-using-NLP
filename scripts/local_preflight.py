"""Local preflight checks before launching full Kaggle training.

This script validates environment readiness and can run a fast local smoke
training pass to catch breakages early.

Usage examples:
  python scripts/local_preflight.py
  python scripts/local_preflight.py --prepare-data
  python scripts/local_preflight.py --prepare-data --check-model-download
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List

from textSummarizer.config.configuration import ConfigurationManager

REQUIRED_MODULES: List[str] = [
    "torch",
    "transformers",
    "datasets",
    "evaluate",
    "fastapi",
    "pandas",
    "yaml",
]


def _run_command(command: List[str], description: str) -> None:
    print(f"\n[RUN] {description}")
    print("      " + " ".join(command))
    start = time.time()
    result = subprocess.run(command, check=False)
    elapsed = time.time() - start
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}) after {elapsed:.1f}s: {' '.join(command)}"
        )
    print(f"      OK ({elapsed:.1f}s)")


def _check_imports(modules: Iterable[str]) -> None:
    print("\n[CHECK] Python dependencies")
    for module in modules:
        importlib.import_module(module)
        print(f"      OK import {module}")


def _check_disk_space(min_free_gb: float) -> None:
    print("\n[CHECK] Disk space")
    usage = shutil.disk_usage(Path.cwd())
    free_gb = usage.free / (1024 ** 3)
    print(f"      Free disk: {free_gb:.2f} GB")
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"Insufficient disk space: {free_gb:.2f} GB < required {min_free_gb:.2f} GB"
        )


def _check_config_and_paths() -> None:
    print("\n[CHECK] Configuration and writable artifacts")
    config = ConfigurationManager()
    artifacts_root = Path(config.config.artifacts_root)
    artifacts_root.mkdir(parents=True, exist_ok=True)

    probe_file = artifacts_root / ".preflight_write_probe"
    probe_file.write_text("ok", encoding="utf-8")
    probe_file.unlink()

    trainer_cfg = config.get_model_trainer_config()
    eval_cfg = config.get_model_evaluation_config()

    print(f"      OK artifacts_root: {artifacts_root}")
    print(f"      OK train root: {trainer_cfg.root_dir}")
    print(f"      OK eval root: {eval_cfg.root_dir}")


def _check_model_download_access() -> None:
    print("\n[CHECK] Model/tokenizer accessibility")
    from transformers import AutoTokenizer

    config = ConfigurationManager().get_model_trainer_config()
    AutoTokenizer.from_pretrained(config.model_ckpt)
    print(f"      OK tokenizer access for checkpoint: {config.model_ckpt}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run local preflight checks and optional smoke training before Kaggle launch"
        )
    )
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="Run stages 1-3 to prepare full tokenized artifacts before smoke training",
    )
    parser.add_argument(
        "--skip-smoke-train",
        action="store_true",
        help="Skip stage-4 smoke training run",
    )
    parser.add_argument(
        "--check-model-download",
        action="store_true",
        help="Verify tokenizer download access for configured checkpoint",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=6.0,
        help="Minimum required free disk space in GB (default: 6)",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("LOCAL PRELAUNCH PREFLIGHT")
    print("=" * 72)

    _check_imports(REQUIRED_MODULES)
    _check_disk_space(args.min_free_gb)
    _check_config_and_paths()

    if args.check_model_download:
        _check_model_download_access()

    if args.prepare_data:
        _run_command(
            [sys.executable, "main.py", "--stage", "1", "--to", "3"],
            "Pipeline data preparation (stages 1-3)",
        )

    if not args.skip_smoke_train:
        _run_command(
            [
                sys.executable,
                "main.py",
                "--stage",
                "4",
                "--to",
                "4",
                "--smoke-train",
            ],
            "Stage 4 smoke training",
        )

    print("\n" + "=" * 72)
    print("PREFLIGHT SUCCESS: local environment is ready for Kaggle training")
    print("=" * 72)


if __name__ == "__main__":
    main()

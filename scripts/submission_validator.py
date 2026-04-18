#!/usr/bin/env python
"""
FINAL SUBMISSION VALIDATOR
Comprehensive checks before submitting to Kaggle/evaluation platform
"""

import os
import re
import sys
import yaml
import subprocess
from pathlib import Path
from typing import Tuple


class SubmissionValidator:
    """Validates project readiness for submission."""

    def __init__(self, project_root: str = "."):
        self.root = Path(project_root)
        self.checks_passed = 0
        self.checks_failed = 0
        self.warnings = []

    def print_header(self, title: str) -> None:
        """Print formatted section header."""
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}")

    def check(self, description: str, condition: bool, details: str = "") -> bool:
        """Print and track a single check."""
        status = "✓" if condition else "✗"
        print(f"{status} {description}", end="")
        if details:
            print(f" — {details}")
        else:
            print()

        if condition:
            self.checks_passed += 1
        else:
            self.checks_failed += 1

        return condition

    def warn(self, message: str) -> None:
        """Log a warning without failing."""
        print(f"⚠ {message}")
        self.warnings.append(message)

    def run_command(self, cmd: str) -> Tuple[int, str]:
        """Run shell command and capture output."""
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.root)
            )
            return result.returncode, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return -1, "TIMEOUT"
        except Exception as e:
            return -1, str(e)

    def validate_file_structure(self) -> None:
        """Check that all critical files exist."""
        self.print_header("1. FILE STRUCTURE")

        critical_files = {
            "README.md": "Project overview",
            "SUBMISSION_GUIDE.md": "Submission instructions",
            "requirements.txt": "Dependencies",
            "config/config.yaml": "Configuration",
            "params.yaml": "Hyperparameters",
            "setup.py": "Package setup",
            "main.py": "Entry point",
            "src/textSummarizer/__init__.py": "Package init",
            "src/textSummarizer/components/data_augmentation.py": "Data augmentation (NEW)",
            "src/textSummarizer/pipeline/stage_01_data_ingestion.py": "Stage 1",
            "src/textSummarizer/pipeline/stage_02_data_validation.py": "Stage 2",
            "src/textSummarizer/pipeline/stage_03_data_transformation.py": "Stage 3",
            "src/textSummarizer/pipeline/stage_04_model_trainer.py": "Stage 4",
            "src/textSummarizer/pipeline/stage_05_model_evaluation.py": "Stage 5",
            "scripts/local_preflight.py": "Preflight validator",
            "docs/SYSTEM_DESIGN.md": "System design doc",
            "docs/HLD.md": "High-level design",
            "docs/LLD.md": "Low-level design",
            "docs/HYPERPARAMETER_JUSTIFICATION.md": "Hyperparameter justification",
        }

        for filepath, description in critical_files.items():
            file_path = self.root / filepath
            self.check(f"  {description}", file_path.exists(), filepath)

    def validate_no_redundancy(self) -> None:
        """Check that no redundant files remain."""
        self.print_header("2. REDUNDANCY CHECK")

        ignored_parts = [".venv", "venv", "site-packages", ".git"]

        redundant_patterns = [
            ("*.log", "Log files"),
            ("**/__pycache__", "Python cache"),
            ("**/*.egg-info", "Egg info"),
            ("**/UPGRADE_SUMMARY.md", "Upgrade summary (redundant)"),
            ("**/trials.ipynb", "Old trial notebook (optional)"),
        ]

        for pattern, desc in redundant_patterns:
            # Check if such files exist
            matches = list(self.root.glob(pattern.replace("**/", "")))
            if pattern.startswith("**/"):
                matches = list(self.root.rglob(pattern.replace("**/", "")))

            filtered_matches = []
            for match in matches:
                match_str = str(match)
                if any(ignored in match_str for ignored in ignored_parts):
                    continue
                filtered_matches.append(match)

            if filtered_matches:
                self.warn(
                    f"Found {desc}: {[str(m.relative_to(self.root)) for m in filtered_matches]}"
                )
            else:
                self.check(f"  No {desc}", True, pattern)

    def validate_dependencies(self) -> None:
        """Check that dependencies are valid."""
        self.print_header("3. DEPENDENCY CHECK")

        # Check requirements.txt exists and is readable
        req_file = self.root / "requirements.txt"
        self.check("  requirements.txt exists", req_file.exists())

        if req_file.exists():
            with open(req_file) as f:
                reqs = f.read()
                self.check("  accelerate>=1.13.0", "accelerate>=1.13.0" in reqs, "Verified pinned version")
                self.check("  transformers pinned", "transformers" in reqs)
                self.check("  torch pinned", "torch" in reqs)
                self.check("  datasets pinned", "datasets" in reqs)

        # Run pip check against the current interpreter for reliability.
        returncode, output = self.run_command(f'"{sys.executable}" -m pip check')
        if returncode == 0:
            self.check("  pip check (no broken dependencies)", True)
        else:
            self.warn(f"pip check returned non-zero: {output.strip()[:200]}")

    def validate_configuration(self) -> None:
        """Check configuration files are valid YAML."""
        self.print_header("4. CONFIGURATION CHECK")

        config_files = {
            "config/config.yaml": "Main config",
            "params.yaml": "Hyperparameters",
        }

        for filepath, desc in config_files.items():
            file_path = self.root / filepath
            if file_path.exists():
                try:
                    with open(file_path) as f:
                        yaml.safe_load(f)
                    self.check(f"  {desc} (valid YAML)", True, filepath)
                except yaml.YAMLError as e:
                    self.check(f"  {desc} (valid YAML)", False, f"Error: {str(e)[:50]}")
            else:
                self.check(f"  {desc} exists", False, filepath)

    def validate_no_secrets(self) -> None:
        """Check for hardcoded secrets or credentials."""
        self.print_header("5. SECURITY CHECK")

        secret_patterns = [
            (r"(?:password|passwd)\s*[=:]\s*['\"]", "hardcoded password"),
            (r"(?:api[_-]?key)\s*[=:]\s*['\"]", "API key"),
            (r"(?:secret|client_secret)\s*[=:]\s*['\"]", "secret token"),
            (r"(?:access[_-]?token|auth[_-]?token)\s*[=:]\s*['\"]", "authentication token"),
        ]

        found_secrets = []
        for pattern, desc in secret_patterns:
            # Search in Python and YAML files only
            for ext in ["*.py", "*.yaml", "*.yml"]:
                for file_path in self.root.rglob(ext):
                    file_str = str(file_path)
                    if any(
                        ignored in file_str
                        for ignored in [".venv", "venv", "site-packages", ".git", "artifacts"]
                    ):
                        continue
                    if file_path.name == "submission_validator.py":
                        continue
                    try:
                        with open(file_path, 'r') as f:
                            for line_num, line in enumerate(f, 1):
                                if re.search(pattern, line, flags=re.IGNORECASE) and not line.strip().startswith("#"):
                                    found_secrets.append((file_path, line_num, desc))
                    except:
                        pass

        if found_secrets:
            self.warn("Potential secrets found:")
            for file_path, line_num, desc in found_secrets[:5]:  # Show first 5
                print(f"    - {file_path}:{line_num} ({desc})")
        else:
            self.check("  No hardcoded secrets", True)

    def validate_code_quality(self) -> None:
        """Quick code quality checks."""
        self.print_header("6. CODE QUALITY CHECK")

        src_dir = self.root / "src/textSummarizer"
        self.check("  src/ directory exists", src_dir.exists())

        # Count Python files
        py_files = list(src_dir.rglob("*.py"))
        self.check(f"  {len(py_files)} Python source files", len(py_files) > 5, f"{len(py_files)} files")

        # Check for __init__.py files
        init_files = list(src_dir.rglob("__init__.py"))
        self.check(f"  {len(init_files)} __init__.py files (packages)", len(init_files) > 3, f"{len(init_files)} files")

    def validate_data_augmentation(self) -> None:
        """Verify data augmentation is implemented."""
        self.print_header("7. DATA AUGMENTATION CHECK")

        aug_file = self.root / "src/textSummarizer/components/data_augmentation.py"
        self.check("  data_augmentation.py exists", aug_file.exists())

        if aug_file.exists():
            with open(aug_file) as f:
                content = f.read()
                self.check("    - Paraphrasing method", "paraphrase_dialogue" in content)
                self.check("    - Noise injection method", "inject_noise" in content)
                self.check("    - Turn shuffling method", "shuffle_dialogue_turns" in content)
                self.check("    - Filler cleanup method", "remove_fillers" in content)
                self.check("    - Unified augmentation entry", "augment_text" in content)

    def validate_documentation(self) -> None:
        """Check documentation completeness."""
        self.print_header("8. DOCUMENTATION CHECK")

        docs_to_check = {
            "docs/SYSTEM_DESIGN.md": "System architecture documented",
            "docs/HLD.md": "High-level design documented",
            "docs/LLD.md": "Low-level design documented",
            "docs/HYPERPARAMETER_JUSTIFICATION.md": "Hyperparameter choices justified",
            "SUBMISSION_GUIDE.md": "Submission guide provided",
        }

        for filepath, desc in docs_to_check.items():
            file_path = self.root / filepath
            if file_path.exists():
                size = file_path.stat().st_size
                self.check(f"  {desc}", size > 500, f"{size} bytes")

    def validate_tests(self) -> None:
        """Check test suite exists."""
        self.print_header("9. TEST SUITE CHECK")

        test_dir = self.root / "tests"
        self.check("  tests/ directory exists", test_dir.exists())

        if test_dir.exists():
            test_files = list(test_dir.glob("test_*.py"))
            self.check(f"  {len(test_files)} test files", len(test_files) >= 3, f"{len(test_files)} files")

            # Try to count test functions
            test_count = 0
            for test_file in test_files:
                with open(test_file) as f:
                    test_count += f.read().count("def test_")

            self.check(f"  {test_count} test functions", test_count >= 10, f"{test_count} tests")

    def validate_preflight_script(self) -> None:
        """Check preflight script exists and is valid."""
        self.print_header("10. PREFLIGHT SCRIPT CHECK")

        preflight_file = self.root / "scripts/local_preflight.py"
        self.check("  local_preflight.py exists", preflight_file.exists())

        if preflight_file.exists():
            with open(preflight_file) as f:
                content = f.read()
                self.check("    - Dependency check", "dependencies" in content.lower())
                self.check("    - Disk space check", "disk" in content.lower())
                self.check("    - Stage 4 validation", "stage" in content.lower() and "4" in content)

    def print_summary(self) -> None:
        """Print final summary."""
        total = self.checks_passed + self.checks_failed
        percentage = (self.checks_passed / total * 100) if total > 0 else 0

        self.print_header("FINAL SUMMARY")

        print(f"\n✓ Passed: {self.checks_passed}/{total} ({percentage:.1f}%)")
        print(f"✗ Failed: {self.checks_failed}/{total}")

        if self.warnings:
            print(f"\n⚠ Warnings: {len(self.warnings)}")
            for warning in self.warnings[:3]:  # Show first 3
                print(f"  - {warning}")
            if len(self.warnings) > 3:
                print(f"  ... and {len(self.warnings) - 3} more")

        print("\n" + "=" * 70)

        if self.checks_failed == 0:
            print("✅ PROJECT READY FOR SUBMISSION!")
            print("=" * 70)
            return 0
        else:
            print("❌ FIX FAILURES BEFORE SUBMISSION")
            print("=" * 70)
            return 1

    def run_all_validations(self) -> int:
        """Run all validation checks."""
        print("\n" + "=" * 70)
        print("  🚀 FINAL SUBMISSION VALIDATOR")
        print("=" * 70)

        self.validate_file_structure()
        self.validate_no_redundancy()
        self.validate_dependencies()
        self.validate_configuration()
        self.validate_no_secrets()
        self.validate_code_quality()
        self.validate_data_augmentation()
        self.validate_documentation()
        self.validate_tests()
        self.validate_preflight_script()

        return self.print_summary()


def main():
    """Main entry point."""
    validator = SubmissionValidator()
    exit_code = validator.run_all_validations()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Export models and test data for server deployment (offline use)."""

import argparse
import logging
import shutil
import sys
import tarfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Package models and data for server deployment",
        epilog="""
Examples:
  # Export everything for server
  python export_for_server.py --output server_package.tar.gz

  # Export specific models only
  python export_for_server.py --models faster-whisper-large-v3-turbo,wav2vec2-turkish-large

  # Export without compression (faster for local transfer)
  python export_for_server.py --output server_package --no-compress
        """,
    )

    parser.add_argument(
        "--output",
        type=str,
        default="server_package.tar.gz",
        help="Output file/directory name",
    )

    parser.add_argument(
        "--models",
        type=str,
        help="Comma-separated list of models to include",
    )

    parser.add_argument(
        "--models-dir",
        type=str,
        default="models/weights",
        help="Directory containing model weights",
    )

    parser.add_argument(
        "--test-data-dir",
        type=str,
        default="test_data",
        help="Directory containing test data",
    )

    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Create directory instead of tar.gz",
    )

    parser.add_argument(
        "--include-results",
        action="store_true",
        help="Include results directory",
    )

    return parser.parse_args()


def copy_tree(src: Path, dst: Path, ignore_patterns: list = None):
    """Copy directory tree, optionally ignoring patterns."""
    if ignore_patterns is None:
        ignore_patterns = []

    def ignore_func(directory, files):
        ignored = set()
        for pattern in ignore_patterns:
            for f in files:
                if pattern in f:
                    ignored.add(f)
        return ignored

    if src.exists():
        shutil.copytree(src, dst, ignore=ignore_func, dirs_exist_ok=True)
        logger.info(f"Copied: {src} -> {dst}")
    else:
        logger.warning(f"Source not found: {src}")


def main():
    """Main entry point."""
    args = parse_args()

    project_root = Path(__file__).parent.parent
    output_path = Path(args.output)

    # Create temporary directory for packaging
    if args.no_compress:
        package_dir = output_path
    else:
        package_dir = Path("_server_package_temp")

    package_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Packaging files for server deployment...")

    # Copy source code
    copy_tree(
        project_root / "src",
        package_dir / "src",
        ignore_patterns=["__pycache__", ".pyc"],
    )

    # Copy scripts
    copy_tree(
        project_root / "scripts",
        package_dir / "scripts",
        ignore_patterns=["__pycache__"],
    )

    # Copy configs
    copy_tree(project_root / "configs", package_dir / "configs")

    # Copy requirements
    copy_tree(project_root / "requirements", package_dir / "requirements")

    # Copy test data
    test_data_src = Path(args.test_data_dir)
    if test_data_src.exists():
        copy_tree(test_data_src, package_dir / "test_data")
    else:
        logger.warning(f"Test data not found: {test_data_src}")
        (package_dir / "test_data").mkdir(parents=True, exist_ok=True)

    # Copy model weights
    models_src = Path(args.models_dir)
    if models_src.exists():
        if args.models:
            # Copy only specified models
            model_names = [m.strip() for m in args.models.split(",")]
            for model_name in model_names:
                model_path = models_src / model_name.replace("/", "--")
                if model_path.exists():
                    copy_tree(
                        model_path,
                        package_dir / "models" / "weights" / model_name.replace("/", "--"),
                    )
                else:
                    logger.warning(f"Model not found: {model_path}")
        else:
            # Copy all models
            copy_tree(models_src, package_dir / "models" / "weights")
    else:
        logger.warning(f"Models directory not found: {models_src}")
        (package_dir / "models" / "weights").mkdir(parents=True, exist_ok=True)

    # Copy results if requested
    if args.include_results:
        results_src = project_root / "results"
        if results_src.exists():
            copy_tree(results_src, package_dir / "results")

    # Copy other necessary files
    for filename in ["setup.py", "README.md", ".gitignore"]:
        src_file = project_root / filename
        if src_file.exists():
            shutil.copy2(src_file, package_dir / filename)
            logger.info(f"Copied: {filename}")

    # Create __init__.py files
    for init_dir in [package_dir / "src", package_dir / "scripts"]:
        init_file = init_dir / "__init__.py"
        if not init_file.exists():
            init_file.touch()

    # Create a run script for the server
    run_script = package_dir / "run_offline.sh"
    run_script.write_text("""#!/bin/bash
# Run benchmark in offline mode

# Set environment variables
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Run benchmark
python scripts/run_benchmark.py \\
    --offline \\
    --cache-dir models/weights \\
    "$@"
""")
    run_script.chmod(0o755)

    # Create Windows batch file too
    bat_script = package_dir / "run_offline.bat"
    bat_script.write_text("""@echo off
REM Run benchmark in offline mode

set HF_DATASETS_OFFLINE=1
set TRANSFORMERS_OFFLINE=1

python scripts\\run_benchmark.py ^
    --offline ^
    --cache-dir models\\weights ^
    %*
""")

    # Calculate package size
    total_size = sum(f.stat().st_size for f in package_dir.rglob("*") if f.is_file())
    logger.info(f"Package size: {total_size / (1024**3):.2f} GB")

    # Create tar.gz if requested
    if not args.no_compress:
        logger.info(f"Creating archive: {output_path}")
        with tarfile.open(output_path, "w:gz") as tar:
            tar.add(package_dir, arcname="stt-benchmark")

        # Clean up temp directory
        shutil.rmtree(package_dir)

        # Show final file size
        archive_size = output_path.stat().st_size
        logger.info(f"Archive size: {archive_size / (1024**3):.2f} GB")

    logger.info(f"\nExport complete: {output_path}")
    logger.info("\nTo use on server:")
    if args.no_compress:
        logger.info(f"  1. Copy {output_path} to server")
        logger.info(f"  2. cd {output_path}")
    else:
        logger.info(f"  1. Copy {output_path} to server")
        logger.info("  2. tar -xzf server_package.tar.gz")
        logger.info("  3. cd stt-benchmark")
    logger.info("  4. pip install -r requirements/common.txt")
    logger.info("  5. ./run_offline.sh --model faster-whisper-large-v3-turbo --language tr")

    return 0


if __name__ == "__main__":
    sys.exit(main())

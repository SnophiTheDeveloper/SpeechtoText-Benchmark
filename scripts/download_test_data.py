#!/usr/bin/env python3
"""Download test data from Common Voice and other datasets."""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download test data for STT benchmarking",
        epilog="""
Examples:
  # Download Turkish Common Voice samples
  python download_test_data.py --language tr --num-samples 10

  # Download English samples from LibriSpeech
  python download_test_data.py --language en --dataset librispeech

  # Create sample test data (for testing without downloading)
  python download_test_data.py --create-sample --language tr
        """,
    )

    parser.add_argument(
        "--language",
        type=str,
        default="tr",
        choices=["tr", "en"],
        help="Language to download (default: tr)",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="common_voice",
        choices=["common_voice", "librispeech"],
        help="Dataset to download from",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to download (default: 10)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="test_data",
        help="Output directory (default: test_data)",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test", "validation"],
        help="Dataset split to use (default: test)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample selection",
    )

    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create sample test data with placeholder files",
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing test data instead of downloading",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    output_dir = Path(args.output_dir)

    # Verify existing data
    if args.verify:
        from src.data.loader import validate_test_data

        data_dir = output_dir / args.language
        logger.info(f"Verifying test data in: {data_dir}")

        report = validate_test_data(data_dir)

        if report["valid"]:
            logger.info("Test data is valid!")
            logger.info(f"  Total files: {report['stats']['total_files']}")
            logger.info(f"  Valid files: {report['stats']['valid_files']}")
            logger.info(
                f"  Total duration: {report['stats']['total_duration_seconds']:.1f}s"
            )
        else:
            logger.error("Test data validation failed!")
            for error in report["errors"]:
                logger.error(f"  - {error}")

        for warning in report.get("warnings", []):
            logger.warning(f"  - {warning}")

        return 0 if report["valid"] else 1

    # Create sample data
    if args.create_sample:
        from src.data.downloader import create_sample_test_data

        logger.info(f"Creating sample test data for {args.language}...")
        result_dir = create_sample_test_data(
            output_dir=output_dir / args.language,
            language=args.language,
        )
        logger.info(f"Sample data created at: {result_dir}")
        return 0

    # Download real data
    logger.info(f"Downloading {args.num_samples} samples from {args.dataset}...")
    logger.info(f"Language: {args.language}")
    logger.info(f"Split: {args.split}")

    try:
        if args.dataset == "common_voice":
            from src.data.downloader import download_common_voice_samples

            result_dir = download_common_voice_samples(
                language=args.language,
                num_samples=args.num_samples,
                output_dir=output_dir,
                seed=args.seed,
                split=args.split,
            )
        elif args.dataset == "librispeech":
            if args.language != "en":
                logger.warning(
                    "LibriSpeech only supports English. Switching to English."
                )

            from src.data.downloader import download_librispeech_samples

            result_dir = download_librispeech_samples(
                num_samples=args.num_samples,
                output_dir=output_dir / "english",
                seed=args.seed,
            )
        else:
            logger.error(f"Unknown dataset: {args.dataset}")
            return 1

        logger.info(f"Download complete! Data saved to: {result_dir}")

        # Verify the downloaded data
        from src.data.downloader import verify_download

        if verify_download(result_dir):
            logger.info("Download verified successfully!")
        else:
            logger.warning("Download verification failed - some files may be missing")

        return 0

    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Install with: pip install datasets soundfile")
        return 1
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

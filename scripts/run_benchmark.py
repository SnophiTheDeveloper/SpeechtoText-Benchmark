#!/usr/bin/env python3
"""CLI script for running STT benchmarks."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmark.runner import BenchmarkRunner
from src.benchmark.utils import setup_logging, get_device_info
from src.models.whisper_model import FasterWhisperModel, DistilWhisperModel, OpenAIWhisperModel
from src.models.wav2vec2_model import Wav2Vec2Model
from src.models.hubert_model import HuBERTModel
from src.models.vosk_model import VoskModel


# Base directory for local models
MODELS_DIR = Path("models")
WEIGHTS_DIR = MODELS_DIR / "weights"


def get_local_model_path(model_name: str) -> Optional[Path]:
    """Check if model exists locally in models/ or models/weights/."""
    # Check direct path first (for custom models like whisper-small-tr-ct2-int8)
    direct_path = MODELS_DIR / model_name
    if direct_path.exists():
        return direct_path

    # Check weights directory
    weights_path = WEIGHTS_DIR / model_name
    if weights_path.exists():
        return weights_path

    # Check with HuggingFace-style name conversion (Systran--faster-whisper-small)
    hf_style_name = model_name.replace("/", "--")
    hf_path = WEIGHTS_DIR / hf_style_name
    if hf_path.exists():
        return hf_path

    return None


def create_faster_whisper_model(model_size: str, device: str) -> FasterWhisperModel:
    """Create FasterWhisperModel, preferring local path if available."""
    hf_id = f"Systran/faster-whisper-{model_size}"
    local_path = get_local_model_path(f"faster-whisper-{model_size}") or get_local_model_path(hf_id.replace("/", "--"))

    if local_path:
        return FasterWhisperModel(model_size, model_path=local_path, device=device)
    return FasterWhisperModel(model_size, device=device)


def create_vosk_model(model_name: str, device: str) -> VoskModel:
    """Create VoskModel from local path."""
    local_path = get_local_model_path(model_name)
    if not local_path:
        raise FileNotFoundError(
            f"Vosk model not found: {model_name}. "
            f"Download with: python scripts/download_models.py --model {model_name}"
        )
    return VoskModel(model_name, model_path=local_path, device=device)


# Model registry
MODEL_REGISTRY = {
    # Whisper models (local-first, fallback to HuggingFace)
    "faster-whisper-large-v3-turbo": lambda device: create_faster_whisper_model("large-v3-turbo", device),
    "faster-whisper-large-v3": lambda device: create_faster_whisper_model("large-v3", device),
    "faster-whisper-medium": lambda device: create_faster_whisper_model("medium", device),
    "faster-whisper-small": lambda device: create_faster_whisper_model("small", device),
    "faster-whisper-base": lambda device: create_faster_whisper_model("base", device),
    "distil-whisper-large-v3": lambda device: DistilWhisperModel("distil-whisper/distil-large-v3", device=device),
    "distil-whisper-tr": lambda device: DistilWhisperModel("Sercan/distil-whisper-large-v3-tr", device=device),
    "openai-whisper-large-v3-turbo": lambda device: OpenAIWhisperModel("openai/whisper-large-v3-turbo", device=device),
    # Custom finetuned models (local CT2 format)
    "whisper-small-tr-finetuned": lambda device: FasterWhisperModel(
        "whisper-small-tr-ct2-int8",
        model_path=MODELS_DIR / "whisper-small-tr-ct2-int8",
        device=device,
        compute_type="int8",
    ),
    # Wav2Vec2 models
    "wav2vec2-turkish-large": lambda device: Wav2Vec2Model("m3hrdadfi/wav2vec2-large-xlsr-turkish", device=device),
    "wav2vec2-turkish-base": lambda device: Wav2Vec2Model("cahya/wav2vec2-base-turkish", device=device),
    "wav2vec2-xlsr-53": lambda device: Wav2Vec2Model("facebook/wav2vec2-large-xlsr-53", device=device),
    # HuBERT models
    "hubert-large-ft": lambda device: HuBERTModel("facebook/hubert-large-ls960-ft", device=device),
    # Vosk models (offline only, must be downloaded first)
    "vosk-model-small-tr": lambda device: create_vosk_model("vosk-model-small-tr-0.3", device),
    "vosk-model-small-en": lambda device: create_vosk_model("vosk-model-small-en-us-0.15", device),
}

# Model families
MODEL_FAMILIES = {
    "whisper": [
        "faster-whisper-large-v3-turbo",
        "faster-whisper-large-v3",
        "faster-whisper-medium",
        "faster-whisper-small",
        "whisper-small-tr-finetuned",
        "distil-whisper-tr",
    ],
    "wav2vec2": [
        "wav2vec2-turkish-large",
        "wav2vec2-turkish-base",
    ],
    "hubert": [
        "hubert-large-ft",
    ],
    "vosk": [
        "vosk-model-small-tr",
        "vosk-model-small-en",
    ],
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run STT benchmark on speech-to-text models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test a single model
  python run_benchmark.py --model faster-whisper-large-v3-turbo --language tr

  # Test all whisper models
  python run_benchmark.py --family whisper --language tr

  # Test with GPU
  python run_benchmark.py --model faster-whisper-large-v3 --device cuda

  # Offline mode (for server without internet)
  python run_benchmark.py --model faster-whisper-large-v3 --offline --cache-dir ./models/weights

  # List available models
  python run_benchmark.py --list-models
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Model name to benchmark (see --list-models)",
    )

    parser.add_argument(
        "--family",
        type=str,
        choices=list(MODEL_FAMILIES.keys()),
        help="Test all models in a family",
    )

    parser.add_argument(
        "--language",
        type=str,
        default="tr",
        help="Language code (default: tr)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (cpu, cuda, cuda:0)",
    )

    parser.add_argument(
        "--test-data",
        type=str,
        default="test_data/tr",
        help="Path to test data directory",
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Path to results directory",
    )

    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run in offline mode (use cached models only)",
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default="models/weights",
        help="Model cache directory for offline mode",
    )

    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Run streaming/latency test instead of batch test",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Chunk size in ms for streaming test",
    )

    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def list_models():
    """Print available models."""
    print("\n=== Available Models ===\n")

    for family, models in MODEL_FAMILIES.items():
        print(f"{family.upper()}:")
        for model in models:
            print(f"  - {model}")
        print()

    print("=== Device Info ===\n")
    info = get_device_info()
    print(f"CUDA Available: {info['cuda_available']}")
    if info["cuda_available"]:
        for device in info["cuda_devices"]:
            print(f"  GPU {device['index']}: {device['name']}")


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    logger = logging.getLogger(__name__)

    # List models and exit
    if args.list_models:
        list_models()
        return 0

    # Validate arguments
    if not args.model and not args.family:
        print("Error: Either --model or --family must be specified")
        print("Use --list-models to see available options")
        return 1

    # Get models to test
    if args.model:
        if args.model not in MODEL_REGISTRY:
            print(f"Error: Unknown model '{args.model}'")
            print("Use --list-models to see available options")
            return 1
        model_names = [args.model]
    else:
        model_names = MODEL_FAMILIES.get(args.family, [])

    # Check test data exists
    test_data_path = Path(args.test_data)
    if not test_data_path.exists():
        print(f"Error: Test data directory not found: {test_data_path}")
        print("Run scripts/download_test_data.py first")
        return 1

    # Create runner
    runner = BenchmarkRunner(
        test_data_dir=args.test_data,
        results_dir=args.results_dir,
        device=args.device,
    )

    # Run benchmark for each model
    for model_name in model_names:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing: {model_name}")
        logger.info(f"{'='*50}")

        try:
            # Create model instance
            model = MODEL_REGISTRY[model_name](args.device)

            # Set offline path if specified
            if args.offline:
                cache_path = Path(args.cache_dir) / model_name.replace("/", "--")
                if cache_path.exists():
                    model.model_path = cache_path
                else:
                    logger.warning(
                        f"Model not found in cache: {cache_path}, "
                        "will attempt to download"
                    )

            # Run test
            if args.streaming:
                results = runner.run_streaming_test(
                    model,
                    language=args.language,
                    chunk_size_ms=args.chunk_size,
                )
                if "avg_first_word_latency_ms" in results:
                    logger.info(
                        f"Avg First Word Latency: "
                        f"{results['avg_first_word_latency_ms']:.1f}ms"
                    )
            else:
                results = runner.run(model, language=args.language)
                logger.info(
                    f"Results: WER={results.aggregate_metrics['wer']:.2%}, "
                    f"CER={results.aggregate_metrics['cer']:.2%}, "
                    f"RTF={results.aggregate_metrics['avg_rtf']:.3f}"
                )

        except Exception as e:
            logger.error(f"Failed to test {model_name}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            continue

    logger.info("\nBenchmark complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

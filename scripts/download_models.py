#!/usr/bin/env python3
"""Download and cache STT models for offline use."""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Model definitions with their HuggingFace IDs
MODELS = {
    "whisper": {
        "faster-whisper-large-v3-turbo": {
            "id": "Systran/faster-whisper-large-v3-turbo",
            "type": "faster-whisper",
        },
        "faster-whisper-large-v3": {
            "id": "Systran/faster-whisper-large-v3",
            "type": "faster-whisper",
        },
        "faster-whisper-medium": {
            "id": "Systran/faster-whisper-medium",
            "type": "faster-whisper",
        },
        "faster-whisper-small": {
            "id": "Systran/faster-whisper-small",
            "type": "faster-whisper",
        },
        "distil-whisper-large-v3": {
            "id": "distil-whisper/distil-large-v3",
            "type": "transformers",
        },
        "distil-whisper-tr": {
            "id": "Sercan/distil-whisper-large-v3-tr",
            "type": "transformers",
        },
        "openai-whisper-large-v3-turbo": {
            "id": "openai/whisper-large-v3-turbo",
            "type": "transformers",
        },
    },
    "wav2vec2": {
        "wav2vec2-turkish-large": {
            "id": "m3hrdadfi/wav2vec2-large-xlsr-turkish",
            "type": "transformers",
        },
        "wav2vec2-turkish-base": {
            "id": "cahya/wav2vec2-base-turkish",
            "type": "transformers",
        },
    },
    "hubert": {
        "hubert-large-ft": {
            "id": "facebook/hubert-large-ls960-ft",
            "type": "transformers",
        },
    },
    "vosk": {
        "vosk-model-small-tr-0.3": {
            "url": "https://huggingface.co/rhasspy/vosk-models/resolve/main/tr/vosk-model-small-tr-0.3.zip",
            "type": "vosk",
        },
        "vosk-model-small-en-us-0.15": {
            "url": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
            "type": "vosk",
        },
    },
}


def download_faster_whisper_model(model_id: str, cache_dir: Path) -> bool:
    """Download a faster-whisper model.

    Args:
        model_id: HuggingFace model ID
        cache_dir: Directory to cache the model

    Returns:
        True if successful
    """
    try:
        from faster_whisper import WhisperModel

        logger.info(f"Downloading faster-whisper model: {model_id}")

        # This will download and cache the model
        model = WhisperModel(model_id, device="cpu", compute_type="int8")

        # Verify it loaded
        if model is not None:
            logger.info(f"Successfully downloaded: {model_id}")
            del model
            return True

    except ImportError:
        logger.error("faster-whisper not installed. Run: pip install faster-whisper")
    except Exception as e:
        logger.error(f"Failed to download {model_id}: {e}")

    return False


def download_transformers_model(model_id: str, cache_dir: Path) -> bool:
    """Download a transformers model.

    Args:
        model_id: HuggingFace model ID
        cache_dir: Directory to cache the model

    Returns:
        True if successful
    """
    try:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        from huggingface_hub import snapshot_download

        logger.info(f"Downloading transformers model: {model_id}")

        # Download the model snapshot
        local_dir = cache_dir / model_id.replace("/", "--")
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )

        logger.info(f"Successfully downloaded to: {local_dir}")
        return True

    except ImportError:
        logger.error(
            "transformers or huggingface_hub not installed. "
            "Run: pip install transformers huggingface_hub"
        )
    except Exception as e:
        logger.error(f"Failed to download {model_id}: {e}")

    return False


def download_wav2vec2_model(model_id: str, cache_dir: Path) -> bool:
    """Download a Wav2Vec2 model.

    Args:
        model_id: HuggingFace model ID
        cache_dir: Directory to cache the model

    Returns:
        True if successful
    """
    try:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        from huggingface_hub import snapshot_download

        logger.info(f"Downloading Wav2Vec2 model: {model_id}")

        local_dir = cache_dir / model_id.replace("/", "--")
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )

        logger.info(f"Successfully downloaded to: {local_dir}")
        return True

    except ImportError:
        logger.error("transformers not installed. Run: pip install transformers")
    except Exception as e:
        logger.error(f"Failed to download {model_id}: {e}")

    return False


def download_vosk_model(model_name: str, url: str, cache_dir: Path) -> bool:
    """Download and extract a Vosk model.

    Args:
        model_name: Model name (e.g., "vosk-model-small-tr-0.3")
        url: URL to download the model zip
        cache_dir: Directory to cache the model

    Returns:
        True if successful
    """
    import zipfile
    import tempfile
    try:
        import requests
    except ImportError:
        logger.error("requests not installed. Run: pip install requests")
        return False

    try:
        model_dir = cache_dir / model_name

        if model_dir.exists():
            logger.info(f"Model already exists: {model_dir}")
            return True

        logger.info(f"Downloading Vosk model: {model_name}")
        logger.info(f"URL: {url}")

        # Download to temp file
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_file:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    percent = (downloaded / total_size) * 100
                    print(f"\rDownloading: {percent:.1f}%", end="", flush=True)

            print()  # New line after progress
            tmp_path = tmp_file.name

        # Extract
        logger.info(f"Extracting to: {cache_dir}")
        with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
            zip_ref.extractall(cache_dir)

        # Clean up
        Path(tmp_path).unlink()

        logger.info(f"Successfully downloaded: {model_name}")
        return True

    except Exception as e:
        logger.error(f"Failed to download {model_name}: {e}")
        return False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download STT models for offline use",
        epilog="""
Examples:
  # Download all models
  python download_models.py --all

  # Download specific model
  python download_models.py --model faster-whisper-large-v3-turbo

  # Download all whisper models
  python download_models.py --family whisper

  # List available models
  python download_models.py --list
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Specific model to download",
    )

    parser.add_argument(
        "--family",
        type=str,
        choices=["whisper", "wav2vec2", "hubert", "vosk"],
        help="Download all models from a family",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available models",
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default="models/weights",
        help="Directory to cache models",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and exit",
    )

    return parser.parse_args()


def list_models():
    """Print available models."""
    print("\n=== Available Models ===\n")
    for family, models in MODELS.items():
        print(f"{family.upper()}:")
        for name, info in models.items():
            print(f"  - {name}: {info['id']}")
        print()


def main():
    """Main entry point."""
    args = parse_args()

    if args.list:
        list_models()
        return 0

    # Create cache directory
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Determine which models to download
    models_to_download = []

    if args.all:
        for family in MODELS.values():
            for name, info in family.items():
                models_to_download.append((name, info))
    elif args.family:
        family = MODELS.get(args.family, {})
        for name, info in family.items():
            models_to_download.append((name, info))
    elif args.model:
        # Find the model
        found = False
        for family in MODELS.values():
            if args.model in family:
                models_to_download.append((args.model, family[args.model]))
                found = True
                break
        if not found:
            logger.error(f"Model not found: {args.model}")
            list_models()
            return 1
    else:
        logger.error("Specify --model, --family, or --all")
        return 1

    # Download models
    success_count = 0
    for name, info in models_to_download:
        logger.info(f"\n{'='*50}")
        logger.info(f"Downloading: {name}")
        logger.info(f"{'='*50}")

        model_type = info["type"]

        if model_type == "faster-whisper":
            model_id = info["id"]
            success = download_faster_whisper_model(model_id, cache_dir)
        elif model_type == "vosk":
            url = info["url"]
            success = download_vosk_model(name, url, cache_dir)
        else:
            model_id = info["id"]
            success = download_transformers_model(model_id, cache_dir)

        if success:
            success_count += 1

    logger.info(f"\n\nDownloaded {success_count}/{len(models_to_download)} models")
    return 0 if success_count == len(models_to_download) else 1


if __name__ == "__main__":
    sys.exit(main())

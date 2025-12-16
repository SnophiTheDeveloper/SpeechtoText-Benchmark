"""Download utilities for test datasets (e.g., Common Voice)."""

import json
import logging
import shutil
from pathlib import Path
from typing import Optional, List
import random

logger = logging.getLogger(__name__)


def download_common_voice_samples(
    language: str = "tr",
    num_samples: int = 10,
    output_dir: str | Path = "test_data",
    seed: int = 42,
    split: str = "test",
) -> Path:
    """Download sample audio files from Common Voice dataset.

    Uses the HuggingFace datasets library to download samples.

    Args:
        language: Language code (e.g., "tr" for Turkish, "en" for English)
        num_samples: Number of samples to download
        output_dir: Output directory for downloaded files
        seed: Random seed for sample selection
        split: Dataset split to use ("train", "test", "validation")

    Returns:
        Path to output directory with audio files and transcripts

    Raises:
        ImportError: If datasets library not installed
        ConnectionError: If download fails
    """
    try:
        from datasets import load_dataset
        import soundfile as sf
    except ImportError:
        raise ImportError(
            "datasets and soundfile required. Install with: "
            "pip install datasets soundfile"
        )

    output_dir = Path(output_dir) / language
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {num_samples} samples from Common Voice {language}...")

    # Map language codes to Common Voice dataset names
    cv_language_map = {
        "tr": "tr",
        "en": "en",
    }
    cv_lang = cv_language_map.get(language, language)

    try:
        # Load dataset (streaming to avoid downloading everything)
        # Try newer Common Voice 17.0 first (Parquet format), fallback to 13.0
        try:
            dataset = load_dataset(
                "mozilla-foundation/common_voice_17_0",
                cv_lang,
                split=split,
                streaming=True,
            )
        except Exception:
            # Fallback to version 13.0 without trust_remote_code
            dataset = load_dataset(
                "mozilla-foundation/common_voice_13_0",
                cv_lang,
                split=split,
                streaming=True,
            )

        # Shuffle and take samples
        dataset = dataset.shuffle(seed=seed)

        files = []
        for i, sample in enumerate(dataset.take(num_samples)):
            filename = f"sample_{i+1:03d}.wav"
            audio_path = audio_dir / filename

            # Get audio data
            audio_array = sample["audio"]["array"]
            sample_rate = sample["audio"]["sampling_rate"]

            # Save as WAV
            sf.write(audio_path, audio_array, sample_rate)

            # Get transcript
            transcript = sample.get("sentence", sample.get("text", ""))

            files.append({
                "filename": filename,
                "transcript": transcript,
                "duration_seconds": len(audio_array) / sample_rate,
                "speaker_id": sample.get("client_id", f"speaker_{i}"),
                "original_id": sample.get("path", ""),
            })

            logger.debug(f"Downloaded: {filename}")

        # Save transcripts
        transcripts_data = {
            "dataset": f"common_voice_{cv_lang}_sample",
            "source": "mozilla-foundation/common_voice_13_0",
            "split": split,
            "num_samples": len(files),
            "files": files,
        }

        transcripts_path = output_dir / "transcripts.json"
        with open(transcripts_path, "w", encoding="utf-8") as f:
            json.dump(transcripts_data, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Downloaded {len(files)} samples to {output_dir}"
        )
        return output_dir

    except Exception as e:
        logger.error(f"Failed to download Common Voice samples: {e}")
        raise


def download_librispeech_samples(
    num_samples: int = 10,
    output_dir: str | Path = "test_data/english",
    seed: int = 42,
) -> Path:
    """Download sample audio files from LibriSpeech dataset.

    Args:
        num_samples: Number of samples to download
        output_dir: Output directory
        seed: Random seed

    Returns:
        Path to output directory
    """
    try:
        from datasets import load_dataset
        import soundfile as sf
    except ImportError:
        raise ImportError(
            "datasets and soundfile required. Install with: "
            "pip install datasets soundfile"
        )

    output_dir = Path(output_dir)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {num_samples} samples from LibriSpeech...")

    try:
        dataset = load_dataset(
            "librispeech_asr",
            "clean",
            split="test",
            streaming=True,
        )

        dataset = dataset.shuffle(seed=seed)

        files = []
        for i, sample in enumerate(dataset.take(num_samples)):
            filename = f"libri_{i+1:03d}.wav"
            audio_path = audio_dir / filename

            audio_array = sample["audio"]["array"]
            sample_rate = sample["audio"]["sampling_rate"]

            sf.write(audio_path, audio_array, sample_rate)

            files.append({
                "filename": filename,
                "transcript": sample["text"].lower(),  # LibriSpeech is uppercase
                "duration_seconds": len(audio_array) / sample_rate,
                "speaker_id": str(sample.get("speaker_id", f"speaker_{i}")),
            })

        transcripts_data = {
            "dataset": "librispeech_test_clean_sample",
            "source": "librispeech_asr",
            "num_samples": len(files),
            "files": files,
        }

        transcripts_path = output_dir / "transcripts.json"
        with open(transcripts_path, "w", encoding="utf-8") as f:
            json.dump(transcripts_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Downloaded {len(files)} samples to {output_dir}")
        return output_dir

    except Exception as e:
        logger.error(f"Failed to download LibriSpeech samples: {e}")
        raise


def create_sample_test_data(
    output_dir: str | Path = "test_data/sample",
    language: str = "tr",
) -> Path:
    """Create sample test data with placeholder files for testing.

    Creates empty audio files and sample transcripts for testing
    the benchmark infrastructure without downloading real data.

    Args:
        output_dir: Output directory
        language: Language code

    Returns:
        Path to output directory
    """
    output_dir = Path(output_dir)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Sample transcripts
    if language == "tr":
        samples = [
            ("sample_001.wav", "merhaba dünya nasılsın"),
            ("sample_002.wav", "bugün hava çok güzel"),
            ("sample_003.wav", "türkçe konuşma tanıma testi"),
        ]
    else:
        samples = [
            ("sample_001.wav", "hello world how are you"),
            ("sample_002.wav", "the weather is nice today"),
            ("sample_003.wav", "english speech recognition test"),
        ]

    files = []
    for filename, transcript in samples:
        # Create placeholder audio file (silent)
        try:
            import numpy as np
            import soundfile as sf

            # 2 seconds of silence at 16kHz
            duration = 2.0
            sample_rate = 16000
            audio = np.zeros(int(duration * sample_rate))
            sf.write(audio_dir / filename, audio, sample_rate)

            files.append({
                "filename": filename,
                "transcript": transcript,
                "duration_seconds": duration,
            })
        except ImportError:
            # Just create empty file if soundfile not available
            (audio_dir / filename).touch()
            files.append({
                "filename": filename,
                "transcript": transcript,
                "duration_seconds": 2.0,
            })

    transcripts_data = {
        "dataset": f"sample_{language}",
        "files": files,
    }

    transcripts_path = output_dir / "transcripts.json"
    with open(transcripts_path, "w", encoding="utf-8") as f:
        json.dump(transcripts_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Created sample test data at {output_dir}")
    return output_dir


def verify_download(data_dir: str | Path) -> bool:
    """Verify that downloaded data is valid.

    Args:
        data_dir: Path to data directory

    Returns:
        True if valid, False otherwise
    """
    from .loader import validate_test_data

    report = validate_test_data(data_dir)
    return report["valid"] and report["stats"].get("valid_files", 0) > 0

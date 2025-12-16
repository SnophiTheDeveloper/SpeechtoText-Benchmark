"""Data loading utilities for test audio and transcripts."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TestDataset:
    """Container for test dataset information."""

    dataset_name: str
    files: List[Dict[str, Any]]
    base_path: Path

    @property
    def num_files(self) -> int:
        return len(self.files)

    @property
    def total_duration(self) -> float:
        return sum(f.get("duration_seconds", 0) for f in self.files)


def load_test_data(data_dir: str | Path) -> TestDataset:
    """Load test data from directory.

    Expects directory structure:
        data_dir/
            audio/
                sample_001.wav
                sample_002.wav
            transcripts.json

    Args:
        data_dir: Path to test data directory

    Returns:
        TestDataset object with file information

    Raises:
        FileNotFoundError: If transcripts.json not found
        ValueError: If transcripts.json is invalid
    """
    data_dir = Path(data_dir)

    # Load transcripts
    transcripts_path = data_dir / "transcripts.json"
    if not transcripts_path.exists():
        raise FileNotFoundError(f"Transcripts file not found: {transcripts_path}")

    with open(transcripts_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Validate structure
    if "files" not in data:
        raise ValueError("transcripts.json must contain 'files' key")

    # Check that audio files exist
    audio_dir = data_dir / "audio"
    valid_files = []

    for file_info in data["files"]:
        if "filename" not in file_info or "transcript" not in file_info:
            logger.warning(f"Skipping invalid file entry: {file_info}")
            continue

        audio_path = audio_dir / file_info["filename"]
        if audio_path.exists():
            valid_files.append(file_info)
        else:
            logger.warning(f"Audio file not found: {audio_path}")

    logger.info(
        f"Loaded {len(valid_files)}/{len(data['files'])} files from {data_dir}"
    )

    return TestDataset(
        dataset_name=data.get("dataset", data_dir.name),
        files=valid_files,
        base_path=data_dir,
    )


def load_audio(
    audio_path: str | Path,
    target_sr: int = 16000,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """Load audio file and return waveform.

    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate (default 16kHz for most STT models)
        mono: Whether to convert to mono

    Returns:
        Tuple of (audio_array, sample_rate)
    """
    try:
        import librosa

        audio, sr = librosa.load(audio_path, sr=target_sr, mono=mono)
        return audio, sr

    except ImportError:
        # Fallback to soundfile
        import soundfile as sf

        audio, sr = sf.read(audio_path)

        # Convert to mono if needed
        if mono and len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Resample if needed
        if sr != target_sr:
            try:
                import librosa

                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                sr = target_sr
            except ImportError:
                logger.warning(
                    f"librosa not installed, audio at {sr}Hz instead of {target_sr}Hz"
                )

        return audio, sr


def get_audio_duration(audio_path: str | Path) -> float:
    """Get duration of audio file in seconds.

    Args:
        audio_path: Path to audio file

    Returns:
        Duration in seconds
    """
    try:
        import librosa

        duration = librosa.get_duration(path=audio_path)
        return duration
    except ImportError:
        import soundfile as sf

        with sf.SoundFile(audio_path) as f:
            return len(f) / f.samplerate


def load_audio_chunks(
    audio_path: str | Path,
    chunk_size_ms: int = 500,
    target_sr: int = 16000,
) -> List[np.ndarray]:
    """Load audio and split into chunks for streaming simulation.

    Args:
        audio_path: Path to audio file
        chunk_size_ms: Size of each chunk in milliseconds
        target_sr: Target sample rate

    Returns:
        List of audio chunks as numpy arrays
    """
    audio, sr = load_audio(audio_path, target_sr=target_sr)

    # Calculate chunk size in samples
    chunk_samples = int(sr * chunk_size_ms / 1000)

    # Split into chunks
    chunks = []
    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i : i + chunk_samples]
        chunks.append(chunk)

    return chunks


def create_test_transcript_file(
    audio_dir: str | Path,
    transcripts: Dict[str, str],
    output_path: Optional[str | Path] = None,
    dataset_name: str = "custom_dataset",
) -> Path:
    """Create a transcripts.json file from a dictionary of transcripts.

    Useful for creating test datasets manually.

    Args:
        audio_dir: Directory containing audio files
        transcripts: Dictionary mapping filename to transcript
        output_path: Path for output file (default: audio_dir/../transcripts.json)
        dataset_name: Name for the dataset

    Returns:
        Path to created transcripts.json file
    """
    audio_dir = Path(audio_dir)

    if output_path is None:
        output_path = audio_dir.parent / "transcripts.json"
    else:
        output_path = Path(output_path)

    files = []
    for filename, transcript in transcripts.items():
        file_path = audio_dir / filename
        if file_path.exists():
            duration = get_audio_duration(file_path)
            files.append({
                "filename": filename,
                "transcript": transcript,
                "duration_seconds": duration,
            })
        else:
            logger.warning(f"Audio file not found: {file_path}")

    data = {
        "dataset": dataset_name,
        "files": files,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Created transcripts file with {len(files)} entries: {output_path}")
    return output_path


def validate_test_data(data_dir: str | Path) -> Dict[str, Any]:
    """Validate test data directory structure and contents.

    Args:
        data_dir: Path to test data directory

    Returns:
        Validation report dictionary
    """
    data_dir = Path(data_dir)
    report = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {},
    }

    # Check transcripts.json
    transcripts_path = data_dir / "transcripts.json"
    if not transcripts_path.exists():
        report["valid"] = False
        report["errors"].append("transcripts.json not found")
        return report

    # Load and validate JSON
    try:
        with open(transcripts_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        report["valid"] = False
        report["errors"].append(f"Invalid JSON: {e}")
        return report

    if "files" not in data:
        report["valid"] = False
        report["errors"].append("'files' key missing in transcripts.json")
        return report

    # Check audio directory
    audio_dir = data_dir / "audio"
    if not audio_dir.exists():
        report["valid"] = False
        report["errors"].append("audio directory not found")
        return report

    # Validate each file
    missing_audio = []
    missing_transcripts = []
    empty_transcripts = []
    total_duration = 0

    for file_info in data["files"]:
        filename = file_info.get("filename")
        transcript = file_info.get("transcript")

        if not filename:
            missing_transcripts.append(str(file_info))
            continue

        audio_path = audio_dir / filename
        if not audio_path.exists():
            missing_audio.append(filename)
            continue

        if not transcript:
            empty_transcripts.append(filename)

        # Get duration
        try:
            duration = get_audio_duration(audio_path)
            total_duration += duration
        except Exception:
            report["warnings"].append(f"Could not read duration: {filename}")

    # Update report
    if missing_audio:
        report["warnings"].append(f"Missing audio files: {missing_audio}")

    if empty_transcripts:
        report["warnings"].append(f"Empty transcripts: {empty_transcripts}")

    report["stats"] = {
        "total_files": len(data["files"]),
        "valid_files": len(data["files"]) - len(missing_audio),
        "missing_audio": len(missing_audio),
        "empty_transcripts": len(empty_transcripts),
        "total_duration_seconds": total_duration,
    }

    return report

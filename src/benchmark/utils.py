"""Utility functions for benchmark operations."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import hashlib

logger = logging.getLogger(__name__)


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None
) -> None:
    """Configure logging for the benchmark.

    Args:
        level: Logging level
        log_file: Optional file path to write logs
    """
    handlers: List[logging.Handler] = [logging.StreamHandler()]

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def generate_result_filename(
    model_name: str,
    language: str,
    timestamp: Optional[datetime] = None
) -> str:
    """Generate a timestamped filename for results.

    Args:
        model_name: Name of the model
        language: Language code
        timestamp: Optional timestamp (defaults to now)

    Returns:
        Filename string like "faster-whisper-large-v3_tr_2024-11-30_14-30.json"
    """
    if timestamp is None:
        timestamp = datetime.now()

    # Clean model name for filename
    clean_name = model_name.replace("/", "_").replace("\\", "_")
    time_str = timestamp.strftime("%Y-%m-%d_%H-%M")

    return f"{clean_name}_{language}_{time_str}.json"


def save_results(
    results: Dict[str, Any],
    output_dir: Path,
    filename: Optional[str] = None
) -> Path:
    """Save benchmark results to JSON file.

    Args:
        results: Results dictionary to save
        output_dir: Directory to save results
        filename: Optional filename (auto-generated if not provided)

    Returns:
        Path to saved file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = generate_result_filename(
            results.get("model_name", "unknown"),
            results.get("language", "unknown"),
        )

    output_path = output_dir / filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {output_path}")
    return output_path


def load_results(file_path: Path) -> Dict[str, Any]:
    """Load benchmark results from JSON file.

    Args:
        file_path: Path to results file

    Returns:
        Results dictionary
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_audio_hash(audio_path: Path) -> str:
    """Generate MD5 hash of audio file for identification.

    Args:
        audio_path: Path to audio file

    Returns:
        MD5 hash string
    """
    hash_md5 = hashlib.md5()
    with open(audio_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string like "2m 30s" or "1h 15m 30s"
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def format_percentage(value: float) -> str:
    """Format a ratio as percentage.

    Args:
        value: Value between 0 and 1 (or higher for error rates)

    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.2f}%"


def get_device_info() -> Dict[str, Any]:
    """Get information about available compute devices.

    Returns:
        Dictionary with device information
    """
    info = {
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_devices": [],
    }

    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_devices"] = [
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_total": torch.cuda.get_device_properties(i).total_memory,
                }
                for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        pass

    return info


def validate_audio_file(file_path: Path) -> bool:
    """Validate that a file is a valid audio file.

    Args:
        file_path: Path to file to validate

    Returns:
        True if valid audio file
    """
    valid_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    return file_path.exists() and file_path.suffix.lower() in valid_extensions


def ensure_sample_rate(audio_path: Path, target_sr: int = 16000) -> Path:
    """Ensure audio file has the correct sample rate.

    If resampling is needed, saves to a temp file.

    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate

    Returns:
        Path to audio file with correct sample rate
    """
    try:
        import librosa
        import soundfile as sf
        import tempfile

        y, sr = librosa.load(audio_path, sr=None)

        if sr == target_sr:
            return audio_path

        # Resample
        y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

        # Save to temp file
        temp_path = Path(tempfile.gettempdir()) / f"resampled_{audio_path.name}"
        sf.write(temp_path, y_resampled, target_sr)

        logger.debug(f"Resampled {audio_path} from {sr}Hz to {target_sr}Hz")
        return temp_path

    except ImportError:
        logger.warning("librosa/soundfile not installed, skipping sample rate check")
        return audio_path


class Timer:
    """Context manager for timing operations."""

    def __init__(self):
        self.start_time: float = 0
        self.end_time: float = 0
        self.elapsed: float = 0

    def __enter__(self):
        import time
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        import time
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time

    @property
    def elapsed_ms(self) -> float:
        return self.elapsed * 1000

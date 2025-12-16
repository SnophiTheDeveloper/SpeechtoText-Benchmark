"""Abstract base class for all STT models."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Generator, List
from pathlib import Path
import logging
import time

logger = logging.getLogger(__name__)


class BaseSTTModel(ABC):
    """Abstract base class for all STT models.

    All STT model wrappers should inherit from this class and implement
    the abstract methods for consistent interface across different models.

    Attributes:
        model_name: HuggingFace model ID or local name
        model_path: Local path for offline loading (for server without internet)
        device: Target device ("cpu", "cuda", or "cuda:0")
        model: The loaded model instance
    """

    def __init__(
        self,
        model_name: str,
        model_path: Optional[Path] = None,
        device: str = "cpu"
    ):
        """Initialize the STT model wrapper.

        Args:
            model_name: HuggingFace model ID or local name
            model_path: Local path for offline loading (for server without internet)
            device: Target device ("cpu", "cuda", or "cuda:0")
        """
        self.model_name = model_name
        self.model_path = Path(model_path) if model_path else None
        self.device = device
        self.model = None
        self._processor = None  # For models that need a processor/tokenizer

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory.

        Should check local cache first, then download if needed.
        For offline environments (server), should raise an error
        if model is not found locally.

        Raises:
            FileNotFoundError: If model not found and offline mode
            RuntimeError: If model loading fails
        """
        pass

    @abstractmethod
    def transcribe(self, audio_path: str, language: str = "tr") -> Dict[str, Any]:
        """Transcribe an audio file.

        Args:
            audio_path: Path to the audio file
            language: Language code ("tr" for Turkish, "en" for English)

        Returns:
            Dictionary containing:
                - text: Transcribed text
                - segments: List of segments with timestamps (if available)
                - language: Detected or specified language
                - duration_seconds: Audio duration
                - processing_time_seconds: Time taken to transcribe

        Raises:
            FileNotFoundError: If audio file not found
            RuntimeError: If transcription fails
        """
        pass

    @abstractmethod
    def transcribe_streaming(
        self,
        audio_path: str,
        chunk_size_ms: int = 500,
        language: str = "tr"
    ) -> Generator[Dict[str, Any], None, None]:
        """Simulated streaming transcription for latency testing.

        Processes audio in chunks and yields results incrementally.
        Used to measure first-word latency and streaming performance.

        Args:
            audio_path: Path to the audio file
            chunk_size_ms: Size of each audio chunk in milliseconds
            language: Language code

        Yields:
            Dictionary containing:
                - text: Partial transcription so far
                - is_final: Whether this is the final result
                - chunk_index: Current chunk number
                - latency_ms: Time since start of processing
        """
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata and status.

        Returns:
            Dictionary containing model information
        """
        return {
            "name": self.model_name,
            "device": self.device,
            "is_loaded": self.model is not None,
            "model_path": str(self.model_path) if self.model_path else None,
            "family": self._get_model_family(),
        }

    def _get_model_family(self) -> str:
        """Get the model family name (whisper, wav2vec2, hubert).

        Override in subclasses if needed.
        """
        return "unknown"

    def unload(self) -> None:
        """Unload the model from memory to free resources."""
        if self.model is not None:
            del self.model
            self.model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        logger.info(f"Model {self.model_name} unloaded")

    def __enter__(self):
        """Context manager entry - load model."""
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - unload model."""
        self.unload()
        return False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name='{self.model_name}', device='{self.device}')"


def get_model_cache_path(model_name: str, base_cache_dir: Path) -> Optional[Path]:
    """Get the local cache path for a model.

    Checks multiple locations for cached model files:
    1. First check base_cache_dir/model_name
    2. Then check HuggingFace default cache

    Args:
        model_name: Model identifier
        base_cache_dir: Base directory for model cache

    Returns:
        Path to cached model if found, None otherwise
    """
    # Check custom cache directory
    custom_path = base_cache_dir / model_name.replace("/", "--")
    if custom_path.exists():
        logger.info(f"Found model in custom cache: {custom_path}")
        return custom_path

    # Check HuggingFace cache
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    hf_model_path = hf_cache / f"models--{model_name.replace('/', '--')}"
    if hf_model_path.exists():
        logger.info(f"Found model in HuggingFace cache: {hf_model_path}")
        return hf_model_path

    # For faster-whisper models, check their specific cache
    faster_whisper_cache = Path.home() / ".cache" / "huggingface" / "hub"
    if "whisper" in model_name.lower():
        whisper_path = faster_whisper_cache / f"models--Systran--faster-whisper-{model_name}"
        if whisper_path.exists():
            logger.info(f"Found model in faster-whisper cache: {whisper_path}")
            return whisper_path

    logger.debug(f"Model {model_name} not found in cache")
    return None


def check_internet_connection() -> bool:
    """Check if internet connection is available.

    Returns:
        True if connected, False otherwise
    """
    import socket
    try:
        socket.create_connection(("huggingface.co", 443), timeout=5)
        return True
    except OSError:
        return False

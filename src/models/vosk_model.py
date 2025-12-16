"""Vosk model wrapper for offline speech recognition."""

import json
import logging
import time
import wave
from pathlib import Path
from typing import Dict, Any, Optional, Generator

from .base import BaseSTTModel

logger = logging.getLogger(__name__)


class VoskModel(BaseSTTModel):
    """Wrapper for Vosk speech recognition models.

    Vosk is a lightweight offline speech recognition toolkit that works
    without internet connection. Models are pre-downloaded and stored locally.

    Supported models:
        - vosk-model-small-tr-0.3 (Turkish, ~45MB)
        - vosk-model-en-us-0.22 (English, ~1.8GB)
        - vosk-model-small-en-us-0.15 (English small, ~40MB)

    Example:
        >>> model = VoskModel("vosk-model-small-tr-0.3", model_path=Path("models/vosk-model-small-tr-0.3"))
        >>> model.load()
        >>> result = model.transcribe("audio.wav", language="tr")
        >>> print(result["text"])
    """

    def __init__(
        self,
        model_name: str,
        model_path: Optional[Path] = None,
        device: str = "cpu",
        sample_rate: int = 16000,
    ):
        """Initialize Vosk model.

        Args:
            model_name: Model name (e.g., "vosk-model-small-tr-0.3")
            model_path: Local path to the extracted model directory
            device: Device (Vosk only supports CPU)
            sample_rate: Audio sample rate (default 16000)
        """
        super().__init__(model_name, model_path, device)
        self.sample_rate = sample_rate
        self._recognizer = None

    def _get_model_family(self) -> str:
        return "vosk"

    def load(self) -> None:
        """Load the Vosk model."""
        try:
            from vosk import Model, SetLogLevel
            SetLogLevel(-1)  # Suppress Vosk logging
        except ImportError:
            raise ImportError(
                "vosk not installed. Install with: pip install vosk"
            )

        if self.model_path and self.model_path.exists():
            model_source = str(self.model_path)
            logger.info(f"Loading Vosk model from: {model_source}")
        else:
            raise FileNotFoundError(
                f"Vosk model not found at {self.model_path}. "
                "Download the model and extract it first."
            )

        self.model = Model(model_source)
        logger.info(f"Loaded Vosk model: {self.model_name}")

    def transcribe(self, audio_path: str, language: str = "tr") -> Dict[str, Any]:
        """Transcribe an audio file using Vosk.

        Args:
            audio_path: Path to the audio file (must be WAV format, 16kHz mono)
            language: Language code (informational only, model is language-specific)

        Returns:
            Dictionary with transcription results
        """
        if self.model is None:
            self.load()

        from vosk import KaldiRecognizer
        import soundfile as sf
        import numpy as np

        start_time = time.perf_counter()

        # Load audio
        audio, sr = sf.read(audio_path)

        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Resample if needed
        if sr != self.sample_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate

        # Convert to int16
        audio_int16 = (audio * 32767).astype(np.int16)
        audio_duration = len(audio) / sr

        # Create recognizer
        rec = KaldiRecognizer(self.model, self.sample_rate)
        rec.SetWords(True)

        # Process audio in chunks
        chunk_size = 4000  # samples per chunk
        for i in range(0, len(audio_int16), chunk_size):
            chunk = audio_int16[i:i + chunk_size].tobytes()
            rec.AcceptWaveform(chunk)

        # Get final result
        result = json.loads(rec.FinalResult())
        text = result.get("text", "")

        processing_time = time.perf_counter() - start_time

        return {
            "text": text,
            "segments": self._parse_words(result.get("result", [])),
            "language": language,
            "duration_seconds": audio_duration,
            "processing_time_seconds": processing_time,
        }

    def _parse_words(self, words: list) -> list:
        """Parse Vosk word results into segments."""
        if not words:
            return []

        segments = []
        for word in words:
            segments.append({
                "text": word.get("word", ""),
                "start": word.get("start", 0),
                "end": word.get("end", 0),
                "confidence": word.get("conf", 1.0),
            })
        return segments

    def transcribe_streaming(
        self,
        audio_path: str,
        chunk_size_ms: int = 500,
        language: str = "tr"
    ) -> Generator[Dict[str, Any], None, None]:
        """Simulated streaming transcription for latency testing.

        Args:
            audio_path: Path to the audio file
            chunk_size_ms: Size of each audio chunk in milliseconds
            language: Language code

        Yields:
            Dictionary with partial transcription results
        """
        if self.model is None:
            self.load()

        from vosk import KaldiRecognizer
        import soundfile as sf
        import numpy as np

        start_time = time.perf_counter()

        # Load audio
        audio, sr = sf.read(audio_path)

        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Resample if needed
        if sr != self.sample_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate

        # Convert to int16
        audio_int16 = (audio * 32767).astype(np.int16)

        # Create recognizer
        rec = KaldiRecognizer(self.model, self.sample_rate)
        rec.SetWords(True)

        # Calculate chunk size in samples
        chunk_samples = int(self.sample_rate * chunk_size_ms / 1000)

        accumulated_text = ""
        chunk_index = 0

        for i in range(0, len(audio_int16), chunk_samples):
            chunk = audio_int16[i:i + chunk_samples].tobytes()

            if rec.AcceptWaveform(chunk):
                result = json.loads(rec.Result())
                partial_text = result.get("text", "")
                if partial_text:
                    accumulated_text += " " + partial_text if accumulated_text else partial_text
            else:
                result = json.loads(rec.PartialResult())
                partial_text = result.get("partial", "")

            latency_ms = (time.perf_counter() - start_time) * 1000

            yield {
                "text": accumulated_text + (" " + partial_text if partial_text else ""),
                "is_final": False,
                "chunk_index": chunk_index,
                "latency_ms": latency_ms,
            }
            chunk_index += 1

        # Final result
        final_result = json.loads(rec.FinalResult())
        final_text = final_result.get("text", "")
        if final_text:
            accumulated_text += " " + final_text if accumulated_text else final_text

        yield {
            "text": accumulated_text.strip(),
            "is_final": True,
            "chunk_index": chunk_index,
            "latency_ms": (time.perf_counter() - start_time) * 1000,
        }

    def unload(self) -> None:
        """Unload the model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self._recognizer is not None:
            del self._recognizer
            self._recognizer = None
        logger.info(f"Model {self.model_name} unloaded")

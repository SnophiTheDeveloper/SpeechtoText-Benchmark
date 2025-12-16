"""Wav2Vec2 model wrapper for speech-to-text."""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Generator

import numpy as np

from .base import BaseSTTModel, check_internet_connection

logger = logging.getLogger(__name__)


class Wav2Vec2Model(BaseSTTModel):
    """Wrapper for Wav2Vec2 models from HuggingFace.

    Wav2Vec2 models are CTC-based models that require a processor
    for tokenization.

    Supported models:
        - m3hrdadfi/wav2vec2-large-xlsr-turkish
        - cahya/wav2vec2-base-turkish
        - facebook/wav2vec2-large-xlsr-53 (multilingual)

    Example:
        >>> model = Wav2Vec2Model("m3hrdadfi/wav2vec2-large-xlsr-turkish")
        >>> model.load()
        >>> result = model.transcribe("audio.wav", language="tr")
        >>> print(result["text"])
    """

    # Common Turkish Wav2Vec2 models
    TURKISH_MODELS = {
        "wav2vec2-turkish-large": "m3hrdadfi/wav2vec2-large-xlsr-turkish",
        "wav2vec2-turkish-base": "cahya/wav2vec2-base-turkish",
        "wav2vec2-xlsr-53": "facebook/wav2vec2-large-xlsr-53",
    }

    def __init__(
        self,
        model_name: str = "m3hrdadfi/wav2vec2-large-xlsr-turkish",
        model_path: Optional[Path] = None,
        device: str = "cpu",
    ):
        """Initialize Wav2Vec2 model.

        Args:
            model_name: HuggingFace model ID or short name
            model_path: Local path for offline loading
            device: Target device ("cpu" or "cuda")
        """
        # Map short names to full model IDs
        if model_name in self.TURKISH_MODELS:
            model_name = self.TURKISH_MODELS[model_name]

        super().__init__(model_name, model_path, device)

    def _get_model_family(self) -> str:
        return "wav2vec2"

    def load(self) -> None:
        """Load the Wav2Vec2 model and processor."""
        try:
            import torch
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        except ImportError:
            raise ImportError(
                "transformers and torch required. Install with: "
                "pip install transformers torch torchaudio"
            )

        # Determine model source
        if self.model_path and self.model_path.exists():
            model_source = str(self.model_path)
            logger.info(f"Loading model from local path: {model_source}")
        else:
            model_source = self.model_name
            if not check_internet_connection():
                raise RuntimeError(
                    f"Model {model_source} not found locally and no internet connection"
                )
            logger.info(f"Loading model from HuggingFace: {model_source}")

        # Load processor
        self._processor = Wav2Vec2Processor.from_pretrained(model_source)

        # Load model
        self.model = Wav2Vec2ForCTC.from_pretrained(model_source)
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Loaded Wav2Vec2 model: {self.model_name}")

    def transcribe(self, audio_path: str, language: str = "tr") -> Dict[str, Any]:
        """Transcribe audio file using Wav2Vec2.

        Args:
            audio_path: Path to audio file
            language: Language code (mainly for logging, model is language-specific)

        Returns:
            Transcription result dictionary
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        import torch
        from ..data.loader import load_audio, get_audio_duration

        start_time = time.perf_counter()

        # Load audio at 16kHz (Wav2Vec2 requirement)
        audio, sr = load_audio(audio_path, target_sr=16000)
        duration = len(audio) / sr

        # Process audio
        inputs = self._processor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )

        # Move to device
        input_values = inputs.input_values.to(self.device)

        # Inference
        with torch.no_grad():
            logits = self.model(input_values).logits

        # Decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self._processor.batch_decode(predicted_ids)[0]

        processing_time = time.perf_counter() - start_time

        return {
            "text": transcription.strip(),
            "segments": [],  # CTC models don't provide timestamps easily
            "language": language,
            "duration_seconds": duration,
            "processing_time_seconds": processing_time,
        }

    def transcribe_streaming(
        self,
        audio_path: str,
        chunk_size_ms: int = 500,
        language: str = "tr",
    ) -> Generator[Dict[str, Any], None, None]:
        """Simulated streaming transcription.

        Note: Wav2Vec2 CTC models are not ideal for streaming as they
        work best with complete utterances. This is a simulation for
        latency measurement purposes.

        Args:
            audio_path: Path to audio file
            chunk_size_ms: Chunk size in milliseconds
            language: Language code

        Yields:
            Partial transcription results
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        import torch
        from ..data.loader import load_audio_chunks

        chunks = load_audio_chunks(audio_path, chunk_size_ms)
        start_time = time.perf_counter()
        accumulated_audio = []

        for i, chunk in enumerate(chunks):
            accumulated_audio = np.concatenate([accumulated_audio, chunk]) if len(accumulated_audio) > 0 else chunk

            # Process accumulated audio
            inputs = self._processor(
                accumulated_audio,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            )

            input_values = inputs.input_values.to(self.device)

            with torch.no_grad():
                logits = self.model(input_values).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self._processor.batch_decode(predicted_ids)[0]

            latency = (time.perf_counter() - start_time) * 1000

            yield {
                "text": transcription.strip(),
                "chunk_index": i,
                "is_final": i == len(chunks) - 1,
                "latency_ms": latency,
                "accumulated_duration_ms": (i + 1) * chunk_size_ms,
            }

    def transcribe_with_lm(
        self,
        audio_path: str,
        lm_path: Optional[str] = None,
        language: str = "tr",
    ) -> Dict[str, Any]:
        """Transcribe with optional language model for better accuracy.

        Args:
            audio_path: Path to audio file
            lm_path: Path to KenLM language model (optional)
            language: Language code

        Returns:
            Transcription result dictionary
        """
        # Basic transcription if no LM provided
        if lm_path is None:
            return self.transcribe(audio_path, language)

        try:
            from pyctcdecode import build_ctcdecoder
            import torch
            from ..data.loader import load_audio, get_audio_duration
        except ImportError:
            logger.warning("pyctcdecode not installed, falling back to greedy decoding")
            return self.transcribe(audio_path, language)

        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        start_time = time.perf_counter()

        # Load audio
        audio, sr = load_audio(audio_path, target_sr=16000)
        duration = len(audio) / sr

        # Process audio
        inputs = self._processor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )

        input_values = inputs.input_values.to(self.device)

        # Get logits
        with torch.no_grad():
            logits = self.model(input_values).logits

        # Build decoder with LM
        vocab = list(self._processor.tokenizer.get_vocab().keys())
        decoder = build_ctcdecoder(vocab, kenlm_model_path=lm_path)

        # Decode with LM
        logits_np = logits.cpu().numpy()[0]
        transcription = decoder.decode(logits_np)

        processing_time = time.perf_counter() - start_time

        return {
            "text": transcription.strip(),
            "segments": [],
            "language": language,
            "duration_seconds": duration,
            "processing_time_seconds": processing_time,
            "used_lm": True,
        }


class Wav2Vec2TurkishModel(Wav2Vec2Model):
    """Convenience class for Turkish Wav2Vec2 models."""

    def __init__(
        self,
        model_name: str = "m3hrdadfi/wav2vec2-large-xlsr-turkish",
        model_path: Optional[Path] = None,
        device: str = "cpu",
    ):
        super().__init__(model_name, model_path, device)

    def transcribe(self, audio_path: str, language: str = "tr") -> Dict[str, Any]:
        """Transcribe with Turkish language default."""
        return super().transcribe(audio_path, language="tr")

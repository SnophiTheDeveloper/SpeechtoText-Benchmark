"""HuBERT model wrapper for speech-to-text."""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Generator

import numpy as np

from .base import BaseSTTModel, check_internet_connection

logger = logging.getLogger(__name__)


class HuBERTModel(BaseSTTModel):
    """Wrapper for HuBERT (Hidden-Unit BERT) models from HuggingFace.

    HuBERT uses self-supervised learning and is fine-tuned for ASR.
    Similar interface to Wav2Vec2 as they share the same architecture.

    Note: HuBERT Turkish models may be limited. Check HuggingFace for
    available models like asafaya/hubert-* series.

    Example:
        >>> model = HuBERTModel("facebook/hubert-large-ls960-ft")
        >>> model.load()
        >>> result = model.transcribe("audio.wav", language="en")
        >>> print(result["text"])
    """

    # Known HuBERT models
    KNOWN_MODELS = {
        "hubert-large-ft": "facebook/hubert-large-ls960-ft",
        "hubert-xlarge-ft": "facebook/hubert-xlarge-ls960-ft",
    }

    def __init__(
        self,
        model_name: str = "facebook/hubert-large-ls960-ft",
        model_path: Optional[Path] = None,
        device: str = "cpu",
    ):
        """Initialize HuBERT model.

        Args:
            model_name: HuggingFace model ID or short name
            model_path: Local path for offline loading
            device: Target device ("cpu" or "cuda")
        """
        # Map short names to full model IDs
        if model_name in self.KNOWN_MODELS:
            model_name = self.KNOWN_MODELS[model_name]

        super().__init__(model_name, model_path, device)

    def _get_model_family(self) -> str:
        return "hubert"

    def load(self) -> None:
        """Load the HuBERT model and processor."""
        try:
            import torch
            from transformers import HubertForCTC, Wav2Vec2Processor
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

        # Load processor (HuBERT uses Wav2Vec2Processor)
        self._processor = Wav2Vec2Processor.from_pretrained(model_source)

        # Load model
        self.model = HubertForCTC.from_pretrained(model_source)
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Loaded HuBERT model: {self.model_name}")

    def transcribe(self, audio_path: str, language: str = "en") -> Dict[str, Any]:
        """Transcribe audio file using HuBERT.

        Args:
            audio_path: Path to audio file
            language: Language code (HuBERT models are typically language-specific)

        Returns:
            Transcription result dictionary
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        import torch
        from ..data.loader import load_audio

        start_time = time.perf_counter()

        # Load audio at 16kHz
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
        language: str = "en",
    ) -> Generator[Dict[str, Any], None, None]:
        """Simulated streaming transcription.

        Note: HuBERT CTC models are not ideal for streaming.
        This is a simulation for latency measurement.

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
        accumulated_audio = np.array([])

        for i, chunk in enumerate(chunks):
            accumulated_audio = (
                np.concatenate([accumulated_audio, chunk])
                if len(accumulated_audio) > 0
                else chunk
            )

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


class HuBERTTurkishModel(HuBERTModel):
    """Wrapper for Turkish HuBERT models.

    Note: Turkish HuBERT models may be limited on HuggingFace.
    Check for models from asafaya or other Turkish NLP contributors.

    If no Turkish HuBERT is available, consider using:
    - Wav2Vec2 Turkish models instead
    - Fine-tuning a multilingual HuBERT on Turkish data
    """

    # Potential Turkish HuBERT models (check HuggingFace for availability)
    TURKISH_MODELS = {
        # Add Turkish models as they become available
        # "hubert-tr": "asafaya/hubert-tr",
    }

    def __init__(
        self,
        model_name: str = "facebook/hubert-large-ls960-ft",
        model_path: Optional[Path] = None,
        device: str = "cpu",
    ):
        """Initialize Turkish HuBERT model.

        Args:
            model_name: HuggingFace model ID
            model_path: Local path for offline loading
            device: Target device
        """
        if model_name in self.TURKISH_MODELS:
            model_name = self.TURKISH_MODELS[model_name]

        super().__init__(model_name, model_path, device)

        if "turkish" not in model_name.lower() and "tr" not in model_name.lower():
            logger.warning(
                f"Model {model_name} may not be optimized for Turkish. "
                "Consider using a Turkish-specific model or Wav2Vec2 Turkish models."
            )


def search_turkish_hubert_models() -> list:
    """Search HuggingFace for Turkish HuBERT models.

    Returns:
        List of model IDs that might be Turkish HuBERT models
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.warning("huggingface_hub not installed, cannot search models")
        return []

    api = HfApi()

    # Search for Turkish HuBERT models
    search_queries = [
        "hubert turkish",
        "hubert tr",
        "asafaya hubert",
    ]

    found_models = set()

    for query in search_queries:
        try:
            models = api.list_models(
                search=query,
                task="automatic-speech-recognition",
                limit=10,
            )
            for model in models:
                if "hubert" in model.modelId.lower():
                    found_models.add(model.modelId)
        except Exception as e:
            logger.debug(f"Search failed for '{query}': {e}")

    return list(found_models)

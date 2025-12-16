"""Whisper model wrappers (faster-whisper and transformers-based)."""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Generator, List

from .base import BaseSTTModel, get_model_cache_path, check_internet_connection

logger = logging.getLogger(__name__)


class FasterWhisperModel(BaseSTTModel):
    """Wrapper for faster-whisper models (CTranslate2 based).

    Faster-whisper provides significantly faster inference compared to
    the original OpenAI Whisper implementation.

    Supported models:
        - large-v3-turbo (fastest large model)
        - large-v3
        - medium
        - small
        - base
        - tiny

    Example:
        >>> model = FasterWhisperModel("large-v3-turbo", device="cuda")
        >>> model.load()
        >>> result = model.transcribe("audio.wav", language="tr")
        >>> print(result["text"])
    """

    # Model size to HuggingFace ID mapping
    MODEL_MAPPING = {
        "large-v3-turbo": "Systran/faster-whisper-large-v3-turbo",
        "large-v3": "Systran/faster-whisper-large-v3",
        "medium": "Systran/faster-whisper-medium",
        "small": "Systran/faster-whisper-small",
        "base": "Systran/faster-whisper-base",
        "tiny": "Systran/faster-whisper-tiny",
    }

    def __init__(
        self,
        model_name: str = "large-v3-turbo",
        model_path: Optional[Path] = None,
        device: str = "cpu",
        compute_type: str = "auto",
        num_workers: int = 1,
    ):
        """Initialize faster-whisper model.

        Args:
            model_name: Model size or HuggingFace model ID
            model_path: Local path for offline loading
            device: "cpu" or "cuda"
            compute_type: Computation type ("float16", "int8", "auto")
            num_workers: Number of workers for parallel processing
        """
        super().__init__(model_name, model_path, device)
        self.compute_type = compute_type
        self.num_workers = num_workers
        self._model_id = self.MODEL_MAPPING.get(model_name, model_name)

    def _get_model_family(self) -> str:
        return "whisper"

    def load(self) -> None:
        """Load the faster-whisper model."""
        try:
            from faster_whisper import WhisperModel
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ImportError(
                "faster-whisper not installed. Install with: pip install faster-whisper"
            )

        # Determine compute type based on device
        if self.compute_type == "auto":
            if self.device.startswith("cuda"):
                compute_type = "float16"
            else:
                compute_type = "int8"
        else:
            compute_type = self.compute_type

        # Local weights directory
        weights_dir = Path("models/weights")
        weights_dir.mkdir(parents=True, exist_ok=True)

        # Determine model source - prefer local, then download to local
        if self.model_path and self.model_path.exists():
            model_source = str(self.model_path)
            logger.info(f"Loading model from local path: {model_source}")
        else:
            # Check if already downloaded to weights directory
            local_model_dir = weights_dir / self._model_id.replace("/", "--")

            if local_model_dir.exists():
                model_source = str(local_model_dir)
                logger.info(f"Loading model from local cache: {model_source}")
            else:
                # Download from HuggingFace to local weights directory
                if not check_internet_connection():
                    raise RuntimeError(
                        f"Model {self._model_id} not found locally and no internet connection"
                    )
                logger.info(f"Downloading model from HuggingFace: {self._model_id}")
                logger.info(f"Saving to: {local_model_dir}")

                # Download to local directory
                snapshot_download(
                    repo_id=self._model_id,
                    local_dir=local_model_dir,
                    local_dir_use_symlinks=False,
                )
                model_source = str(local_model_dir)
                logger.info(f"Model downloaded and saved to: {model_source}")

        # Load model
        device = "cuda" if self.device.startswith("cuda") else "cpu"
        device_index = 0
        if ":" in self.device:
            device_index = int(self.device.split(":")[1])

        self.model = WhisperModel(
            model_source,
            device=device,
            device_index=device_index,
            compute_type=compute_type,
            num_workers=self.num_workers,
        )

        logger.info(f"Loaded faster-whisper model: {self.model_name}")

    def transcribe(self, audio_path: str, language: str = "tr") -> Dict[str, Any]:
        """Transcribe audio file using faster-whisper.

        Args:
            audio_path: Path to audio file
            language: Language code

        Returns:
            Transcription result dictionary
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        start_time = time.perf_counter()

        # Transcribe
        segments, info = self.model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
        )

        # Collect segments
        segments_list = []
        full_text_parts = []

        for segment in segments:
            seg_dict = {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
            }

            # Add word-level timestamps if available
            if segment.words:
                seg_dict["words"] = [
                    {
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                        "probability": word.probability,
                    }
                    for word in segment.words
                ]

            segments_list.append(seg_dict)
            full_text_parts.append(segment.text)

        processing_time = time.perf_counter() - start_time

        return {
            "text": "".join(full_text_parts).strip(),
            "segments": segments_list,
            "language": info.language,
            "language_probability": info.language_probability,
            "duration_seconds": info.duration,
            "processing_time_seconds": processing_time,
        }

    def transcribe_streaming(
        self,
        audio_path: str,
        chunk_size_ms: int = 500,
        language: str = "tr",
    ) -> Generator[Dict[str, Any], None, None]:
        """Simulated streaming transcription.

        Processes audio in chunks to measure latency characteristics.

        Args:
            audio_path: Path to audio file
            chunk_size_ms: Chunk size in milliseconds
            language: Language code

        Yields:
            Partial transcription results
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        try:
            from ..data.loader import load_audio_chunks
        except ImportError:
            raise RuntimeError("Data loader not available")

        # Load audio in chunks
        chunks = load_audio_chunks(audio_path, chunk_size_ms)

        start_time = time.perf_counter()
        accumulated_audio = []

        for i, chunk in enumerate(chunks):
            accumulated_audio.extend(chunk)

            # Create temporary audio for transcription
            import numpy as np
            import tempfile
            import soundfile as sf

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                sf.write(tmp_path, np.array(accumulated_audio), 16000)

            try:
                # Transcribe accumulated audio
                segments, info = self.model.transcribe(
                    tmp_path,
                    language=language,
                    beam_size=1,  # Faster for streaming
                    vad_filter=False,
                )

                text_parts = [seg.text for seg in segments]
                current_text = "".join(text_parts).strip()

                latency = (time.perf_counter() - start_time) * 1000

                yield {
                    "text": current_text,
                    "chunk_index": i,
                    "is_final": i == len(chunks) - 1,
                    "latency_ms": latency,
                    "accumulated_duration_ms": (i + 1) * chunk_size_ms,
                }

            finally:
                # Clean up temp file
                import os
                try:
                    os.unlink(tmp_path)
                except:
                    pass


class DistilWhisperModel(BaseSTTModel):
    """Wrapper for Distil-Whisper models (transformers-based).

    Distil-Whisper models are distilled versions that are faster
    while maintaining good accuracy.

    Supported models:
        - distil-whisper/distil-large-v3 (English)
        - Sercan/distil-whisper-large-v3-tr (Turkish)
    """

    def __init__(
        self,
        model_name: str = "distil-whisper/distil-large-v3",
        model_path: Optional[Path] = None,
        device: str = "cpu",
    ):
        """Initialize Distil-Whisper model.

        Args:
            model_name: HuggingFace model ID
            model_path: Local path for offline loading
            device: Target device
        """
        super().__init__(model_name, model_path, device)

    def _get_model_family(self) -> str:
        return "whisper"

    def load(self) -> None:
        """Load the Distil-Whisper model."""
        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        except ImportError:
            raise ImportError(
                "transformers and torch required. Install with: "
                "pip install transformers torch"
            )

        # Determine model source
        if self.model_path and self.model_path.exists():
            model_source = str(self.model_path)
        else:
            model_source = self.model_name
            if not check_internet_connection():
                raise RuntimeError(
                    f"Model {model_source} not found locally and no internet connection"
                )

        # Determine torch dtype
        if self.device.startswith("cuda"):
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        # Load model and processor
        self._processor = AutoProcessor.from_pretrained(model_source)

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_source,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )

        model.to(self.device)

        # Create pipeline
        self.model = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=self._processor.tokenizer,
            feature_extractor=self._processor.feature_extractor,
            max_new_tokens=128,
            torch_dtype=torch_dtype,
            device=self.device,
        )

        logger.info(f"Loaded Distil-Whisper model: {self.model_name}")

    def transcribe(self, audio_path: str, language: str = "tr") -> Dict[str, Any]:
        """Transcribe audio using Distil-Whisper.

        Args:
            audio_path: Path to audio file
            language: Language code

        Returns:
            Transcription result dictionary
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        from ..data.loader import get_audio_duration

        start_time = time.perf_counter()

        # Transcribe
        generate_kwargs = {"language": language}

        result = self.model(
            audio_path,
            generate_kwargs=generate_kwargs,
            return_timestamps=True,
        )

        processing_time = time.perf_counter() - start_time
        duration = get_audio_duration(audio_path)

        # Parse result
        text = result.get("text", "")
        chunks = result.get("chunks", [])

        segments = []
        if chunks:
            for chunk in chunks:
                segments.append({
                    "start": chunk.get("timestamp", [None, None])[0],
                    "end": chunk.get("timestamp", [None, None])[1],
                    "text": chunk.get("text", ""),
                })

        return {
            "text": text.strip(),
            "segments": segments,
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
        """Simulated streaming transcription."""
        # Similar implementation to FasterWhisperModel
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        from ..data.loader import load_audio_chunks
        import numpy as np
        import tempfile
        import soundfile as sf
        import os

        chunks = load_audio_chunks(audio_path, chunk_size_ms)
        start_time = time.perf_counter()
        accumulated_audio = []

        for i, chunk in enumerate(chunks):
            accumulated_audio.extend(chunk)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                sf.write(tmp_path, np.array(accumulated_audio), 16000)

            try:
                result = self.model(tmp_path, generate_kwargs={"language": language})
                current_text = result.get("text", "").strip()

                latency = (time.perf_counter() - start_time) * 1000

                yield {
                    "text": current_text,
                    "chunk_index": i,
                    "is_final": i == len(chunks) - 1,
                    "latency_ms": latency,
                    "accumulated_duration_ms": (i + 1) * chunk_size_ms,
                }
            finally:
                try:
                    os.unlink(tmp_path)
                except:
                    pass


class OpenAIWhisperModel(BaseSTTModel):
    """Wrapper for OpenAI Whisper models via transformers.

    For models like openai/whisper-large-v3-turbo.
    """

    def __init__(
        self,
        model_name: str = "openai/whisper-large-v3-turbo",
        model_path: Optional[Path] = None,
        device: str = "cpu",
    ):
        super().__init__(model_name, model_path, device)

    def _get_model_family(self) -> str:
        return "whisper"

    def load(self) -> None:
        """Load OpenAI Whisper model via transformers."""
        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        except ImportError:
            raise ImportError(
                "transformers and torch required. Install with: "
                "pip install transformers torch"
            )

        model_source = (
            str(self.model_path)
            if self.model_path and self.model_path.exists()
            else self.model_name
        )

        if not self.model_path and not check_internet_connection():
            raise RuntimeError(
                f"Model {model_source} not found locally and no internet connection"
            )

        torch_dtype = (
            torch.float16 if self.device.startswith("cuda") else torch.float32
        )

        self._processor = AutoProcessor.from_pretrained(model_source)

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_source,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(self.device)

        self.model = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=self._processor.tokenizer,
            feature_extractor=self._processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=self.device,
        )

        logger.info(f"Loaded OpenAI Whisper model: {self.model_name}")

    def transcribe(self, audio_path: str, language: str = "tr") -> Dict[str, Any]:
        """Transcribe audio."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        from ..data.loader import get_audio_duration

        start_time = time.perf_counter()

        result = self.model(
            audio_path,
            generate_kwargs={"language": language},
            return_timestamps=True,
        )

        processing_time = time.perf_counter() - start_time
        duration = get_audio_duration(audio_path)

        text = result.get("text", "")
        chunks = result.get("chunks", [])

        segments = []
        if chunks:
            for chunk in chunks:
                timestamps = chunk.get("timestamp", [None, None])
                segments.append({
                    "start": timestamps[0] if timestamps else None,
                    "end": timestamps[1] if len(timestamps) > 1 else None,
                    "text": chunk.get("text", ""),
                })

        return {
            "text": text.strip(),
            "segments": segments,
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
        """Simulated streaming transcription."""
        # Reuse DistilWhisper implementation
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        from ..data.loader import load_audio_chunks
        import numpy as np
        import tempfile
        import soundfile as sf
        import os

        chunks = load_audio_chunks(audio_path, chunk_size_ms)
        start_time = time.perf_counter()
        accumulated_audio = []

        for i, chunk in enumerate(chunks):
            accumulated_audio.extend(chunk)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                sf.write(tmp_path, np.array(accumulated_audio), 16000)

            try:
                result = self.model(tmp_path, generate_kwargs={"language": language})
                current_text = result.get("text", "").strip()

                latency = (time.perf_counter() - start_time) * 1000

                yield {
                    "text": current_text,
                    "chunk_index": i,
                    "is_final": i == len(chunks) - 1,
                    "latency_ms": latency,
                    "accumulated_duration_ms": (i + 1) * chunk_size_ms,
                }
            finally:
                try:
                    os.unlink(tmp_path)
                except:
                    pass

"""RealtimeSTT wrapper for real-time testing (local/server only).

This module provides integration with RealtimeSTT library for
actual real-time microphone input testing.

Note: This is only for local or server environments with microphone access.
Not suitable for Google Colab.
"""

import logging
import time
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RealtimeTestResult:
    """Container for real-time test results."""

    model_name: str
    duration_seconds: float
    transcriptions: List[Dict[str, Any]] = field(default_factory=list)
    first_word_latency_ms: Optional[float] = None
    avg_chunk_latency_ms: Optional[float] = None
    total_words: int = 0
    errors: List[str] = field(default_factory=list)


class RealtimeSTTTester:
    """Wrapper for testing RealtimeSTT with various backends.

    This class provides utilities for testing real-time speech recognition
    with microphone input.

    Example:
        >>> tester = RealtimeSTTTester(model="large-v3-turbo")
        >>> tester.start_recording()
        >>> # Speak into microphone...
        >>> results = tester.stop_and_get_results()
    """

    def __init__(
        self,
        model: str = "large-v3-turbo",
        language: str = "tr",
        device: str = "cuda",
    ):
        """Initialize RealtimeSTT tester.

        Args:
            model: Whisper model size for RealtimeSTT
            language: Language code
            device: Device to use
        """
        self.model = model
        self.language = language
        self.device = device
        self.recorder = None
        self._transcriptions: List[Dict[str, Any]] = []
        self._start_time: Optional[float] = None
        self._first_word_time: Optional[float] = None

    def _check_realtimestt_installed(self) -> bool:
        """Check if RealtimeSTT is installed."""
        try:
            import RealtimeSTT
            return True
        except ImportError:
            return False

    def _on_transcription(self, text: str) -> None:
        """Callback for transcription results."""
        current_time = time.perf_counter()

        if self._start_time is None:
            return

        latency = (current_time - self._start_time) * 1000

        # Record first word time
        if self._first_word_time is None and text.strip():
            self._first_word_time = current_time

        self._transcriptions.append({
            "text": text,
            "timestamp": current_time,
            "latency_ms": latency,
        })

        logger.debug(f"Transcription: {text} (latency: {latency:.0f}ms)")

    def start_recording(self) -> bool:
        """Start real-time recording and transcription.

        Returns:
            True if started successfully
        """
        if not self._check_realtimestt_installed():
            logger.error(
                "RealtimeSTT not installed. Install with: pip install RealtimeSTT"
            )
            return False

        try:
            from RealtimeSTT import AudioToTextRecorder

            # Configure recorder
            self.recorder = AudioToTextRecorder(
                model=self.model,
                language=self.language,
                compute_type="float16" if "cuda" in self.device else "int8",
                device=self.device,
                on_recording_start=lambda: logger.info("Recording started"),
                on_recording_stop=lambda: logger.info("Recording stopped"),
                on_transcription_start=lambda: None,
            )

            self._transcriptions = []
            self._start_time = time.perf_counter()
            self._first_word_time = None

            # Start recording
            self.recorder.start()
            logger.info("Real-time recording started")
            return True

        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            return False

    def stop_and_get_results(self) -> RealtimeTestResult:
        """Stop recording and return results.

        Returns:
            RealtimeTestResult with all transcriptions
        """
        if self.recorder is None:
            return RealtimeTestResult(
                model_name=self.model,
                duration_seconds=0,
                errors=["Recording was not started"],
            )

        try:
            # Stop recording
            self.recorder.stop()
            end_time = time.perf_counter()
            duration = end_time - self._start_time if self._start_time else 0

            # Calculate metrics
            first_word_latency = None
            if self._first_word_time and self._start_time:
                first_word_latency = (self._first_word_time - self._start_time) * 1000

            avg_latency = None
            if self._transcriptions:
                latencies = [t["latency_ms"] for t in self._transcriptions]
                avg_latency = sum(latencies) / len(latencies)

            total_words = sum(
                len(t["text"].split()) for t in self._transcriptions
            )

            return RealtimeTestResult(
                model_name=self.model,
                duration_seconds=duration,
                transcriptions=self._transcriptions,
                first_word_latency_ms=first_word_latency,
                avg_chunk_latency_ms=avg_latency,
                total_words=total_words,
            )

        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
            return RealtimeTestResult(
                model_name=self.model,
                duration_seconds=0,
                errors=[str(e)],
            )

        finally:
            self.recorder = None

    def test_with_audio_file(
        self,
        audio_path: str,
        playback_speed: float = 1.0,
    ) -> RealtimeTestResult:
        """Simulate real-time test using an audio file.

        Plays audio file through virtual audio device or simulates
        chunk-by-chunk processing.

        Args:
            audio_path: Path to audio file
            playback_speed: Speed multiplier (1.0 = real-time)

        Returns:
            RealtimeTestResult
        """
        # This is a simplified version that uses streaming instead of actual playback
        from ..models.whisper_model import FasterWhisperModel

        model = FasterWhisperModel(self.model, device=self.device)
        model.load()

        self._transcriptions = []
        self._start_time = time.perf_counter()
        self._first_word_time = None

        try:
            for result in model.transcribe_streaming(
                audio_path, chunk_size_ms=500, language=self.language
            ):
                self._on_transcription(result.get("text", ""))

                # Simulate real-time delay
                if playback_speed > 0:
                    time.sleep(0.5 / playback_speed)

        except Exception as e:
            logger.error(f"Error during file test: {e}")

        finally:
            model.unload()

        end_time = time.perf_counter()

        first_word_latency = None
        if self._first_word_time and self._start_time:
            first_word_latency = (self._first_word_time - self._start_time) * 1000

        avg_latency = None
        if self._transcriptions:
            latencies = [t["latency_ms"] for t in self._transcriptions]
            avg_latency = sum(latencies) / len(latencies)

        return RealtimeTestResult(
            model_name=self.model,
            duration_seconds=end_time - self._start_time,
            transcriptions=self._transcriptions,
            first_word_latency_ms=first_word_latency,
            avg_chunk_latency_ms=avg_latency,
            total_words=sum(len(t["text"].split()) for t in self._transcriptions),
        )


def check_audio_devices() -> Dict[str, Any]:
    """Check available audio input devices.

    Returns:
        Dictionary with device information
    """
    result = {
        "pyaudio_available": False,
        "devices": [],
        "default_input": None,
    }

    try:
        import pyaudio

        result["pyaudio_available"] = True

        p = pyaudio.PyAudio()

        # Get device count
        device_count = p.get_device_count()

        for i in range(device_count):
            device_info = p.get_device_info_by_index(i)
            if device_info.get("maxInputChannels", 0) > 0:
                result["devices"].append({
                    "index": i,
                    "name": device_info.get("name"),
                    "sample_rate": device_info.get("defaultSampleRate"),
                    "input_channels": device_info.get("maxInputChannels"),
                })

        # Get default input device
        try:
            default_input = p.get_default_input_device_info()
            result["default_input"] = {
                "index": default_input.get("index"),
                "name": default_input.get("name"),
            }
        except:
            pass

        p.terminate()

    except ImportError:
        logger.warning("PyAudio not installed")
    except Exception as e:
        logger.error(f"Error checking audio devices: {e}")

    return result

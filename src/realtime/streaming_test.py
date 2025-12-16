"""Streaming/real-time test utilities for STT models."""

import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Generator
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StreamingResult:
    """Container for streaming test results."""

    model_name: str
    audio_file: str
    chunk_size_ms: int
    total_chunks: int
    first_word_latency_ms: Optional[float]
    total_latency_ms: float
    final_text: str
    chunk_results: List[Dict[str, Any]]


def simulate_streaming_input(
    audio_path: str,
    chunk_size_ms: int = 500,
    sample_rate: int = 16000,
) -> Generator[np.ndarray, None, None]:
    """Simulate streaming audio input by yielding chunks.

    Args:
        audio_path: Path to audio file
        chunk_size_ms: Size of each chunk in milliseconds
        sample_rate: Target sample rate

    Yields:
        Audio chunks as numpy arrays
    """
    from ..data.loader import load_audio

    # Load full audio
    audio, sr = load_audio(audio_path, target_sr=sample_rate)

    # Calculate chunk size in samples
    chunk_samples = int(sample_rate * chunk_size_ms / 1000)

    # Yield chunks
    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i : i + chunk_samples]
        yield chunk

        # Simulate real-time delay
        # time.sleep(chunk_size_ms / 1000)  # Uncomment for real-time simulation


def run_streaming_benchmark(
    model,
    audio_path: str,
    chunk_sizes_ms: List[int] = [250, 500, 1000],
    language: str = "tr",
) -> Dict[str, StreamingResult]:
    """Run streaming benchmark with different chunk sizes.

    Args:
        model: STT model with transcribe_streaming method
        audio_path: Path to audio file
        chunk_sizes_ms: List of chunk sizes to test
        language: Language code

    Returns:
        Dictionary mapping chunk_size to StreamingResult
    """
    results = {}

    for chunk_size in chunk_sizes_ms:
        logger.info(f"Testing chunk size: {chunk_size}ms")

        chunk_results = []
        first_word_latency = None
        start_time = time.perf_counter()

        try:
            for result in model.transcribe_streaming(
                audio_path, chunk_size_ms=chunk_size, language=language
            ):
                chunk_results.append(result)

                # Record first word latency
                if first_word_latency is None and result.get("text", "").strip():
                    first_word_latency = result.get("latency_ms", 0)

            total_latency = (time.perf_counter() - start_time) * 1000

            # Get final text
            final_text = ""
            if chunk_results:
                final_result = chunk_results[-1]
                final_text = final_result.get("text", "")

            results[chunk_size] = StreamingResult(
                model_name=model.model_name,
                audio_file=audio_path,
                chunk_size_ms=chunk_size,
                total_chunks=len(chunk_results),
                first_word_latency_ms=first_word_latency,
                total_latency_ms=total_latency,
                final_text=final_text,
                chunk_results=chunk_results,
            )

        except NotImplementedError:
            logger.warning(f"Streaming not supported for {model.model_name}")
            break
        except Exception as e:
            logger.error(f"Streaming test failed: {e}")
            continue

    return results


def analyze_streaming_results(results: Dict[int, StreamingResult]) -> Dict[str, Any]:
    """Analyze streaming benchmark results.

    Args:
        results: Dictionary of StreamingResult by chunk size

    Returns:
        Analysis summary
    """
    if not results:
        return {"error": "No results to analyze"}

    analysis = {
        "model_name": list(results.values())[0].model_name,
        "chunk_sizes_tested": list(results.keys()),
        "by_chunk_size": {},
    }

    for chunk_size, result in results.items():
        analysis["by_chunk_size"][chunk_size] = {
            "first_word_latency_ms": result.first_word_latency_ms,
            "total_latency_ms": result.total_latency_ms,
            "total_chunks": result.total_chunks,
            "final_text_length": len(result.final_text),
        }

    # Find optimal chunk size (lowest first word latency)
    valid_results = [
        (cs, r) for cs, r in results.items() if r.first_word_latency_ms is not None
    ]
    if valid_results:
        optimal = min(valid_results, key=lambda x: x[1].first_word_latency_ms)
        analysis["optimal_chunk_size_ms"] = optimal[0]
        analysis["optimal_first_word_latency_ms"] = optimal[1].first_word_latency_ms

    return analysis


class StreamingLatencyProfiler:
    """Profile streaming latency characteristics of a model."""

    def __init__(self, model, sample_rate: int = 16000):
        """Initialize profiler.

        Args:
            model: STT model to profile
            sample_rate: Audio sample rate
        """
        self.model = model
        self.sample_rate = sample_rate
        self.results: List[Dict[str, Any]] = []

    def profile(
        self,
        audio_files: List[str],
        chunk_sizes_ms: List[int] = [250, 500, 1000],
        language: str = "tr",
    ) -> Dict[str, Any]:
        """Run profiling on multiple audio files.

        Args:
            audio_files: List of audio file paths
            chunk_sizes_ms: Chunk sizes to test
            language: Language code

        Returns:
            Profiling summary
        """
        all_results = []

        for audio_file in audio_files:
            logger.info(f"Profiling: {audio_file}")

            file_results = run_streaming_benchmark(
                self.model,
                audio_file,
                chunk_sizes_ms=chunk_sizes_ms,
                language=language,
            )

            all_results.append({
                "file": audio_file,
                "results": file_results,
            })

        # Aggregate statistics
        aggregate = {
            "model_name": self.model.model_name,
            "num_files": len(audio_files),
            "chunk_sizes_tested": chunk_sizes_ms,
            "statistics": {},
        }

        for chunk_size in chunk_sizes_ms:
            latencies = []
            first_word_latencies = []

            for file_result in all_results:
                if chunk_size in file_result["results"]:
                    result = file_result["results"][chunk_size]
                    latencies.append(result.total_latency_ms)
                    if result.first_word_latency_ms is not None:
                        first_word_latencies.append(result.first_word_latency_ms)

            if latencies:
                aggregate["statistics"][chunk_size] = {
                    "avg_total_latency_ms": np.mean(latencies),
                    "std_total_latency_ms": np.std(latencies),
                    "avg_first_word_latency_ms": (
                        np.mean(first_word_latencies) if first_word_latencies else None
                    ),
                    "std_first_word_latency_ms": (
                        np.std(first_word_latencies) if first_word_latencies else None
                    ),
                }

        self.results = all_results
        return aggregate

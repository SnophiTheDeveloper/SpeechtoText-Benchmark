"""Benchmark runner for STT model evaluation."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from tqdm import tqdm

from .metrics import calculate_metrics, calculate_aggregate_metrics, TranscriptionMetrics
from .utils import (
    save_results,
    generate_result_filename,
    Timer,
    validate_audio_file,
    get_device_info,
)
from ..models.base import BaseSTTModel
from ..data.loader import load_test_data, TestDataset

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    model_name: str
    model_family: str
    language: str
    device: str
    timestamp: str
    test_data_info: Dict[str, Any]
    aggregate_metrics: Dict[str, float]
    per_file_results: List[Dict[str, Any]]
    errors: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_name": self.model_name,
            "model_family": self.model_family,
            "language": self.language,
            "device": self.device,
            "timestamp": self.timestamp,
            "test_data_info": self.test_data_info,
            "aggregate_metrics": self.aggregate_metrics,
            "per_file_results": self.per_file_results,
            "errors": self.errors,
        }


class BenchmarkRunner:
    """Main benchmark runner for STT model evaluation.

    Runs batch tests on audio files and calculates metrics.

    Example:
        >>> runner = BenchmarkRunner(
        ...     test_data_dir="test_data/turkish",
        ...     results_dir="results/whisper",
        ...     device="cuda"
        ... )
        >>> model = FasterWhisperModel("large-v3-turbo")
        >>> results = runner.run(model, language="tr")
        >>> print(f"WER: {results.aggregate_metrics['wer']:.2%}")
    """

    def __init__(
        self,
        test_data_dir: str | Path,
        results_dir: str | Path,
        device: str = "cpu",
    ):
        """Initialize the benchmark runner.

        Args:
            test_data_dir: Directory containing test audio and transcripts
            results_dir: Directory to save results
            device: Compute device ("cpu", "cuda", "cuda:0")
        """
        self.test_data_dir = Path(test_data_dir)
        self.results_dir = Path(results_dir)
        self.device = device

        # Create results directory if needed
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        model: BaseSTTModel,
        language: str = "tr",
        save: bool = True,
        show_progress: bool = True,
    ) -> BenchmarkResult:
        """Run benchmark on a model.

        Args:
            model: STT model to benchmark
            language: Language code
            save: Whether to save results to file
            show_progress: Whether to show progress bar

        Returns:
            BenchmarkResult with all metrics
        """
        logger.info(f"Starting benchmark for {model.model_name}")
        timestamp = datetime.now()

        # Load test data
        try:
            test_data = load_test_data(self.test_data_dir)
        except Exception as e:
            logger.error(f"Failed to load test data: {e}")
            raise

        # Load model
        logger.info(f"Loading model {model.model_name}...")
        with Timer() as load_timer:
            model.load()
        logger.info(f"Model loaded in {load_timer.elapsed:.2f}s")

        # Process each file
        per_file_results: List[Dict[str, Any]] = []
        metrics_list: List[TranscriptionMetrics] = []
        errors: List[Dict[str, str]] = []

        files_iter = test_data.files
        if show_progress:
            files_iter = tqdm(files_iter, desc=f"Testing {model.model_name}")

        for file_info in files_iter:
            audio_path = self.test_data_dir / "audio" / file_info["filename"]

            # Skip if file doesn't exist
            if not audio_path.exists():
                error_msg = f"Audio file not found: {audio_path}"
                logger.warning(error_msg)
                errors.append({"file": file_info["filename"], "error": error_msg})
                continue

            # Validate audio file
            if not validate_audio_file(audio_path):
                error_msg = f"Invalid audio file: {audio_path}"
                logger.warning(error_msg)
                errors.append({"file": file_info["filename"], "error": error_msg})
                continue

            # Transcribe
            try:
                result = model.transcribe(str(audio_path), language=language)

                # Calculate metrics
                metrics = calculate_metrics(
                    reference=file_info["transcript"],
                    hypothesis=result["text"],
                    processing_time_seconds=result["processing_time_seconds"],
                    audio_duration_seconds=result.get(
                        "duration_seconds", file_info.get("duration_seconds", 1.0)
                    ),
                    language=language,
                )

                metrics_list.append(metrics)

                per_file_results.append({
                    "file": file_info["filename"],
                    "ground_truth": file_info["transcript"],
                    "prediction": result["text"],
                    "wer": metrics.wer,
                    "cer": metrics.cer,
                    "duration_seconds": result.get(
                        "duration_seconds", file_info.get("duration_seconds")
                    ),
                    "processing_time_seconds": result["processing_time_seconds"],
                    "rtf": metrics.rtf,
                })

            except Exception as e:
                error_msg = f"Transcription error: {str(e)}"
                logger.error(f"Error processing {file_info['filename']}: {error_msg}")
                errors.append({"file": file_info["filename"], "error": error_msg})
                continue

        # Calculate aggregate metrics
        aggregate_metrics = calculate_aggregate_metrics(metrics_list)

        # Get test data info
        test_data_info = {
            "dataset": test_data.dataset_name,
            "num_files": len(test_data.files),
            "num_processed": len(per_file_results),
            "num_errors": len(errors),
            "total_duration_seconds": sum(
                f.get("duration_seconds", 0) for f in test_data.files
            ),
        }

        # Get model info
        model_info = model.get_model_info()

        # Create result
        result = BenchmarkResult(
            model_name=model.model_name,
            model_family=model_info.get("family", "unknown"),
            language=language,
            device=self.device,
            timestamp=timestamp.isoformat(),
            test_data_info=test_data_info,
            aggregate_metrics=aggregate_metrics,
            per_file_results=per_file_results,
            errors=errors,
        )

        # Save results
        if save:
            output_path = save_results(
                result.to_dict(),
                self.results_dir / model_info.get("family", "unknown"),
            )
            logger.info(f"Results saved to {output_path}")

        # Unload model to free memory
        model.unload()

        logger.info(
            f"Benchmark complete: WER={aggregate_metrics['wer']:.2%}, "
            f"CER={aggregate_metrics['cer']:.2%}, "
            f"RTF={aggregate_metrics['avg_rtf']:.3f}"
        )

        return result

    def run_streaming_test(
        self,
        model: BaseSTTModel,
        language: str = "tr",
        chunk_size_ms: int = 500,
        save: bool = True,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """Run streaming/latency test on a model.

        Tests first-word latency and streaming performance.

        Args:
            model: STT model to test
            language: Language code
            chunk_size_ms: Chunk size in milliseconds
            save: Whether to save results
            show_progress: Whether to show progress bar

        Returns:
            Dictionary with streaming test results
        """
        logger.info(f"Starting streaming test for {model.model_name}")
        timestamp = datetime.now()

        # Load test data
        test_data = load_test_data(self.test_data_dir)

        # Load model
        model.load()

        results = {
            "model_name": model.model_name,
            "chunk_size_ms": chunk_size_ms,
            "language": language,
            "timestamp": timestamp.isoformat(),
            "files": [],
        }

        files_iter = test_data.files
        if show_progress:
            files_iter = tqdm(files_iter, desc=f"Streaming test {model.model_name}")

        for file_info in files_iter:
            audio_path = self.test_data_dir / "audio" / file_info["filename"]

            if not audio_path.exists():
                continue

            try:
                file_result = {
                    "file": file_info["filename"],
                    "chunks": [],
                    "first_word_latency_ms": None,
                    "total_latency_ms": 0,
                }

                first_word_received = False

                for chunk_result in model.transcribe_streaming(
                    str(audio_path), chunk_size_ms=chunk_size_ms, language=language
                ):
                    file_result["chunks"].append(chunk_result)

                    # Record first word latency
                    if not first_word_received and chunk_result.get("text", "").strip():
                        file_result["first_word_latency_ms"] = chunk_result.get(
                            "latency_ms", 0
                        )
                        first_word_received = True

                    if chunk_result.get("is_final"):
                        file_result["total_latency_ms"] = chunk_result.get(
                            "latency_ms", 0
                        )
                        file_result["final_text"] = chunk_result.get("text", "")

                results["files"].append(file_result)

            except NotImplementedError:
                logger.warning(
                    f"Streaming not supported for {model.model_name}"
                )
                break
            except Exception as e:
                logger.error(f"Streaming error for {file_info['filename']}: {e}")
                continue

        # Calculate aggregate streaming metrics
        if results["files"]:
            first_word_latencies = [
                f["first_word_latency_ms"]
                for f in results["files"]
                if f["first_word_latency_ms"] is not None
            ]
            if first_word_latencies:
                results["avg_first_word_latency_ms"] = sum(first_word_latencies) / len(
                    first_word_latencies
                )

        model.unload()

        if save:
            filename = f"{model.model_name.replace('/', '_')}_streaming_{timestamp.strftime('%Y-%m-%d_%H-%M')}.json"
            output_path = self.results_dir / "streaming" / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        return results

    def compare_models(
        self,
        models: List[BaseSTTModel],
        language: str = "tr",
    ) -> Dict[str, Any]:
        """Run benchmark on multiple models and compare results.

        Args:
            models: List of models to compare
            language: Language code

        Returns:
            Comparison dictionary with ranked results
        """
        all_results = []

        for model in models:
            try:
                result = self.run(model, language=language, save=True)
                all_results.append(result)
            except Exception as e:
                logger.error(f"Failed to benchmark {model.model_name}: {e}")
                continue

        # Sort by WER
        all_results.sort(key=lambda r: r.aggregate_metrics.get("wer", float("inf")))

        comparison = {
            "language": language,
            "device": self.device,
            "timestamp": datetime.now().isoformat(),
            "rankings": [
                {
                    "rank": i + 1,
                    "model_name": r.model_name,
                    "model_family": r.model_family,
                    "wer": r.aggregate_metrics.get("wer"),
                    "cer": r.aggregate_metrics.get("cer"),
                    "avg_rtf": r.aggregate_metrics.get("avg_rtf"),
                }
                for i, r in enumerate(all_results)
            ],
        }

        # Save comparison
        comp_path = self.results_dir / f"comparison_{language}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.json"
        with open(comp_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)

        return comparison

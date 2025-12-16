"""Metrics calculation for STT benchmark evaluation.

Uses jiwer library for WER and CER calculations.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

try:
    from jiwer import wer, cer, process_words, process_characters
except ImportError:
    raise ImportError("jiwer is required for metrics. Install with: pip install jiwer")

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionMetrics:
    """Container for transcription evaluation metrics."""

    wer: float  # Word Error Rate
    cer: float  # Character Error Rate
    rtf: float  # Real-Time Factor (processing_time / audio_duration)
    latency_ms: float  # Processing latency in milliseconds
    num_words_reference: int
    num_words_hypothesis: int
    num_chars_reference: int
    num_chars_hypothesis: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "wer": self.wer,
            "cer": self.cer,
            "rtf": self.rtf,
            "latency_ms": self.latency_ms,
            "num_words_reference": self.num_words_reference,
            "num_words_hypothesis": self.num_words_hypothesis,
            "num_chars_reference": self.num_chars_reference,
            "num_chars_hypothesis": self.num_chars_hypothesis,
        }


def normalize_text(text: str, language: str = "tr") -> str:
    """Normalize text for comparison.

    Args:
        text: Input text to normalize
        language: Language code for language-specific normalization

    Returns:
        Normalized text
    """
    # Basic normalization
    text = text.lower().strip()

    # Remove extra whitespace
    text = " ".join(text.split())

    # Language-specific normalization
    if language == "tr":
        # Turkish-specific: handle common character variations
        # Keep Turkish characters as-is for proper comparison
        pass
    elif language == "en":
        # English-specific normalization if needed
        pass

    return text


def calculate_wer(
    reference: str,
    hypothesis: str,
    normalize: bool = True,
    language: str = "tr"
) -> float:
    """Calculate Word Error Rate between reference and hypothesis.

    WER = (Substitutions + Insertions + Deletions) / Total Words in Reference

    Args:
        reference: Ground truth transcription
        hypothesis: Model prediction
        normalize: Whether to normalize texts before comparison
        language: Language code for normalization

    Returns:
        WER as a float (0.0 to 1.0+, can exceed 1.0 if many insertions)
    """
    if normalize:
        reference = normalize_text(reference, language)
        hypothesis = normalize_text(hypothesis, language)

    if not reference:
        logger.warning("Empty reference text, returning WER=1.0")
        return 1.0 if hypothesis else 0.0

    try:
        error_rate = wer(reference, hypothesis)
        return error_rate
    except Exception as e:
        logger.error(f"Error calculating WER: {e}")
        return 1.0


def calculate_cer(
    reference: str,
    hypothesis: str,
    normalize: bool = True,
    language: str = "tr"
) -> float:
    """Calculate Character Error Rate between reference and hypothesis.

    CER = (Substitutions + Insertions + Deletions) / Total Characters in Reference

    Args:
        reference: Ground truth transcription
        hypothesis: Model prediction
        normalize: Whether to normalize texts before comparison
        language: Language code for normalization

    Returns:
        CER as a float (0.0 to 1.0+)
    """
    if normalize:
        reference = normalize_text(reference, language)
        hypothesis = normalize_text(hypothesis, language)

    if not reference:
        logger.warning("Empty reference text, returning CER=1.0")
        return 1.0 if hypothesis else 0.0

    try:
        error_rate = cer(reference, hypothesis)
        return error_rate
    except Exception as e:
        logger.error(f"Error calculating CER: {e}")
        return 1.0


def calculate_rtf(
    processing_time_seconds: float,
    audio_duration_seconds: float
) -> float:
    """Calculate Real-Time Factor.

    RTF = Processing Time / Audio Duration
    RTF < 1.0 means faster than real-time
    RTF > 1.0 means slower than real-time

    Args:
        processing_time_seconds: Time taken to process audio
        audio_duration_seconds: Duration of the audio file

    Returns:
        Real-Time Factor
    """
    if audio_duration_seconds <= 0:
        logger.warning("Invalid audio duration, returning RTF=inf")
        return float("inf")

    return processing_time_seconds / audio_duration_seconds


def calculate_latency_ms(processing_time_seconds: float) -> float:
    """Convert processing time to milliseconds.

    Args:
        processing_time_seconds: Processing time in seconds

    Returns:
        Latency in milliseconds
    """
    return processing_time_seconds * 1000


def calculate_metrics(
    reference: str,
    hypothesis: str,
    processing_time_seconds: float,
    audio_duration_seconds: float,
    language: str = "tr",
    normalize: bool = True
) -> TranscriptionMetrics:
    """Calculate all metrics for a single transcription.

    Args:
        reference: Ground truth transcription
        hypothesis: Model prediction
        processing_time_seconds: Time taken to process
        audio_duration_seconds: Audio file duration
        language: Language code
        normalize: Whether to normalize texts

    Returns:
        TranscriptionMetrics object with all calculated metrics
    """
    # Normalize for comparison
    ref_normalized = normalize_text(reference, language) if normalize else reference
    hyp_normalized = normalize_text(hypothesis, language) if normalize else hypothesis

    # Calculate error rates
    wer_score = calculate_wer(reference, hypothesis, normalize, language)
    cer_score = calculate_cer(reference, hypothesis, normalize, language)

    # Calculate timing metrics
    rtf = calculate_rtf(processing_time_seconds, audio_duration_seconds)
    latency = calculate_latency_ms(processing_time_seconds)

    # Count words and characters
    ref_words = ref_normalized.split()
    hyp_words = hyp_normalized.split()

    return TranscriptionMetrics(
        wer=wer_score,
        cer=cer_score,
        rtf=rtf,
        latency_ms=latency,
        num_words_reference=len(ref_words),
        num_words_hypothesis=len(hyp_words),
        num_chars_reference=len(ref_normalized.replace(" ", "")),
        num_chars_hypothesis=len(hyp_normalized.replace(" ", "")),
    )


def calculate_aggregate_metrics(
    metrics_list: List[TranscriptionMetrics]
) -> Dict[str, float]:
    """Calculate aggregate metrics from a list of individual metrics.

    Args:
        metrics_list: List of TranscriptionMetrics objects

    Returns:
        Dictionary with aggregate statistics
    """
    if not metrics_list:
        return {
            "wer": 0.0,
            "cer": 0.0,
            "avg_rtf": 0.0,
            "avg_latency_ms": 0.0,
            "total_words": 0,
            "total_chars": 0,
        }

    # Calculate weighted averages based on reference length
    total_ref_words = sum(m.num_words_reference for m in metrics_list)
    total_ref_chars = sum(m.num_chars_reference for m in metrics_list)

    # For WER: weight by word count
    if total_ref_words > 0:
        weighted_wer = sum(
            m.wer * m.num_words_reference for m in metrics_list
        ) / total_ref_words
    else:
        weighted_wer = 0.0

    # For CER: weight by character count
    if total_ref_chars > 0:
        weighted_cer = sum(
            m.cer * m.num_chars_reference for m in metrics_list
        ) / total_ref_chars
    else:
        weighted_cer = 0.0

    # Simple averages for timing metrics
    avg_rtf = sum(m.rtf for m in metrics_list) / len(metrics_list)
    avg_latency = sum(m.latency_ms for m in metrics_list) / len(metrics_list)

    return {
        "wer": weighted_wer,
        "cer": weighted_cer,
        "avg_rtf": avg_rtf,
        "avg_latency_ms": avg_latency,
        "total_words": total_ref_words,
        "total_chars": total_ref_chars,
        "num_files": len(metrics_list),
    }


def get_detailed_word_errors(
    reference: str,
    hypothesis: str,
    language: str = "tr"
) -> Dict[str, Any]:
    """Get detailed word-level error analysis.

    Args:
        reference: Ground truth transcription
        hypothesis: Model prediction
        language: Language code

    Returns:
        Dictionary with detailed error breakdown
    """
    ref = normalize_text(reference, language)
    hyp = normalize_text(hypothesis, language)

    try:
        output = process_words(ref, hyp)
        return {
            "substitutions": output.substitutions,
            "insertions": output.insertions,
            "deletions": output.deletions,
            "hits": output.hits,
            "wer": output.wer,
            "alignments": [
                {
                    "type": align.type,
                    "ref": align.ref if hasattr(align, "ref") else None,
                    "hyp": align.hyp if hasattr(align, "hyp") else None,
                }
                for align in output.alignments
            ] if hasattr(output, "alignments") else [],
        }
    except Exception as e:
        logger.error(f"Error getting detailed word errors: {e}")
        return {"error": str(e)}

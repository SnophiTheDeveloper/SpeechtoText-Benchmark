"""Setup script for STT Benchmark Framework."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="stt-benchmark",
    version="0.1.0",
    author="STT Benchmark Team",
    description="A modular benchmark framework for Turkish and English speech-to-text models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/USERNAME/stt-benchmark",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "jiwer>=3.0.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "numpy>=1.24.0",
        "soundfile>=0.12.0",
        "librosa>=0.10.0",
    ],
    extras_require={
        "whisper": [
            "faster-whisper>=1.0.0",
            "transformers>=4.35.0",
            "torch>=2.0.0",
        ],
        "wav2vec2": [
            "transformers>=4.35.0",
            "torch>=2.0.0",
            "torchaudio>=2.0.0",
        ],
        "hubert": [
            "transformers>=4.35.0",
            "torch>=2.0.0",
        ],
        "realtime": [
            "RealtimeSTT>=0.1.0",
            "pyaudio>=0.2.13",
            "webrtcvad>=2.0.10",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "stt-benchmark=scripts.run_benchmark:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
    keywords="speech-to-text, stt, benchmark, whisper, wav2vec2, hubert, turkish",
)

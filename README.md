# STT Model Benchmark Framework

Türkçe ve İngilizce speech-to-text modellerini karşılaştırmak için modüler bir benchmark framework'ü.

## Özellikler

- **Çoklu Model Desteği**: Whisper, Wav2Vec2, HuBERT ailelerini destekler
- **Docker-First**: Her şey container'da çalışır, kurulum derdi yok
- **Custom Model Desteği**: Kendi finetuned modellerini (CT2 formatında) ekleyebilirsin
- **Streaming Test**: Real-time latency ölçümü
- **Modüler Mimari**: Yeni modeller kolayca eklenebilir
- **Offline Mod**: İnternet olmadan sunucuda çalışabilir
- **Otomatik Metrikler**: WER, CER, RTF, latency hesaplama

## Desteklenen Modeller

### Whisper Ailesi
| Model | HuggingFace ID / Path | Diller |
|-------|----------------------|--------|
| faster-whisper-large-v3-turbo | Systran/faster-whisper-large-v3-turbo | TR, EN |
| faster-whisper-large-v3 | Systran/faster-whisper-large-v3 | TR, EN |
| faster-whisper-medium | Systran/faster-whisper-medium | TR, EN |
| faster-whisper-small | Systran/faster-whisper-small | TR, EN |
| whisper-small-tr-finetuned | models/whisper-small-tr-ct2-int8 (lokal) | TR |
| distil-whisper-tr | Sercan/distil-whisper-large-v3-tr | TR |

### Wav2Vec2 Ailesi
| Model | HuggingFace ID | Diller |
|-------|----------------|--------|
| wav2vec2-turkish-large | m3hrdadfi/wav2vec2-large-xlsr-turkish | TR |
| wav2vec2-turkish-base | cahya/wav2vec2-base-turkish | TR |
| wav2vec2-xlsr-53 | facebook/wav2vec2-large-xlsr-53 | Multi |

### HuBERT Ailesi
| Model | HuggingFace ID | Diller |
|-------|----------------|--------|
| hubert-large-ft | facebook/hubert-large-ls960-ft | EN |

## Hızlı Başlangıç (Docker)

### Gereksinimler
- Docker Desktop kurulu olmalı
- GPU kullanacaksan: NVIDIA Container Toolkit

### 1. Docker Image Oluştur

```bash
cd docker

# CPU versiyonu (GPU yoksa)
docker-compose build stt-benchmark-cpu

# GPU versiyonu (NVIDIA GPU varsa)
docker-compose build stt-benchmark
```

### 2. Mevcut Modelleri Listele

```bash
docker-compose run --rm stt-benchmark-cpu python scripts/run_benchmark.py --list-models
```

### 3. Benchmark Çalıştır

```bash
# Türkçe test (test_data/tr klasöründe veriler olmalı)
docker-compose run --rm stt-benchmark-cpu python scripts/run_benchmark.py \
    --model faster-whisper-small --language tr --device cpu --test-data test_data/tr

# İngilizce test
docker-compose run --rm stt-benchmark-cpu python scripts/run_benchmark.py \
    --model faster-whisper-small --language en --device cpu --test-data test_data/en

# Custom finetuned model test
docker-compose run --rm stt-benchmark-cpu python scripts/run_benchmark.py \
    --model whisper-small-tr-finetuned --language tr --device cpu --test-data test_data/tr
```

### 4. Streaming/Real-time Test

```bash
# 500ms chunk boyutuyla streaming test
docker-compose run --rm stt-benchmark-cpu python scripts/run_benchmark.py \
    --model faster-whisper-small --language tr --device cpu --test-data test_data/tr \
    --streaming --chunk-size 500
```

## Test Verileri Hazırlama

### Klasör Yapısı
```
test_data/
├── tr/
│   ├── audio/
│   │   ├── sample1.mp3
│   │   └── sample2.mp3
│   └── transcripts.json
└── en/
    ├── audio/
    │   ├── audio1.mp3
    │   └── audio2.mp3
    └── transcripts.json
```

### transcripts.json Formatı
```json
{
  "dataset": "my_dataset",
  "files": [
    {
      "filename": "sample1.mp3",
      "transcript": "Bu bir test cümlesidir."
    },
    {
      "filename": "sample2.mp3",
      "transcript": "İkinci örnek metin."
    }
  ]
}
```

## Custom Model Ekleme

Kendi finetuned modelini eklemek için:

1. Modeli CTranslate2 formatına dönüştür (int8 quantized önerilir)
2. `models/` klasörüne koy (örn: `models/my-model-ct2-int8/`)
3. `scripts/run_benchmark.py`'de MODEL_REGISTRY'ye ekle:

```python
"my-custom-model": lambda device: FasterWhisperModel(
    "my-model-ct2-int8",
    model_path=Path("models/my-model-ct2-int8"),
    device=device,
    compute_type="int8",
),
```

## Proje Yapısı

```
stt-benchmark/
├── src/
│   ├── benchmark/           # Benchmark runner ve metrikler
│   │   ├── runner.py        # Ana benchmark sınıfı
│   │   ├── metrics.py       # WER, CER hesaplama
│   │   └── utils.py         # Yardımcı fonksiyonlar
│   ├── models/              # Model wrappers
│   │   ├── base.py          # Abstract base class
│   │   ├── whisper_model.py # Whisper implementations
│   │   ├── wav2vec2_model.py
│   │   └── hubert_model.py
│   ├── data/                # Veri yükleme
│   │   ├── loader.py        # Test data loader
│   │   └── downloader.py    # Dataset downloader
│   └── realtime/            # Real-time test modülleri
├── docker/                  # Docker dosyaları
│   ├── Dockerfile           # GPU image
│   ├── Dockerfile.cpu       # CPU-only image
│   └── docker-compose.yml   # Compose konfigürasyonu
├── models/                  # Custom/finetuned modeller
├── scripts/                 # CLI script'leri
├── test_data/              # Test audio ve transkriptler
└── results/                # Benchmark sonuçları (JSON)
```

## Benchmark Sonuçları

### Örnek Sonuçlar (CPU, 10 örnek)

| Model | WER | CER | RTF | Yükleme Süresi |
|-------|-----|-----|-----|----------------|
| faster-whisper-small | 51.39% | 21.93% | 0.713 | 52s |
| whisper-small-tr-finetuned | 40.28% | 15.36% | 0.582 | 5.4s |

### Sonuç Formatı

Sonuçlar `results/` klasörüne JSON formatında kaydedilir:

```json
{
    "model_name": "faster-whisper-large-v3-turbo",
    "model_family": "whisper",
    "language": "tr",
    "aggregate_metrics": {
        "wer": 0.12,
        "cer": 0.08,
        "avg_rtf": 0.15,
        "avg_latency_ms": 450
    },
    "per_file_results": [...]
}
```

## Metrikler

| Metrik | Açıklama |
|--------|----------|
| WER | Word Error Rate - Kelime hata oranı (düşük = iyi) |
| CER | Character Error Rate - Karakter hata oranı |
| RTF | Real-Time Factor - İşleme süresi / ses süresi (< 1.0 = gerçek zamandan hızlı) |
| Latency | İşleme süresi (ms) |
| First Word Latency | İlk kelime algılama süresi (streaming için) |

## Model İndirme

```bash
# Tüm Whisper modellerini indir
docker-compose run --rm stt-benchmark-cpu python scripts/download_models.py --family whisper

# Tek model indir
docker-compose run --rm stt-benchmark-cpu python scripts/download_models.py --model faster-whisper-small
```

## Offline Sunucu Kullanımı

İnterneti olmayan sunucuda çalıştırmak için:

```bash
# 1. İnterneti olan makinede modelleri indir
docker-compose run --rm stt-benchmark python scripts/download_models.py --family whisper

# 2. models/ ve test_data/ klasörlerini sunucuya kopyala

# 3. Sunucuda offline modda çalıştır
docker-compose run --rm stt-benchmark-offline python scripts/run_benchmark.py \
    --model faster-whisper-large-v3 --language tr
```

## Geliştirme

```bash
# Interactive shell
docker-compose run --rm stt-benchmark-cpu bash

# Test çalıştır
docker-compose run --rm stt-benchmark-cpu python -c "import src; print('OK')"
```

## Lisans

MIT License

## Referanslar

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [Transformers](https://huggingface.co/transformers)
- [Common Voice](https://commonvoice.mozilla.org/)
- [jiwer](https://github.com/jitsi/jiwer)

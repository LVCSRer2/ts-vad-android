# TS-VAD Android

Real-time **Target Speaker Voice Activity Detection** on Android.

The app detects when a specific enrolled speaker is talking, distinguishing them from other speakers and silence — all running on-device.

## How It Works

1. **Voice Enrollment** — Record 5 seconds of your voice to create a speaker profile
2. **Real-time Detection** — Classify each audio frame as:
   - **Target** (enrolled speaker is talking)
   - **Non-target** (someone else is talking)
   - **Silence**

## Architecture

```
Microphone (16kHz mono)
    │
    ├──► [Mel Spectrogram] ──► [Speaker Encoder] ──► 256-dim d-vector (enrollment)
    │         │
    │         ▼
    └──► [Log-Mel Fbank] + [d-vector] ──► [Personal VAD LSTM] ──► 3-class output
```

### Models

| Model | Architecture | Size | Description |
|-------|-------------|------|-------------|
| Speaker Encoder | 3-layer LSTM (GE2E) | 5.4 MB | [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) d-vector encoder |
| Personal VAD | 2-layer LSTM + FC | 512 KB | Speaker-conditioned VAD ([pirxus/personalVAD](https://github.com/pirxus/personalVAD)) |

### Feature Extraction

- 40-dim Mel filterbank features (librosa-compatible)
- Pre-computed Slaney-normalized mel filterbank loaded from binary
- Exact N-point FFT via Bluestein (Chirp-Z) algorithm for librosa STFT compatibility
- `center=True` padding, power spectrum without normalization

## Project Structure

```
android-app/          # Android application (Kotlin + Jetpack Compose)
├── app/src/main/
│   ├── assets/       # ONNX models + mel filterbank
│   │   ├── speaker_encoder.onnx
│   │   ├── personal_vad.onnx
│   │   └── mel_filterbank.bin
│   └── java/com/example/tsvad/
│       ├── MainActivity.kt
│       ├── MainViewModel.kt
│       ├── audio/
│       │   ├── AudioCapturer.kt      # 16kHz PCM capture
│       │   └── FeatureExtractor.kt   # Mel spectrogram (librosa-compatible)
│       ├── model/
│       │   ├── SpeakerEncoder.kt     # Resemblyzer d-vector extraction
│       │   └── PersonalVAD.kt        # Stateful LSTM inference
│       ├── data/
│       │   └── EmbeddingStore.kt     # Speaker embedding persistence
│       └── ui/
│           ├── EnrollScreen.kt
│           └── DetectionScreen.kt
│
export/               # Python model export scripts
├── personal_vad_model.py    # Personal VAD definition + ONNX export
├── download_models.py       # Download pretrained weights
├── resemblyzer_encoder.onnx # Pre-exported speaker encoder
└── mel_filterbank.bin       # Pre-computed librosa mel filterbank
```

## Build

### Prerequisites

- Android SDK (API 24+)
- JDK 17

### Steps

```bash
cd android-app
./gradlew assembleDebug
```

The APK will be at `android-app/app/build/outputs/apk/debug/app-debug.apk`.

### Model Preparation (optional)

The ONNX models are included in `android-app/app/src/main/assets/`. To re-export from scratch:

```bash
cd export
pip install torch resemblyzer onnx librosa numpy
python download_models.py
python personal_vad_model.py
```

## Tech Stack

- **Kotlin** + **Jetpack Compose** for UI
- **ONNX Runtime Android** for on-device inference
- **AudioRecord API** for real-time 16kHz audio capture
- Pure Kotlin mel spectrogram extraction (no native dependencies)

## References

- [Personal VAD: Speaker-Conditioned Voice Activity Detection](https://arxiv.org/abs/2104.01167) — Google Research
- [pirxus/personalVAD](https://github.com/pirxus/personalVAD) — PyTorch implementation
- [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) — GE2E speaker encoder

## License

This project builds upon:
- Personal VAD model weights from [pirxus/personalVAD](https://github.com/pirxus/personalVAD)
- Speaker encoder from [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) (Apache 2.0)
- ONNX Runtime (MIT License)

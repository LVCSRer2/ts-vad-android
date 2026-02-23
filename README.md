# TS-VAD Android

안드로이드에서 실시간으로 동작하는 **타겟 화자 음성 활동 감지 (Target Speaker Voice Activity Detection)** 앱입니다.

등록된 화자가 말하고 있는지를 다른 화자 및 침묵과 구분하여 감지하며, 모든 추론은 온디바이스로 수행됩니다.

## 사용 방법

1. **음성 등록** — 5초간 자신의 목소리를 녹음하여 화자 프로필 생성
2. **실시간 감지** — 각 오디오 프레임을 3가지로 분류:
   - **Target** (등록된 화자가 말하는 중)
   - **Non-target** (다른 사람이 말하는 중)
   - **Silence** (침묵)

## 아키텍처

```
마이크 (16kHz 모노)
    │
    ├──► [Mel Spectrogram] ──► [Speaker Encoder] ──► 256차원 d-vector (등록 시)
    │         │
    │         ▼
    └──► [Log-Mel Fbank] + [d-vector] ──► [Personal VAD LSTM] ──► 3클래스 출력
```

### 모델

| 모델 | 구조 | 크기 | 설명 |
|------|------|------|------|
| Speaker Encoder | 3-layer LSTM (GE2E) | 5.4 MB | [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) d-vector 인코더 |
| Personal VAD | 2-layer LSTM + FC | 512 KB | 화자 조건부 VAD ([pirxus/personalVAD](https://github.com/pirxus/personalVAD)) |

### 추론 전략

장시간 사용 시 LSTM hidden state drift를 방지하기 위한 **Stateless Sliding Window** 방식:

1. 매 추론마다 LSTM hidden state를 0으로 초기화
2. Context 버퍼 50프레임(500ms)을 25프레임 단위로 재생하여 LSTM warmup (결과 버림)
3. 현재 25프레임(250ms) 청크를 추론하여 예측 결과 사용

이를 통해 앱 실행 시간에 관계없이 안정적인 감지가 가능합니다.

### 특징 추출

- 40차원 Mel filterbank 특징 (librosa 호환)
- 사전 계산된 Slaney 정규화 mel filterbank를 바이너리 파일에서 로드
- **Bluestein (Chirp-Z) 알고리즘**으로 정확한 N-point FFT 수행 — 2의 거듭제곱 zero-padding 없이 librosa STFT 출력과 정확히 일치
- `center=True` 패딩, power spectrum 정규화 없음
- 검증: librosa 대비 Pearson 상관계수 1.0, 최대 오차 ~1e-6

## 프로젝트 구조

```
android-app/                # 안드로이드 앱 (Kotlin + Jetpack Compose)
├── app/src/main/
│   ├── assets/             # ONNX 모델 + mel filterbank
│   │   ├── speaker_encoder.onnx   (5.4 MB)
│   │   ├── personal_vad.onnx      (512 KB)
│   │   └── mel_filterbank.bin     (31 KB)
│   └── java/com/example/tsvad/
│       ├── MainActivity.kt        # 권한 처리, 네비게이션
│       ├── MainViewModel.kt       # 등록 + 감지 오케스트레이션
│       ├── audio/
│       │   ├── AudioCapturer.kt       # 16kHz 모노 PCM 캡처
│       │   └── FeatureExtractor.kt    # Mel spectrogram (Bluestein FFT)
│       ├── model/
│       │   ├── SpeakerEncoder.kt      # Resemblyzer d-vector 추출
│       │   └── PersonalVAD.kt         # LSTM 추론 및 상태 관리
│       ├── data/
│       │   └── EmbeddingStore.kt      # 화자 임베딩 저장
│       └── ui/
│           ├── EnrollScreen.kt        # 5초 음성 녹음 화면
│           └── DetectionScreen.kt     # 실시간 감지 표시 화면
│
export/                     # Python 모델 내보내기 스크립트
├── personal_vad_model.py   # Personal VAD 정의 + ONNX 내보내기
├── download_models.py      # 사전 학습 가중치 다운로드
├── resemblyzer_encoder.onnx
├── personal_vad.onnx
└── mel_filterbank.bin
```

## 빌드

### 사전 요구사항

- Android SDK (API 24+)
- JDK 17

### 빌드 방법

```bash
cd android-app
./gradlew assembleDebug
```

APK 출력 경로: `android-app/app/build/outputs/apk/debug/app-debug.apk`

### 모델 재생성 (선택사항)

ONNX 모델은 `android-app/app/src/main/assets/`에 포함되어 있습니다. 직접 재생성하려면:

```bash
cd export
pip install torch resemblyzer onnx librosa numpy
python download_models.py
python personal_vad_model.py
```

## 기술 스택

- **Kotlin** + **Jetpack Compose** UI
- **ONNX Runtime Android** (`1.20.0`) 온디바이스 추론
- **AudioRecord API** 실시간 16kHz 오디오 캡처
- 순수 Kotlin 특징 추출 (네이티브/JNI 의존성 없음)

## 참고 문헌

- [Personal VAD: Speaker-Conditioned Voice Activity Detection](https://arxiv.org/abs/2104.01167) — Google Research
- [pirxus/personalVAD](https://github.com/pirxus/personalVAD) — PyTorch 구현
- [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) — GE2E 화자 인코더

## 라이선스

이 프로젝트는 다음을 기반으로 합니다:
- Personal VAD 모델 가중치: [pirxus/personalVAD](https://github.com/pirxus/personalVAD)
- 화자 인코더: [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) (Apache 2.0)
- ONNX Runtime (MIT License)

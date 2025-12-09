# Hybrid AI Trading Bot (ML + RL)

이 프로젝트는 **지도 학습(XGBoost + LSTM)** 기반의 시장 예측 모델(Teacher)과 **강화 학습(PPO)** 에이전트(Student)를 결합한 고성능 하이브리드 가상화폐 트레이딩 봇입니다.

단순히 과거 데이터를 학습하는 것을 넘어, **Teacher-Student 아키텍처**를 통해 ML의 예측 확률을 RL의 판단 근거로 활용하며, **추세 추종(Trend Following) 필터**와 **정밀 시뮬레이션(Intra-bar Check)**을 통해 실전에서의 안정성을 극대화하도록 설계되었습니다.

---

## 🚀 주요 특징 (Key Features)

### 1. 하이브리드 Teacher-Student 아키텍처
* **Teacher (ML Model):** 시장의 미세한 패턴을 감지하는 **XGBoost**(정형 데이터 강점)와 **LSTM**(시계열 강점)의 앙상블 모델입니다. 롱/숏/관망 확률을 예측하여 Student에게 신호를 줍니다.
* **Student (RL Agent):** **PPO(Proximal Policy Optimization)** 알고리즘을 사용하여, Teacher의 신호와 시장 데이터를 종합해 최적의 매매 행동을 결정합니다.

### 2. 강력한 리스크 관리 및 추세 추종 (Trend Guardrail)
* **Trend Filter**: **EMA 200** 지표를 기준으로 대세 상승장에서는 숏 진입을 차단하고, 하락장에서는 롱 진입을 차단하여 역추세 매매로 인한 손실을 원천 방지합니다.
* **Dynamic Stops**: 시장 변동성(ATR)에 기반한 동적 Stop Loss(SL) 및 Take Profit(TP)을 설정합니다.
* **Trailing Stop & Break Even**: 수익이 발생하면 SL을 본절 또는 수익 구간으로 이동시켜 이익을 보존합니다.

### 3. 정밀 시뮬레이션 백테스트 (Precision Backtesting)
* **1H Decision / 5m Execution**: 의사결정은 1시간봉(Main)으로 하되, 실제 체결 및 청산 시뮬레이션은 **5분봉(Precision)** 데이터를 사용하여 고가/저가 터치 여부를 정밀하게 판정합니다.
* **현실적인 요소 반영**: 슬리피지(Slippage), 수수료(Commission), 펀딩비(Funding Fee)가 모두 반영된 리얼한 수익률을 계산합니다.

### 4. GPU 가속 및 최적화
* **Mixed Precision (FP16)**: RTX 30 시리즈 이상의 GPU에서 학습 속도를 극대화하기 위해 혼합 정밀도 연산을 적용했습니다.
* **Memory Optimization**: `cuda_malloc_async` 할당자를 사용하여 GPU 메모리 효율을 최적화했습니다.

---

## 📂 시스템 구조 (System Structure)

```text
📂 Crypto_Bot
│
├── config.py              # [설정] 하이퍼파라미터, 경로, 세션 관리
├── main.py                # [실행] 프로그램 진입점 (로깅, TB 실행, 파이프라인 가동)
│
├── 📂 utils/
│   └── data_loader.py     # [데이터] CCXT 수집, 기술적 지표(TA-Lib) 가공, 타겟 생성
│
├── 📂 models/
│   ├── ml_models.py       # [Teacher] XGBoost + LSTM 앙상블 클래스
│   └── rl_env.py          # [Student] Gymnasium 기반 강화학습 환경 (보상 함수 포함)
│
├── 📂 strategies/
│   ├── trading_core.py    # [매매 엔진] 자산 관리, 포지션 진입/청산, 정밀 시뮬레이션 로직
│   └── pipeline_trainer.py# [파이프라인] 데이터 로드 -> ML학습 -> 신호생성 -> RL학습 -> 백테스트 통합 관리
│
├── 📂 logs/               # [로그] 세션별 실행 로그 및 TensorBoard 데이터
└── 📂 models_saved/       # [저장] 학습된 ML/RL 모델 체크포인트
```

---

## 🧠 상세 로직 (Detailed Logic)

### 1. ML Strategy (Teacher)
* **Input**: OHLCV + 기술적 지표(RSI, MACD, BB, etc.)
* **Process**: RobustScaler로 정규화 후, XGBoost와 LSTM이 각각 예측 수행.
* **Output**: `ml_signal` (Long 확률 - Short 확률, -1.0 ~ 1.0). RL 에이전트의 관측값(Observation)으로 제공됩니다.

### 2. RL Strategy (Student)
* **Observation Space**: 시장 데이터 + 계좌 상태(잔고, PnL) + **Teacher's Signal**.
* **Action Space**: `Hold`, `Buy(Long)`, `Sell(Short)`, `Close`.
* **Reward Function**: 
    * 로그 수익률(Log Return) 기반.
    * 샤프 지수(Sharpe Ratio) 보너스 및 MDD 페널티 적용.
    * 추세 추종 성공 시 가산점 부여.

### 3. Execution Logic (Trading Core)
* **EMA 200 필터**: `Price > EMA 200` 일 때 Short 신호 무시 (Trend Following).
* **Intra-bar Logic**: 1시간봉 내부의 5분봉 12개를 순회하며 `Low <= SL` 혹은 `High >= TP` 조건을 체크하여 즉시 청산 로직을 수행합니다.

---

## 🗺️ 프로젝트 로드맵 (Project Roadmap)

### ✅ Phase 1 & 2: 기반 구축 (완료)
* 기본 데이터 파이프라인 및 환경 구축.
* Teacher(ML) - Student(RL) 기본 상호작용 구현.
* GPU 가속 환경(TensorFlow/PyTorch Hybrid) 최적화.

### 🔄 Phase 3: 로직 교정 및 추세 추종 (현재 단계)
* **Trend Filter 적용**: 역추세 매매로 인한 손실 방지 (EMA 200).
* **보상 함수 고도화**: 변동성 페널티 조정 및 추세 추종 보너스 강화.
* **Teacher 모델 강화**: Epoch 증가 및 학습 데이터 확장으로 예측력 향상.

### 🔜 Phase 3.5: 일반화 (Generalization)
* **Multi-Coin Training**: BTC 외 ETH, SOL 등 다양한 코인 데이터를 통합 학습하여 특정 자산에 과적합되는 현상 방지.
* **Feature Engineering**: 시장 심리 지수, 거시 경제 지표 등 외부 데이터 추가.

### 🔜 Phase 4: 하이퍼파라미터 최적화 (Auto-Tuning)
* **Optuna 도입**: Learning Rate, Batch Size, Window Size, Network Depth 등 주요 파라미터를 자동으로 탐색하여 최적의 조합 도출.

### 🔜 Phase 5: 실전 검증 (Live/Paper Trading)
* **Live Trader 구현**: Binance Futures API 연동.
* **Real-time Monitoring**: 텔레그램 봇 알림 및 실시간 대시보드 구축.

---

## 🛠 설치 및 실행 (Installation & Usage)

### 1. 필수 요구 사항 (Prerequisites)
* OS: Windows 10/11 (Native GPU Support)
* GPU: NVIDIA RTX 30 Series 이상 권장
* **CUDA 11.8 & cuDNN 8.x** (필수)

### 2. 설치 (Installation)
```bash
# 가상환경 생성 및 활성화
python -m venv .venv
.\.venv\Scripts\activate

# [중요] PyTorch (CUDA 11.8) 우선 설치
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# 나머지 의존성 설치
pip install -r requirements.txt
```

### 3. 실행 (Usage)
단 하나의 명령어로 데이터 다운로드부터 ML 학습, RL 학습, 백테스트까지 모든 과정이 자동으로 수행됩니다.
```bash
python main.py
```
* 실행 후 자동으로 열리는 **TensorBoard**(`http://localhost:6006`)에서 학습 과정을 모니터링하세요.
* 종료 후 `logs/세션ID/` 폴더에서 `backtest_result.png` 그래프와 `system.log`를 확인할 수 있습니다.

---

## ⚠️ Disclaimer
본 프로젝트는 알고리즘 트레이딩 연구 및 학습 목적으로 개발되었습니다. 실제 투자는 본인의 책임 하에 신중하게 진행해야 하며, 개발자는 이 프로그램 사용으로 인한 금전적 손실에 대해 책임을 지지 않습니다.
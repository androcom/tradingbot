# Hybrid AI Trading Bot (ML + RL + Optimization)

이 프로젝트는 **지도 학습(XGBoost + LSTM)**의 예측력과 **강화 학습(PPO)**의 판단력, 그리고 **수학적 최적화(Optuna)**를 결합한 차세대 알고리즘 트레이딩 시스템입니다.

단순한 과거 데이터 학습을 넘어, **Walk-Forward Validation(전진 분석)**을 통해 미래 참조 편향을 제거하고, **MTF(Multi-Timeframe)** 분석으로 추세를 추종하며, **Optuna**를 통해 매매 규칙을 정밀하게 튜닝하여 실전 수익률을 극대화하도록 설계되었습니다.

---

## 🚀 핵심 경쟁력 (Key Features)

### 1. 무결성 학습 시스템 (Integrity Learning)
* **Walk-Forward Validation**: 학습 과정에서 ML 모델이 미래 데이터를 미리 보는(Look-ahead Bias) 문제를 원천 차단하기 위해 **K-Fold 교차 검증** 방식으로 '정직한(Honest)' 예측 신호를 생성하여 RL에게 제공합니다.
* **Teacher-Student Architecture**: Teacher(ML)가 시장의 확률을 계산하면, Student(RL)가 자금 상황과 리스크를 고려해 최종 진입 여부를 결정합니다.

### 2. 고도화된 전략 엔진 (Advanced Strategy Engine)
* **MTF Trend Filter**: 1시간봉(Main) 매매 시 4시간봉(Aux)의 EMA 추세를 참조하여 역추세 매매를 필터링합니다.
* **Dynamic Risk Sizing**: 고정 레버리지를 사용하지 않고, ATR 기반 손절폭에 따라 **자산의 1%~2% 리스크**만 감수하도록 진입 물량을 자동 조절합니다.
* **Smart Trailing Stop**: Optuna로 최적화된 `Trigger` 및 `Gap` 파라미터를 사용하여 수익을 길게 가져가면서도 급락 시 이익을 보존합니다.

### 3. 하이퍼파라미터 자동 최적화 (Auto-Optimization)
* **ML Optimization**: `Active Ratio(매매 빈도)` 제약 조건을 적용하여, 과적합되지 않으면서도 충분한 매매 기회를 포착하는 최적의 지표 설정값을 찾아냅니다.
* **Logic Optimization**: 학습된 모델을 고정한 채, 시뮬레이션을 통해 손절폭, 익절 타이밍, 트레일링 간격 등의 매매 규칙(Rule)만을 초고속으로 최적화합니다.

---

## 📂 시스템 구조 (System Structure)

```text
📂 Crypto_Bot
│
├── config.py              # [설정] 모든 하이퍼파라미터 및 트레이딩 규칙 중앙 관리
├── main.py                # [실행] 전체 파이프라인(데이터-ML-RL-백테스트) 실행
│
├── 📂 utils/
│   └── data_loader.py     # [데이터] MTF 데이터 병합 및 Shift 처리(미래 참조 방지)
│
├── 📂 models/
│   ├── ml_models.py       # [Teacher] XGBoost + LSTM 앙상블 (Class Weight 적용)
│   └── rl_env.py          # [Student] PPO 강화학습 환경 (수익률 기반 보상 함수)
│
├── 📂 strategies/
│   ├── trading_core.py    # [엔진] 리스크 관리, 포지션 사이징, 정밀 청산 로직
│   └── pipeline_trainer.py# [학습기] K-Fold 신호 생성 및 학습 파이프라인 제어
│
├── 📂 optimization/       # [최적화]
│   ├── optimize_teacher.py # ML 모델 파라미터 튜닝 (Activity Constraint)
│   └── optimize_logic.py   # 매매 로직(SL/TP) 고속 시뮬레이션 튜닝
│
└── 📂 logs/               # [로그] 텐서보드 및 백테스트 결과 그래프 저장
│   └── 📂 optimization/   # 최적화 로그 저장소
│
└── 📂 optimization/           # 최적화 스크립트 모음
    ├── optimize_teacher.py    # [ML] 지표/타겟/모델 파라미터 튜닝 (5-Fold CV)
    └── optimize_logic.py      # [Logic] 손절/익절/트레일링 룰 튜닝 (Fast Sim)
```
---

## 🗺️ 프로젝트 로드맵 (Roadmap)

### ✅ Phase 1 ~ 2: 기반 구축 및 논리적 결함 제거 (완료)
* 데이터 파이프라인 구축 (Binance API, Feature Engineering).
* GPU 가속 환경 설정 (CUDA 세팅).
* 기본 ML(Teacher) + RL(Student) 구조 설계.
* Data Leakage 해결: K-Fold 및 MTF Shift 로직 적용.
* Class Imbalance 해결: Target Threshold 최적화를 통해 Hold 비율 80% → 48%로 정상화.
* Reward Hacking 해결: 거래 회피 방지를 위한 보상 함수(Penalty/Bonus) 수정.

### 🔄 Phase 3: 전략 안정화 및 베이스라인 확보 (현재 진행 중)
* 현재 작업: 수정된 로직으로 대규모 학습(1,000만 스텝) 수행.
* 목표: 논리적 오류 없이 작동하며, 학습 곡선이 우상향하는 "정상적인 봇" 확인.

### 🔜 Phase 4: 성능 고도화
* Optuna Full Optimization: 반복 연산을 통해 전체 파라미터에 대한 최적 조합 확인.
* 지표 다변화 : 시장 심리 지수, 거시 경제 데이터 추가 테스트.

### 🔜 Phase 5: 확장 및 실전
* **Paper Trading**: 바이낸스 테스트넷에서 실시간 가동 확인 (체결 오차 및 슬리피지 검증).
* **Multi-Coin**: BTC 외 메이저 알트코인으로 포트폴리오 다각화.
* **Live Deployment**: 바이낸스 API 연동 및 실시간 텔레그램 봇 구동.

---

## 🛠 실행 방법 (Usage)

### 1. 전체 파이프라인 실행 (학습 + 백테스트)
가장 일반적인 실행 방법입니다. 설정된 값으로 ML/RL 학습을 수행하고 결과를 출력합니다.
```bash
python main.py
```

### 2. 전략 로직 최적화 (선택 사항)
ML 모델 변경 없이 매매 규칙(손절/익절 등)만 빠르게 튜닝하고 싶을 때 사용합니다.
```bash
python optimization/optimize_logic.py
```

---

## 📊 성능 리포트 (Latest Status)
* **ML Accuracy**: ~80% (Balanced Class Weight 적용)
* **Target Distribution**: Hold 82% / Trade 18% (스나이퍼 전략)
* **Backtest**: 하락장 숏 포지션 수익 방어 확인, 추세 추종 로직 안정화 완료.

---

## ⚠️ Disclaimer
본 프로젝트는 알고리즘 트레이딩 연구 및 학습 목적으로 개발되었습니다. 실제 투자는 본인의 책임 하에 신중하게 진행해야 하며, 개발자는 이 프로그램 사용으로 인한 금전적 손실에 대해 책임을 지지 않습니다.
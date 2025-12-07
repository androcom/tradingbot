이 프로젝트는 **지도 학습(XGBoost + LSTM)** 모델과 **규칙 기반의 국면 전환(Regime Switching) 로직**을 결합한 고도화된 AI 트레이딩 봇입니다. 다양한 시장 상황(추세장 vs 횡보장)에 적응하도록 설계되었으며, 전진 분석(Walk-Forward Validation)을 통해 리스크를 관리하면서 수익성을 극대화하는 것을 목표로 합니다.

## 시스템 아키텍처 (System Architecture)

이 시스템은 크게 4가지 핵심 모듈로 구성됩니다:

1.  **데이터 프로세서 (`data_processor.py`)**:
    * 바이낸스(Binance)에서 OHLCV 데이터를 수집합니다.
    * RSI, ADX, 볼린저 밴드 등 기술적 지표를 생성합니다.
    * **신규 피처**: AI 학습 효율을 높이기 위해 RVOL(상대 거래량), 이격도(Disparity), 로그 수익률(Log Returns)을 추가했습니다.
2.  **AI 모델 (`ai_models.py`)**:
    * **앙상블 모델**: 트리 기반의 XGBoost와 시계열 기반의 LSTM을 결합하여 예측 성능을 높입니다.
    * 다음 구간의 포지션 방향(Long/Short/Hold) 확률을 예측합니다.
3.  **유전 알고리즘 최적화기 (`ga_optimizer.py`)**:
    * 14일 주기로 유전 알고리즘(Genetic Algorithm)을 사용하여 매매 파라미터(`SL`, `Risk`, `Entry Threshold`)를 최적화합니다.
    * 목표: 위험 조정 수익률(Risk-Adjusted Return) 극대화.
4.  **트레이딩 엔진 (`trading_engine.py` & `main_backtest.py`)**:
    * AI 신호와 필터링 로직을 기반으로 매매를 실행합니다.
    * 슬리피지(Slippage), 수수료(Commission), 펀딩비(Funding Fees)를 포함한 현실적인 시뮬레이션을 수행합니다.

---

## 핵심 로직 (Hybrid Regime Switching)

봇은 현재 시장의 국면(Regime)을 판단하여 진입 전략을 다르게 가져갑니다.

### A. 시장 국면 판단 (Market Regime Detection)
* **추세장 (Trend Market)**: `ADX > 20` (강한 방향성 존재).
* **횡보장 (Range Market)**: `ADX <= 20` (방향성 없는 지루한 움직임).
* **거래량 필터 (Volume Filter)**: `RVOL > 1.2` (평소 대비 거래량이 20% 이상 터져야 신호 인정).

### B. 진입 로직 (Entry Logic)
| 국면 (Regime) | 조건 (Condition) | 신호 로직 (Signal Logic) |
| :--- | :--- | :--- |
| **추세 모드 (Trend Mode)** | `ADX > 20` AND `Vol > 1.2x` | **철저한 추세 추종 (Follow Trend Only)**.<br>- 상승 추세 (`가격 > EMA200`): AI가 롱 확신 시 진입.<br>- 하락 추세 (`가격 < EMA200`): AI가 숏 확신 시 진입. |
| **횡보 모드 (Range Mode)** | `ADX <= 20` AND `Vol > 1.2x` | **평균 회귀 (Mean Reversion)**.<br>- 롱 진입: `RSI < 30` (과매도) AND AI 롱 확신.<br>- 숏 진입: `RSI > 70` (과매수) AND AI 숏 확신. |

### C. 청산 로직 (Trailing Stop)
고정된 목표가(Take Profit) 대신, 강한 추세에서 수익을 길게 가져가기 위해 **동적 트레일링 스탑**을 사용합니다.

1.  **활성화 (Activation)**: 미실현 수익이 `TP_Ratio * SL_Distance`에 도달하면 발동.
2.  **추적 (Trailing)**: 가격이 유리한 방향으로 갈수록 손절선을 따라 올림(롱) / 내림(숏).
    * **간격 (Distance)**: 최고점/최저점 대비 `0.5 * SL_Distance` 만큼의 여유를 두고 타이트하게 추격.
3.  **강제 손절 (Hard Stop)**: 자산 보호를 위해 고정 손절선은 항상 유지됨.

___
___

## 개발 로드맵 (Development Roadmap)

### 1단계: 견고한 규칙 기반 하이브리드 시스템 (현재)
* 스케일링을 포함한 올바른 Walk-Forward Validation 구현.
* XGBoost와 LSTM 결합 모델 구축.
* 추세장과 횡보장을 구분하여 전략 이원화.
* 거래량 필터(RVOL) 및 트레일링 스탑(Trailing Stop) 적용.
* 수수료와 슬리피지를 반영한 현실적인 PnL 계산 확인.

### 2단계: 강화 학습
* **목표**: 하드코딩된 "진입 로직"을 학습된 에이전트(Agent)로 대체.
* **기술 스택**: `FinRL` 또는 `Stable-Baselines3` (PPO/DQN 알고리즘).
* **행동 공간 (Action Space)**: 에이전트가 언제 추세/역추세 전략을 스위칭할지, 레버리지를 얼마나 쓸지 스스로 결정.
* **보상 함수 (Reward Function)**: 샤프 지수(Sharpe Ratio) 극대화 + MDD 최소화.

### 3단계: 메타 러닝 & 실전 운용
* **목표**: 여러 모델을 동시에 운용하며 자금을 동적으로 배분.
* **메타 모델 (Meta-Model)**: 최근 시장 상황에 따라 가장 성과가 좋은 모델(예: 1단계 로직 vs 2단계 RL 에이전트)을 선택하는 "관리자 AI" 도입.
* **실전 트레이딩**: 바이낸스 API 연동 및 실시간 자동매매 구축.
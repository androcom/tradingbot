# config.py
import os

# --- 파일 및 디렉토리 경로 설정 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

# 필요한 디렉토리가 없으면 생성
for d in [DATA_DIR, MODEL_DIR, LOG_DIR]:
    if not os.path.exists(d): os.makedirs(d)

LOG_FILE = os.path.join(LOG_DIR, 'debug_trade.log')
RESULT_FILE = os.path.join(LOG_DIR, 'backtest_result.csv')
CHART_FILE = os.path.join(LOG_DIR, 'equity_curve.png')

# --- 거래소 및 대상 코인 설정 ---
EXCHANGE_NAME = 'binance'
TARGET_SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT'] 

# --- 데이터 타임프레임 설정 ---
MAIN_TIMEFRAME = '4h'   # 주 거래 타임프레임
AUX_TIMEFRAME = '1d'    # 보조 타임프레임 (일봉)

# --- 데이터 수집 및 백테스트 기간 설정 ---
COLLECT_START = '2015-01-01 00:00:00'
COLLECT_END   = '2025-12-31 00:00:00'
TEST_START    = '2024-01-01 00:00:00'
ONLINE_TRAIN_INTERVAL_DAYS = 60 # 온라인 재학습 주기 (일)

# --- AI 모델 하이퍼파라미터 설정 ---
LSTM_WINDOW = 60        # LSTM 입력 시퀀스 길이
BATCH_SIZE = 128        # 학습 배치 크기
TRAIN_EPOCHS = 10       # 학습 에포크 수
W_XGB = 0.7     # XGBoost 모델 가중치
W_LSTM = 0.3    # LSTM 모델 가중치

# --- 자금 관리 및 리스크 설정 ---
ENABLE_SHORT = True      # 숏 포지션 진입 허용 여부
MIN_LEVERAGE = 1.0       # 최소 레버리지
MAX_LEVERAGE = 5.0       # 최대 레버리지

INITIAL_BALANCE = 10000.0 # 초기 자본금
SLIPPAGE = 0.001         # 슬리피지 (0.1%)
COMMISSION = 0.001       # 수수료 (0.1%)
BANKRUPTCY_LIMIT = 0.2   # 파산 한도 비율
FUNDING_RATE_4H = 0.00005 # 4시간 기준 펀딩비 가정

BASE_RISK_PER_TRADE = 0.010  # 트레이드 당 기본 리스크 (1%)
MAX_RISK_PER_TRADE = 0.040   # 트레이드 당 최대 리스크 (4%)

ENTRY_THRESHOLD = 0.25   # 진입 임계값 (점수)
TARGET_THRESHOLD = 0.005 # 목표 수익률 임계값 (레이블링용)

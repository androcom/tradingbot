# config.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

for d in [DATA_DIR, MODEL_DIR, LOG_DIR]:
    if not os.path.exists(d): os.makedirs(d)

LOG_FILE = os.path.join(LOG_DIR, 'debug_trade.log')
RESULT_FILE = os.path.join(LOG_DIR, 'backtest_result.csv')

EXCHANGE_NAME = 'binance'
TARGET_SYMBOLS = ['BTC/USDT']

# 타임프레임
MAIN_TIMEFRAME = '1h'
AUX_TIMEFRAME = '4h'
PRECISION_TIMEFRAME = '5m'

# 데이터 기간
COLLECT_START = '2018-01-01 00:00:00'
COLLECT_END   = '2025-12-31 00:00:00'
TEST_START    = '2024-01-01 00:00:00'

ONLINE_TRAIN_INTERVAL_DAYS = 14

INDICATOR_WINDOW = 14
BB_WINDOW = 20
BB_STD = 2.0
EMA_WINDOW = 200

LOOK_AHEAD_STEPS = 4
TARGET_THRESHOLD = 0.008

LSTM_WINDOW = 60
BATCH_SIZE = 128
TRAIN_EPOCHS = 5
W_XGB = 0.5
W_LSTM = 0.5

XGB_PARAMS = {
    'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.03,
    'tree_method': 'hist', 'device': 'cuda', 
    'objective': 'multi:softprob', 'eval_metric': 'mlogloss',
    'subsample': 0.8, 'colsample_bytree': 0.8
}

# [수정] 자금 관리: 안정형 설정 (레버리지 2배)
ENABLE_SHORT = True
MIN_LEVERAGE = 1
MAX_LEVERAGE = 2 

INITIAL_BALANCE = 10000.0
SLIPPAGE = 0.0002
COMMISSION = 0.0004
BANKRUPTCY_LIMIT = 0.1
FUNDING_RATE_4H = 0.00005

GA_SETTINGS = {
    'population_size': 40,
    'generations': 8,
    'elitism': 4,
    'mutation_rate': 0.15
}

# [수정] GA 범위: 높은 진입 장벽 (0.55~)
GA_GENE_RANGES = {
    'entry_threshold': (0.55, 0.85),
    'sl_mul': (2.5, 5.0),
    'risk_scale': (0.01, 0.02),
    'tp_ratio': (2.0, 5.0)
}

# [수정] 전략 상수: 추세 기준 강화
GLOBAL_RISK_LIMIT = 0.02
BB_WIDTH_THRESHOLD = 0.002
TS_TRIGGER_PCT = 0.020
BE_TRIGGER_PCT = 0.015

ADX_THRESHOLD = 30          # 20 -> 30 (확실한 추세만)
RVOL_THRESHOLD = 1.2        # 거래량 필터 기준
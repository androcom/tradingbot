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

# 재학습 주기 (14일)
ONLINE_TRAIN_INTERVAL_DAYS = 14

# 지표 설정
INDICATOR_WINDOW = 14
BB_WINDOW = 20
BB_STD = 2.0
EMA_WINDOW = 200

# 라벨링 설정
LOOK_AHEAD_STEPS = 4
TARGET_THRESHOLD = 0.008

# 모델 설정
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

# 자금 관리
ENABLE_SHORT = True
MIN_LEVERAGE = 1
MAX_LEVERAGE = 3

INITIAL_BALANCE = 10000.0
SLIPPAGE = 0.0002
COMMISSION = 0.0004
BANKRUPTCY_LIMIT = 0.1
FUNDING_RATE_4H = 0.00005

# GA 설정
GA_SETTINGS = {
    'population_size': 30,
    'generations': 5,
    'elitism': 3,
    'mutation_rate': 0.2
}

# GA 범위 설정 (수익 추구형)
GA_GENE_RANGES = {
    'entry_threshold': (0.35, 0.65), 
    'sl_mul': (2.0, 4.5),            
    'risk_scale': (0.01, 0.03),      
    'tp_ratio': (1.5, 4.0) # 이 값은 이제 트레일링 시작점으로 사용됨
}

# 전략 상수
GLOBAL_RISK_LIMIT = 0.05    
BB_WIDTH_THRESHOLD = 0.0015 
TS_TRIGGER_PCT = 0.020      
BE_TRIGGER_PCT = 0.015      
ADX_THRESHOLD = 20          

# [신규] 거래량 필터 임계값
RVOL_THRESHOLD = 1.2  # 평소 거래량 대비 1.2배 이상이어야 진입
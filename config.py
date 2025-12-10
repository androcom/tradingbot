import os
from datetime import datetime

# ---------------------------------------------------------
# [System] 경로 및 세션 설정
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
LOG_BASE_DIR = os.path.join(BASE_DIR, 'logs')
MODEL_BASE_DIR = os.path.join(BASE_DIR, 'models_saved')

for d in [DATA_DIR, LOG_BASE_DIR, MODEL_BASE_DIR]:
    os.makedirs(d, exist_ok=True)

class SessionManager:
    def __init__(self):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(LOG_BASE_DIR, self.session_id)
        self.model_dir = os.path.join(MODEL_BASE_DIR, self.session_id)
        self.tensorboard_dir = os.path.join(self.log_dir, "tb_logs")

    def create(self):
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        return {
            'id': self.session_id,
            'root': self.log_dir,
            'tb': self.tensorboard_dir,
            'model': self.model_dir,
            'log_file': os.path.join(self.log_dir, 'system.log')
        }

# ---------------------------------------------------------
# [Data] 데이터 대상 및 기간
# ---------------------------------------------------------
TARGET_COINS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT']
MAIN_SYMBOL = 'BTC/USDT'

TIMEFRAME_MAIN = '1h'
TIMEFRAME_AUX = '4h'   
TIMEFRAME_PRECISION = '5m'

DATE_START = '2019-01-01 00:00:00'
DATE_END   = '2025-12-31 00:00:00'
TEST_SPLIT_DATE = '2024-01-01 00:00:00'

EXCLUDE_COLS = [
    'timestamp', 'open', 'high', 'low', 'close', 'volume', 
    'target_cls', 'target_val', 'date', 'symbol'
]

# ---------------------------------------------------------
# [Feature] 지표 및 타겟
# ---------------------------------------------------------
INDICATOR_WINDOW = 45
LOOK_AHEAD_STEPS = 1
TARGET_THRESHOLD = 0.00233

# ---------------------------------------------------------
# [Model] 모델 하이퍼파라미터
# ---------------------------------------------------------
# 1. XGBoost (Teacher) - Optuna 최적값
XGB_PARAMS = {
    'n_estimators': 400,
    'max_depth': 3,            # 깊이가 얕음 -> 과적합 방지 및 일반화 성능 우수
    'learning_rate': 0.061,
    'n_jobs': -1,
    'random_state': 42,
    'eval_metric': 'mlogloss'
}

# 2. LSTM (Teacher)
ML_SEQ_LEN = 60
ML_EPOCHS = 150
ML_BATCH_SIZE = 256
LSTM_PARAMS = {
    'units_1': 64,
    'units_2': 32,
    'dropout': 0.3
}

# 3. PPO (Student)
RL_TOTAL_TIMESTEPS = 10_000_000 
RL_PPO_PARAMS = {
    'learning_rate': 2e-4,
    'n_steps': 2048,
    'batch_size': 2048,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.005,
    'policy_kwargs': dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )
}

# ---------------------------------------------------------
# [Trading] 거래 규칙 & 리스크 관리
# ---------------------------------------------------------
INITIAL_BALANCE = 10000.0
MAX_LEVERAGE = 5 
FEE_RATE = 0.0006
SLIPPAGE = 0.0002

TRADING_RULES = {
    'trend_window': 200,       
    'sl_atr_multiplier': 3.0,  # 윈도우가 짧아졌으므로(15), 손절폭은 3.0으로 넉넉히 줌 (노이즈 방어)
    'risk_per_trade': 0.01,    # 거래당 1% 리스크 고정

    'min_trade_amount': 10.0,
    'funding_rate_hourly': 0.000025,
    'scale_down_factor': 0.5,   # 역추세일 때 50% 비중으로 진입 (공격성 약간 상향)

    # [신규 최적값 적용] 익절 및 트레일링 설정
    'tp_trigger_atr': 1.3,     # 1.3 ATR 수익 시 본절 발동
    'trailing_gap_atr': 1.6    # 고점 대비 1.6 ATR 간격 유지
}
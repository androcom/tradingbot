import os
from datetime import datetime

# ---------------------------------------------------------
# [System] 경로 및 세션 설정
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
LOG_BASE_DIR = os.path.join(BASE_DIR, 'logs')
MODEL_BASE_DIR = os.path.join(BASE_DIR, 'models_saved')

# 디렉토리 자동 생성
for d in [DATA_DIR, LOG_BASE_DIR, MODEL_BASE_DIR]:
    os.makedirs(d, exist_ok=True)

class SessionManager:
    def __init__(self):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # [수정] 변수명 통일 (session_dir -> log_dir)
        # 로그와 모델 경로를 분리하여 관리
        self.log_dir = os.path.join(LOG_BASE_DIR, self.session_id)
        self.model_dir = os.path.join(MODEL_BASE_DIR, self.session_id)
        self.tensorboard_dir = os.path.join(self.log_dir, "tb_logs")

    def create(self):
        # [수정] __init__에서 정의한 변수명(log_dir) 사용
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        
        return {
            'id': self.session_id,
            'root': self.log_dir,       # 로그 저장소 (session_dir 대신 log_dir 사용)
            'tb': self.tensorboard_dir, # 텐서보드
            'model': self.model_dir,    # 모델 저장소
            'log_file': os.path.join(self.log_dir, 'system.log') # 통합 로그 파일
        }

# ---------------------------------------------------------
# [Data] 데이터 및 지표 설정
# ---------------------------------------------------------
SYMBOL = 'BTC/USDT'
TIMEFRAME_MAIN = '1h'
TIMEFRAME_AUX = '4h'
TIMEFRAME_PRECISION = '5m'

DATE_START = '2019-01-01 00:00:00'
DATE_END   = '2025-12-31 00:00:00'
TEST_SPLIT_DATE = '2024-01-01 00:00:00'

# ML Feature 설정
INDICATOR_WINDOW = 14
LOOK_AHEAD_STEPS = 1
TARGET_THRESHOLD = 0.005

# 학습에서 제외할 컬럼 (Raw Data & Target)
EXCLUDE_COLS = [
    'timestamp', 'open', 'high', 'low', 'close', 'volume', 
    'target_cls', 'target_val' 
]

# ---------------------------------------------------------
# [ML] Supervised Learning (Teacher)
# ---------------------------------------------------------
ML_SEQ_LEN = 60
ML_EPOCHS = 30  # 에포크 30으로 증가
ML_BATCH_SIZE = 64

# ---------------------------------------------------------
# [RL] Reinforcement Learning (Student)
# ---------------------------------------------------------
RL_TOTAL_TIMESTEPS = 1_000_000

# PPO 네트워크 및 파라미터 고도화
RL_PPO_PARAMS = {
    'learning_rate': 3e-4,
    'n_steps': 2048,
    'batch_size': 512,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.01,
    'policy_kwargs': dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )
}

# ---------------------------------------------------------
# [Trading] 매매 로직 설정
# ---------------------------------------------------------
INITIAL_BALANCE = 10000.0
LEVERAGE = 2
FEE_RATE = 0.0004 # 0.04%
SLIPPAGE = 0.0002 # 0.02%
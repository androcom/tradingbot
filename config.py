# config.py
import os

# --- 파일 및 디렉토리 경로 설정 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

for d in [DATA_DIR, MODEL_DIR, LOG_DIR]:
    if not os.path.exists(d): os.makedirs(d)

LOG_FILE = os.path.join(LOG_DIR, 'trade_log.txt')

# --- 거래소 및 대상 코인 설정 ---
EXCHANGE_NAME = 'binance'
TARGET_SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT'] 

# --- 데이터 타임프레임 설정 ---
MAIN_TIMEFRAME = '4h'   
AUX_TIMEFRAME = '1d'    

# --- 데이터 수집 및 백테스트 기간 설정 ---
COLLECT_START = '2015-01-01 00:00:00'
COLLECT_END   = '2025-12-31 00:00:00'
TEST_START    = '2024-01-01 00:00:00'
ONLINE_TRAIN_INTERVAL_DAYS = 60 

# --- AI 모델 하이퍼파라미터 설정 ---
LSTM_WINDOW = 60        
BATCH_SIZE = 256          
TRAIN_EPOCHS = 30
W_XGB = 0.5              
W_LSTM = 0.5             
VOLATILITY_LOOKAHEAD = 12 

# --- 기본 자금 설정 ---
ENABLE_SHORT = True      
INITIAL_BALANCE = 10000.0 
SLIPPAGE = 0.001         
COMMISSION = 0.001       
FUNDING_RATE_4H = 0.00005 

# --- 리스크 관리 한계선 (GA가 아무리 공격적이어도 넘을 수 없는 선) ---
MIN_LEVERAGE = 1       
MAX_LEVERAGE = 5       
MAX_RISK_PER_TRADE_CAP = 0.05  # 자산의 5%

# --- [신규] 유전 알고리즘(GA) 설정 ---
# 학습된 모델을 바탕으로 최적의 파라미터를 찾기 위한 설정
GA_GENERATIONS = 20       # 세대 수 (너무 많으면 느림, 20~30 적당)
GA_POPULATION_SIZE = 50   # 한 세대당 개체 수
GA_ELITISM = 5            # 다음 세대로 무조건 살아남는 상위 개체 수
GA_MUTATION_RATE = 0.1    # 돌연변이 확률

# 유전자가 탐색할 파라미터 범위 (Gene Bounds)
# AI가 이 범위 안에서 스스로 최적값을 찾습니다.
GENE_RANGES = {
    # 1. 기본 진입 장벽 (Confidence Threshold)
    # 범위: 0.2(공격적) ~ 0.6(매우 보수적)
    'base_threshold': (0.2, 0.6),

    # 2. 변동성 민감도 (Volatility Impact)
    # 변동성이 클 때 진입 장벽을 얼마나 높일지 결정
    # 0.0(무시) ~ 3.0(변동성 크면 진입 거의 안 함)
    'vol_impact': (0.0, 3.0),

    # 3. 손절 거리 배수 (StopLoss Multiplier)
    # 예측 변동성의 몇 배를 손절폭으로 잡을지
    # 1.0(타이트함, BTC용) ~ 3.0(널널함, XRP 휩소 방어용)
    'sl_mul': (1.0, 3.0),

    # 4. 익절 비율 (TakeProfit Ratio)
    # 손절폭 대비 익절폭 비율 (손익비)
    # 1.0(1:1) ~ 4.0(1:4 추세 추종)
    'tp_ratio': (1.0, 4.0)
}
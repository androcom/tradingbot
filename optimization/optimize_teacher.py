import optuna
import pandas as pd
import numpy as np
import logging
import sys
import os
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import DataLoader
import config

# 로거 설정 생략 (기존과 동일)
def setup_optimization_logger(name_prefix):
    # ... (기존 코드 사용)
    log_dir = os.path.join(config.LOG_BASE_DIR, 'optimization')
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logging.getLogger(name_prefix)
    logger.setLevel(logging.INFO)
    logger.handlers = [logging.StreamHandler(sys.stdout)]
    return logger

logger = setup_optimization_logger("TeacherOpt")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 데이터 캐싱
RAW_DATA = None

def load_data():
    global RAW_DATA
    loader = DataLoader()
    RAW_DATA = loader.fetch_data(config.MAIN_SYMBOL, config.TIMEFRAME_MAIN, config.DATE_START, config.DATE_END)
    logger.info(f"Loaded {len(RAW_DATA)} rows.")

def calculate_profit(y_pred, prices):
    balance = 1.0
    position = 0; entry = 0
    for i in range(len(y_pred)):
        if position == 0:
            if y_pred[i] == 2: position = 1; entry = prices[i]
            elif y_pred[i] == 0: position = -1; entry = prices[i]
        else:
            if (position == 1 and y_pred[i] != 2) or (position == -1 and y_pred[i] != 0):
                pnl = (prices[i] - entry)/entry if position == 1 else (entry - prices[i])/entry
                balance *= (1 + pnl - 0.0006)
                position = 0
    return balance

def objective(trial):
    # [핵심] Target Threshold 탐색 (낮은 값 위주)
    param_threshold = trial.suggest_float('target_threshold', 0.001, 0.005, log=True)
    param_window = trial.suggest_int('window', 10, 60, step=5)
    
    loader = DataLoader()
    df = RAW_DATA.copy()
    
    # 지표 및 타겟 생성
    df = loader.add_indicators(df, window=param_window)
    df = loader.create_target(df, threshold=param_threshold)
    
    if df.empty: return 0.0

    # [제약 조건] Hold 비율이 50%를 넘으면 가차없이 탈락
    dist = df['target_cls'].value_counts(normalize=True)
    hold_ratio = dist.get(1, 0)
    if hold_ratio > 0.50: # Hold가 너무 많으면 감점
        return -1.0 * hold_ratio 

    # XGBoost 학습 및 검증
    tscv = TimeSeriesSplit(n_splits=3)
    scores = []
    feat_cols = [c for c in df.columns if c not in config.EXCLUDE_COLS]
    
    for tr_idx, te_idx in tscv.split(df):
        train = df.iloc[tr_idx]
        test = df.iloc[te_idx]
        
        model = XGBClassifier(n_estimators=200, max_depth=3, n_jobs=-1, eval_metric='mlogloss')
        model.fit(train[feat_cols], train['target_cls'])
        preds = model.predict(test[feat_cols])
        
        # 단순 수익률 계산 (정확도보다 수익성)
        profit = calculate_profit(preds, test['close'].values)
        scores.append(profit)
        
    avg_profit = np.mean(scores)
    logger.info(f"Trial {trial.number} | Profit: {avg_profit:.4f} | Hold: {hold_ratio:.1%} | Thr: {param_threshold:.5f}")
    
    return avg_profit

if __name__ == "__main__":
    load_data()
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=1000)
    
    print("Best Params:", study.best_params)
    print("Best Value:", study.best_value)
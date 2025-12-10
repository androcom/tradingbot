import optuna
import pandas as pd
import numpy as np
import logging
import sys
import os
import warnings
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import DataLoader
import config

# [설정] 경고 제어
if config.SYSTEM['SUPPRESS_WARNINGS']:
    warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore", category=UserWarning)

optuna.logging.set_verbosity(optuna.logging.WARNING)

LOG_DIR = os.path.join(config.LOG_BASE_DIR, 'optimization')
os.makedirs(LOG_DIR, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
DB_URL = f"sqlite:///{os.path.join(LOG_DIR, 'optuna_study.db')}"

# [수정] 파일 로그 제거, 터미널 로그 간소화
def setup_console_logger():
    logger = logging.getLogger("TeacherOpt")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S'))
        logger.addHandler(handler)
    return logger

logger = setup_console_logger()

RAW_DATA = None
def get_data():
    global RAW_DATA
    if RAW_DATA is None:
        loader = DataLoader()
        RAW_DATA = loader.fetch_data(config.MAIN_SYMBOL, config.TIMEFRAME_MAIN, config.DATE_START, config.DATE_END)
    return RAW_DATA.copy()

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
                balance *= (1 + pnl - config.FEE_RATE)
                position = 0
    return balance

def objective(trial):
    try:
        df = get_data()
        
        param_threshold = trial.suggest_float('target_threshold', 0.003, 0.01, log=True)
        param_window = trial.suggest_int('window', 10, 90, step=5)
        
        # [수정] Config의 Device 설정 사용
        device_type = config.SYSTEM['OPT_TEACHER_DEVICE']
        
        xgb_params = config.XGB_PARAMS.copy()
        xgb_params.update({
            'n_estimators': trial.suggest_int('n_estimators', 200, 500, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'device': device_type,
            'tree_method': 'hist' if device_type == 'cuda' else 'auto',
            'verbosity': 0,
            'early_stopping_rounds': 30
        })

        loader = DataLoader()
        df = loader.add_indicators(df, window=param_window)
        df = loader.create_target(df, threshold=param_threshold)
        
        if df.empty: return 0.0

        dist = df['target_cls'].value_counts(normalize=True)
        hold_ratio = dist.get(1, 0)
        trial.set_user_attr("hold_ratio", hold_ratio)
        
        if hold_ratio > 0.70: return -1.0 * hold_ratio

        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        feat_cols = [c for c in df.columns if c not in config.EXCLUDE_COLS]
        
        for step, (tr_idx, te_idx) in enumerate(tscv.split(df)):
            train = df.iloc[tr_idx]; test = df.iloc[te_idx]
            model = XGBClassifier(**xgb_params)
            model.fit(train[feat_cols], train['target_cls'], eval_set=[(test[feat_cols], test['target_cls'])], verbose=0)
            preds = model.predict(test[feat_cols])
            profit = calculate_profit(preds, test['close'].values)
            scores.append(profit)
            
            trial.report(np.mean(scores), step)
            if trial.should_prune(): raise optuna.TrialPruned()
            
        avg_profit = np.mean(scores)
        
        # 진행 상황은 10번에 한번만 출력 (로그 홍수 방지)
        if trial.number % 10 == 0:
            logger.info(f"Trial {trial.number:04d} | Profit: {avg_profit:.4f} | Hold: {hold_ratio:.1%}")
        
        return avg_profit

    except Exception as e:
        return 0.0

if __name__ == "__main__":
    logger.info(f"🚀 Optimization Started (Device: {config.SYSTEM['OPT_TEACHER_DEVICE']})")
    try: get_data()
    except: sys.exit(1)

    study = optuna.create_study(study_name=f"TeacherOpt_{TIMESTAMP}", storage=DB_URL, direction='maximize', 
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5), load_if_exists=True)
    
    try:
        study.optimize(objective, n_trials=1000, n_jobs=config.SYSTEM['NUM_WORKERS'])
    except KeyboardInterrupt: pass

    # CSV 저장 (포맷 통일)
    df_res = study.trials_dataframe()
    df_res.rename(columns={'value':'profit_score', 'user_attrs_hold_ratio':'hold_ratio'}, inplace=True)
    cols = ['number', 'profit_score', 'hold_ratio', 'params_target_threshold', 'params_window', 'state']
    # 필요한 컬럼이 존재할 때만 선택
    valid_cols = [c for c in cols if c in df_res.columns]
    df_res = df_res[valid_cols + [c for c in df_res.columns if c not in valid_cols]]
    
    csv_path = os.path.join(LOG_DIR, f'TeacherOpt_Result_{TIMESTAMP}.csv')
    df_res.to_csv(csv_path, index=False)
    logger.info(f"✅ Results: {csv_path}")
    if len(study.trials) > 0:
        logger.info(f"🏆 Best: {study.best_value:.4f}")
import optuna
import pandas as pd
import numpy as np
import logging
import sys
import os
from datetime import datetime
from xgboost import XGBClassifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import DataLoader
from strategies.trading_core import TradingCore
import config

# ---------------------------------------------------------
# [Logging Setup]
# ---------------------------------------------------------
def setup_optimization_logger(name_prefix):
    log_dir = os.path.join(config.LOG_BASE_DIR, 'optimization')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'{name_prefix}_{timestamp}.log')
    
    logger = logging.getLogger(name_prefix)
    logger.setLevel(logging.INFO)
    logger.handlers = []

    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(stream_handler)
    
    return logger

logger = setup_optimization_logger("LogicOpt")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------
# [1] 데이터 및 ML 신호 준비 (캐싱)
# ---------------------------------------------------------
logger.info("⏳ Loading Data & Generating Signals (Once)...")
loader = DataLoader()
df = loader.get_ml_data(config.MAIN_SYMBOL)

# Train/Test Split (Test 구간에 최적화)
test_mask = df.index >= config.TEST_SPLIT_DATE
test_df = df[test_mask].copy()

# XGBoost 재학습 및 신호 생성
logger.info("   >> Generating signals for logic optimization...")
feature_cols = [c for c in df.columns if c not in config.EXCLUDE_COLS]
split_idx = int(len(df) * 0.7)
train_sub = df.iloc[:split_idx]
test_sub = df.iloc[split_idx:]

model = XGBClassifier(**config.XGB_PARAMS)
model.fit(train_sub[feature_cols], train_sub['target_cls'])
signals = model.predict(test_sub[feature_cols])

test_sub = test_sub.copy()
test_sub['ml_signal'] = signals

# NumPy 변환 (속도 최적화)
sim_data = {
    'close': test_sub['close'].to_numpy(),
    'high': test_sub['high'].to_numpy(),
    'low': test_sub['low'].to_numpy(),
    'ema_trend': test_sub.get('ema_trend_4h', test_sub.get('ema_trend', np.zeros(len(test_sub)))).to_numpy(),
    'atr': test_sub['atr'].to_numpy(),
    'signal': test_sub['ml_signal'].to_numpy()
}
timestamps = test_sub.index
logger.info(f"✅ Data Ready. Simulation Rows: {len(test_sub)}")

# ---------------------------------------------------------
# [2] Objective Function
# ---------------------------------------------------------
def objective(trial):
    # 1. 튜닝할 파라미터
    sl_mult = trial.suggest_float('sl_atr_multiplier', 2.0, 6.0, step=0.5)
    risk_pct = trial.suggest_float('risk_per_trade', 0.01, 0.05, step=0.005)
    tp_trigger = trial.suggest_float('tp_trigger_atr', 0.8, 3.0, step=0.1)
    trailing_gap = trial.suggest_float('trailing_gap_atr', 1.0, 3.0, step=0.1)
    
    core = TradingCore()
    
    # 규칙 주입
    core.rules['sl_atr_multiplier'] = sl_mult
    core.rules['risk_per_trade'] = risk_pct
    
    # TradingCore._update_stops 메서드 오버라이딩 (동적 파라미터 적용)
    def dynamic_update_stops(self, curr_high, curr_low, entry_price):
        dist = self.position['base_sl_dist']
        
        # SL Dist에서 역산하여 ATR 추정 (기존 로직 유지)
        # base_sl_dist = atr * sl_mult 이므로, atr = base_sl_dist / sl_mult
        atr = dist / sl_mult
        
        if self.position['type'] == 'LONG':
            if curr_high > self.position['highest_price']:
                self.position['highest_price'] = curr_high
                # 본절
                if curr_high > entry_price + (atr * tp_trigger):
                    self.position['sl'] = max(self.position['sl'], entry_price * 1.001)
                # 트레일링
                if curr_high > entry_price + (atr * (tp_trigger + 1.0)):
                    self.position['sl'] = max(self.position['sl'], curr_high - (atr * trailing_gap))
                    
        else: # SHORT
            if curr_low < self.position['lowest_price']:
                self.position['lowest_price'] = curr_low
                # 본절
                if curr_low < entry_price - (atr * tp_trigger):
                    self.position['sl'] = min(self.position['sl'], entry_price * 0.999)
                # 트레일링
                if curr_low < entry_price - (atr * (tp_trigger + 1.0)):
                    self.position['sl'] = min(self.position['sl'], curr_low + (atr * trailing_gap))
    
    # 메서드 바인딩
    core._update_stops = dynamic_update_stops.__get__(core, TradingCore)

    # 3. 고속 시뮬레이션 Loop
    total_steps = len(sim_data['close'])
    
    for i in range(total_steps):
        row = {
            'close': sim_data['close'][i],
            'high': sim_data['high'][i],
            'low': sim_data['low'][i],
            'ema_trend_4h': sim_data['ema_trend'][i],
            'atr': sim_data['atr'][i]
        }
        
        sig = sim_data['signal'][i]
        action = 0
        if sig == 2: action = 1
        elif sig == 0: action = 2
        
        core.process_step(action, row, timestamps[i])
        
        if core.balance < 500: # 파산
            break
            
    final_balance = core.balance
    
    # [상세 로깅]
    logger.info(f"Trial {trial.number:04d} | Bal: ${final_balance:,.2f} | "
                f"SL: {sl_mult}, Risk: {risk_pct:.3f}, TP: {tp_trigger}, Trail: {trailing_gap}")
    
    return final_balance

if __name__ == "__main__":
    logger.info(f"🚀 Starting Trading Logic Optimization...")
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=500)

    logger.info("="*50)
    logger.info("✅ Logic Optimization Finished!")
    logger.info(f"Best Balance: ${study.best_value:,.2f}")
    logger.info("Best Params: ")
    for key, value in study.best_params.items():
        logger.info(f"    {key}: {value}")
    logger.info("="*50)
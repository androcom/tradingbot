import optuna
import pandas as pd
import numpy as np
import logging
import sys
import os
import warnings
from datetime import datetime
from xgboost import XGBClassifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import DataLoader
from strategies.trading_core import TradingCore
import config

if config.SYSTEM['SUPPRESS_WARNINGS']:
    warnings.filterwarnings("ignore")

LOG_DIR = os.path.join(config.LOG_BASE_DIR, 'optimization')
os.makedirs(LOG_DIR, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

def setup_console_logger():
    logger = logging.getLogger("LogicOpt")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S'))
        logger.addHandler(handler)
    return logger

logger = setup_console_logger()
optuna.logging.set_verbosity(optuna.logging.WARNING)

SIM_DATA = None
TIMESTAMPS = None

def prepare_data():
    global SIM_DATA, TIMESTAMPS
    if SIM_DATA is not None: return

    logger.info("⏳ Loading & Signals...")
    loader = DataLoader()
    df = loader.get_ml_data(config.MAIN_SYMBOL)
    test_mask = df.index >= config.TEST_SPLIT_DATE
    
    feat_cols = [c for c in df.columns if c not in config.EXCLUDE_COLS]
    split_idx = int(len(df) * 0.7)
    train_sub = df.iloc[:split_idx]
    test_sub = df[test_mask].copy()

    # [수정] Config 장치 사용
    device_type = config.SYSTEM['OPT_LOGIC_DEVICE']
    xgb_params = config.XGB_PARAMS.copy()
    xgb_params.update({
        'device': device_type,
        'tree_method': 'hist' if device_type == 'cuda' else 'auto'
    })
    
    model = XGBClassifier(**xgb_params)
    model.fit(train_sub[feat_cols], train_sub['target_cls'])
    signals = model.predict(test_sub[feat_cols])

    test_sub['ml_signal'] = signals
    
    SIM_DATA = {
        'close': test_sub['close'].to_numpy(),
        'high': test_sub['high'].to_numpy(),
        'low': test_sub['low'].to_numpy(),
        'ema_trend_4h': test_sub.get('ema_trend_4h', np.zeros(len(test_sub))).to_numpy(),
        'atr': test_sub['atr'].to_numpy(),
        'signal': test_sub['ml_signal'].to_numpy()
    }
    TIMESTAMPS = test_sub.index
    logger.info(f"✅ Data Ready.")

def objective(trial):
    if SIM_DATA is None: prepare_data()

    sl_mult = trial.suggest_float('sl_atr_multiplier', 2.0, 7.0, step=0.5)
    risk_pct = trial.suggest_float('risk_per_trade', 0.01, 0.05, step=0.005)
    tp_trigger = trial.suggest_float('tp_trigger_atr', 0.5, 3.0, step=0.1)
    trailing_gap = trial.suggest_float('trailing_gap_atr', 0.5, 3.0, step=0.1)
    
    core = TradingCore()
    core.rules['sl_atr_multiplier'] = sl_mult
    core.rules['risk_per_trade'] = risk_pct
    
    def dynamic_update_stops(self, curr_high, curr_low, entry_price):
        dist = self.position['base_sl_dist']
        atr = dist / sl_mult
        if self.position['type'] == 'LONG':
            if curr_high > self.position['highest_price']:
                self.position['highest_price'] = curr_high
                if curr_high > entry_price + (atr * tp_trigger):
                    self.position['sl'] = max(self.position['sl'], entry_price * 1.001)
                if curr_high > entry_price + (atr * (tp_trigger + 1.0)):
                    self.position['sl'] = max(self.position['sl'], curr_high - (atr * trailing_gap))
        else:
            if curr_low < self.position['lowest_price']:
                self.position['lowest_price'] = curr_low
                if curr_low < entry_price - (atr * tp_trigger):
                    self.position['sl'] = min(self.position['sl'], entry_price * 0.999)
                if curr_low < entry_price - (atr * (tp_trigger + 1.0)):
                    self.position['sl'] = min(self.position['sl'], curr_low + (atr * trailing_gap))
    
    core._update_stops = dynamic_update_stops.__get__(core, TradingCore)

    total_steps = len(SIM_DATA['close'])
    closes = SIM_DATA['close']; highs = SIM_DATA['high']; lows = SIM_DATA['low']
    trends = SIM_DATA['ema_trend_4h']; atrs = SIM_DATA['atr']; signals = SIM_DATA['signal']
    
    for i in range(total_steps):
        row = {'close': closes[i], 'high': highs[i], 'low': lows[i], 'ema_trend_4h': trends[i], 'atr': atrs[i]}
        sig = signals[i]
        action = 1 if sig == 2 else (2 if sig == 0 else 0)
        core.process_step(action, row, TIMESTAMPS[i])
        if core.balance < 500: break
            
    if trial.number % 50 == 0:
        logger.info(f"Trial {trial.number:04d} | Bal: ${core.balance:,.2f}")
    
    return core.balance

if __name__ == "__main__":
    prepare_data()
    logger.info("🚀 Logic Optimization Started")
    study = optuna.create_study(direction='maximize')
    
    try: study.optimize(objective, n_trials=2000, n_jobs=config.SYSTEM['NUM_WORKERS']) 
    except KeyboardInterrupt: pass

    df_res = study.trials_dataframe()
    df_res.rename(columns={'value':'final_balance'}, inplace=True)
    cols = ['number', 'final_balance', 'params_sl_atr_multiplier', 'params_risk_per_trade', 'state']
    valid_cols = [c for c in cols if c in df_res.columns]
    df_res = df_res[valid_cols + [c for c in df_res.columns if c not in valid_cols]]

    csv_path = os.path.join(LOG_DIR, f'LogicOpt_Result_{TIMESTAMP}.csv')
    df_res.to_csv(csv_path, index=False)
    
    logger.info(f"✅ Results: {csv_path}")
    if len(study.trials) > 0:
        logger.info(f"🏆 Best: ${study.best_value:,.2f}")
# main_backtest.py
import pandas as pd
import numpy as np
from datetime import timedelta
import logging
import os
import matplotlib.pyplot as plt 
import config
from data_processor import DataProcessor
from ai_models import HybridEnsemble
from trading_engine import AccountManager
from sklearn.preprocessing import RobustScaler
from ga_optimizer import GeneticOptimizer

logger = logging.getLogger("BacktestLogger")
logger.setLevel(logging.DEBUG)
logger.propagate = False 
if not os.path.exists(config.LOG_DIR): os.makedirs(config.LOG_DIR)
file_handler = logging.FileHandler(config.LOG_FILE, mode='w', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
logger.addHandler(file_handler)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(console_handler)

def plot_results(df_result, symbol):
    if df_result.empty: return
    safe_symbol = symbol.replace('/', '_')
    chart_path = os.path.join(config.LOG_DIR, f"equity_curve_{safe_symbol}.png")
    
    df_result.index = pd.to_datetime(df_result.index)
    running_max = df_result['balance'].cummax()
    drawdown = (df_result['balance'] - running_max) / running_max * 100
    
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(df_result.index, df_result['balance'], label=f'{symbol} Balance', color='blue')
    plt.title(f'Equity Curve - {symbol}')
    plt.grid(True)
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.fill_between(df_result.index, drawdown, 0, color='red', alpha=0.3)
    plt.plot(df_result.index, drawdown, color='red', label='Drawdown (%)')
    plt.title('Drawdown')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(chart_path)
    logger.info(f"   [Result] Chart saved to {chart_path}")
    plt.close()

def run_strategy(symbol):
    logger.info(f"\n{'='*60}\n>> STARTING PHASE-1 STABLE STRATEGY: {symbol}\n{'='*60}")

    dp = DataProcessor(symbol)
    full_df = dp.prepare_multi_timeframe_data()
    
    if len(full_df) < config.LSTM_WINDOW + 200:
        logger.error("!! Not enough data.")
        return

    exclude_cols = ['timestamp', 'target_cls', 'open', 'high', 'low', 'close', 'volume', 'atr_origin', 'ema_200_origin', 'rsi_origin', 'rvol_origin']
    features = [c for c in full_df.columns if c not in exclude_cols]
    
    df_5m = dp.load_precision_data(config.TEST_START)

    current_test_start = pd.to_datetime(config.TEST_START)
    final_end = pd.to_datetime(config.COLLECT_END)
    train_interval = timedelta(days=config.ONLINE_TRAIN_INTERVAL_DAYS)
    
    model = HybridEnsemble(symbol)
    optimizer = GeneticOptimizer()
    
    account = AccountManager(balance=config.INITIAL_BALANCE, leverage=1)
    equity_curve = [{'time': current_test_start, 'balance': config.INITIAL_BALANCE}]
    
    current_params = {
        'entry_threshold': 0.6, 'sl_mul': 3.0, 'risk_scale': 0.01, 'tp_ratio': 2.0
    }

    while current_test_start < final_end:
        current_test_end = current_test_start + train_interval
        if current_test_end > final_end: current_test_end = final_end
        
        logger.info(f"\n>> [Period] {current_test_start} ~ {current_test_end}")

        # A. 데이터 분할
        train_start_lookback = current_test_start - timedelta(days=365) 
        train_mask = (full_df.index >= train_start_lookback) & (full_df.index < current_test_start)
        test_mask = (full_df.index >= current_test_start) & (full_df.index < current_test_end)
        
        df_train_raw = full_df.loc[train_mask].copy()
        df_test_raw = full_df.loc[test_mask].copy()
        
        if len(df_train_raw) < config.LSTM_WINDOW * 2 or df_test_raw.empty:
            current_test_start = current_test_end
            continue

        # B. 스케일링
        scaler = RobustScaler()
        df_train_scaled = df_train_raw.copy()
        df_train_scaled[features] = scaler.fit_transform(df_train_raw[features])
        df_test_scaled = df_test_raw.copy()
        df_test_scaled[features] = scaler.transform(df_test_raw[features])
        df_train_scaled.fillna(0, inplace=True)
        df_test_scaled.fillna(0, inplace=True)

        # C. AI 모델 학습 및 저장
        X_train_seq, y_train_seq = dp.create_sequences(df_train_scaled, features, config.LSTM_WINDOW)
        X_train_flat = df_train_scaled.iloc[config.LSTM_WINDOW:][features].values
        y_train_flat = df_train_scaled.iloc[config.LSTM_WINDOW:]['target_cls'].values
        
        logger.info("   [AI] Training Model...")
        model.train(X_train_seq, y_train_seq, X_train_flat, y_train_flat)
        model.save_models()
        
        # D. GA 최적화
        train_ai_scores = model.batch_predict(X_train_seq, X_train_flat)
        logger.info("   [GA] Optimizing Parameters...")
        best_params = optimizer.optimize(
            df_train_raw.iloc[config.LSTM_WINDOW:], 
            train_ai_scores
        )
        current_params = best_params
        logger.info(f"   [GA] Best: Th={best_params['entry_threshold']:.2f}, SL={best_params['sl_mul']:.1f}, Risk={best_params['risk_scale']:.3f}")

        # E. 백테스트
        X_test_seq, _ = dp.create_sequences(
            pd.concat([df_train_scaled.iloc[-config.LSTM_WINDOW:], df_test_scaled]), 
            features, 
            config.LSTM_WINDOW
        )
        X_test_flat = df_test_scaled[features].values
        test_ai_scores = model.batch_predict(X_test_seq, X_test_flat)
        
        for i in range(len(df_test_raw)):
            if account.balance <= 0: break
            
            bar_idx = i
            if bar_idx >= len(test_ai_scores): break
            
            curr_row = df_test_raw.iloc[bar_idx]
            if bar_idx == len(df_test_raw) - 1: continue 
            next_bar = df_test_raw.iloc[bar_idx + 1]
            
            score = test_ai_scores[bar_idx]
            
            # 지표
            current_atr = curr_row.get('atr_origin', curr_row['atr'])
            current_price = curr_row['close']
            current_ema = curr_row.get('ema_200_origin', curr_row['ema_200'])
            current_bb_w = curr_row['bb_width']
            current_adx = curr_row['adx'] 
            current_rsi = curr_row.get('rsi_origin', curr_row['rsi'])
            current_rvol = curr_row.get('rvol_origin', curr_row['rvol'])

            signal = 'HOLD'
            
            # [수정] 안정형 전략: Super Trend Guard + Strict Reversal
            is_volume_supported = current_rvol > config.RVOL_THRESHOLD

            if current_bb_w > config.BB_WIDTH_THRESHOLD:
                is_trend_mode = current_adx > config.ADX_THRESHOLD # 30
                is_super_trend = current_adx > 45 
                is_uptrend = current_price > current_ema
                
                # Trend Mode
                if is_trend_mode:
                    if is_volume_supported:
                        if is_uptrend:
                            if score > current_params['entry_threshold']: signal = 'OPEN_LONG'
                        else: 
                            if config.ENABLE_SHORT and score < -current_params['entry_threshold']: signal = 'OPEN_SHORT'
                
                # Range Mode (Super Trend일 땐 역추세 금지)
                elif not is_super_trend:
                    is_oversold = current_rsi < 25 # 엄격
                    is_overbought = current_rsi > 75
                    
                    if is_oversold and score > current_params['entry_threshold']: signal = 'OPEN_LONG'
                    elif config.ENABLE_SHORT and is_overbought and score < -current_params['entry_threshold']: signal = 'OPEN_SHORT'

            exec_price = next_bar['open']
            dyn_lev = 1
            qty = 0
            
            if signal != 'HOLD':
                dyn_lev = account.get_dynamic_leverage(score, current_params['entry_threshold'])
                qty = account.get_position_qty(
                    exec_price, score, current_atr, dyn_lev,
                    risk_scale=current_params['risk_scale'],
                    sl_mult=current_params['sl_mul']
                )
            
            account.execute_trade(signal, exec_price, qty, dyn_lev, next_bar.name)
            
            bar_start = next_bar.name
            bar_end = bar_start + timedelta(hours=1)
            try: precision_candles = df_5m.loc[bar_start:bar_end]
            except: precision_candles = pd.DataFrame()

            account.update_pnl_and_check_exit(
                next_bar['close'], next_bar['high'], next_bar['low'], next_bar.name,
                current_atr, precision_candles,
                sl_mult=current_params['sl_mul'],
                tp_ratio=current_params['tp_ratio']
            )
            
            equity_curve.append({'time': next_bar.name, 'balance': account.balance})

        current_test_start = current_test_end
        if account.balance < config.INITIAL_BALANCE * 0.2:
            logger.warning(">> BANKRUPTCY STOP.")
            break

    df_result = pd.DataFrame(equity_curve).set_index('time')
    df_result = df_result[~df_result.index.duplicated(keep='last')]
    df_result.to_csv(os.path.join(config.LOG_DIR, f"final_result_{symbol.replace('/','_')}.csv"))
    
    plot_results(df_result, symbol)
    
    final_bal = account.balance
    total_ret = ((final_bal - config.INITIAL_BALANCE) / config.INITIAL_BALANCE) * 100
    logger.info(f"Final Balance: {final_bal:.2f} ({total_ret:+.2f}%)")

if __name__ == "__main__":
    for s in config.TARGET_SYMBOLS:
        run_strategy(s)
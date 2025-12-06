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
from ga_optimizer import GeneticOptimizer
from sklearn.preprocessing import RobustScaler

# 로거 설정 (기존과 동일)
logger = logging.getLogger("BacktestLogger")
logger.setLevel(logging.INFO)
logger.propagate = False 
if not os.path.exists(config.LOG_DIR): os.makedirs(config.LOG_DIR)
file_handler = logging.FileHandler(config.LOG_FILE, mode='w', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
logger.addHandler(file_handler)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(console_handler)

def plot_results(df_result, symbol):
    if df_result.empty: return
    safe_symbol = symbol.replace('/', '_')
    chart_path = os.path.join(config.LOG_DIR, f"equity_{safe_symbol}.png")
    
    df_result.index = pd.to_datetime(df_result.index)
    plt.figure(figsize=(12, 6))
    plt.plot(df_result.index, df_result['balance'], label=f'{symbol} Balance')
    plt.title(f'Equity Curve - {symbol}')
    plt.grid(True)
    plt.legend()
    plt.savefig(chart_path)
    plt.close()

def run_strategy(symbol):
    logger.info("\n" + "="*60)
    logger.info(f">> STARTING STRATEGY FOR: {symbol}")
    logger.info("="*60)

    dp = DataProcessor(symbol)
    full_df = dp.prepare_multi_timeframe_data()
    
    if len(full_df) < config.LSTM_WINDOW + 200:
        logger.error("!! Not enough data.")
        return

    exclude_cols = ['timestamp', 'target_cls', 'target_vol', 'open', 'high', 'low', 'close', 'volume']
    feature_cols = [c for c in full_df.columns if c not in exclude_cols]
    
    current_train_end = pd.to_datetime(config.TEST_START)
    final_end = pd.to_datetime(config.COLLECT_END)
    interval = timedelta(days=config.ONLINE_TRAIN_INTERVAL_DAYS)
    
    model = HybridEnsemble(symbol)
    account = AccountManager(balance=config.INITIAL_BALANCE)
    ga = GeneticOptimizer() # GA 옵티마이저 생성
    
    equity_curve = [{'time': current_train_end, 'balance': config.INITIAL_BALANCE}]
    
    # 초기 유전자는 기본값으로 시작
    current_genes = {
        'base_threshold': 0.2, 'vol_impact': 1.0, 'sl_mul': 1.0, 'tp_ratio': 2.0
    }

    while current_train_end < final_end:
        chunk_end = current_train_end + interval
        if chunk_end > final_end: chunk_end = final_end
        
        logger.info(f"\n>> [Period] {current_train_end} ~ {chunk_end} | Symbol: {symbol}")
        
        # 1. 데이터 분리
        train_mask = full_df.index < current_train_end
        test_mask = (full_df.index >= current_train_end) & (full_df.index < chunk_end)
        
        train_df = full_df.loc[train_mask]
        test_df = full_df.loc[test_mask]
        
        if test_df.empty: break
        if len(train_df) < config.LSTM_WINDOW + 200:
            current_train_end = chunk_end; continue

        # 2. 스케일링
        scaler = RobustScaler()
        X_train_raw = train_df[feature_cols].values
        X_test_raw = test_df[feature_cols].values
        
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)
        
        y_train_cls = train_df['target_cls'].values
        y_train_vol = train_df['target_vol'].values

        # 3. 모델 학습 (LSTM/XGB)
        # 시퀀스 생성
        X_train_seq, y_train_seq_cls, y_train_seq_vol = dp.create_sequences(
            X_train_scaled, y_train_cls, y_train_vol, config.LSTM_WINDOW
        )
        
        # XGB용 평탄화 데이터 (LSTM 윈도우 이후부터 사용 가능)
        X_train_flat = X_train_scaled[config.LSTM_WINDOW:]
        y_train_flat_cls = y_train_cls[config.LSTM_WINDOW:]
        
        model.train(
            X_train_seq, y_train_seq_cls, y_train_seq_vol, 
            X_train_flat, y_train_flat_cls, 
            features_name=feature_cols, is_update=(account.position is not None)
        )

        # -------------------------------------------------------------
        # [NEW] GA 최적화 단계 (최근 학습 데이터의 일부를 사용하여 최적화)
        # -------------------------------------------------------------
        logger.info("   >> Running GA Optimization on recent data...")
        
        # GA용 Calibration 데이터 (Train의 마지막 3개월 정도 사용)
        calib_size = 90 * 6 # 90일 * 4시간봉(6개/일) = 540개
        if len(X_train_scaled) > calib_size + config.LSTM_WINDOW:
            X_calib = X_train_scaled[-calib_size:]
            # Calibration용 OHLCV (가격 데이터 필요)
            calib_ohlcv = train_df[['open', 'high', 'low', 'close']].values[-calib_size:]
            
            # Calibration 데이터에 대해 AI 예측 수행
            # (학습된 모델이 과거 데이터를 보고 어떤 점수를 줬을지 계산)
            dummy_y = np.zeros(len(X_calib))
            X_calib_seq, _, _ = dp.create_sequences(X_calib, dummy_y, dummy_y, config.LSTM_WINDOW)
            
            # 시퀀스 생성으로 인해 앞부분 잘림 보정
            calib_ohlcv_cut = calib_ohlcv[config.LSTM_WINDOW:] 
            X_calib_flat = X_calib[config.LSTM_WINDOW:]
            
            # 예측 (Score, Vol)
            calib_scores, calib_vols = model.predict(X_calib_seq, X_calib_flat)
            
            # GA 입력용 통합 배열 생성
            ga_input_preds = np.column_stack((calib_scores, calib_vols))
            
            # GA 실행
            best_gene, best_fit = ga.optimize(calib_ohlcv_cut, ga_input_preds)
            current_genes = best_gene
            logger.info(f"   >> [GA Best Gene] Fit: {best_fit:.1f} | Genes: {current_genes}")
        else:
            logger.warning("   >> Not enough data for GA. Using previous genes.")

        # -------------------------------------------------------------
        # 4. 실전(Test) 매매 시뮬레이션
        # -------------------------------------------------------------
        combined_scaled = np.concatenate([X_train_scaled[-config.LSTM_WINDOW:], X_test_scaled], axis=0)
        dummy_y = np.zeros(len(combined_scaled))
        X_test_seq, _, _ = dp.create_sequences(combined_scaled, dummy_y, dummy_y, config.LSTM_WINDOW)
        X_test_flat = X_test_scaled

        # Test 구간 예측
        ai_scores, pred_vols = model.predict(X_test_seq, X_test_flat)

        for i in range(len(test_df) - 1):
            if account.balance <= 0: break
            
            curr_row = test_df.iloc[i]
            next_bar = test_df.iloc[i+1]
            timestamp = next_bar.name
            
            score = ai_scores[i]
            pred_vol = pred_vols[i]
            exec_price = next_bar['open']
            
            # SL/TP 청산 체크
            sl = account.position['sl_price'] if account.position else 0
            tp = account.position['tp_price'] if account.position else 0
            
            account.update_pnl_and_check_exit(
                next_bar['close'], next_bar['high'], next_bar['low'], 
                timestamp, sl, tp
            )
            
            # 신규 진입 (GA가 찾아낸 current_genes 사용)
            if not account.position:
                lev, qty, sl_dist, tp_dist = account.calculate_trade_parameters(
                    score, pred_vol, exec_price, current_genes
                )
                
                # 진입 결정 (calculate_trade_parameters가 0을 리턴하면 진입 안 함)
                if qty > 0:
                    signal = 'OPEN_LONG' if score > 0 else 'OPEN_SHORT'
                    if score < 0 and not config.ENABLE_SHORT: signal = 'HOLD'
                    
                    if signal != 'HOLD':
                        account.execute_trade(signal, exec_price, qty, lev, sl_dist, tp_dist, timestamp)
                
            equity_curve.append({'time': timestamp, 'balance': account.balance})

        current_train_end = chunk_end
        if account.balance <= 0: 
            logger.error("!!! BANKRUPT !!!")
            break

    # 최종 정리
    if account.position:
        account._force_close(full_df.iloc[-1]['close'], full_df.index[-1], 'FinalClose')
        equity_curve.append({'time': full_df.index[-1], 'balance': account.balance})

    df_result = pd.DataFrame(equity_curve).set_index('time')
    df_result = df_result[~df_result.index.duplicated(keep='last')]
    
    final_bal = df_result.iloc[-1]['balance']
    roi = ((final_bal/config.INITIAL_BALANCE)-1)*100
    logger.info(f"\n   >> [RESULT] {symbol} Balance: ${final_bal:,.2f} ({roi:+.2f}%)")
    
    safe_symbol = symbol.replace('/', '_')
    df_result.to_csv(os.path.join(config.LOG_DIR, f"result_{safe_symbol}.csv"))
    plot_results(df_result, symbol)

if __name__ == "__main__":
    # data_processor.py와 ai_models.py는 기존 코드 그대로 사용 (import 문제 없음)
    for target_symbol in config.TARGET_SYMBOLS:
        run_strategy(target_symbol)
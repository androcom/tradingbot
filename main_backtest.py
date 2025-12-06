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

# 로깅 설정 초기화
logger = logging.getLogger("BacktestLogger")
logger.setLevel(logging.DEBUG)
logger.propagate = False 
if not os.path.exists(config.LOG_DIR): os.makedirs(config.LOG_DIR)

# 파일 핸들러 설정 (상세 로그 저장)
file_handler = logging.FileHandler(config.LOG_FILE, mode='w', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
file_handler.setFormatter(file_fmt)
logger.addHandler(file_handler)

# 콘솔 핸들러 설정 (주요 정보 출력)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_fmt = logging.Formatter('%(message)s')
console_handler.setFormatter(console_fmt)
logger.addHandler(console_handler)

# 백테스트 결과 그래프 그리기 및 저장
def plot_results(df_result, symbol):
    if df_result.empty: return

    safe_symbol = symbol.replace('/', '_')
    chart_path = os.path.join(config.LOG_DIR, f"equity_curve_{safe_symbol}.png")

    df_result.index = pd.to_datetime(df_result.index)
    running_max = df_result['balance'].cummax()
    drawdown = (df_result['balance'] - running_max) / running_max * 100

    plt.figure(figsize=(12, 8))
    
    # 자산 곡선 (Equity Curve)
    plt.subplot(2, 1, 1)
    plt.plot(df_result.index, df_result['balance'], label=f'{symbol} Balance', color='blue')
    plt.title(f'Equity Curve - {symbol}')
    plt.grid(True)
    plt.legend()
    
    # 낙폭 (Drawdown)
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

# 단일 종목에 대한 전략 실행 함수
def run_strategy(symbol):
    logger.info("\n" + "="*60)
    logger.info(f">> STARTING STRATEGY FOR: {symbol}")
    logger.info(f">> Timeframe: {config.MAIN_TIMEFRAME} | Leverage: {config.MIN_LEVERAGE}~{config.MAX_LEVERAGE}x")
    logger.info("="*60)

    # 1. 데이터 준비 및 전처리
    dp = DataProcessor(symbol)
    full_df = dp.prepare_multi_timeframe_data()
    
    if len(full_df) < config.LSTM_WINDOW + 100:
        logger.error(f"!! Not enough data for {symbol}.")
        return

    features = [c for c in full_df.columns if c not in ['timestamp', 'target_cls', 'open', 'high', 'low', 'close', 'volume']]
    logger.info(f"   - Features: {features}")

    current_train_end = pd.to_datetime(config.TEST_START)
    final_end = pd.to_datetime(config.COLLECT_END)
    interval = timedelta(days=config.ONLINE_TRAIN_INTERVAL_DAYS)
    
    # 2. AI 모델 초기화 (매번 새로운 모델 인스턴스 생성)
    model = HybridEnsemble(symbol)
    
    # 3. 초기 학습 (테스트 시작 전 데이터로 학습)
    logger.info(f"   [Init] Training new model from scratch for {symbol}...")
    init_train_df = full_df[full_df.index < current_train_end]
    
    if not init_train_df.empty and len(init_train_df) > config.LSTM_WINDOW:
        X_init_seq, y_init_seq = dp.create_sequences(init_train_df, features, config.LSTM_WINDOW)
        X_init_flat = init_train_df.iloc[config.LSTM_WINDOW:][features].values
        y_init_flat = init_train_df.iloc[config.LSTM_WINDOW:]['target_cls'].values
        
        model.train(X_init_seq, y_init_seq, X_init_flat, y_init_flat, is_update=False)
    else:
        logger.warning("   [Warning] Not enough initial data for training. Skipping...")

    # 4. 백테스트 시뮬레이션 루프 시작
    account = AccountManager(balance=config.INITIAL_BALANCE, leverage=1.0)
    equity_curve = [{'time': current_train_end, 'balance': config.INITIAL_BALANCE}]
    
    while current_train_end < final_end:
        chunk_end = current_train_end + interval
        if chunk_end > final_end: chunk_end = final_end
        
        logger.info(f"\n>> [Period] {current_train_end} ~ {chunk_end} | Symbol: {symbol}")
        
        train_mask = full_df.index < current_train_end
        test_mask = (full_df.index >= current_train_end) & (full_df.index < chunk_end)
        
        train_df = full_df.loc[train_mask]
        test_df = full_df.loc[test_mask]
        
        if test_df.empty: break
        if len(train_df) < config.LSTM_WINDOW: 
            current_train_end = chunk_end; continue

        # 온라인 재학습 (주기적으로 최신 데이터 반영)
        X_train_seq, y_train_seq = dp.create_sequences(train_df, features, config.LSTM_WINDOW)
        X_train_flat = train_df.iloc[config.LSTM_WINDOW:][features].values
        y_train_flat = train_df.iloc[config.LSTM_WINDOW:]['target_cls'].values
        
        model.train(X_train_seq, y_train_seq, X_train_flat, y_train_flat, is_update=True)

        # 테스트 구간에 대한 예측 수행
        concat_df = pd.concat([train_df.iloc[-config.LSTM_WINDOW:], test_df])
        X_test_seq, _ = dp.create_sequences(concat_df, features, config.LSTM_WINDOW)
        X_test_flat = test_df[features].values
        
        # 기술적 지표 기반 점수 (RSI 과매수/과매도)
        tech_scores = np.where(test_df['rsi'] < 30, 1, np.where(test_df['rsi'] > 70, -1, 0))
        # AI 모델 예측 점수
        ai_scores = model.batch_predict(X_test_seq, X_test_flat, tech_scores)

        # 바(Bar) 단위 매매 루프
        for i in range(len(test_df) - 1):
            if account.balance <= 0: break
            
            curr_row = test_df.iloc[i]
            next_bar = test_df.iloc[i+1]
            
            score = ai_scores[i]            
            current_atr = curr_row['atr']   
            
            signal = 'HOLD'
            if score > config.ENTRY_THRESHOLD: 
                signal = 'OPEN_LONG'
            elif config.ENABLE_SHORT and score < -config.ENTRY_THRESHOLD: 
                signal = 'OPEN_SHORT'
            
            exec_price = next_bar['open']
            qty = 0
            dyn_leverage = 1.0
            
            # 진입 신호 발생 시 수량 및 레버리지 계산
            if signal != 'HOLD':
                dyn_leverage = account.get_dynamic_leverage(score)
                qty = account.get_position_qty(exec_price, score, current_atr, dyn_leverage)
            
            # 주문 실행
            account.execute_trade(signal, exec_price, qty, dyn_leverage, next_bar.name)
            
            # 포지션 평가 및 청산 체크 (손절/익절)
            account.update_pnl_and_check_exit(
                next_bar['close'], 
                next_bar['high'], 
                next_bar['low'], 
                next_bar.name, 
                current_atr
            )
            
            equity_curve.append({'time': next_bar.name, 'balance': account.balance})

        current_train_end = chunk_end
        if account.balance <= 0: break

    # 백테스트 종료 후 남은 포지션 강제 청산
    if account.position:
        last_bar = full_df.iloc[-1]
        account._force_close(last_bar['close'], full_df.index[-1], 'FinalClose')
        equity_curve.append({'time': full_df.index[-1], 'balance': account.balance})

    # 최종 결과 계산 및 저장
    df_result = pd.DataFrame(equity_curve).set_index('time')
    df_result = df_result[~df_result.index.duplicated(keep='last')]
    
    final_bal = df_result.iloc[-1]['balance']
    roi = ((final_bal/config.INITIAL_BALANCE)-1)*100
    
    logger.info(f"\n   >> [RESULT] {symbol} Balance: ${final_bal:,.2f} ({roi:+.2f}%)")
    
    safe_symbol = symbol.replace('/', '_')
    result_path = os.path.join(config.LOG_DIR, f"backtest_result_{safe_symbol}.csv")
    df_result.to_csv(result_path)
    
    plot_results(df_result, symbol)

if __name__ == "__main__":
    # 설정된 모든 대상 종목에 대해 전략 실행
    for target_symbol in config.TARGET_SYMBOLS:
        run_strategy(target_symbol)

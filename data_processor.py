# data_processor.py
import pandas as pd
import numpy as np
import ccxt
import ta
import logging
import time
import os
from sklearn.preprocessing import RobustScaler
import config

logger = logging.getLogger("BacktestLogger")

class DataProcessor:
    def __init__(self, symbol):
        self.exchange = getattr(ccxt, config.EXCHANGE_NAME)()
        self.symbol = symbol 
        self.scaler = RobustScaler()

    def fetch_ohlcv(self, timeframe, start_str, end_str):
        symbol_str = self.symbol.replace('/', '_')
        file_name = f"{symbol_str}_{timeframe}.csv"
        file_path = os.path.join(config.DATA_DIR, file_name)
        
        df = pd.DataFrame()

        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            except: pass

        if df.empty:
            since = self.exchange.parse8601(start_str)
            end_ts = self.exchange.parse8601(end_str)
            all_ohlcv = []
            logger.info(f"   [Data] Fetching new {timeframe} data for {self.symbol}...")
            while since < end_ts:
                try:
                    ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe, since, limit=1000)
                    if not ohlcv: break
                    last_ts = ohlcv[-1][0]
                    if last_ts == since: break 
                    since = last_ts + 1
                    all_ohlcv.extend(ohlcv)
                    time.sleep(self.exchange.rateLimit / 1000)
                except:
                    time.sleep(5); continue
            
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df.to_csv(file_path)

        df = df[~df.index.duplicated(keep='last')].sort_index()
        mask = (df.index >= pd.to_datetime(start_str)) & (df.index < pd.to_datetime(end_str))
        return df.loc[mask]

    def add_technical_indicators(self, df, suffix=''):
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        # 기본 지표
        df[f'rsi{suffix}'] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
        bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
        df[f'bb_width{suffix}'] = bb.bollinger_wband()
        df[f'atr{suffix}'] = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
        macd = ta.trend.MACD(close=close)
        df[f'macd_diff{suffix}'] = macd.macd_diff() 
        df[f'adx{suffix}'] = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14).adx()
        df[f'cci{suffix}'] = ta.trend.CCIIndicator(high=high, low=low, close=close, window=20).cci()
        
        # 추가 지표 (성능 개선용)
        df[f'log_ret{suffix}'] = np.log(close / close.shift(1))
        df[f'obv{suffix}'] = ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
        df[f'mfi{suffix}'] = ta.volume.MFIIndicator(high=high, low=low, close=close, volume=volume, window=14).money_flow_index()
        
        ichimoku = ta.trend.IchimokuIndicator(high=high, low=low)
        df[f'ichi_trend{suffix}'] = np.where(close > ichimoku.ichimoku_a(), 1, -1)
        df[f'close_aux{suffix}'] = close.pct_change().rolling(window=10).std()
        
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        return df

    def create_targets(self, df):
        # 1. 방향 타겟 (Classification)
        # 단순 수익률 대신 변동성 대비 수익률을 사용하여 노이즈 제거
        future_ret = df['close'].shift(-1) / df['close'] - 1.0
        # 동적 임계값: 최근 변동성(ATR) 이상 움직여야 추세로 인정
        threshold = (df['atr'] / df['close'])
        
        conditions = [
            (future_ret < -threshold), 
            (future_ret > threshold)   
        ]
        choices = [0, 2] # 0: Short, 2: Long
        df['target_cls'] = np.select(conditions, choices, default=1)

        # 2. 변동성 타겟 (Regression) - [신규]
        # 향후 N봉 동안의 최대 변동폭(High-Low) 비율을 예측하여 SL/TP 설정에 활용
        lookahead = config.VOLATILITY_LOOKAHEAD
        future_high = df['high'].rolling(window=lookahead).max().shift(-lookahead)
        future_low = df['low'].rolling(window=lookahead).min().shift(-lookahead)
        
        # 현재가 대비 최대 변동 비율
        volatility_ratio = (future_high - future_low) / df['close']
        df['target_vol'] = volatility_ratio
        
        df.dropna(subset=['target_cls', 'target_vol'], inplace=True)
        return df

    def prepare_multi_timeframe_data(self):
        main_df = self.fetch_ohlcv(config.MAIN_TIMEFRAME, config.COLLECT_START, config.COLLECT_END)
        aux_df = self.fetch_ohlcv(config.AUX_TIMEFRAME, config.COLLECT_START, config.COLLECT_END)
        
        if main_df.empty or aux_df.empty: return pd.DataFrame()

        main_df = self.add_technical_indicators(main_df, suffix='')
        aux_df = self.add_technical_indicators(aux_df, suffix='_1d')

        aux_safe = aux_df.shift(1) 
        aux_resampled = aux_safe.reindex(main_df.index, method='ffill')
        
        aux_cols = [c for c in aux_resampled.columns if c.endswith('_1d')]
        merged_df = pd.concat([main_df, aux_resampled[aux_cols]], axis=1)
        
        merged_df.dropna(inplace=True)
        merged_df = self.create_targets(merged_df)

        logger.info(f"   - Data prepared. Shape: {merged_df.shape}")
        return merged_df

    def create_sequences(self, data, target_cls, target_vol, window_size):
        # Multi-Output 생성을 위한 시퀀스
        X, y_cls, y_vol = [], [], []
        for i in range(len(data) - window_size):
            X.append(data[i : i + window_size])
            y_cls.append(target_cls[i + window_size])
            y_vol.append(target_vol[i + window_size])
        return np.array(X), np.array(y_cls), np.array(y_vol)
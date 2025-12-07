# data_processor.py
import pandas as pd
import numpy as np
import ccxt
import ta
import logging
import time
import os
import config

logger = logging.getLogger("BacktestLogger")

class DataProcessor:
    def __init__(self, symbol):
        self.exchange = getattr(ccxt, config.EXCHANGE_NAME)()
        self.symbol = symbol 
        
    def _download_range(self, timeframe, start_ts, end_ts):
        all_ohlcv = []
        since = start_ts
        while since < end_ts:
            try:
                ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe, since, limit=1000)
                if not ohlcv: break
                last_fetched_ts = ohlcv[-1][0]
                if last_fetched_ts == since: break 
                since = last_fetched_ts + 1
                all_ohlcv.extend(ohlcv)
                time.sleep(self.exchange.rateLimit / 1000)
            except Exception:
                time.sleep(5)
                continue
        if not all_ohlcv: return pd.DataFrame()
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df

    def fetch_ohlcv(self, timeframe, start_str, end_str):
        symbol_str = self.symbol.replace('/', '_')
        file_name = f"{symbol_str}_{timeframe}.parquet"
        file_path = os.path.join(config.DATA_DIR, file_name)
        
        target_start_ts = int(self.exchange.parse8601(start_str))
        target_end_ts = int(self.exchange.parse8601(end_str))
        
        df = pd.DataFrame()
        is_updated = False

        if os.path.exists(file_path):
            try:
                df = pd.read_parquet(file_path)
                df.sort_index(inplace=True)
            except: df = pd.DataFrame()

        if df.empty:
            if timeframe != config.PRECISION_TIMEFRAME:
                logger.info(f"   [Data] New Download ({timeframe})...")
            df = self._download_range(timeframe, target_start_ts, target_end_ts)
            is_updated = True
        else:
            file_start_ts = int(df.index[0].timestamp() * 1000)
            file_end_ts = int(df.index[-1].timestamp() * 1000)
            
            if target_start_ts < (file_start_ts - 1000 * 60 * 60 * 4):
                if timeframe != config.PRECISION_TIMEFRAME:
                    logger.info(f"   [Data] Prepending data ({timeframe})...")
                front_df = self._download_range(timeframe, target_start_ts, file_start_ts)
                if not front_df.empty:
                    df = pd.concat([front_df, df])
                    is_updated = True

            if file_end_ts < (target_end_ts - 1000 * 60 * 60 * 4):
                if timeframe != config.PRECISION_TIMEFRAME:
                    logger.info(f"   [Data] Appending data ({timeframe})...")
                back_df = self._download_range(timeframe, file_end_ts + 1, target_end_ts)
                if not back_df.empty:
                    df = pd.concat([df, back_df])
                    is_updated = True

        if is_updated and not df.empty:
            df = df[~df.index.duplicated(keep='last')]
            df.sort_index(inplace=True)
            df.to_parquet(file_path)
            if timeframe != config.PRECISION_TIMEFRAME:
                logger.info(f"   [Data] Synced: {file_name}")
        
        if df.empty: return df
        mask = (df.index >= pd.to_datetime(start_str)) & (df.index < pd.to_datetime(end_str))
        return df.loc[mask]

    def add_technical_indicators(self, df, suffix=''):
        win = config.INDICATOR_WINDOW
        
        df[f'rsi{suffix}'] = ta.momentum.RSIIndicator(close=df['close'], window=win).rsi()
        bb = ta.volatility.BollingerBands(close=df['close'], window=config.BB_WINDOW, window_dev=config.BB_STD)
        df[f'bb_width{suffix}'] = bb.bollinger_wband()
        
        atr_val = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=win).average_true_range()
        df[f'atr{suffix}'] = atr_val
        
        ema_val = ta.trend.EMAIndicator(close=df['close'], window=config.EMA_WINDOW).ema_indicator()
        df[f'ema_200{suffix}'] = ema_val

        # [수정] RVOL 및 고급 피처 추가
        df[f'vol_ma{suffix}'] = df['volume'].rolling(window=20).mean()
        df[f'rvol{suffix}'] = df['volume'] / (df[f'vol_ma{suffix}'] + 1e-9)
        df[f'disparity{suffix}'] = df['close'] / (df[f'ema_200{suffix}'] + 1e-9)
        df[f'log_ret{suffix}'] = np.log(df['close'] / df['close'].shift(1))

        if suffix == '':
            df['atr_origin'] = atr_val
            df['ema_200_origin'] = ema_val
            df['rsi_origin'] = df['rsi']
            df['rvol_origin'] = df['rvol']

        macd = ta.trend.MACD(close=df['close'])
        df[f'macd{suffix}'] = macd.macd()
        df[f'macd_diff{suffix}'] = macd.macd_diff() 
        df[f'adx{suffix}'] = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=win).adx()
        df[f'cci{suffix}'] = ta.trend.CCIIndicator(high=df['high'], low=df['low'], close=df['close'], window=20).cci()
        
        stoch = ta.momentum.StochRSIIndicator(close=df['close'], window=14, smooth1=3, smooth2=3)
        df[f'stoch_k{suffix}'] = stoch.stochrsi_k()
        df[f'willr{suffix}'] = ta.momentum.WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close']).williams_r()
        df[f'obv{suffix}'] = ta.volume.OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
        df[f'mfi{suffix}'] = ta.volume.MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume']).money_flow_index()
        
        ichimoku = ta.trend.IchimokuIndicator(high=df['high'], low=df['low'])
        df[f'cloud_thickness{suffix}'] = ichimoku.ichimoku_a() - ichimoku.ichimoku_b()

        if suffix == '':
            roll_win = 24 if config.MAIN_TIMEFRAME == '1h' else 6 
            df['rolling_24h_change'] = (df['close'] - df['close'].shift(roll_win)) / df['close'].shift(roll_win)
        
        df[f'close_aux{suffix}'] = df['close'].pct_change().rolling(window=10).std()
        
        df.dropna(inplace=True)
        return df

    def create_target_labels(self, df):
        future_ret = (df['close'].shift(-config.LOOK_AHEAD_STEPS) - df['close']) / df['close']
        threshold = config.TARGET_THRESHOLD
        
        conditions = [ (future_ret < -threshold), (future_ret > threshold) ]
        choices = [0, 2]
        
        df['target_cls'] = np.select(conditions, choices, default=1)
        df.dropna(subset=['target_cls'], inplace=True)
        df['target_cls'] = df['target_cls'].astype(int)
        return df

    def prepare_multi_timeframe_data(self):
        main_df = self.fetch_ohlcv(config.MAIN_TIMEFRAME, config.COLLECT_START, config.COLLECT_END)
        aux_df = self.fetch_ohlcv(config.AUX_TIMEFRAME, config.COLLECT_START, config.COLLECT_END)
        
        if main_df.empty or aux_df.empty: return pd.DataFrame()

        main_df = self.add_technical_indicators(main_df, suffix='')
        aux_df = self.add_technical_indicators(aux_df, suffix='_1d')

        aux_safe = aux_df.shift(1) 
        aux_resampled = aux_safe.reindex(main_df.index, method='ffill')
        
        aux_features = [c for c in aux_resampled.columns if '_1d' in c and ('rsi' in c or 'atr' in c or 'macd' in c)]
        merged_df = pd.concat([main_df, aux_resampled[aux_features]], axis=1)
        
        merged_df.dropna(inplace=True)
        merged_df = self.create_target_labels(merged_df)
        return merged_df

    def load_precision_data(self, start_date):
        logger.info(f"   [Precision] Checking updates for {config.PRECISION_TIMEFRAME} data...")
        return self.fetch_ohlcv(config.PRECISION_TIMEFRAME, start_date, config.COLLECT_END)

    def create_sequences(self, df, features, window_size):
        data = df[features].values.astype(np.float32)
        target = df['target_cls'].values.astype(np.int32)
        
        num_samples = len(data) - window_size
        if num_samples <= 0: return np.array([]), np.array([])

        X = np.lib.stride_tricks.sliding_window_view(data, window_shape=(window_size, len(features)))
        X = X.squeeze(axis=1) 
        y = target[window_size:]
        
        if len(X) > len(y): X = X[:len(y)]
        elif len(y) > len(X): y = y[:len(X)]
        return X, y
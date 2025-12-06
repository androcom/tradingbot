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
        self.main_tf = config.MAIN_TIMEFRAME
        self.aux_tf = config.AUX_TIMEFRAME
        
    # OHLCV 데이터 가져오기 (로컬 파일 확인 후 없으면 거래소 다운로드)
    def fetch_ohlcv(self, timeframe, start_str, end_str):
        symbol_str = self.symbol.replace('/', '_')
        file_name = f"{symbol_str}_{timeframe}.csv"
        file_path = os.path.join(config.DATA_DIR, file_name)
        
        df = pd.DataFrame()

        # 1. 로컬에 저장된 데이터 파일이 있는지 확인
        if os.path.exists(file_path):
            logger.info(f"   [Data] Found local file: {file_name}. Loading...")
            try:
                df = pd.read_csv(file_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            except Exception as e:
                logger.warning(f"   [Warning] Load failed: {e}. Downloading...")

        # 2. 로컬 데이터가 없으면 거래소 API를 통해 다운로드
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
                except Exception as e:
                    logger.error(f"   [Error] Fetching failed: {e}")
                    time.sleep(5)
                    continue
            
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df.to_csv(file_path)
            logger.info(f"   [Data] Saved to {file_path}")

        # 중복 제거 및 정렬, 요청 기간 필터링
        df = df[~df.index.duplicated(keep='last')]
        df = df.sort_index()
        mask = (df.index >= pd.to_datetime(start_str)) & (df.index < pd.to_datetime(end_str))
        return df.loc[mask]

    # 기술적 지표 추가
    def add_technical_indicators(self, df, suffix=''):
        # 1. 상대강도지수 (RSI)
        df[f'rsi{suffix}'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
        # 2. 볼린저 밴드 폭 (BB Width)
        bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
        df[f'bb_width{suffix}'] = bb.bollinger_wband()
        # 3. 평균 진폭 범위 (ATR) - 변동성 측정
        atr_indicator = ta.volatility.AverageTrueRange(
            high=df['high'], low=df['low'], close=df['close'], window=14
        )
        df[f'atr{suffix}'] = atr_indicator.average_true_range()
        
        # 4. 이동평균 수렴확산 (MACD)
        macd = ta.trend.MACD(close=df['close'])
        df[f'macd{suffix}'] = macd.macd()
        df[f'macd_diff{suffix}'] = macd.macd_diff() 
        
        # 5. 평균 방향성 지수 (ADX) - 추세 강도
        df[f'adx{suffix}'] = ta.trend.ADXIndicator(
            high=df['high'], low=df['low'], close=df['close'], window=14
        ).adx()
        
        # 6. 상품 채널 지수 (CCI)
        df[f'cci{suffix}'] = ta.trend.CCIIndicator(
            high=df['high'], low=df['low'], close=df['close'], window=20
        ).cci()

        # 7. 종가 변동성 보조 지표
        df[f'close_aux{suffix}'] = df['close'].pct_change().rolling(window=10).std()
        
        df.dropna(inplace=True)
        return df

    # 학습용 타겟 레이블 생성 (미래 수익률 기반)
    def create_target_labels(self, df):
        future_ret = df['close'].shift(-1) / df['close'] - 1.0
        threshold = config.TARGET_THRESHOLD
        
        conditions = [
            (future_ret < -threshold), # 하락
            (future_ret > threshold)   # 상승
        ]
        choices = [0, 2] # 0: 하락, 2: 상승
        
        # 기본값 1: 횡보
        df['target_cls'] = np.select(conditions, choices, default=1)
        df.dropna(subset=['target_cls'], inplace=True)
        df['target_cls'] = df['target_cls'].astype(int)
        return df

    # 다중 타임프레임 데이터 준비 및 병합
    def prepare_multi_timeframe_data(self):
        main_df = self.fetch_ohlcv(config.MAIN_TIMEFRAME, config.COLLECT_START, config.COLLECT_END)
        aux_df = self.fetch_ohlcv(config.AUX_TIMEFRAME, config.COLLECT_START, config.COLLECT_END)
        
        if main_df.empty or aux_df.empty: return pd.DataFrame()

        main_df = self.add_technical_indicators(main_df, suffix='')
        aux_df = self.add_technical_indicators(aux_df, suffix='_1d')

        # --- [미래 참조 편향(Look-ahead Bias) 방지] ---
        # 1. 보조 데이터(일봉)를 하루 뒤로 밀어서 현재 시점에 미래 데이터를 참조하지 않도록 함
        aux_safe = aux_df.shift(1) 

        # 2. 메인 타임프레임 인덱스에 맞춰 보조 데이터를 Forward Fill (직전 값 유지)
        aux_resampled = aux_safe.reindex(main_df.index, method='ffill')
        # -----------------------------

        # 필요한 보조 지표만 선택하여 병합
        aux_features = [c for c in aux_resampled.columns if '_1d' in c and ('rsi' in c or 'atr' in c or 'adx' in c)]
        
        merged_df = pd.concat([main_df, aux_resampled[aux_features]], axis=1)
        merged_df.dropna(inplace=True)
        merged_df = self.create_target_labels(merged_df)
        
        logger.info(f"   - Data prepared for {self.symbol}. Shape: {merged_df.shape}")
        return merged_df

    # LSTM 학습을 위한 시퀀스 데이터 생성
    def create_sequences(self, df, features, window_size):
        X, y = [], []
        data = df[features].values
        target = df['target_cls'].values
        
        for i in range(len(data) - window_size):
            X.append(data[i : i + window_size])
            y.append(target[i + window_size])
            
        return np.array(X), np.array(y)

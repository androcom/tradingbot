import pandas as pd
import numpy as np
import ccxt
import ta
import os
import time
import config

class DataLoader:
    def __init__(self, logger=None):
        self.exchange = ccxt.binance()
        self.logger = logger

    def log(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def _download_range(self, symbol, timeframe, start_ts, end_ts):
        """CCXT를 사용해 지정된 범위의 OHLCV 데이터 다운로드"""
        all_ohlcv = []
        since = start_ts
        limit = 1000  # Binance limit
        
        while since < end_ts:
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit=limit)
                if not ohlcv:
                    break
                
                last_fetched_ts = ohlcv[-1][0]
                
                # 더 이상 새로운 데이터가 없으면 중단
                if last_fetched_ts == since:
                    break
                    
                since = last_fetched_ts + 1
                all_ohlcv.extend(ohlcv)
                
                # Rate Limit 준수
                time.sleep(self.exchange.rateLimit / 1000)
                
                # 진행 상황 로그 (선택 사항)
                # end_date_str = pd.to_datetime(last_fetched_ts, unit='ms')
                # print(f"   Downloading... current: {end_date_str}", end='\r')
                
            except Exception as e:
                self.log(f"[Data] Error during download: {e}")
                time.sleep(5)
                continue
                
        if not all_ohlcv:
            return pd.DataFrame()
            
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # 중복 제거 및 정렬
        df = df[~df.index.duplicated(keep='last')]
        df.sort_index(inplace=True)
        
        return df

    def fetch_data(self, symbol, timeframe, start_str, end_str):
        """데이터 다운로드, 로컬 캐싱 및 로드"""
        safe_symbol = symbol.replace('/', '_')
        file_path = os.path.join(config.DATA_DIR, f"{safe_symbol}_{timeframe}.parquet")
        
        # 타임스탬프 변환 (ms 단위)
        target_start_ts = self.exchange.parse8601(start_str)
        target_end_ts = self.exchange.parse8601(end_str)

        df = pd.DataFrame()
        is_new_download = False

        # 1. 로컬 파일 로드 시도
        if os.path.exists(file_path):
            try:
                df = pd.read_parquet(file_path)
                if not df.empty:
                    file_start_ts = int(df.index[0].timestamp() * 1000)
                    file_end_ts = int(df.index[-1].timestamp() * 1000)
                    
                    # 데이터가 요청 범위보다 부족하면 추가 다운로드 로직이 필요하지만,
                    # 여기서는 파일이 존재하면 우선 사용하는 것으로 단순화 (필요시 기존 코드의 append 로직 사용)
                    # 만약 파일이 있는데 기간이 너무 짧다면 삭제 후 다시 받는 것을 권장
                    self.log(f"[Data] Loaded local file: {file_path} ({len(df)} rows)")
                    
                    # 요청된 기간만큼 필터링
                    mask = (df.index >= pd.to_datetime(start_str)) & (df.index < pd.to_datetime(end_str))
                    return df.loc[mask]
            except Exception as e:
                self.log(f"[Data] File load error, redownloading... {e}")
                df = pd.DataFrame()

        # 2. 파일이 없거나 로드 실패 시 다운로드
        if df.empty:
            self.log(f"[Data] Downloading {symbol} ({timeframe}) from {start_str} to {end_str}...")
            df = self._download_range(symbol, timeframe, target_start_ts, target_end_ts)
            
            if not df.empty:
                df.to_parquet(file_path)
                self.log(f"[Data] Download complete & Saved to {file_path}")
                is_new_download = True
            else:
                self.log("[Data] Download failed or no data found.")

        return df

    def add_indicators(self, df):
        """기술적 지표 추가 (Look-ahead Bias 엄격 금지)"""
        if df.empty:
            return df
        
        df = df.copy()
        
        # 1. 기본 지표
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        
        # 2. 볼린저 밴드
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_w'] = bb.bollinger_wband()
        df['bb_p'] = bb.bollinger_pband()
        df['bb_h'] = bb.bollinger_hband()
        df['bb_l'] = bb.bollinger_lband()

        # 3. 이동평균 및 괴리율
        df['ema_200'] = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()
        df['dist_ema'] = (df['close'] - df['ema_200']) / (df['ema_200'] + 1e-9)

        # 4. 거래량 변동성
        df['vol_ma'] = df['volume'].rolling(20).mean()
        df['rvol'] = df['volume'] / (df['vol_ma'] + 1e-9)

        # [중요] 지표 계산으로 인한 NaN 제거 (앞부분 200개 등)
        df.dropna(inplace=True)
        
        return df

    def create_target(self, df):
        """
        ML 학습용 Target 생성.
        [주의] 이 함수는 Feature 생성 후 호출되며, 결과 컬럼은 Feature로 사용 금지
        """
        if df.empty:
            return df
            
        df = df.copy()
        
        # 미래 N봉 뒤의 수익률 (Look-ahead)
        # config.LOOK_AHEAD_STEPS가 1이면 바로 다음 봉을 예측
        future_ret = df['close'].shift(-config.LOOK_AHEAD_STEPS) / df['close'] - 1
        
        # 0: 하락, 1: 횡보, 2: 상승
        # Threshold: 0.005 (0.5%)
        conditions = [
            (future_ret < -config.TARGET_THRESHOLD),
            (future_ret > config.TARGET_THRESHOLD)
        ]
        choices = [0, 2] # 0: Short, 2: Long
        
        df['target_cls'] = np.select(conditions, choices, default=1) # 1: Hold
        df['target_val'] = future_ret # 회귀 분석용 (Optional)
        
        # 마지막 N개 행은 미래 데이터(Target)를 알 수 없으므로 제거
        df.dropna(subset=['target_cls', 'target_val'], inplace=True)
        
        # target_cls를 정수형으로 변환
        df['target_cls'] = df['target_cls'].astype(int)
        
        return df

    def get_ml_data(self):
        """ML 학습을 위한 전체 데이터셋 준비 (다운로드 -> 지표 -> 타겟)"""
        df = self.fetch_data(config.SYMBOL, config.TIMEFRAME_MAIN, config.DATE_START, config.DATE_END)
        
        if df.empty:
            self.log("!! [Error] Main DataFrame is empty. Cannot proceed.")
            return df
            
        df = self.add_indicators(df)
        df = self.create_target(df) 
        
        return df
    
    def get_precision_data(self):
        """백테스트용 5분봉 데이터"""
        return self.fetch_data(config.SYMBOL, config.TIMEFRAME_PRECISION, config.DATE_START, config.DATE_END)
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from collections import deque
import config

class CryptoEnv(gym.Env):
    def __init__(self, df, trading_logic, precision_df=None, debug=False):
        super(CryptoEnv, self).__init__()
        
        self.df = df
        self.precision_df = precision_df
        self.logic = trading_logic # strategies/trading_core.py의 인스턴스
        self.debug = debug
        
        # RL 학습에 사용할 Feature Columns
        self.features = [c for c in df.columns if c not in config.EXCLUDE_COLS]
        
        # Action Space: 0:Hold, 1:Long, 2:Short, 3:Close
        self.action_space = spaces.Discrete(4)
        
        # Observation Space
        # [Market Features] + [Position Info (3개: Size, EntryPrice, UnrealizedPnL%)]
        obs_dim = len(self.features) + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # 보상 계산을 위한 기록용 버퍼
        self.returns_buffer = deque(maxlen=30)
        self.max_equity = config.INITIAL_BALANCE

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.logic.reset() # Account Reset
        
        self.returns_buffer.clear()
        self.max_equity = config.INITIAL_BALANCE
        
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        market_obs = row[self.features].values.astype(np.float32)
        
        # [수정] 절대 가격(entry_price) 대신 '진입가 대비 현재가 비율' 사용
        if self.logic.position:
            pos_size = float(self.logic.position['size'])
            # Entry Price 자체는 학습에 방해됨 (값이 너무 큼). 
            # 대신 현재가/진입가 비율을 사용.
            entry_price = float(self.logic.position['price'])
            price_ratio = (row['close'] / entry_price) if entry_price > 0 else 1.0
            pnl_pct = self.logic.get_unrealized_pnl_pct(row['close'])
        else:
            pos_size = 0.0
            price_ratio = 1.0
            pnl_pct = 0.0
            
        pos_obs = np.array([pos_size, price_ratio, pnl_pct], dtype=np.float32)
        
        return np.concatenate([market_obs, pos_obs])

    def step(self, action):
        # 1. 현재 스텝 데이터 확인
        row = self.df.iloc[self.current_step]
        current_ts = row.name
        current_close = row['close']
        
        # 이전 자산 가치 (보상 계산용)
        # Unrealized PnL을 포함한 총 자산(Equity) 기준
        prev_equity = self.logic.balance + self.logic.get_unrealized_pnl(current_close)
        
        # 2. 5분봉 정밀 데이터 매칭
        precision_candles = None
        if self.precision_df is not None:
            try:
                end_ts = current_ts + pd.Timedelta(minutes=59)
                precision_candles = self.precision_df.loc[current_ts:end_ts]
            except KeyError:
                pass 

        # 3. 매매 로직 실행 (Trading Core 위임)
        self.logic.process_step(action, row, current_ts, precision_candles)
        
        # 4. 자산 가치 재평가
        curr_equity = self.logic.balance + self.logic.get_unrealized_pnl(current_close)
        
        # 5. 보상 계산 (개선된 함수 호출)
        reward = self._calculate_reward(prev_equity, curr_equity)
        
        # 6. 종료 조건 체크
        self.current_step += 1
        terminated = False
        truncated = False
        info = {}
        
        # 데이터 끝 도달
        if self.current_step >= len(self.df) - 1:
            terminated = True
            info['final_balance'] = curr_equity # 백테스트 그래프용
            
        # 파산 체크 (원금의 40% 미만)
        if curr_equity < config.INITIAL_BALANCE * 0.4:
            terminated = True
            reward = -10.0 # 파산 페널티
            info['final_balance'] = curr_equity
        
        return self._get_obs(), reward, terminated, truncated, info

    def _calculate_reward(self, prev_equity, curr_equity):
        # 1. 수익률 보상 (로그 수익률)
        # 스케일링을 키워서(100 -> 200) 작은 수익도 크게 느끼게 함
        log_return = np.log(curr_equity / prev_equity) if prev_equity > 0 else 0
        step_reward = log_return * 200 
        
        # 2. [수정] Winning Bonus 강화 (수익 나면 더 큰 칭찬)
        if log_return > 0:
            step_reward *= 2.0  # 1.5 -> 2.0 (수익에 대한 갈망 유도)

        # 3. [삭제] 변동성 페널티 제거
        # 초반 학습에는 방해만 됩니다. 단순하게 갑시다.
        # if len(self.returns_buffer) > 10: ... (삭제)

        # 4. [완화] MDD 페널티 대폭 축소
        # 고점 대비 하락에 대한 공포를 줄여줍니다.
        if curr_equity > self.max_equity:
            self.max_equity = curr_equity
            step_reward += 0.5 # 고점 갱신 보너스 추가 (신규)
        
        # Drawdown 페널티는 정말 심각할 때만(10% 이상) 부여
        drawdown = (self.max_equity - curr_equity) / self.max_equity
        if drawdown > 0.1: 
            step_reward -= (drawdown * 1.0) # 계수 완화

        # 5. [신규] '버티기' 보상 (Time Survival Reward)
        # 깡통 차지 않고 살아있는 것만으로도 아주 작은 점수를 줌
        step_reward += 0.001 

        return np.clip(step_reward, -5.0, 5.0)

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
        
        # RL 학습에 사용할 Feature Columns (설정에서 제외된 컬럼 뺌)
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
        self.logic.reset() # 자산 및 포지션 초기화
        
        self.returns_buffer.clear()
        self.max_equity = config.INITIAL_BALANCE
        
        return self._get_obs(), {}

    def _get_obs(self):
        """현재 상태(Observation) 생성"""
        # 1. Market Data
        row = self.df.iloc[self.current_step]
        market_obs = row[self.features].values.astype(np.float32)
        
        # 2. Position Data
        # 정규화를 위해 가격 정보는 비율로 변환하거나 스케일링이 필요하지만,
        # 여기서는 모델(PPO)의 VecNormalize가 처리하도록 Raw값 전달
        if self.logic.position:
            pos_size = float(self.logic.position['size'])
            entry_price = float(self.logic.position['price'])
            # 현재 가격 기준 미실현 수익률
            pnl_pct = self.logic.get_unrealized_pnl_pct(row['close'])
        else:
            pos_size = 0.0
            entry_price = 0.0
            pnl_pct = 0.0
            
        pos_obs = np.array([pos_size, entry_price, pnl_pct], dtype=np.float32)
        
        return np.concatenate([market_obs, pos_obs])

    def step(self, action):
        # 1. 현재 스텝 데이터 확인
        row = self.df.iloc[self.current_step]
        current_ts = row.name # Index must be DatetimeIndex
        current_close = row['close']
        
        # 이전 자산 가치 (보상 계산용)
        prev_equity = self.logic.balance + self.logic.get_unrealized_pnl(current_close)
        
        # 2. 5분봉 정밀 데이터 매칭 (Precision Slicing)
        precision_candles = None
        if self.precision_df is not None:
            try:
                # 현재 1시간봉의 시작 시간 ~ 끝 시간 (59분)
                end_ts = current_ts + pd.Timedelta(minutes=59)
                # pandas의 시간 인덱스 슬라이싱 활용 (빠름)
                precision_candles = self.precision_df.loc[current_ts:end_ts]
            except KeyError:
                pass # 데이터가 없으면 Fallback 로직(TradingCore 내부) 사용

        # 3. 매매 로직 실행 (Trading Core 위임)
        # 여기서 5분봉 데이터를 넘겨주어 고가/저가 터치 여부를 정밀 계산
        self.logic.process_step(action, row, current_ts, precision_candles)
        
        # 4. 자산 가치 재평가
        curr_equity = self.logic.balance + self.logic.get_unrealized_pnl(current_close)
        
        # 5. 보상 계산
        reward = self._calculate_reward(prev_equity, curr_equity)
        
        # 6. 종료 조건 체크
        self.current_step += 1
        terminated = False
        truncated = False
        
        # 데이터 끝 도달
        if self.current_step >= len(self.df) - 1:
            terminated = True
            
        # 파산 체크 (원금의 40% 미만이면 종료)
        if curr_equity < config.INITIAL_BALANCE * 0.4:
            terminated = True
            reward = -10.0 # 과도한 -100 대신 적절한 페널티 부여
        
        return self._get_obs(), reward, terminated, truncated, {}

    def _calculate_reward(self, prev_equity, curr_equity):
        """
        [보상 함수 로직]
        1. Log Returns: 복리 효과 반영
        2. Volatility Penalty: 변동성이 크면 보상 차감 (Sortino 스타일)
        3. Drawdown Penalty: 최고점 대비 하락폭 페널티
        """
        if prev_equity <= 0: return 0
        
        # 로그 수익률 사용
        log_return = np.log(curr_equity / prev_equity)
        self.returns_buffer.append(log_return)
        
        # 1. 기본 보상 (수익률 스케일링)
        step_reward = log_return * 100 
        
        # 2. 변동성 페널티 (안정성 추구)
        if len(self.returns_buffer) > 5:
            std = np.std(self.returns_buffer)
            # 변동성이 클수록 페널티 (수익이 나더라도 불안정하면 점수 깎음)
            step_reward -= (std * 0.1)
            
        # 3. MDD 페널티 (Drawdown 관리)
        if curr_equity > self.max_equity:
            self.max_equity = curr_equity
        
        drawdown = (self.max_equity - curr_equity) / self.max_equity
        if drawdown > 0.05: # 5% 이상 하락 시 추가 페널티
            step_reward -= (drawdown * 0.2)

        # 4. Reward Clipping (학습 안정화)
        # 지나치게 큰 보상/벌점은 PPO 학습을 방해하므로 자름
        step_reward = np.clip(step_reward, -1.0, 1.0)
        
        return step_reward

    def render(self, mode='human'):
        """간단한 상태 출력"""
        if self.logic.position:
            print(f"Step: {self.current_step} | Bal: {self.logic.balance:.2f} | "
                  f"Pos: {self.logic.position['type']} ({self.logic.get_unrealized_pnl_pct(0)*100:.2f}%)")
        else:
            print(f"Step: {self.current_step} | Bal: {self.logic.balance:.2f} | Pos: None")
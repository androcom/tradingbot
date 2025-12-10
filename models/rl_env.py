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
        self.logic = trading_logic
        self.debug = debug
        
        self.features = [c for c in df.columns if c not in config.EXCLUDE_COLS and 'ml_signal' not in c]
        self.action_space = spaces.Discrete(4)
        
        # Observation: Market + Position + ML_Signal
        obs_dim = len(self.features) + 3 + 1 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        self.returns_buffer = deque(maxlen=30)
        self.max_equity = config.INITIAL_BALANCE
        
        # [신규] 기본 보상 파라미터 (Optuna가 없을 때를 대비한 기본값)
        self.reward_params = {
            'profit_scale': 200.0,
            'teacher_bonus': 0.05,
            'teacher_penalty': 0.1,
            'mdd_penalty_factor': 1.0,
            'new_high_bonus': 0.5  # 기본값
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.logic.reset()
        self.returns_buffer.clear()
        self.max_equity = config.INITIAL_BALANCE
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        market_obs = row[self.features].values.astype(np.float32)
        
        if self.logic.position:
            pos_size = float(self.logic.position['size'])
            entry_price = float(self.logic.position['entry_price'])
            price_ratio = (row['close'] / entry_price) if entry_price > 0 else 1.0
            pnl_pct = self.logic.get_unrealized_pnl_pct(row['close'])
        else:
            pos_size = 0.0
            price_ratio = 1.0
            pnl_pct = 0.0
            
        ml_sig = row.get('ml_signal', 0.0)
        pos_obs = np.array([pos_size, price_ratio, pnl_pct, ml_sig], dtype=np.float32)
        
        return np.concatenate([market_obs, pos_obs])

    def step(self, action):
        row = self.df.iloc[self.current_step]
        current_ts = row.name
        current_close = row['close']
        ml_signal = row.get('ml_signal', 0.0)
        
        prev_equity = self.logic.balance + self.logic.get_unrealized_pnl(current_close)
        
        precision_candles = None
        if self.precision_df is not None:
            try:
                end_ts = current_ts + pd.Timedelta(minutes=59)
                precision_candles = self.precision_df.loc[current_ts:end_ts]
            except KeyError: pass 

        self.logic.process_step(action, row, current_ts, precision_candles)
        
        curr_equity = self.logic.balance + self.logic.get_unrealized_pnl(current_close)
        
        reward = self._calculate_reward(prev_equity, curr_equity, action, ml_signal)
        
        self.current_step += 1
        terminated = False
        truncated = False
        info = {}
        
        if self.current_step >= len(self.df) - 1:
            terminated = True
            info['final_balance'] = curr_equity
            
        if curr_equity < config.INITIAL_BALANCE * 0.4:
            terminated = True
            reward = -10.0
            info['final_balance'] = curr_equity
        
        return self._get_obs(), reward, terminated, truncated, info

    def _calculate_reward(self, prev_equity, curr_equity, action, ml_signal):
        p = self.reward_params # 파라미터 딕셔너리 사용
        
        log_return = np.log(curr_equity / prev_equity) if prev_equity > 0 else 0
        step_reward = log_return * p['profit_scale']
        
        # Teacher Forcing
        if ml_signal > 0.5:
            if action == 1: step_reward += p['teacher_bonus']
            elif action == 2: step_reward -= p['teacher_penalty']
        elif ml_signal < -0.5:
            if action == 2: step_reward += p['teacher_bonus']
            elif action == 1: step_reward -= p['teacher_penalty']
            
        # New High Bonus (변수 적용)
        if curr_equity > self.max_equity:
            self.max_equity = curr_equity
            step_reward += p['new_high_bonus'] # [수정됨]
        
        # MDD Penalty
        drawdown = (self.max_equity - curr_equity) / self.max_equity
        if drawdown > 0.1: step_reward -= (drawdown * p['mdd_penalty_factor'])

        return np.clip(step_reward, -5.0, 5.0)
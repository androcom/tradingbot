import os
import sys
import logging
import warnings
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

import config
from utils.data_loader import DataLoader
from models.ml_models import HybridLearner
from models.rl_env import CryptoEnv
from strategies.trading_core import TradingCore

warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3")

# [수정] RL 로그 콜백 (Running Average로 버그 수정)
class RLLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RLLoggingCallback, self).__init__(verbose)
        self.logger_obj = logging.getLogger()
        self.last_mean_reward = 0.0
        self.last_mean_length = 0.0

    def _on_training_start(self) -> None:
        self.logger_obj.info(f"   [RL_TRAIN] Training STARTED... (Total Timesteps: {self.model._total_timesteps})")

    def _on_step(self) -> bool:
        # 매 스텝마다 최신 정보가 있으면 업데이트해둠
        if len(self.model.ep_info_buffer) > 0:
            self.last_mean_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
            self.last_mean_length = np.mean([ep_info['l'] for ep_info in self.model.ep_info_buffer])
        return True

    def _on_training_end(self) -> None:
        self.logger_obj.info(f"   [RL_TRAIN] Training FINISHED | Final Ep_Rew_Mean: {self.last_mean_reward:.2f} | Final Ep_Len_Mean: {self.last_mean_length:.2f}")

class PipelineTrainer:
    def __init__(self, session_paths):
        self.paths = session_paths
        self.logger = logging.getLogger()
        self.loader = DataLoader(self.logger)
        self.model_dir = self.paths['model']
        self.scaler = RobustScaler()

    def log(self, msg):
        self.logger.info(msg)
        
    def log_parameters(self):
        """[수정] Config의 모든 파라미터를 로그에 출력"""
        self.log("[CONFIG] FULL CONFIGURATION:")
        for attr in dir(config):
            if not attr.startswith("__"):
                val = getattr(config, attr)
                if isinstance(val, (int, float, str, list, dict)):
                    self.log(f"   - {attr}: {val}")
        self.log("-" * 30)

    def _get_optimal_n_envs(self):
        try:
            cpu_count = multiprocessing.cpu_count()
            return max(4, min(16, cpu_count - 2))
        except Exception:
            return 4

    def run_all(self):
        self.log(f"\n{'='*60}")
        self.log(f"🚀 PIPELINE START: Session {self.paths['id']}")
        self.log(f"{'='*60}\n")
        
        self.log_parameters()

        # 1. Load Data
        self.log(f"[Phase 1] Loading Data for {config.MAIN_SYMBOL} (MTF Mode)...")
        full_df = self.loader.get_ml_data(config.MAIN_SYMBOL)
        
        if full_df.empty:
            self.log("!! [Error] No data found.")
            return

        # 2. Scaler Fit (Train Only)
        train_idx_mask = full_df.index < config.TEST_SPLIT_DATE
        feature_cols = [c for c in full_df.columns if c not in config.EXCLUDE_COLS]
        
        self.log("   - Fitting Scaler on Train Data...")
        self.scaler.fit(full_df.loc[train_idx_mask, feature_cols])

        # 3. Generate Signals
        self.log("\n[Phase 2 & 3] Generating 'Honest' ML Signals (Walk-Forward)...")
        full_df['ml_signal'] = 0.0
        
        train_df = full_df[train_idx_mask].copy()
        test_df = full_df[~train_idx_mask].copy()
        
        kf = KFold(n_splits=5, shuffle=False)
        fold = 1
        for tr_idx, val_idx in kf.split(train_df):
            self.log(f"   >> Processing Fold {fold}/5 for Honest Training Signals...")
            fold_train = train_df.iloc[tr_idx]
            fold_val = train_df.iloc[val_idx]
            
            X_flat_tr, X_seq_tr, y_tr = self._prepare_ml_inputs(fold_train, feature_cols, is_training=True)
            X_flat_val, X_seq_val, _ = self._prepare_ml_inputs(fold_val, feature_cols, is_training=False)
            
            if len(X_flat_tr) == 0 or len(X_flat_val) == 0: continue

            temp_model = HybridLearner(self.model_dir)
            temp_model.train(X_flat_tr, y_tr, X_seq_tr, y_tr)
            signals = temp_model.predict_proba(X_flat_val, X_seq_val)
            
            valid_start_idx = config.ML_SEQ_LEN
            target_indices = fold_val.index[valid_start_idx:]
            min_len = min(len(target_indices), len(signals))
            full_df.loc[target_indices[:min_len], 'ml_signal'] = signals[:min_len]
            fold += 1

        self.log("   >> Training Final Model for Test Set Predictions...")
        X_flat_all_tr, X_seq_all_tr, y_all_tr = self._prepare_ml_inputs(train_df, feature_cols, is_training=True)
        final_model = HybridLearner(self.model_dir)
        final_model.train(X_flat_all_tr, y_all_tr, X_seq_all_tr, y_all_tr)
        
        X_flat_test, X_seq_test, _ = self._prepare_ml_inputs(test_df, feature_cols, is_training=False)
        test_signals = final_model.predict_proba(X_flat_test, X_seq_test)
        
        test_indices = test_df.index[config.ML_SEQ_LEN:]
        min_len_test = min(len(test_indices), len(test_signals))
        full_df.loc[test_indices[:min_len_test], 'ml_signal'] = test_signals[:min_len_test]
        full_df['ml_signal'] = full_df['ml_signal'].fillna(0)
        
        self.log(f"   - Signal Generation Complete.")

        # 4. Train RL
        self.log("\n[Phase 4] Training RL Agent (Student)...")
        self._train_rl(full_df[train_idx_mask])

        # 5. Backtest
        self.log("\n[Phase 5] Running Precision Backtest...")
        precision_df = self.loader.get_precision_data(config.MAIN_SYMBOL)
        if not precision_df.empty:
            precision_df = precision_df[precision_df.index >= config.TEST_SPLIT_DATE]
            self.log(f"   - Precision Data Loaded: {len(precision_df)} rows")
        
        self._run_backtest(full_df[~train_idx_mask], precision_df)
        
        self.log(f"\n{'='*60}")
        self.log(f"✅ PIPELINE FINISHED.")
        self.log(f"{'='*60}\n")

    def _prepare_ml_inputs(self, df, features, is_training=False):
        data_scaled = self.scaler.transform(df[features])
        window_size = config.ML_SEQ_LEN
        
        if len(data_scaled) <= window_size:
            return np.array([]), np.array([]), np.array([])

        X_seq = np.lib.stride_tricks.sliding_window_view(data_scaled, window_shape=(window_size, len(features)))
        X_seq = X_seq.squeeze(axis=1)
        if X_seq.ndim == 4: X_seq = X_seq[:, 0, :, :]
             
        X_flat = data_scaled[window_size:]
        
        if 'target_cls' in df.columns and is_training:
            y = df['target_cls'].values[window_size:]
            min_len = min(len(X_seq), len(X_flat), len(y))
            return X_flat[:min_len], X_seq[:min_len], y[:min_len]
        else:
            min_len = min(len(X_seq), len(X_flat))
            return X_flat[:min_len], X_seq[:min_len], None

    def _attach_ml_signal(self, df, features):
        # Already handled in run_all logic
        return df

    def _train_rl(self, df):
        def make_env():
            return CryptoEnv(df, TradingCore(), precision_df=None, debug=False)

        n_envs = self._get_optimal_n_envs()
        self.log(f"   - Parallel Environments: {n_envs}")

        env = SubprocVecEnv([make_env for _ in range(n_envs)])
        env = VecMonitor(env)
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., gamma=config.RL_PPO_PARAMS['gamma'])
        
        checkpoint_callback = CheckpointCallback(
            save_freq=100000, save_path=self.paths['model'], name_prefix='rl_model'
        )
        logging_callback = RLLoggingCallback()

        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1, 
            tensorboard_log=self.paths['tb'], 
            device='cuda',
            **config.RL_PPO_PARAMS
        )

        model.learn(
            total_timesteps=config.RL_TOTAL_TIMESTEPS, 
            callback=[checkpoint_callback, logging_callback],
            tb_log_name="PPO_Main",
            progress_bar=True
        )
        
        model.save(os.path.join(self.paths['model'], "final_agent"))
        env.save(os.path.join(self.paths['model'], "vec_normalize.pkl"))
        env.close()

    def _run_backtest(self, df, precision_df):
        env = CryptoEnv(df, TradingCore(), precision_df=precision_df, debug=True)
        dummy_env = DummyVecEnv([lambda: env])
        
        vec_norm_path = os.path.join(self.paths['model'], "vec_normalize.pkl")
        if os.path.exists(vec_norm_path):
            norm_env = VecNormalize.load(vec_norm_path, dummy_env)
            norm_env.training = False 
            norm_env.norm_reward = False
        else:
            norm_env = dummy_env

        model = PPO.load(os.path.join(self.paths['model'], "final_agent"))
        
        obs = norm_env.reset()
        done = [False]
        
        history_dates = []
        history_bal = []
        history_price = []
        
        timestamps = df.index.to_numpy()
        prices = df['close'].to_numpy()
        total_steps = len(df)
        
        current_step = 0
        final_recorded_bal = config.INITIAL_BALANCE

        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = norm_env.step(action)
            
            if done[0]:
                info = infos[0]
                if 'final_balance' in info:
                    final_recorded_bal = info['final_balance']
                break

            if current_step < total_steps:
                try:
                    ts = timestamps[current_step]
                    price = float(prices[current_step])
                    real_env = norm_env.envs[0]
                    bal = real_env.logic.balance + real_env.logic.get_unrealized_pnl(price)
                    history_dates.append(ts)
                    history_bal.append(bal)
                    history_price.append(price)
                except IndexError:
                    pass
            current_step += 1

        save_path = os.path.join(self.paths['root'], 'backtest_result.png')
        self._plot_backtest(history_dates, history_bal, history_price, save_path)
        
        # [추가] 거래 기록 CSV 저장
        real_env = norm_env.envs[0]
        if real_env.logic.history:
            trade_df = pd.DataFrame(real_env.logic.history)
            csv_path = os.path.join(self.paths['root'], 'trade_history.csv')
            trade_df.to_csv(csv_path, index=False)
            self.log(f"   - Trade history saved to {csv_path}")
        
        roi = (final_recorded_bal - config.INITIAL_BALANCE) / config.INITIAL_BALANCE * 100
        self.log(f"   - Final Balance: ${final_recorded_bal:,.2f} (ROI: {roi:+.2f}%)")

    def _plot_backtest(self, dates, balances, prices, save_path):
        if not dates or not balances: return
        fig, ax1 = plt.subplots(figsize=(12, 6))
        color = 'tab:blue'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Balance ($)', color=color)
        ax1.plot(dates, balances, color=color, label='Balance', linewidth=1.5)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.3)
        ax2 = ax1.twinx()
        color = 'tab:gray'
        ax2.set_ylabel('BTC Price', color=color)
        ax2.plot(dates, prices, color=color, alpha=0.3, label='Price')
        ax2.tick_params(axis='y', labelcolor=color)
        plt.title('RL Agent Backtest Result')
        fig.tight_layout()
        plt.savefig(save_path)
        plt.close()
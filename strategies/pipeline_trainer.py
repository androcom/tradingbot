import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler

# Stable Baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

# Internal Modules
import config
from utils.data_loader import DataLoader
from models.ml_models import HybridLearner
from models.rl_env import CryptoEnv
from strategies.trading_core import TradingCore

class PipelineTrainer:
    def __init__(self, session_paths):
        self.paths = session_paths
        # 전역 로거 사용
        self.logger = logging.getLogger()
        self.loader = DataLoader(self.logger)
        self.ml_model = HybridLearner(self.paths['model'])
        self.scaler = RobustScaler()

    def log(self, msg):
        self.logger.info(msg)

    def run_all(self):
        self.log(f"\n{'='*60}")
        self.log(f"🚀 PIPELINE START: Session {self.paths['id']}")
        self.log(f"📁 Logs: {self.paths['root']}")
        self.log(f"📁 Models: {self.paths['model']}")
        self.log(f"{'='*60}\n")

        # [추가] 이번 세션의 하이퍼파라미터 기록 (디버깅용 핵심 정보)
        self.log_parameters()

        # 1. 데이터 로드
        self.log("[Phase 1] Loading & Preparing Data...")
        full_df = self.loader.get_ml_data()
        
        train_df = full_df[full_df.index < config.TEST_SPLIT_DATE].copy()
        test_df = full_df[full_df.index >= config.TEST_SPLIT_DATE].copy()
        
        self.log(f"   - Train Set : {len(train_df)} rows")
        self.log(f"   - Test Set  : {len(test_df)} rows")

        # 2. ML 학습
        self.log("\n[Phase 2] Training ML Model (Teacher)...")
        feature_cols = [c for c in full_df.columns if c not in config.EXCLUDE_COLS]
        
        self.log("   - Fitting Scaler...")
        self.scaler.fit(train_df[feature_cols])
        
        X_flat_tr, X_seq_tr, y_tr = self._prepare_ml_inputs(train_df, feature_cols, is_training=True)
        self.ml_model.train(X_flat_tr, y_tr, X_seq_tr, y_tr)
        
        # 3. 신호 생성
        self.log("\n[Phase 3] Generating ML Signals for RL...")
        train_df = self._attach_ml_signal(train_df, feature_cols)
        test_df = self._attach_ml_signal(test_df, feature_cols)
        self.log(f"   - ML Signals attached.")

        # 4. RL 학습
        self.log("\n[Phase 4] Training RL Agent (Student)...")
        self._train_rl(train_df)

        # 5. 백테스트
        self.log("\n[Phase 5] Running Precision Backtest...")
        precision_df = self.loader.get_precision_data()
        if not precision_df.empty:
            precision_df = precision_df[precision_df.index >= config.TEST_SPLIT_DATE]
            self.log(f"   - Precision Data Loaded: {len(precision_df)} rows")
        else:
            self.log("   ! Warning: Precision data missing. Running in Fallback mode.")
            
        self._run_backtest(test_df, precision_df)
        
        self.log(f"\n{'='*60}")
        self.log(f"✅ PIPELINE FINISHED.")
        self.log(f"{'='*60}\n")

    def log_parameters(self):
        """현재 설정된 핵심 파라미터들을 로그에 기록"""
        self.log("[CONFIG] Hyperparameters:")
        self.log(f"   - ML Epochs: {config.ML_EPOCHS}")
        self.log(f"   - ML Sequence: {config.ML_SEQ_LEN}")
        self.log(f"   - RL Timesteps: {config.RL_TOTAL_TIMESTEPS}")
        self.log(f"   - RL LR: {config.RL_PPO_PARAMS['learning_rate']}")
        self.log(f"   - Batch Size: {config.RL_PPO_PARAMS['batch_size']}")
        self.log(f"   - Leverage: {config.LEVERAGE}x")
        self.log(f"   - Fee/Slippage: {config.FEE_RATE}/{config.SLIPPAGE}")
        self.log("-" * 30)

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
        df = df.copy()
        X_flat, X_seq, _ = self._prepare_ml_inputs(df, features, is_training=False)
        
        padding = np.zeros(config.ML_SEQ_LEN)
        if len(X_seq) > 0:
            signals = self.ml_model.predict_proba(X_flat, X_seq)
            full_signals = np.concatenate([padding, signals])
        else:
            full_signals = np.zeros(len(df))
            
        if len(full_signals) > len(df): full_signals = full_signals[:len(df)]
        elif len(full_signals) < len(df):
            diff = len(df) - len(full_signals)
            full_signals = np.concatenate([full_signals, np.zeros(diff)])
            
        df['ml_signal'] = full_signals
        df['ml_signal'] = df['ml_signal'].fillna(0)
        return df

    def _train_rl(self, df):
        def make_env():
            return CryptoEnv(df, TradingCore(), precision_df=None, debug=False)

        n_envs = 4
        env = SubprocVecEnv([make_env for _ in range(n_envs)])
        env = VecMonitor(env)
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., gamma=config.RL_PPO_PARAMS['gamma'])
        
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=self.paths['tb'], **config.RL_PPO_PARAMS)
        
        checkpoint_callback = CheckpointCallback(
            save_freq=50000, save_path=self.paths['model'], name_prefix='rl_model'
        )

        model.learn(
            total_timesteps=config.RL_TOTAL_TIMESTEPS, 
            callback=checkpoint_callback,
            tb_log_name="PPO_Main",
            progress_bar=True
        )
        
        model.save(os.path.join(self.paths['model'], "final_agent"))
        env.save(os.path.join(self.paths['model'], "vec_normalize.pkl"))
        env.close()

    def _run_backtest(self, df, precision_df):
        # 1. 단일 환경 생성
        env = CryptoEnv(df, TradingCore(), precision_df=precision_df, debug=True)
        
        # 2. DummyVecEnv로 래핑 (VecNormalize 로드용)
        dummy_env = DummyVecEnv([lambda: env])
        
        vec_norm_path = os.path.join(self.paths['model'], "vec_normalize.pkl")
        if os.path.exists(vec_norm_path):
            norm_env = VecNormalize.load(vec_norm_path, dummy_env)
            norm_env.training = False 
            norm_env.norm_reward = False
            self.log("   - Loaded VecNormalize stats.")
        else:
            norm_env = dummy_env
            self.log("   ! Warning: VecNormalize stats not found.")

        model = PPO.load(os.path.join(self.paths['model'], "final_agent"))
        
        obs = norm_env.reset()
        done = [False]
        
        history_dates = []
        history_bal = []
        history_price = []
        
        self.log("   - Simulating steps...")
        
        # 메인 루프 (인덱스 보정)
        # VecNormalize를 통과한 환경(norm_env)에서 step을 진행하되,
        # 내부 데이터 접근은 실제 환경 인스턴스(env)를 통해 직접 수행하여 동기화 보장
        
        current_step = 0
        total_steps = len(df)
        
        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = norm_env.step(action)
            
            # 데이터 수집 (현재 스텝 기준)
            if current_step < total_steps:
                try:
                    ts = df.index[current_step]
                    price = df.iloc[current_step]['close']
                    # TradingCore의 balance는 다음 스텝을 위해 업데이트된 상태임
                    bal = env.logic.balance 
                    
                    history_dates.append(ts)
                    history_bal.append(bal)
                    history_price.append(price)
                except IndexError:
                    pass
            
            current_step += 1

        save_path = os.path.join(self.paths['root'], 'backtest_result.png')
        self._plot_backtest(history_dates, history_bal, history_price, save_path)
        self.log(f"   - Graph saved to {save_path}")
        
        final_bal = history_bal[-1] if history_bal else config.INITIAL_BALANCE
        roi = (final_bal - config.INITIAL_BALANCE) / config.INITIAL_BALANCE * 100
        self.log(f"   - Final Balance: ${final_bal:,.2f} (ROI: {roi:+.2f}%)")

    def _plot_backtest(self, dates, balances, prices, save_path):
        if not dates or not balances:
            self.log("   ! No history to plot.")
            return

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
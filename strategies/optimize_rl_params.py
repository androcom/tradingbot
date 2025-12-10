import optuna
import os
import sys
import logging
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.data_loader import DataLoader
from strategies.trading_core import TradingCore
from models.rl_env import CryptoEnv

# [설정] 로그 끄기
config.SYSTEM['SUPPRESS_WARNINGS'] = True
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 데이터 미리 로드 (캐싱)
LOADER = DataLoader()
DF_MAIN = LOADER.get_ml_data(config.MAIN_SYMBOL)

def objective(trial):
    # 1. 최적화할 보상 파라미터 (Reward Engineering)
    reward_params = {
        'profit_scale': trial.suggest_int('profit_scale', 100, 500, step=50),
        'teacher_bonus': trial.suggest_float('teacher_bonus', 0.0, 0.2),
        'teacher_penalty': trial.suggest_float('teacher_penalty', 0.0, 0.3),
        'mdd_penalty_factor': trial.suggest_float('mdd_penalty', 0.5, 2.0),
        
        # [추가] 전고점 갱신 보너스 (기존 0.5 하드코딩 -> 최적화)
        'new_high_bonus': trial.suggest_float('new_high_bonus', 0.1, 1.0)
    }
    
    # 2. RL 하이퍼파라미터
    lr = trial.suggest_float('learning_rate', 1e-5, 5e-4, log=True)
    
    # 3. 환경 생성 함수
    def make_env():
        env = CryptoEnv(DF_MAIN, TradingCore(), precision_df=None, debug=False)
        env.reward_params = reward_params # 파라미터 주입
        return env

    # 4. 약식 학습 (Short Training)
    # CPU 코어 수에 맞춰 n_envs 조절 (예: 6~8)
    n_envs = config.SYSTEM['NUM_WORKERS'] 
    train_steps = 300000 
    
    env = SubprocVecEnv([make_env for _ in range(n_envs)])
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=0.99)
    
    # RL Device 설정 (config 참조)
    device = config.SYSTEM['MAIN_RL_DEVICE']
    model = PPO("MlpPolicy", env, learning_rate=lr, verbose=0, device=device, n_steps=1024, batch_size=1024)
    
    try:
        model.learn(total_timesteps=train_steps)
        # 평가: 최근 100 에피소드 평균 보상
        mean_reward = np.mean([ep['r'] for ep in env.ep_info_buffer])
        return mean_reward
        
    except Exception as e:
        return -99999
    finally:
        env.close()

if __name__ == "__main__":
    print("🚀 Starting RL Reward Optimization (Short Training Mode)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50) # 50번 시도
    
    print("🏆 Best Reward Params:", study.best_params)
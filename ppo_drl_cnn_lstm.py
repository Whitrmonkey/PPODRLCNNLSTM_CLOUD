# PPO + CNN-LSTM + Optuna for ETH/USDT 15m Futures Trading (GPU Optimized with Custom Policy)

import os
import gym
import optuna
import torch
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from ta import add_all_ta_features
from gym import spaces
from tqdm import tqdm
import matplotlib.pyplot as plt

# === CONFIG ===
TRAIN_PATH = "/PPODRLCNNLSTM_CLOUD/content/eth_usdt_15m.csv"
TEST_PATH = "/PPODRLCNNLSTM_CLOUD/content/eth_usdt_15m_trade.csv"
MODEL_PATH = "PPODRLCNNLSTM_Cloud.zip"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === ENVIRONMENT ===
class CryptoTradingEnv(gym.Env):
    def __init__(self, df, window_size=96):
        super(CryptoTradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                             shape=(window_size, df.shape[1]), dtype=np.float32)
        self.reset()

    def reset(self):
        self.current_step = self.window_size
        self.position = 0
        self.entry_price = 0
        self.balance = 1000
        self.trades = []
        return self._get_observation()

    def _get_observation(self):
        return self.df.iloc[self.current_step - self.window_size:self.current_step].values.astype(np.float32)

    def step(self, action):
        reward = 0
        done = False
        price = self.df.iloc[self.current_step]['close']

        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = price
        elif action == 2 and self.position == 0:
            self.position = -1
            self.entry_price = price
        elif action == 0 and self.position != 0:
            pnl = (price - self.entry_price) * self.position
            reward = pnl
            self.balance += pnl
            self.trades.append(pnl)
            self.position = 0

        self.current_step += 1
        if self.current_step >= len(self.df):
            done = True

        return self._get_observation(), reward, done, {}

# === FEATURE EXTRACTOR ===
class CNNLSTMFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, cnn_filters, lstm_units):
        super().__init__(observation_space, features_dim=lstm_units)
        n_input_channels = observation_space.shape[1]
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=n_input_channels, out_channels=cnn_filters, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.lstm = nn.LSTM(input_size=cnn_filters, hidden_size=lstm_units, batch_first=True)

    def forward(self, obs):
        x = obs.permute(0, 2, 1).to(torch.float32)
        x = self.cnn(x)
        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        return lstm_out[:, -1, :]

# === CUSTOM POLICY ===
class CustomCNNLSTMPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        cnn_filters = kwargs.pop("cnn_filters")
        lstm_units = kwargs.pop("lstm_units")
        super(CustomCNNLSTMPolicy, self).__init__(*args,
            features_extractor_class=CNNLSTMFeatureExtractor,
            features_extractor_kwargs=dict(
                cnn_filters=cnn_filters,
                lstm_units=lstm_units
            ),
            **kwargs)

# === LOAD & PROCESS DATA ===
def load_data(path):
    print(f"📥 Loading data from: {path}")
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    df = add_all_ta_features(df, open='open', high='high', low='low', close='close', volume='volume', fillna=True)
    df['ema50'] = df['trend_ema_fast']
    df['ema200'] = df['trend_ema_slow']
    df['ema50_cross_ema200'] = (df['ema50'] > df['ema200']).astype(int)
    df['breakout'] = (df['close'] > df['close'].rolling(20).max()).astype(int)
    df.dropna(inplace=True)
    print(f"✅ Data loaded: {df.shape[0]} rows")
    return df

# === OBJECTIVE ===
def optimize(trial):
    print(f"🚀 Starting Trial #{trial.number}")
    cnn_filters = trial.suggest_categorical("cnn_filters", [16, 32, 64, 128])
    lstm_units = trial.suggest_categorical("lstm_units", [32, 64, 128, 256])
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    n_steps = trial.suggest_categorical("n_steps", [256, 512, 1024])
    gamma = trial.suggest_float("gamma", 0.90, 0.999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 1.0)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.01)

    train_df = load_data(TRAIN_PATH)
    env = DummyVecEnv([lambda: CryptoTradingEnv(train_df)])

    model = PPO(CustomCNNLSTMPolicy, env, verbose=0, learning_rate=lr, n_steps=n_steps,
                gamma=gamma, gae_lambda=gae_lambda, clip_range=clip_range,
                ent_coef=ent_coef, device=DEVICE,
                policy_kwargs=dict(cnn_filters=cnn_filters, lstm_units=lstm_units),
                tensorboard_log="./tensorboard/")

    model.learn(total_timesteps=800_000)

    test_df = load_data(TEST_PATH)
    test_env = DummyVecEnv([lambda: CryptoTradingEnv(test_df)])
    obs = test_env.reset()
    total_reward, n_trades = 0, 0

    while True:
        action, _ = model.predict(obs)
        obs, reward, done, _ = test_env.step(action)
        total_reward += reward[0]
        n_trades += 1
        if done[0]:
            break

    print(f"🎯 Trial #{trial.number} — Avg Reward: {total_reward / n_trades:.4f}")
    return total_reward / n_trades

if __name__ == '__main__':
    print("🔧 Starting Optuna optimization...")
    study = optuna.create_study(direction="maximize")
    study.optimize(optimize, n_trials=30)
    print("✅ Best Trial Params:", study.best_trial.params)

    print("🔁 Retraining final model with best hyperparameters...")
    best_params = study.best_trial.params
    train_df = load_data(TRAIN_PATH)
    env = DummyVecEnv([lambda: CryptoTradingEnv(train_df)])

    model = PPO(CustomCNNLSTMPolicy, env, verbose=1, learning_rate=best_params['learning_rate'],
                n_steps=best_params['n_steps'], gamma=best_params['gamma'],
                gae_lambda=best_params['gae_lambda'], clip_range=best_params['clip_range'],
                ent_coef=best_params['ent_coef'], device=DEVICE,
                policy_kwargs=dict(cnn_filters=best_params['cnn_filters'], lstm_units=best_params['lstm_units']))
    model.learn(total_timesteps=800_000)
    model.save(MODEL_PATH)
    print(f"✅ Final model saved as {MODEL_PATH}")


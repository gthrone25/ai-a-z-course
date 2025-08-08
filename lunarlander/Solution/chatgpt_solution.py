"""
dqn_lunar_lander.py

Deep Q-Learning (DQN) for OpenAI Gym LunarLander-v2 using PyTorch.
Includes:
 - Replay buffer
 - Epsilon-greedy policy
 - Target network updates
 - Checkpoint saving
 - Video recording of a trained agent

Usage:
  python dqn_lunar_lander.py           # trains (default small run), saves model and creates video
  Modify hyperparameters below as desired.

Note: Make sure gym (with box2d), torch, numpy are installed.
"""

import os
import random
import collections
from dataclasses import dataclass
from typing import Tuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------
# Hyperparameters
# ----------------------
ENV_NAME = "LunarLander-v2"
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_EPISODES = 800          # Increase for better performance
MAX_STEPS = 1000
BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-3
REPLAY_SIZE = 100000
INIT_MEMORY = 2000
TARGET_UPDATE_FREQ = 1000   # in steps
UPDATE_FREQ = 4             # optimize every N env steps
EPS_START = 1.0
EPS_END = 0.02
EPS_DECAY = 30000           # decay over this many steps
SAVE_PATH = "dqn_lunarlander.pth"
VIDEO_FOLDER = "videos"
RECORD_VIDEO = True         # set False if you don't want to record
EVAL_EPISODES = 5

# ----------------------
# Utilities / Replay Buffer
# ----------------------
Transition = collections.namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states = torch.tensor(np.vstack([b.state for b in batch]), dtype=torch.float32, device=DEVICE)
        actions = torch.tensor([b.action for b in batch], dtype=torch.long, device=DEVICE).unsqueeze(1)
        rewards = torch.tensor([b.reward for b in batch], dtype=torch.float32, device=DEVICE).unsqueeze(1)
        next_states = torch.tensor(np.vstack([b.next_state for b in batch]), dtype=torch.float32, device=DEVICE)
        dones = torch.tensor([b.done for b in batch], dtype=torch.float32, device=DEVICE).unsqueeze(1)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# ----------------------
# Q-Network
# ----------------------
class DQN(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden=[128, 128]):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ----------------------
# Epsilon schedule
# ----------------------
@dataclass
class EpsilonScheduler:
    start: float
    end: float
    decay: int
    steps_done: int = 0

    def get_epsilon(self) -> float:
        self.steps_done += 1
        # exponential decay to epsilon_end over decay steps
        eps = self.end + (self.start - self.end) * np.exp(-1.0 * self.steps_done / (self.decay / 5.0))
        # alternative linear: eps = max(self.end, self.start - (self.steps_done / self.decay) * (self.start-self.end))
        return float(eps)

# ----------------------
# Training function
# ----------------------
def train():
    # env
    env = gym.make(ENV_NAME)
    env.seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # networks
    policy_net = DQN(obs_dim, action_dim).to(DEVICE)
    target_net = DQN(obs_dim, action_dim).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay = ReplayBuffer(REPLAY_SIZE)
    eps_sched = EpsilonScheduler(EPS_START, EPS_END, EPS_DECAY)

    # initialize replay
    state = env.reset()
    for _ in range(INIT_MEMORY):
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        replay.push(state, action, reward, next_state, done)
        state = next_state if not done else env.reset()

    total_steps = 0
    losses = []
    episode_rewards = []

    def select_action(s):
        eps = eps_sched.get_epsilon()
        if random.random() < eps:
            return env.action_space.sample()
        with torch.no_grad():
            s_t = torch.tensor(s, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            qvals = policy_net(s_t)
            return int(qvals.argmax().item())

    for ep in range(1, NUM_EPISODES + 1):
        state = env.reset()
        ep_reward = 0.0
        for t in range(MAX_STEPS):
            action = select_action(state)
            next_state, reward, done, _ = env.step(action)
            replay.push(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
            total_steps += 1

            # learn every UPDATE_FREQ steps
            if total_steps % UPDATE_FREQ == 0 and len(replay) >= BATCH_SIZE:
                states, actions, rewards, next_states, dones = replay.sample(BATCH_SIZE)
                # Q(s,a)
                q_values = policy_net(states).gather(1, actions)
                # target: r + gamma * max_a' Q_target(s', a') * (1 - done)
                with torch.no_grad():
                    next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
                    target = rewards + (1.0 - dones) * GAMMA * next_q_values
                loss = nn.functional.mse_loss(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            # update target network
            if total_steps % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        episode_rewards.append(ep_reward)
        avg_reward = np.mean(episode_rewards[-100:])

        if ep % 10 == 0 or ep == 1:
            print(f"Episode {ep:4d} | Steps {total_steps:7d} | EpReward {ep_reward:7.2f} | Avg100 {avg_reward:7.2f} | Eps {eps_sched.get_epsilon():.3f}")

        # checkpoint
        if ep % 100 == 0:
            torch.save(policy_net.state_dict(), f"{SAVE_PATH}.ep{ep}")

    # final save
    torch.save(policy_net.state_dict(), SAVE_PATH)
    print("Training complete. Model saved to:", SAVE_PATH)
    env.close()
    return policy_net

# ----------------------
# Play & Record video
# ----------------------
def evaluate_and_record(model: nn.Module, episodes=5, record=True, video_folder=VIDEO_FOLDER):
    # create env for recording
    env = gym.make(ENV_NAME)
    env.seed(SEED + 123)
    if record:
        os.makedirs(video_folder, exist_ok=True)
        # RecordVideo wrapper for gym >=0.21: use record_video wrapper
        try:
            from gym.wrappers import RecordVideo
            env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda x: True)
        except Exception:
            # older gym versions:
            env = gym.wrappers.Monitor(env, video_folder, force=True)
    model.eval()
    total = []
    for ep in range(episodes):
        s = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            with torch.no_grad():
                s_t = torch.tensor(s, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                q = model(s_t)
                a = int(q.argmax().item())
            s, r, done, _ = env.step(a)
            ep_ret += r
        print(f"Eval episode {ep+1} reward: {ep_ret:.2f}")
        total.append(ep_ret)
    env.close()
    print(f"Avg eval reward over {episodes}: {np.mean(total):.2f}")

# ----------------------
# Main
# ----------------------
if __name__ == "__main__":
    # Simple orchestrator: train then evaluate & record
    # If a saved model exists, you can skip training by loading it
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--eval", action="store_true", help="Run evaluation and record video")
    parser.add_argument("--load", type=str, default="", help="Path to saved model to load (skip training)")
    args = parser.parse_args()

    if args.load:
        # load model and evaluate
        env = gym.make(ENV_NAME)
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        net = DQN(obs_dim, action_dim).to(DEVICE)
        state_dict = torch.load(args.load, map_location=DEVICE)
        net.load_state_dict(state_dict)
        print("Loaded model:", args.load)
        evaluate_and_record(net, episodes=EVAL_EPISODES, record=RECORD_VIDEO, video_folder=VIDEO_FOLDER)
    else:
        if args.train:
            trained_net = train()
        else:
            # default behavior: train then record
            trained_net = train()
            evaluate_and_record(trained_net, episodes=EVAL_EPISODES, record=RECORD_VIDEO, video_folder=VIDEO_FOLDER)

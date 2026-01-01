import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import math
import random
import numpy as np
import torch.nn.functional as F
from gymnasium.vector import AsyncVectorEnv

from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import time
from config import *

import gymnasium as gym
from models.convs.plainPPO import PPONet
from utils import *



CURRICULUM_NAMING, CURRICULUM = map_envs(CURRICULUM)


class Teacher():
    def __init__(self):
        self.env_dict = {id: {
            "fast_average": 0.0,
            "slow_average": 0.0,
        } for id in CURRICULUM.keys()}
        self.lps = torch.zeros(len(self.env_dict))
        self.alpha = 0.1
        self.beta = 0.05
        self.counts = {id: 0 for id in CURRICULUM.keys()}

    def update_env(self, earned_return, env_id):
        env = self.env_dict[env_id]
        env["fast_average"] = earned_return * self.alpha + (1 - self.alpha) * env["fast_average"]
        env["slow_average"] = earned_return * self.beta + (1 - self.beta) * env["slow_average"]
        self.lps[env_id] = math.fabs(env["fast_average"] - env["slow_average"])

    def get_env(self):
        chosen_env = None
        if random.random() < 0.2:
            chosen_env = random.choice(list(self.env_dict.keys()))
        else:
            safe_lps = self.lps + 1e-6
            probabilities = safe_lps / safe_lps.sum()
            chosen_env = Categorical(probs=probabilities).sample().item()

        self.counts[chosen_env] += 1
        return chosen_env


class MetaEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.current_taks = random.choice(list(CURRICULUM.values()))
        self.env = self.current_taks()
        self.changed = False
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def set_task(self, task_func):
        self.changed = task_func == self.current_taks
        self.current_taks = task_func

    def reset(self, **kwargs):
        if self.changed:
            self.env = self.current_taks()
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)


def make_meta_env():
    return MetaEnv()


if __name__ == "__main__":

    tb = SummaryWriter(LOG_DIR, flush_secs=30)

    LOAD_VERSION = 0
    NUM_ITERATIONS = 100000


    EPS = 0.2
    POLICY_COEFF = 1
    VALUE_COEFF = 1
    ALPHA = 0.005
    BETA = 0.01
    learning_model = PPONet(EMBEDDING_DIM, HIDDEN_DIMS, NUM_ACTIONS).to(DEVICE)
    learning_model.eval()
    optimizer = torch.optim.Adam(learning_model.parameters(), lr=LEARNING_RATE)
    teacher = Teacher()
    vector_env = AsyncVectorEnv(
        [make_meta_env for _ in range(NUM_ENVS)],
        shared_memory=False,
        autoreset_mode="Disabled"
    )

    i = 0
    now = time.time()
    var = 1
    mean = 0
    num_episodes = 0

    individual_logs = {key: 0 for key in CURRICULUM.keys()}

    while True:
        i += 1
        num_episodes += NUM_ENVS
        states, actions, rewards, dones, old_log_probs = [], [], [], [], []
        learning_model.eval()
        env_id = teacher.get_env()
        vector_env.call("set_task", CURRICULUM[env_id])
        max_steps = CURRICULUM_STEPS[CURRICULUM_NAMING[env_id]]
        with torch.no_grad():
            finished_scores = []
            obs, _ = vector_env.reset()
            h = learning_model.init_hidden(NUM_ENVS, device=DEVICE)
            for t in range(max_steps):
                ob = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(1)
                v_ext, p, h, _ = learning_model(ob, h)
                dist = Categorical(logits=p)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                action = action.squeeze(1).cpu().numpy()
                old_log_probs.append(log_prob)
                states.append(obs)
                obs, reward, terminated, truncated, _ = vector_env.step(action)

                actions.append(action)
                rewards.append(reward)
                dones.append(terminated)

            states = torch.tensor(np.array(states), dtype=torch.float32, device=DEVICE)  # (T, num_envs, C,H,W)
            actions = torch.tensor(np.array(actions), dtype=torch.long, device=DEVICE)  # (T, num_envs)
            rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=DEVICE)  # (T, num_envs)
            dones = torch.tensor(np.array(dones), dtype=torch.long, device=DEVICE)  # (T, num_envs)

            states = states.permute(1, 0, 2, 3, 4)  # (batch, time, C, H, W)
            actions = actions.permute(1, 0)  # (batch, time)
            rewards = rewards.permute(1, 0)  # (batch, time)
            dones = dones.permute(1, 0)  # (batch, time)
            old_log_probs = torch.stack(old_log_probs).squeeze(-1).permute(1, 0)

        first_done = dones.argmax(dim=1)
        has_done = dones.any(dim=1)

        lengths = torch.where(has_done, first_done + 1, torch.tensor(max_steps, device=DEVICE))
        mask = torch.arange(max_steps, device=DEVICE).unsqueeze(0) < lengths.unsqueeze(1)

        with torch.no_grad():
            last_obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(1)
            next_v_ext, _, _, _ = learning_model(last_obs_tensor, h)

            next_v_ext = next_v_ext.squeeze(-1).squeeze(-1)

            current_return_ext = next_v_ext * (1.0 - dones[:, -1])
            extrinsic_returns = torch.zeros_like(rewards)

            for t in reversed(range(max_steps)):
                bootstrap_mask = 1.0 - dones[:, t]

                extrinsic_returns[:, t] = rewards[:, t] + GAMMA * current_return_ext * bootstrap_mask
                current_return_ext = extrinsic_returns[:, t]

        extrinsic_returns = extrinsic_returns.detach()

        ##training

        policy_losses = []
        entropy_losses = []

        extrinsic_losses = []

        learning_model.train()

        learning_hidden = learning_model.init_hidden(NUM_ENVS, DEVICE)
        for c in range(0, max_steps, CHUNK_SIZE):
            chunk_mask = mask[:, c: c + CHUNK_SIZE]
            if chunk_mask.sum() == 0:
                continue
            state_chunk = states[:, c:c + CHUNK_SIZE, :]
            rewards_chunk = rewards[:, c:c + CHUNK_SIZE]
            old_log_probs_chunk = old_log_probs[:, c:c + CHUNK_SIZE]
            dones_chunk = dones[:, c:c + CHUNK_SIZE]
            actions_chunk = actions[:, c:c + CHUNK_SIZE]

            extrinsic_chunk = extrinsic_returns[:, c:c + CHUNK_SIZE]

            extrinsic_val, policy_logits, hidden, _ = learning_model(state_chunk, learning_hidden)

            extrinsic_val = extrinsic_val.squeeze(-1)

            value_loss_ext = F.mse_loss(extrinsic_val[chunk_mask], extrinsic_chunk[chunk_mask])

            advantage_extrinsic = extrinsic_chunk - extrinsic_val
            advantage = (advantage_extrinsic).detach()

            distributions = Categorical(logits=policy_logits)
            log_probs = distributions.log_prob(actions_chunk)
            entropies = distributions.entropy()

            log_probs_flat = log_probs[chunk_mask]
            old_log_probs_flat = old_log_probs_chunk[chunk_mask]
            advantage_flat = advantage[chunk_mask]
            entropy_flat = entropies[chunk_mask]

            ratio = torch.exp(log_probs_flat - old_log_probs_flat)

            surr1 = ratio * advantage_flat
            surr2 = torch.clamp(ratio, 1.0 - EPS, 1.0 + EPS) * advantage_flat

            entropy = entropy_flat.mean()
            policy_loss = -torch.min(surr1, surr2).mean() - BETA * entropy

            combined_loss = POLICY_COEFF * policy_loss + VALUE_COEFF * value_loss_ext
            policy_losses.append(policy_loss.item())

            entropy_losses.append(entropy.item())

            extrinsic_losses.append(value_loss_ext.item())

            combined_loss.backward()

            learning_hidden = hidden.detach()
        torch.nn.utils.clip_grad_norm_(learning_model.parameters(), 0.1)
        optimizer.step()
        optimizer.zero_grad()
        mean_return = rewards[mask].mean()
        lp = teacher.update_env(mean_return, env_id)

        if i % 10 == 0:
            env_name = f"{CURRICULUM_NAMING[env_id]}"
            for t_id in teacher.env_dict.keys():
                name_env = CURRICULUM_NAMING[t_id]
                tb.add_scalar(f"Teacher_LP/{name_env}", teacher.lps[t_id].item(), global_step=num_episodes)
                smooth_performance = teacher.env_dict[t_id]["fast_average"]
                tb.add_scalar(f"Global performance/{name_env}", smooth_performance  , global_step=num_episodes)

            steps = individual_logs[env_id]
            smooth_performance = teacher.env_dict[env_id]["fast_average"]
            individual_logs[env_id] += 1

            tb.add_scalar("Losses/Policy", np.mean(policy_losses), global_step=num_episodes)
            tb.add_scalar("Losses/Entropy", np.mean(entropy_losses), global_step=num_episodes)
            tb.add_scalar("Losses/Extrinsic", np.mean(extrinsic_losses), global_step=num_episodes)
        if i % UPDATE_PRINT == 0:
            print("Synced")
            duration = time.time() - now
            now = time.time()
            print("duration:", duration)
        if i % SAVE_EVERY == 0:
            torch.save(
                {
                    "model_state_dict": learning_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "iteration": i,
                },
                os.path.join(CHECKPOINTS_DIR, f"model{(i // SAVE_EVERY) + LOAD_VERSION}.pt"),
            )
            print("saved")

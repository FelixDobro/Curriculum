import os
import sys
from pathlib import Path
from typing import List
sys.path.append(str(Path(__file__).resolve().parent.parent))
import math
import random
import numpy as np
import torch.nn.functional as F
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import time
from config import *
from models.convs.cell import PPONetCell
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
        self.beta = 0.01
        self.counts = {id: 0 for id in CURRICULUM.keys()}

    def update_env(self, earned_return, env_id):
        env = self.env_dict[env_id]
        env["fast_average"] = earned_return * self.alpha + (1 - self.alpha) * env["fast_average"]
        env["slow_average"] = earned_return * self.beta + (1 - self.beta) * env["slow_average"]
        self.lps[env_id] = math.fabs(env["fast_average"] - env["slow_average"])

    def get_env(self):
        chosen_env = None
        if random.random() < 0.1:
            chosen_env = random.choice(list(self.env_dict.keys()))
        else:
            safe_lps = self.lps + 1e-6
        
            chosen_env = Categorical(probs=safe_lps).sample().item()

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
     
        self.current_taks = task_func
        self.env = task_func()
    def reset(self, **kwargs):
        
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)

def make_meta_env():
    return MetaEnv()


if __name__ == "__main__":

    tb = SummaryWriter(LOG_DIR)

 
    EPS = 0.2
    POLICY_COEFF = 1
    VALUE_COEFF = 1
    BETA = 0.001
    learning_model = PPONetCell(EMBEDDING_DIM, HIDDEN_DIMS, NUM_ACTIONS).to(DEVICE)
    learning_model.eval()
    optimizer = torch.optim.Adam(learning_model.parameters(), lr=LEARNING_RATE)
    teacher = Teacher()

    vector_env = SyncVectorEnv(
        [make_meta_env for _ in range(NUM_ENVS)]
    )

    i = 0
    now = time.time()
    var = 1
    mean = 0
    num_episodes_played = 0
    num_samples = 0
    successes = 0

    select_count = {key: 0 for key in CURRICULUM.keys()}
    while True:
        i += 1

        states, actions, rewards, dones, old_log_probs, values = [], [], [], [], [], []
        learning_model.eval()
        env_id = teacher.get_env()
        vector_env.call("set_task", CURRICULUM[env_id])
        max_steps = CURRICULUM_STEPS[CURRICULUM_NAMING[env_id]]
     
        
        with torch.no_grad():
            learning_model.eval()
        
            obs, _ = vector_env.reset() 
            h = learning_model.init_hidden(NUM_ENVS, device=DEVICE)
            
            for t in range(max_steps):
                ob = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
        
                v_ext, p, h = learning_model(ob, h)
                dist = Categorical(logits=p)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                action_num = action.cpu().numpy()
                old_log_probs.append(log_prob)
                
                states.append(ob)
                obs, reward, terminated, truncated, _ = vector_env.step(action_num)
                num_episodes_played += np.sum(terminated | truncated)


                h = h * (1.0 - torch.tensor(terminated, dtype=torch.float32, device=DEVICE)[:,None])
                values.append(v_ext.squeeze(1))
                actions.append(action)
                rewards.append(reward)
                dones.append(terminated)
           
    
            states = torch.stack(states)  # (T, num_envs, C,H,W)
            actions = torch.stack(actions)  # (T, num_envs)
            values = torch.stack(values)
            rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=DEVICE)  # (T, num_envs)
            dones = torch.tensor(np.array(dones), dtype=torch.long, device=DEVICE)  # (T, num_envs)
            states = states.permute(1, 0, 2, 3, 4)  # (batch, time, C, H, W)
            actions = actions.permute(1, 0)  # (batch, time)
            rewards = rewards.permute(1, 0)  # (batch, time)
            dones = dones.permute(1, 0)  # (batch, time)
            old_log_probs = torch.stack(old_log_probs).squeeze(-1).permute(1, 0)
            values = values.squeeze(-1).permute(1,0)


            
            last_obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
            next_v_ext, _, _, = learning_model(last_obs_tensor, h)
            next_v_ext = next_v_ext.squeeze(-1).squeeze(-1)

            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
          
          

            for t in reversed(range(max_steps)):
                if t == max_steps - 1:
                    nextnonterminal = 1.0 - dones[:, -1] 
                    nextvalues = next_v_ext
                else:
                    nextnonterminal = 1.0 - dones[:, t]
                    nextvalues = values[:, t + 1]
                
        
                delta = rewards[:, t] + GAMMA * nextvalues * nextnonterminal - values[:, t]
                advantages[:, t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam

            extrinsic_returns = advantages + values
            frozen_advantages = advantages.detach()

        ##training
        for _ in range(PPO_EPOCHS):
            policy_losses = []
            entropy_losses = []
            extrinsic_losses = []
            learning_model.train()
            learning_hidden = learning_model.init_hidden(NUM_ENVS, DEVICE)

            
            for c in range(0, max_steps, CHUNK_SIZE):
                 
                state_chunk = states[:, c:c + CHUNK_SIZE, :]
                rewards_chunk = rewards[:, c:c + CHUNK_SIZE]
                advantage = frozen_advantages[:, c:c + CHUNK_SIZE]
                old_log_probs_chunk = old_log_probs[:, c:c + CHUNK_SIZE]
                dones_chunk = dones[:, c:c + CHUNK_SIZE]
                actions_chunk = actions[:, c:c + CHUNK_SIZE]
                extrinsic_chunk = extrinsic_returns[:, c:c + CHUNK_SIZE]

  
                policy_logits_list = []
                extrinsic_val_list = []

                policy_logits_list, extrinsic_val_list = forward_pass(
                    state_chunk=state_chunk,
                    learning_model=learning_model,
                    dones_chunk=dones_chunk,
                    learning_hidden=learning_hidden
                )
                    
                policy_logits = torch.stack(policy_logits_list, dim=1)
                extrinsic_val = torch.stack(extrinsic_val_list, dim=1)
                value_loss_ext = F.mse_loss(extrinsic_val, extrinsic_chunk)

            
                distributions = Categorical(logits=policy_logits)
                log_probs = distributions.log_prob(actions_chunk)
                entropies = distributions.entropy()

                log_probs_flat = log_probs
                old_log_probs_flat = old_log_probs_chunk
                advantage_flat = advantage
                entropy_flat = entropies

                ratio = torch.exp(log_probs_flat - old_log_probs_flat)

                surr1 = ratio * advantage_flat
                surr2 = torch.clamp(ratio, 1.0 - EPS, 1.0 + EPS) * advantage_flat

                entropy = entropy_flat.mean()
                policy_loss = -torch.min(surr1, surr2).mean() - BETA * entropy

                combined_loss = POLICY_COEFF * policy_loss + VALUE_COEFF * value_loss_ext
                
                policy_losses.append(policy_loss.item())

                entropy_losses.append(entropy.item())

                extrinsic_losses.append(value_loss_ext.item())
                next_hidden = learning_hidden.detach()
   
                num_chunks = max(1,max_steps // CHUNK_SIZE)
                (combined_loss / num_chunks).backward()
                learning_hidden = next_hidden

            torch.nn.utils.clip_grad_norm_(learning_model.parameters(), 0.1)
            optimizer.step()
            optimizer.zero_grad()

        #mean_return = rewards[mask].mean()
        
        num_samples += max_steps
        successes += dones.sum().item()
     
        success_ratio = successes / num_episodes_played
        lp = teacher.update_env(success_ratio, env_id)
        successes = 0
        num_episodes_played = 0
        
        env_name = f"{CURRICULUM_NAMING[env_id]}"
        goal_reward = CURRICULUM_REWARDS["goal"]
        select_count[env_id] += 1

     
        for t_id in teacher.env_dict.keys():
            name_env = CURRICULUM_NAMING[t_id]
            tb.add_scalar(f"Teacher_LP/{name_env}", teacher.lps[t_id].item(), global_step=num_samples)
            p_fast = teacher.env_dict[t_id]["fast_average"]
            tb.add_scalar(f"p_fast/{name_env}", p_fast  , global_step=num_samples)
            p_slow = teacher.env_dict[t_id]["slow_average"]
            tb.add_scalar(f"p_slow/{name_env}", p_slow  , global_step=num_samples)
        tb.add_scalar(f"Success ratio %/{env_name}",success_ratio, global_step=num_samples)
        tb.add_scalar(f"# Selected/{env_name}", select_count[env_id], global_step=num_samples)


        tb.add_scalar("Losses/Policy", np.mean(policy_losses), global_step=num_samples)
        tb.add_scalar("Losses/Entropy", np.mean(entropy_losses), global_step=num_samples)
        tb.add_scalar("Losses/Extrinsic", np.mean(extrinsic_losses), global_step=num_samples)
    
            
        if i % SAVE_EVERY == 0:
            torch.save(
                {
                    "model_state_dict": learning_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "iteration": i,
                },
                os.path.join(CHECKPOINTS_DIR, f"model{(i // SAVE_EVERY)}.pt"),
            )
            print("saved")

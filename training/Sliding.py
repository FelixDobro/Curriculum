import os
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parent.parent))
import math
import random
import numpy as np
import torch.nn.functional as F
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from config import *

import gymnasium as gym
from models.convs.plainPPO import PPONet
from utils import *
from models.convs.cell import PPONetCell



class Teacher():
    def __init__(self):
        self.size = MIN_SIZE

    def update(self, mean):
      
        if mean >= UPDTAE_UP:
            self.size = min(self.size + 1, MAX_SIZE)
        if mean <= UPDATE_DOWN:
            self.size = max(self.size-1, MIN_SIZE)

    def sample(self):
        return random.randint(self.size, self.size + WINDOW_SIZE - 1)
        
        
class MetaEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.env = Maze(size=MIN_SIZE, n_obstacles=(MIN_SIZE**2)//3, max_steps=SLIDING_STEPS, render_mode="rgb_array")
        self.env = ConvWrapper(self.env)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def set_env(self, size):
        self.env = Maze(size=size, n_obstacles=(size**2)//3, max_steps=SLIDING_STEPS, render_mode="rgb_array")
        self.env = ConvWrapper(self.env)
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    def step(self, action):
        return self.env.step(action)


def make_meta_env():
    return MetaEnv()


if __name__ == "__main__":

    for run, seed in enumerate(SEEDS[:NUM_ROUNDS]):
        print(f"Run: {run} \n")

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        Path.mkdir(CHECKPOINTS_DIR / Path(f"_run_{run}"), exist_ok=True, parents=True)

        tb = SummaryWriter(LOG_DIR / Path(f"_run_{run}"))
 
        EPS = 0.2
        POLICY_COEFF = 1
        VALUE_COEFF = 1
        BETA = 0.001
        learning_model = PPONetCell(EMBEDDING_DIM, HIDDEN_DIMS, NUM_ACTIONS).to(DEVICE)
        learning_model.eval()
        optimizer = torch.optim.Adam(learning_model.parameters(), lr=LEARNING_RATE)
    
        vector_env = SyncVectorEnv(
            [make_meta_env for _ in range(NUM_ENVS)]
        )

        teacher = Teacher()

        i = 1

    
        num_episodes = 0
        num_samples = 0

        return_summary = []
        num_episodes_played = 0
        successes = 0
        with tqdm(total=NUM_SAMPLES_PER_ROUND) as pbar:
            while num_samples < NUM_SAMPLES_PER_ROUND:
                i += 1
            
                num_episodes += NUM_ENVS
                states, actions, rewards, dones, old_log_probs, values = [], [], [], [], [], []
            
                with torch.no_grad():
                    learning_model.eval()
                
                    obs, _ = vector_env.reset() 
                    h = learning_model.init_hidden(NUM_ENVS, device=DEVICE)
                    
                    for t in range(SLIDING_STEPS):
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
                
                

                    for t in reversed(range(SLIDING_STEPS)):
                        if t == SLIDING_STEPS - 1:
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

                    
                    for c in range(0, SLIDING_STEPS, CHUNK_SIZE):
                        
                        state_chunk = states[:, c:c + CHUNK_SIZE, :]
                        rewards_chunk = rewards[:, c:c + CHUNK_SIZE]
                        advantage = frozen_advantages[:, c:c + CHUNK_SIZE]
                        old_log_probs_chunk = old_log_probs[:, c:c + CHUNK_SIZE]
                        dones_chunk = dones[:, c:c + CHUNK_SIZE]
                        actions_chunk = actions[:, c:c + CHUNK_SIZE]
                        extrinsic_chunk = extrinsic_returns[:, c:c + CHUNK_SIZE]

        
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
                        num_chunks = max(1,SLIDING_STEPS // CHUNK_SIZE)
                        (combined_loss / num_chunks).backward()

                        learning_hidden = next_hidden

                    torch.nn.utils.clip_grad_norm_(learning_model.parameters(), 0.1)
                    optimizer.step()
                    optimizer.zero_grad()


                increment = SLIDING_STEPS * NUM_ENVS
                num_samples += increment
                pbar.update(increment)
                successes += dones.sum().item()

                with torch.no_grad():
                    mean_return = successes / num_episodes_played 
                    tb.add_scalar("Success Rate (%)", mean_return, global_step=num_samples)
                    if num_episodes_played > PERFORMANCE_SAMPLE_SIZE:
                        teacher.update(successes / num_episodes_played)
                        vector_env.call("set_env", teacher.sample())

                        successes = 0
                        num_episodes_played = 0
                

                        # if teacher.size == MAX_SIZE and mean_return > 0.98:
                        #     BETA = max(MIN_BETA, BETA * 0.9) 

                        #     for param_group in optimizer.param_groups:
                        #         LEARNING_RATE = max(MIN_LR, param_group['lr'] * 0.95)
                        #         param_group['lr'] = LEARNING_RATE
                    
                
            
            
                
                tb.add_scalar("Size of env", teacher.size, global_step=num_samples)
                tb.add_scalar("Beta", BETA, global_step=num_samples)
                tb.add_scalar("Learning Rate", LEARNING_RATE, global_step=num_samples)
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
                    

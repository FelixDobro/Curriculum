import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from torch.distributions import Categorical
from models.convs.plainPPO import PPONet
from utils import *
from config import *
import gymnasium as gym



@torch.no_grad()
def get_success_ratio(model, env, num_episodes):
    successes = 0
    for iterations in range(num_episodes):
        print(iterations)
        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        h = model.init_hidden(1)
        
        terminated = False
        truncated = False

        while not (terminated or truncated):
            _, logits,h,_ = model(obs, h)

            action = Categorical(logits=logits).sample().item()
            obs, reward, terminated, truncated, _ = env.step(action)
            
            obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        successes += 1 if terminated else 0
    
    return successes / num_episodes * 100
        

if __name__ == "__main__":

    model_props = torch.load(f"{MODEL_DIR}")
    state_dict = model_props["model_state_dict"]

    model = PPONet(EMBEDDING_DIM, HIDDEN_DIMS, NUM_ACTIONS)
    model.load_state_dict(state_dict)
    model.eval()


    curriculum_names, curriculum_functions = map_envs(names=CURRICULUM)

    result = {}
    for env_id, env_function in curriculum_functions.items():
        env_name = curriculum_names[env_id]
        env = env_function()
        print(f"Evaluating: {env_name}")
        ratio = get_success_ratio(model,env, NUM_EPISODES_EVAL)
        print(f"{env_name}: {ratio}")
        result[env_name] = ratio
        env.close()
    print(result)
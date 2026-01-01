import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from torch.distributions import Categorical
from models.convs.plainPPO import PPONet
from utils import *
from config import *
import gymnasium as gym



@torch.no_grad()
def run_gru_episode(model, env):
    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    h = model.init_hidden(1)
    done = False

    while not done:
        _, logits,h,_ = model(obs, h)
        action = Categorical(logits=logits).sample().item()
        obs, _, terminated, truncated, _ = env.step(action)
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        done = terminated or truncated



if __name__ == "__main__":

    model_props = torch.load(f"{MODEL_DIR}")
    state_dict = model_props["model_state_dict"]

    model = PPONet(EMBEDDING_DIM, HIDDEN_DIMS, NUM_ACTIONS)
    model.load_state_dict(state_dict)
    model.eval()


    curriculum_names, curriculum_functions = map_envs(names=CURRICULUM)
    print(curriculum_names)
    print(curriculum_functions)

    for env_id, env_function in curriculum_functions.items():
        env_name = curriculum_names[env_id]
        video_folder = f"videos/{env_name}"
        env = env_function()
        env = gym.wrappers.RecordVideo(
            env,
            video_folder,
            episode_trigger=lambda episode_id: True,
            fps=FPS
        )
        print(f"Recording: {env_name} dir {video_folder}...")

        for i in range(NUM_VIDEOS):
            run_gru_episode(model, env)

        env.close()
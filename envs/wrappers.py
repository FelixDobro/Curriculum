import numpy as np
from gymnasium import ObservationWrapper, spaces


class ConvWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        old_space = env.observation_space["image"]  # (H,W,C)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(old_space.shape[2], old_space.shape[0], old_space.shape[1]),  # (C,H,W)
            dtype=np.uint8
        )

    def observation(self, obs):
        return np.transpose(obs["image"], (2, 0, 1)).copy()


class RelativePositionWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Original Image-Shape
        img_shape = self.env.observation_space["image"].shape
        img_size = int(np.prod(img_shape))

        # Neue Observation Space (Bild + dx + dy)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(img_size + 1,),  # +2 für dx, dy
            dtype=np.float32,
        )

import numpy as np
import torch
from torch import nn
import gym
import os
import random

import dmc2gym
import safety_gym
from safety_gym.envs.engine import Engine
from gym.envs.registration import register


def make_env(cfg):
    """Helper function to create dm_control environment"""
    if cfg.env == "ball_in_cup_catch":
        domain_name = "ball_in_cup"
        task_name = "catch"
    else:
        domain_name = cfg.env.split("_")[0]
        task_name = "_".join(cfg.env.split("_")[1:])

    env = dmc2gym.make(
        domain_name=domain_name,
        task_name=task_name,
        seed=cfg.seed,
        visualize_reward=True,
    )
    env.seed(cfg.seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env


def make_safety_env(cfg):
    """Helper function to create safety environment"""
    if "custom" in cfg.env:
        return make_custom_env(cfg)
    env_split = cfg.env.split("_")
    env_name = f"Safexp-{env_split[0].capitalize()}{env_split[1].capitalize()}{env_split[-1]}-v0"

    env = gym.make(env_name, render_mode="rgb_array")
    env.seed(cfg.seed)

    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env


def make_custom_env(cfg):
    """Custom environments used in the paper (taken from the official implementation)"""
    if "static" in cfg.env:
        config1 = {
            "placements_extents": [-1.5, -1.5, 1.5, 1.5],
            "robot_base": "xmls/point.xml",
            "task": "goal",
            "goal_size": 0.3,
            "goal_keepout": 0.305,
            "goal_locations": [(0.79413827, -0.28269969)],
            "observe_goal_lidar": True,
            'observe_box_lidar': True,
            "observe_hazards": True,
            "constrain_hazards": True,
            'observe_vases': True,
            "lidar_max_dist": 3,
            "lidar_num_bins": 16,
            "hazards_num": 8,
            "hazards_size": 0.2,
            "hazards_keepout": 0.18,
            "hazards_locations": [(-0.01160915, -0.13139404), (-1.09901719, -0.99530323), (0.03081399, -1.00922507), (0.6083558 ,  0.7765939), \
                (1.31398354,  0.4422078), (-0.39260381, -0.42062426), ( 0.95638437, -1.15024822), (-0.11601288,  0.59559585)]
        }
        register(
            id="StaticEnv-v0",
            entry_point="safety_gym.envs.mujoco:Engine",
            max_episode_steps=1000,
            kwargs={"config": config1},
        )
        env = gym.make("StaticEnv-v0", render_mode="rgb_array")
    else:
        config2 = {
            "placements_extents": [-1.5, -1.5, 1.5, 1.5],
            "robot_base": "xmls/point.xml",
            "task": "goal",
            "goal_size": 0.3,
            "goal_keepout": 0.305,
            "observe_goal_lidar": True,
            "observe_hazards": True,
            "constrain_hazards": True,
            "lidar_max_dist": 3,
            "lidar_num_bins": 16,
            "hazards_num": 3,
            "hazards_size": 0.3,
            "hazards_keepout": 0.305,
        }
        register(
            id="DynamicEnv-v0",
            entry_point="safety_gym.envs.mujoco:Engine",
            max_episode_steps=1000,
            kwargs={"config": config2},
        )
        env = gym.make("DynamicEnv-v0", render_mode="rgb_array")

    return env


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


class train_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(True)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_dir(*path_parts):
    dir_path = os.path.join(*path_parts)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod)
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()

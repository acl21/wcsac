import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import utils


class DoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth):
        super().__init__()

        self.Q1 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
        self.Q2 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_critic/{k}_hist', v, step)

        assert len(self.Q1) == len(self.Q2)
        for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
            assert type(m1) == type(m2)
            if type(m1) is nn.Linear:
                logger.log_param(f'train_critic/q1_fc{i}', m1, step)
                logger.log_param(f'train_critic/q2_fc{i}', m2, step)


class SafetyCritic(nn.Module):
    """Safety Critic Network for estimating Long Term Costs (Mean and Variance)"""
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth):
        super().__init__()

        self.QC = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
        self.VC = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        qc = self.QC(obs_action)
        vc = self.VC(obs_action)

        self.outputs['qc'] = qc
        self.outputs['vc'] = vc

        return qc, vc

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_safety_critic/{k}_hist', v, step)

        assert len(self.QC) == len(self.VC)
        for i, (m1, m2) in enumerate(zip(self.QC, self.VC)):
            assert type(m1) == type(m2)
            if type(m1) is nn.Linear:
                logger.log_param(f'train_safety_critic/qc_fc{i}', m1, step)
                logger.log_param(f'train_safety_critic/vc_fc{i}', m2, step)
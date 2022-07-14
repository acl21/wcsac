import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist
import math

from agent import Agent
import os
import utils
from itertools import chain

import hydra


class WCSACAgent(Agent):
    """WCSAC algorithm."""
    def __init__(self, obs_dim, action_dim, action_range, device, critic_cfg, safety_critic_cfg,
                 actor_cfg, discount, init_temperature, alpha_lr, alpha_betas,
                 actor_lr, actor_betas, actor_update_frequency, critic_lr,
                 critic_betas, critic_tau, critic_target_update_frequency,
                 batch_size, learnable_temperature, cost_limit, max_episode_len,
                 risk_level, damp_scale, lr_scale):
        super().__init__()

        self.action_range = action_range
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.learnable_temperature = learnable_temperature  # boolean

        # Safety related params
        self.max_episode_len = max_episode_len
        self.cost_limit = cost_limit  # d in Eq. 10
        self.risk_level = risk_level  # alpha in Eq. 9 / risk averse = 0, risk neutral = 1
        normal = tdist.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        self.pdf_cdf = normal.log_prob(normal.icdf(torch.tensor(self.risk_level))).exp() / self.risk_level  # precompute CVaR Value for st. normal distribution
        self.pdf_cdf = self.pdf_cdf.cuda()
        self.damp_scale = damp_scale
        self.cost_lr_scale = lr_scale

        # Reward critic
        self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Safety critic
        self.safety_critic = hydra.utils.instantiate(safety_critic_cfg).to(self.device)
        self.safety_critic_target = hydra.utils.instantiate(safety_critic_cfg).to(self.device)
        self.safety_critic_target.load_state_dict(self.safety_critic.state_dict())

        # Actor
        self.actor = hydra.utils.instantiate(actor_cfg).to(self.device)

        # Entropy temperature (beta in the paper)
        self.log_alpha = torch.tensor(np.log(np.clip(init_temperature, 1e-8, 1e8))).to(self.device)
        self.log_alpha.requires_grad = True

        # Cost temperature (kappa in the paper)
        self.log_beta = torch.tensor(np.log(np.clip(init_temperature, 1e-8, 1e8))).to(self.device)
        self.log_beta.requires_grad = True

        # Set target entropy to -|A|
        self.target_entropy = -action_dim

        # Set target cost
        self.target_cost = self.cost_limit * (1 - self.discount**self.max_episode_len) / (1 - self.discount) / self.max_episode_len

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr,
                                                betas=actor_betas)

        self.all_critics_optimizer = torch.optim.Adam(chain(self.critic.parameters(), self.safety_critic.parameters()),
                                                 lr=critic_lr,
                                                 betas=critic_betas)

        # Alpha (entropy weight) optimizer 
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr,
                                                    betas=alpha_betas)

        # Beta (safety weight) optimizer
        self.log_beta_optimizer = torch.optim.Adam([self.log_beta],
                                                    lr=alpha_lr*self.cost_lr_scale,
                                                    betas=alpha_betas)

        self.train()
        self.critic_target.train()
        self.safety_critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        self.safety_critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def beta(self):
        return self.log_beta.exp()

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def update_critic(self, obs, action, reward, cost, next_obs, not_done, logger, step):
        # Get next action from current pi_theta(*|next_obs)
        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)

        # Q1, Q2 targets
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        # Current QC and VC estimates
        current_QC, current_VC = self.safety_critic(obs, action)
        current_VC = torch.clamp(current_VC, min=1e-8, max=1e8)

        # QC, VC targets
        # use next_action as an approximation
        next_QC, next_VC = self.safety_critic_target(next_obs, next_action)
        next_VC = torch.clamp(next_VC, min=1e-8, max=1e8)

        target_QC = cost + (not_done * self.discount * next_QC)
        target_VC = cost**2 - current_QC**2 + 2 * self.discount * cost * next_QC +\
            self.discount**2 * next_VC + self.discount**2 * next_QC**2  # Eq. 8 in the paper 
        target_QC = target_QC.detach()
        target_VC = torch.clamp(target_VC.detach(), min=1e-8, max=1e8)

        # Critic Loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        logger.log('train/critic_loss', critic_loss, step)

        # Safety Critic Loss
        safety_critic_loss = F.mse_loss(current_QC, target_QC) +\
            torch.mean(current_VC + target_VC - 2 * torch.sqrt(current_VC * target_VC))
        logger.log('train/safety_critic_loss', safety_critic_loss, step)

        # Jointly optimize Reward and Safety Critics
        total_loss = critic_loss + safety_critic_loss
        self.all_critics_optimizer.zero_grad()
        total_loss.backward()
        self.all_critics_optimizer.step()

        # self.critic.log(logger, step)
        # self.safety_critic.log(logger, step)
        
    def update_actor_and_alpha_and_beta(self, obs, action_taken, logger, step):
        # Get updated action from current pi_theta(*|obs)
        dist = self.actor(obs)
        action = dist.rsample()  # uses reparametrization trick
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)

        # Reward Critic
        actor_Q1, actor_Q2 = self.critic(obs, action)
        actor_Q = torch.min(actor_Q1, actor_Q2)

        # Safety Critic with actor actions
        actor_QC, actor_VC = self.safety_critic(obs, action)
        actor_VC = torch.clamp(actor_VC, min=1e-8, max=1e8)

        # Safety Critic with actual actions
        current_QC, current_VC = self.safety_critic(obs, action_taken)
        current_VC = torch.clamp(current_VC, min=1e-8, max=1e8)

        # CVaR + Damp impact of safety constraint in actor update / not used if damp_scale = 0
        cvar = current_QC + self.pdf_cdf.cuda() * torch.sqrt(current_VC)  # Eq. 9 in the paper
        damp = self.damp_scale * torch.mean(self.target_cost - cvar)

        # Actor Loss
        actor_loss = torch.mean(self.alpha.detach() * log_prob - (actor_Q + (self.beta.detach() - damp) * (actor_QC + self.pdf_cdf.cuda() * torch.sqrt(actor_VC))).detach())

        logger.log('train/actor_loss', actor_loss, step)
        logger.log('train/actor_entropy', -log_prob.mean(), step)
        logger.log('train/actor_cost', torch.mean(actor_QC + self.pdf_cdf.cuda() * torch.sqrt(actor_VC)), step)

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # self.actor.log(logger, step)

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = torch.mean(self.alpha * (-log_prob - self.target_entropy).detach())
            logger.log('train/alpha_loss', alpha_loss, step)
            logger.log('train/alpha_value', self.alpha, step)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()
            
            self.log_beta_optimizer.zero_grad()
            beta_loss = torch.mean(self.beta * (self.target_cost - cvar).detach())
            logger.log('train/beta_loss', beta_loss, step)
            logger.log('train/beta_value', self.beta, step)
            beta_loss.backward()
            self.log_beta_optimizer.step()

    def update(self, replay_buffer, logger, step):
        obs, action, reward, cost, next_obs, not_done, not_done_no_max = replay_buffer.sample(self.batch_size)

        logger.log('train/batch_reward', reward.mean(), step)
        logger.log('train/batch_cost', cost.mean(), step)
        
        self.update_critic(obs, action, reward, cost, next_obs, not_done_no_max,
                           logger, step)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha_and_beta(obs, action, logger, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)
            utils.soft_update_params(self.safety_critic, self.safety_critic_target, self.critic_tau)

    def save(self, path):
        torch.save(self.actor.trunk.state_dict(), os.path.join(path, 'actor.pt'))
        torch.save(self.critic.Q1.state_dict(), os.path.join(path, 'critic_q1.pt'))
        torch.save(self.critic.Q2.state_dict(), os.path.join(path, 'critic_q2.pt'))
        torch.save(self.safety_critic.QC.state_dict(), os.path.join(path, 'safety_critic_qc.pt'))
        torch.save(self.safety_critic.VC.state_dict(), os.path.join(path, 'safety_critic_vc.pt'))


    def load(self, path):
        self.actor.trunk.load_state_dict(torch.load(os.path.join(path, 'actor.pt')))
        self.critic.Q1.load_state_dict(torch.load(os.path.join(path, 'critic_q1.pt')))
        self.critic.Q2.load_state_dict(torch.load(os.path.join(path, 'critic_q2.pt')))
        self.safety_critic.QC.load_state_dict(torch.load(os.path.join(path, 'safety_critic_qc.pt')))
        self.safety_critic.VC.load_state_dict(torch.load(os.path.join(path, 'safety_critic_vc.pt')))

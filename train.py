#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
import time
import pickle as pkl

from video import VideoRecorder
from logger import Logger
from replay_buffer import ReplayBuffer
import utils

import hydra


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg

        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency,
                             agent=cfg.agent.name)

        assert 1>= cfg.risk_level >= 0, f"risk_level must be between 0 and 1 (inclusive), got: {cfg.risk_level}"
        assert cfg.seed != -1, f"seed must be provided, got default seed: {cfg.seed}"
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = utils.make_safety_env(cfg)

        cfg.agent.params.obs_dim = int(self.env.observation_space.shape[0])
        cfg.agent.params.action_dim = int(self.env.action_space.shape[0])
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)

        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          int(cfg.replay_buffer_capacity),
                                          self.device)

        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None)
        self.step = 0
        if cfg.restart_path != 'dummy':
            self.agent.load(cfg.restart_path)

    def evaluate(self):
        average_episode_reward = 0
        average_episode_cost = 0
        average_episode_goals_met = 0
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            self.agent.reset()
            self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            episode_cost = 0
            episode_goals_met = 0
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, info = self.env.step(action)
                self.video_recorder.record(self.env)
                episode_reward += reward
                episode_cost += info.get('cost', 0)
                episode_goals_met += 1 if info.get('goal_met', False) else 0

            average_episode_reward += episode_reward
            average_episode_cost += episode_cost
            average_episode_goals_met += episode_goals_met
            self.video_recorder.save(f'{self.step}.mp4')
        average_episode_reward /= self.cfg.num_eval_episodes
        average_episode_cost /= self.cfg.num_eval_episodes
        average_episode_goals_met /= self.cfg.num_eval_episodes
        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.log('eval/episode_cost', average_episode_cost,
                        self.step)
        self.logger.log('eval/episode_goals_met', average_episode_goals_met,
                        self.step)
        self.logger.dump(self.step)
        self.agent.save(self.work_dir)

    def run(self):
        episode, episode_reward, episode_cost, done = 0, 0, 0, True
        start_time = time.time()
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration',
                                    time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()

                self.logger.log('train/episode_reward', episode_reward,
                                self.step)
                self.logger.log('train/episode_cost', episode_cost,
                                self.step)

                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                episode_cost = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                self.agent.update(self.replay_buffer, self.logger, self.step)

            next_obs, reward, done, info = self.env.step(action)
            cost = info.get('cost', 0)
            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env.num_steps else done
            episode_reward += reward
            episode_cost += cost

            self.replay_buffer.add(obs, action, reward, cost, next_obs, done,
                                   done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1
        self.agent.save(os.path.join(self.work_dir, 'agent_final.pt'))


@hydra.main(config_path='config/train.yaml', strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()

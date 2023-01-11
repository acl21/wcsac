#!/usr/bin/env python3
import torch
import os
import time

from video import VideoRecorder
from logger import Logger
from replay_buffer import ReplayBuffer
import utils

import hydra


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg

        self.logger = Logger(
            self.work_dir,
            save_tb=cfg.log_save_tb,
            log_frequency=cfg.log_frequency,
            agent=cfg.agent.name,
        )

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = utils.make_env(cfg)
        # self.env = utils.make_safety_env(cfg)
        # self.env = utils.make_custom_env(cfg)

        if "run" in cfg.env:
            self.env_max_steps = self.env._max_episode_steps
        else:
            self.env_max_steps = self.env.num_steps

        cfg.agent.agent.obs_dim = int(self.env.observation_space.shape[0])
        cfg.agent.agent.action_dim = int(self.env.action_space.shape[0])
        cfg.agent.agent.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max()),
        ]
        self.agent = hydra.utils.instantiate(cfg.agent.agent, _recursive_=False)

        self.replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            int(cfg.replay_buffer_capacity),
            self.device,
        )

        self.video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None)
        self.step = 0

    def evaluate(self):
        average_episode_reward = 0
        for episode in range(self.cfg.num_eval_episodes):
            obs, _ = self.env.reset()
            self.agent.reset()
            self.video_recorder.init(enabled=(episode == 0))
            done, truncated = False, False
            episode_reward = 0
            while not done or truncated:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, truncated, _ = self.env.step(action)
                self.video_recorder.record(self.env)
                episode_reward += reward

            average_episode_reward += episode_reward
            self.video_recorder.save(f"{self.step}.mp4")
        average_episode_reward /= self.cfg.num_eval_episodes
        self.logger.log("eval/episode_reward", average_episode_reward, self.step)
        self.logger.dump(self.step)

    def run(self):
        episode, episode_reward, done, truncated = 0, 0, True, True
        start_time = time.time()
        while self.step < self.cfg.num_train_steps:
            if done or truncated:
                if self.step > 0:
                    self.logger.log("train/duration", time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log("eval/episode", episode, self.step)
                    self.evaluate()

                self.logger.log("train/episode_reward", episode_reward, self.step)

                obs, _ = self.env.reset()
                self.agent.reset()
                done, truncated = False, False
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log("train/episode", episode, self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    print("INSIDE EVAL")
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                self.agent.update(self.replay_buffer, self.logger, self.step)

            next_obs, reward, done, truncated, _ = self.env.step(action)

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env_max_steps else done
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, 1, next_obs, done, done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1


@hydra.main(config_path='config', config_name='train', version_base=None)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
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

        assert 1 >= cfg.risk_level >= 0, f"risk_level must be between 0 and 1 (inclusive), got: {cfg.risk_level}"
        assert cfg.seed != -1, f"seed must be provided, got default seed: {cfg.seed}"
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = utils.make_safety_env(cfg)

        cfg.agent.params.obs_dim = int(self.env.observation_space.shape[0])
        cfg.agent.params.action_dim = int(self.env.action_space.shape[0])
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max()),
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)

        self.replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            int(cfg.replay_buffer_capacity),
            self.device,
        )

        self.video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None)
        self.step = 0
        if cfg.restart_path != "dummy":
            self.agent.load(cfg.restart_path)

        utils.make_dir(self.work_dir, "model_weights")

    def evaluate(self):
        mean_reward = 0
        mean_cost = 0
        mean_goals_met = 0
        mean_hazard_touches = 0
        cost_limit_violations = 0
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            self.agent.reset()
            self.video_recorder.init(enabled=(episode == 0))
            done = False
            ep_reward = 0
            ep_cost = 0
            ep_goals_met = 0
            ep_hazard_touches = 0

            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, info = self.env.step(action)
                self.video_recorder.record(self.env)
                ep_reward += reward
                ep_cost += info.get("cost", 0)
                ep_goals_met += 1 if info.get("goal_met", False) else 0
                ep_hazard_touches += 1 if (info.get("cost_hazards", 0) > 0) else 0

            mean_reward += ep_reward
            mean_cost += ep_cost
            mean_goals_met += ep_goals_met
            mean_hazard_touches += ep_hazard_touches
            cost_limit_violations += 1 if (ep_cost > self.cfg.agent.params.cost_limit) else 0
            self.video_recorder.save(f"{self.step}.mp4")

        mean_reward /= self.cfg.num_eval_episodes
        mean_cost /= self.cfg.num_eval_episodes
        mean_goals_met /= self.cfg.num_eval_episodes
        mean_hazard_touches /= self.cfg.num_eval_episodes
        self.logger.log("eval/mean_reward", mean_reward, self.step)
        self.logger.log("eval/mean_cost", mean_cost, self.step)
        self.logger.log("eval/mean_goals_met", mean_goals_met, self.step)
        self.logger.log("eval/hazard_touches", mean_hazard_touches, self.step)
        self.logger.log("eval/cost_limit_violations", cost_limit_violations, self.step)
        self.logger.dump(self.step)
        self.agent.save(self.work_dir)
        self.agent.save_actor(os.path.join(self.work_dir, "model_weights"), self.step)

    def run(self):
        episode, ep_reward, ep_cost, total_cost, done = 0, 0, 0, 0, True
        start_time = time.time()
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log("train/duration", time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if (self.step > 0 and self.step % self.cfg.eval_frequency == 0):
                    self.logger.log("eval/episode", episode, self.step)
                    self.evaluate()

                self.logger.log("train/episode_reward", ep_reward, self.step)
                self.logger.log("train/episode_cost", ep_cost, self.step)
                if self.step > 0:
                    self.logger.log("train/cost_rate", total_cost / self.step, self.step)

                obs = self.env.reset()
                self.agent.reset()
                done = False
                ep_reward = 0
                ep_cost = 0
                ep_step = 0
                episode += 1

                self.logger.log("train/episode", episode, self.step)

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
            cost = info.get("cost", 0)
            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if ep_step + 1 == self.env.num_steps else done
            ep_reward += reward
            ep_cost += cost
            total_cost += cost
            self.replay_buffer.add(obs, action, reward, cost, next_obs, done, done_no_max)

            obs = next_obs
            ep_step += 1
            self.step += 1
        self.agent.save(self.work_dir)
        self.logger.log("eval/episode", episode, self.step)
        self.evaluate()


@hydra.main(config_path="config/train.yaml", strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()

#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gym
import parl
import time
import numpy as np
from es import ES
from filter import MeanStdFilter
from mujoco_agent import MujocoAgent
from mujoco_model import MujocoModel
from noise import SharedNoiseTable


@parl.remote_class
class Actor(object):
    def __init__(self, config):
        self.config = config

        self.env = gym.make(self.config['env_name'])
        self.config['obs_dim'] = self.env.observation_space.shape[0]
        self.config['act_dim'] = self.env.action_space.shape[0]

        self.obs_filter = MeanStdFilter(self.config['obs_dim'])
        self.noise = SharedNoiseTable(self.config['noise_size'])

        model = MujocoModel(self.config['act_dim'])
        algorithm = ES(model, hyperparas=self.config)
        self.agent = MujocoAgent(algorithm, self.config)

    def _play_one_episode(self, add_noise=False):
        rewards = []
        t = 0
        obs = self.env.reset()
        while True:
            obs = self.obs_filter(obs[None])

            action = self.agent.predict(obs)
            if add_noise and isinstance(self.env.action_space, gym.spaces.Box):
                action += np.random.randn(*action.shape) * self.config['action_noise_std']

            obs, reward, done, _ = self.env.step(action)
            rewards.append(reward)
            t += 1
            if done:
                break

        rewards = np.array(rewards, dtype=np.float32)
        return rewards, t

    def sample(self, flat_params):
        noise_indices, returns, lengths = [], [], []
        eval_returns, eval_lengths = [], []

        # Perform some rollouts with noise.
        task_tstart = time.time()
        while (len(noise_indices) == 0
               or time.time() - task_tstart < self.config['min_task_runtime']):

            if np.random.uniform() < self.config["eval_prob"]:
                # Do an evaluation run with no perturbation.
                self.agent.set_flat_params(flat_params)
                rewards, length = self._play_one_episode(add_noise=False)
                eval_returns.append(rewards.sum())
                eval_lengths.append(length)
            else:
                # Do a regular run with parameter perturbations.
                noise_index = self.noise.sample_index(self.agent.params_total_size)

                perturbation = self.config["noise_stdev"] * self.noise.get(
                    noise_index, self.agent.params_total_size)

                # mirrored sampling: evaluate pairs of perturbations \epsilon, −\epsilon
                self.agent.set_flat_params(flat_params + perturbation)
                rewards_pos, lengths_pos = self._play_one_episode(add_noise=True)

                self.agent.set_flat_params(flat_params - perturbation)
                rewards_neg, lengths_neg = self._play_one_episode(add_noise=True)

                noise_indices.append(noise_index)
                returns.append([rewards_pos.sum(), rewards_neg.sum()])
                lengths.append([lengths_pos, lengths_neg])

        return {'noise_indices': noise_indices,
            'noisy_returns': returns,
            'noisy_lengths': lengths,
            'eval_returns': eval_returns,
            'eval_lengths': eval_lengths}
    
    def get_filter(self, flush_after=False):
        return_filter = self.obs_filter.as_serializable()
        if flush_after:
            self.obs_filter.clear_buffer()
        return return_filter

    def set_filter(self, new_filter):
        self.obs_filter.sync(new_filter)

if __name__ == '__main__':
    from es_config import config

    actor = Actor(config)
    actor.as_remote(config['server_ip'], config['server_port'])

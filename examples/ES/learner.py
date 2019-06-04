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
import os
import parl
import numpy as np
import threading
import utils
from es import ES
from filter import MeanStdFilter
from mujoco_agent import MujocoAgent
from mujoco_model import MujocoModel
from noise import SharedNoiseTable
from optimizers import Adam
from parl import RemoteManager
from parl.utils import logger, CSVLogger
from six.moves import queue


class Learner(object):
    def __init__(self, config):
        self.config = config

        env = gym.make(self.config['env_name'])
        self.config['obs_dim'] = env.observation_space.shape[0]
        self.config['act_dim'] = env.action_space.shape[0]

        self.obs_filter = MeanStdFilter(self.config['obs_dim'])
        self.noise = SharedNoiseTable(self.config['noise_size'])

        model = MujocoModel(self.config['act_dim'])
        algorithm = ES(model, hyperparas=self.config)
        self.agent = MujocoAgent(algorithm, self.config)
        self.latest_flat_params = self.agent.get_flat_params()

        self.optimizer = Adam(self.agent.params_total_size, self.config['stepsize'])

        self.sample_total_episodes = 0
        self.sample_total_steps = 0

        self.actors_signal_input_queues = []
        self.actors_output_queues = []
        self.run_remote_manager()
        
        self.reward_list = []

        self.csv_logger = CSVLogger(
            os.path.join(logger.get_dir(), 'result.csv'))

    def run_remote_manager(self):
        """ Accept connection of new remote actor and start sampling of the remote actor.
        """
        remote_manager = RemoteManager(port=self.config['server_port'])
        logger.info('Waiting for {} remote actors to connect.'.format(
            self.config['actor_num']))

        self.remote_count = 0
        for i in range(self.config['actor_num']):
            remote_actor = remote_manager.get_remote()
            signal_queue = queue.Queue()
            output_queue = queue.Queue()
            self.actors_signal_input_queues.append(signal_queue)
            self.actors_output_queues.append(output_queue)

            self.remote_count += 1
            logger.info('Remote actor count: {}'.format(self.remote_count))

            remote_thread = threading.Thread(
                target=self.run_remote_sample,
                args=(remote_actor, signal_queue, output_queue))
            remote_thread.setDaemon(True)
            remote_thread.start()

        logger.info('All remote actors are ready, begin to learn.')

    def run_remote_sample(self, remote_actor, signal_queue, output_queue):
        """ Sample data from remote actor or get filters of remote actor. 
        """
        while True:
            info = signal_queue.get()
            if info['signal'] == 'sample':
                result = remote_actor.sample(info['latest_flat_params'])
                from box import Box
                result = Box(result, frozen_box=True)
                output_queue.put(result)
            elif info['signal'] == 'get_filter':
                actor_filter = remote_actor.get_filter(flush_after=True)
                output_queue.put(actor_filter)
            elif info['signal'] == 'set_filter':
                remote_actor.set_filter(info['latest_filter'])
            else:
                raise NotImplementedError

    def step(self):
        """
        1. kick off all actors to synchronize parameters and observation filter and sample data;
        2. sample data from remote actors;
        3. update parameters and observation filter.
        """

        num_episodes, num_timesteps = 0, 0
        results = []
        
        while num_episodes < self.config['min_episodes_per_batch'] or \
                num_timesteps < self.config['min_steps_per_batch']:
            # Send sample signal to all actors
            for q in self.actors_signal_input_queues:
                q.put({
                    'signal': 'sample',
                    'latest_flat_params': self.latest_flat_params,
                    })

            # Collect results from all actors
            for q in self.actors_output_queues:
                result = q.get()
                results.append(result)
                # Update the number of episodes and the number of timesteps
                # keeping in mind that result.noisy_lengths is a list of lists,
                # where the inner lists have length 2.
                num_episodes += sum(len(pair) for pair in result['noisy_lengths'])
                num_timesteps += sum(sum(pair) for pair in result['noisy_lengths'])
        
        all_noise_indices = []
        all_training_returns = []
        all_training_lengths = []
        all_eval_returns = []
        all_eval_lengths = []

        # Loop over the results.
        for result in results:
            all_eval_returns += result.eval_returns
            all_eval_lengths += result.eval_lengths

            all_noise_indices += result.noise_indices
            all_training_returns += result.noisy_returns
            all_training_lengths += result.noisy_lengths

        assert len(all_eval_returns) == len(all_eval_lengths)
        assert (len(all_noise_indices) == len(all_training_returns) ==
                len(all_training_lengths))

        self.sample_total_episodes += num_episodes
        self.sample_total_steps += num_timesteps

        # Assemble the results.
        eval_returns = np.array(all_eval_returns)
        eval_lengths = np.array(all_eval_lengths)
        noise_indices = np.array(all_noise_indices)
        noisy_returns = np.array(all_training_returns)
        noisy_lengths = np.array(all_training_lengths)

        # Process the returns. (rank transformation to the returns)
        # normalize returns to (-0.5, 0.5)
        proc_noisy_returns = utils.compute_centered_ranks(noisy_returns)

        # Compute and take a step.
        # mirrored sampling: evaluate pairs of perturbations \epsilon, −\epsilon
        g, count = utils.batched_weighted_sum(
            proc_noisy_returns[:, 0] - proc_noisy_returns[:, 1],
            (self.noise.get(index, self.agent.params_total_size)
             for index in noise_indices),
            batch_size=500)
        g /= noisy_returns.size
        assert (g.shape == self.latest_flat_params.shape and g.dtype == np.float32
                and count == len(noise_indices))
        
        # Compute the new weights theta.
        theta, update_ratio = self.optimizer.update(self.latest_flat_params,
                -g + self.config["l2_coeff"] * self.latest_flat_params)

        # Update the parameters of policy.
        self.latest_flat_params = theta
        # Update obs filter
        self._update_filter()

        # Store the rewards
        if len(all_eval_returns) > 0:
            self.reward_list.append(np.mean(eval_returns))
        reward_mean = np.mean(self.reward_list[-self.config['report_length']:])

        metrics = {
            "weights_norm": np.square(theta).sum(),
            "grad_norm": np.square(g).sum(),
            "update_ratio": update_ratio,
            "episodes_this_iter": noisy_lengths.size,
            "sample_total_episodes": self.sample_total_episodes,
            'sample_total_steps': self.sample_total_steps,
            "episode_reward_mean": reward_mean,
            "episode_len_mean": eval_lengths.mean(),
            "timesteps_this_iter": noisy_lengths.sum(),
        }

        self.log_metrics(metrics)
        return metrics
    
    def _update_filter(self):
        # Send get_filter signal to all actors
        for q in self.actors_signal_input_queues:
            q.put({'signal': 'get_filter'})

        filters = []
        # Collect filters from  all actors and update global filter
        for q in self.actors_output_queues:
            actor_filter = q.get()
            self.obs_filter.apply_changes(actor_filter)

        # Send set_filter signal to all actors
        for q in self.actors_signal_input_queues:
            q.put({'signal': 'set_filter', 'latest_filter': self.obs_filter.as_serializable()})

    def log_metrics(self, metrics):
        logger.info(metrics)
        self.csv_logger.log_dict(metrics)
    
    def close(self):
        self.csv_logger.close()

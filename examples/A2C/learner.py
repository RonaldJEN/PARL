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
import numpy as np
import os
import queue
import six
import time
import threading
from atari_model import AtariModel
from atari_agent import AtariAgent
from collections import defaultdict
from parl import RemoteManager
from parl.algorithms import A3C
from parl.env.atari_wrappers import wrap_deepmind
from parl.utils import logger, CSVLogger, get_gpu_count
from parl.utils.scheduler import PiecewiseScheduler
from parl.utils.time_stat import TimeStat
from parl.utils.window_stat import WindowStat


class Learner(object):
    def __init__(self, config):
        self.config = config

        #=========== Create Agent ==========
        env = gym.make(config['env_name'])
        env = wrap_deepmind(env, dim=config['env_dim'], obs_format='NCHW')
        obs_shape = env.observation_space.shape
        act_dim = env.action_space.n
        self.config['obs_shape'] = obs_shape
        self.config['act_dim'] = act_dim

        model = AtariModel(act_dim)
        algorithm = A3C(model, hyperparas=config)
        self.agent = AtariAgent(algorithm, config)

        if self.agent.gpu_id >= 0:
            assert get_gpu_count() == 1, 'Only support training in single GPU,\
                    Please set environment variable: `export CUDA_VISIBLE_DEVICES=[GPU_ID_YOU_WANT_TO_USE]` .'

        else:
            cpu_num = os.environ.get('CPU_NUM')
            assert cpu_num is not None and cpu_num == '1', 'Only support training in single CPU,\
                    Please set environment variable:  `export CPU_NUM=1`.'

        #========== Learner ==========

        self.total_loss_stat = WindowStat(100)
        self.pi_loss_stat = WindowStat(100)
        self.vf_loss_stat = WindowStat(100)
        self.entropy_stat = WindowStat(100)
        self.lr = None
        self.entropy_coeff = None

        self.learn_time_stat = TimeStat(100)
        self.start_time = None

        #========== Remote Actor ===========
        self.remote_count = 0
        self.sample_data_queue = queue.Queue()

        self.remote_metrics_queue = queue.Queue()
        self.sample_total_steps = 0

        self.params_queues = []
        self.run_remote_manager()

        self.csv_logger = CSVLogger(
            os.path.join(logger.get_dir(), 'result.csv'))

    def run_remote_manager(self):
        """ Accept connection of new remote actor and start sampling of the remote actor.
        """
        remote_manager = RemoteManager(port=self.config['server_port'])
        logger.info('Waiting for {} remote actors to connect.'.format(
            self.config['actor_num']))

        for i in six.moves.range(self.config['actor_num']):
            remote_actor = remote_manager.get_remote()
            params_queue = queue.Queue()
            self.params_queues.append(params_queue)

            self.remote_count += 1
            logger.info('Remote actor count: {}'.format(self.remote_count))

            remote_thread = threading.Thread(
                target=self.run_remote_sample,
                args=(remote_actor, params_queue))
            remote_thread.setDaemon(True)
            remote_thread.start()

        logger.info('All remote actors are ready, begin to learn.')
        self.start_time = time.time()

    def run_remote_sample(self, remote_actor, params_queue):
        """ Sample data from remote actor and update parameters of remote actor.
        """
        cnt = 0
        while True:
            latest_params = params_queue.get()
            remote_actor.set_params(latest_params)
            batch = remote_actor.sample()

            self.sample_data_queue.put(batch)

            cnt += 1
            if cnt % self.config['get_remote_metrics_interval'] == 0:
                metrics = remote_actor.get_metrics()
                if metrics:
                    self.remote_metrics_queue.put(metrics)

    def step(self):
        """
        1. kick off all actors to synchronize parameters and sample data;
        2. collect sample data of all actors;
        3. update parameters. 
        """

        latest_params = self.agent.get_params()
        for params_queue in self.params_queues:
            params_queue.put(latest_params)

        train_batch = defaultdict(list)
        for i in range(self.config['actor_num']):
            sample_data = self.sample_data_queue.get()
            for key, value in sample_data.items():
                train_batch[key].append(value)

            self.sample_total_steps += sample_data['obs'].shape[0]

        for key, value in train_batch.items():
            train_batch[key] = np.concatenate(value)

        with self.learn_time_stat:
            total_loss, pi_loss, vf_loss, entropy, lr, entropy_coeff = self.agent.learn(
                obs_np=train_batch['obs'],
                actions_np=train_batch['actions'],
                advantages_np=train_batch['advantages'],
                target_values_np=train_batch['target_values'])

        self.total_loss_stat.add(total_loss)
        self.pi_loss_stat.add(pi_loss)
        self.vf_loss_stat.add(vf_loss)
        self.entropy_stat.add(entropy)
        self.lr = lr
        self.entropy_coeff = entropy_coeff

    def log_metrics(self):
        """ Log metrics of learner and actors
        """
        if self.start_time is None:
            return

        metrics = []
        while True:
            try:
                metric = self.remote_metrics_queue.get_nowait()
                metrics.append(metric)
            except queue.Empty:
                break

        episode_rewards, episode_steps = [], []
        for x in metrics:
            episode_rewards.extend(x['episode_rewards'])
            episode_steps.extend(x['episode_steps'])
        max_episode_rewards, mean_episode_rewards, min_episode_rewards, \
                max_episode_steps, mean_episode_steps, min_episode_steps =\
                None, None, None, None, None, None
        if episode_rewards:
            mean_episode_rewards = np.mean(np.array(episode_rewards).flatten())
            max_episode_rewards = np.max(np.array(episode_rewards).flatten())
            min_episode_rewards = np.min(np.array(episode_rewards).flatten())

            mean_episode_steps = np.mean(np.array(episode_steps).flatten())
            max_episode_steps = np.max(np.array(episode_steps).flatten())
            min_episode_steps = np.min(np.array(episode_steps).flatten())

        metric = {
            'Sample steps': self.sample_total_steps,
            'max_episode_rewards': max_episode_rewards,
            'mean_episode_rewards': mean_episode_rewards,
            'min_episode_rewards': min_episode_rewards,
            'max_episode_steps': max_episode_steps,
            'mean_episode_steps': mean_episode_steps,
            'min_episode_steps': min_episode_steps,
            'total_loss': self.total_loss_stat.mean,
            'pi_loss': self.pi_loss_stat.mean,
            'vf_loss': self.vf_loss_stat.mean,
            'entropy': self.entropy_stat.mean,
            'learn_time_s': self.learn_time_stat.mean,
            'elapsed_time_s': int(time.time() - self.start_time),
            'lr': self.lr,
            'entropy_coeff': self.entropy_coeff,
        }

        logger.info(metric)
        self.csv_logger.log_dict(metric)

    def should_stop(self):
        return self.sample_total_steps >= self.config['max_sample_steps']

    def close(self):
        self.csv_logger.close()

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

import numpy as np
import paddle.fluid as fluid
import parl.layers as layers
from parl.framework.agent_base import Agent
from utils import unflatten

class MujocoAgent(Agent):
    def __init__(self, algorithm, config):
        self.config = config
        super(MujocoAgent, self).__init__(algorithm)

        params = self.get_params()
        self.params_shapes = [x.shape for x in params]
        self.params_total_size = np.sum([np.prod(x) for x in self.params_shapes])


        exec_strategy = fluid.ExecutionStrategy()
        exec_strategy.use_experimental_executor = True
        exec_strategy.num_threads = 4
        build_strategy = fluid.BuildStrategy()
        build_strategy.remove_unnecessary_lock = True
        
        self.parallel_executor = fluid.ParallelExecutor(
                use_cuda=False,
                main_program=self.predict_program,
                build_strategy=build_strategy,
                exec_strategy=exec_strategy
                )


    def build_program(self):
        self.predict_program = fluid.Program()

        with fluid.program_guard(self.predict_program):
            obs = layers.data(
                name='obs', shape=[self.config['obs_dim']], dtype='float32')
            self.predict_action = self.alg.predict(obs)

    def predict(self, obs):
        obs = obs.astype('float32')
        obs = np.expand_dims(obs, axis=0)
        act = self.parallel_executor.run(
            feed={'obs': obs},
            fetch_list=[self.predict_action.name])[0]
        return act
    
    def get_flat_params(self):
        params = self.get_params()
        flat_params = np.concatenate([x.flatten() for x in params])
        return flat_params
    
    def set_flat_params(self, flat_params):
        params = unflatten(flat_params, self.params_shapes)
        self.set_params(params)

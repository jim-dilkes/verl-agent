# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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

import ray
from agent_system.environments.env_package.babyai.babyai_text import BabyAITextEnv
import numpy as np

class BabyAIWorker:
    """
    Ray remote actor. Each actor holds its own independent BabyAITextEnv instance.
    """

    def __init__(self, env_kwargs):
        """Initialize the BabyAI environment in this worker"""
        self.env = BabyAITextEnv(**env_kwargs)

    def step(self, action_text):
        """Execute a step in the environment. The env parses the raw text action itself."""
        obs, reward, done, info = self.env.step(action_text)
        info['won'] = info['success']
        return obs, reward, done, info

    def reset(self, seed_for_reset):
        """Reset the environment with given seed"""
        obs, mission = self.env.reset(seed=seed_for_reset)
        info = {'mission': mission, 'won': False}
        return obs, info


class BabyAIMultiProcessEnv:
    """
    Ray-based wrapper for the BabyAI text environment.
    Each Ray actor creates an independent BabyAITextEnv instance.
    The main process communicates with Ray actors to collect step/reset results.
    """

    def __init__(self,
                 seed=0,
                 env_num=1,
                 group_n=1,
                 resources_per_worker={"num_cpus": 0.1},
                 is_train=True,
                 env_kwargs=None):
        """
        - env_num: Number of different environments
        - group_n: Number of same environments in each group (for GRPO and GiGPO)
        - env_kwargs: Dictionary of parameters for initializing BabyAITextEnv
        - seed: Random seed for reproducibility
        """
        super().__init__()

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        self.is_train = is_train
        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        np.random.seed(seed)

        if env_kwargs is None:
            env_kwargs = {}

        # Create Ray remote actors
        env_worker = ray.remote(**resources_per_worker)(BabyAIWorker)
        self.workers = []
        for i in range(self.num_processes):
            worker = env_worker.remote(env_kwargs)
            self.workers.append(worker)

    def step(self, actions):
        """
        Perform step in parallel.
        :param actions: list[str], length must match self.num_processes
        :return:
            obs_list, reward_list, done_list, info_list
            Each is a list of length self.num_processes
        """
        assert len(actions) == self.num_processes

        # Send step commands to all workers
        futures = []
        for worker, action in zip(self.workers, actions):
            future = worker.step.remote(action)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return obs_list, reward_list, done_list, info_list

    def reset(self):
        """
        Perform reset in parallel.
        :return: obs_list and info_list, the initial observations for each environment
        """
        # randomly generate self.env_num seeds
        if self.is_train:
            seeds = np.random.randint(0, 2**16 - 1, size=self.env_num)
        else:
            seeds = np.random.randint(2**16, 2**32 - 1, size=self.env_num)

        # repeat the seeds for each group
        seeds = np.repeat(seeds, self.group_n)
        seeds = seeds.tolist()

        # Send reset commands to all workers
        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.reset.remote(seeds[i])
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list = []
        info_list = []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)
        return obs_list, info_list

    def close(self):
        """
        Close all Ray actors
        """
        for worker in self.workers:
            ray.kill(worker)

    def __del__(self):
        self.close()


def build_babyai_envs(
        seed=0,
        env_num=1,
        group_n=1,
        resources_per_worker={"num_cpus": 0.1},
        is_train=True,
        env_kwargs=None):
    return BabyAIMultiProcessEnv(seed, env_num, group_n, resources_per_worker, is_train, env_kwargs=env_kwargs)

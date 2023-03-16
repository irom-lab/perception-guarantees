"""Subprocess vectorized environment

This module contains a vectorized environment that runs multiple environments in parallel in subprocesses. It is modified from the stable-baselines3 library.

Please contact the author(s) of this library if you have any questions.
Authors: Allen Z. Ren (allen.ren@princeton.edu)
"""

import torch
from nav_sim.env.subproc_vec_env import SubprocVecEnv


def make_vec_envs(
    env_class, seed, num_process, cpu_offset, device,
    pickle_option='cloudpickle', start_method='forkserver', **kwargs
):
    """
    Create a wrapped vectorized environment.
    
    Args:
        env_class (class): the environment class
        seed (int): the random seed for the first env; increment by 1 for each subsequent env
        num_process (int): the number of environments to run in parallel
        cpu_offset (int): the starting CPU thread ID
        device (torch.device): the device to store the observations and actions
        vec_env_type (class): the vectorized environment class
        vec_env_cfg (dict): the configuration for the vectorized environment
        **kwargs: the arguments for the environment class
    """
    envs = [env_class(**kwargs) for _ in range(num_process)]
    for rank, env in enumerate(envs):
        env.seed(seed + rank)
    envs = VecEnvBase(envs, cpu_offset, device, pickle_option, start_method)
    return envs


class VecEnvBase(SubprocVecEnv):
    """Vectorized environment class
    
    Args:
        venv (list): the list of environments
        cpu_offset (int): the starting CPU thread ID
        device (torch.device): the device to store the observations and actions
        pickle_option (str): the option for the pickle library
        start_method (str): the method to start the subprocesses
    """

    def __init__(
        self, venv, cpu_offset, device, pickle_option='cloudpickle',
        start_method=None
    ):
        super(VecEnvBase, self).__init__(
            venv, cpu_offset, pickle_option=pickle_option,
            start_method=start_method
        )
        self.device = device

    def reset(self, tasks):
        """Reset the environment
        
        Args:
            tasks (list): the list of tasks
        
        Returns:
            obs (torch.Tensor): the observation
        """
        args_all = [(task,) for task in tasks]
        obs = self.reset_arg(args_all)
        return torch.from_numpy(obs).to(self.device)

    def reset_one(self, index, task):
        """Reset one environment
        
        Args:
            index (int): the index of the environment
            task (dict): the task
        
        Returns:
            obs (torch.Tensor): the observation
        """
        obs = self.env_method('reset', task=task, indices=[index])[0]
        return torch.from_numpy(obs).to(self.device)

    # Overrides
    def step_async(self, actions):
        """Step the environment asynchronously
        
        In reality, the action space can be anything... - e.g., a trajectory plus the initial joint angles for the pushing task. We could also super this in each  class to check the action space carefully. vec_env
        
        Args:
            actions (torch.Tensor): the actions to take
        """
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        super().step_async(actions)

    # Overrides
    def step_wait(self):
        """Wait for the environment to step
        
        Returns:
            obs (torch.Tensor): the observations
            reward (torch.Tensor): the rewards
            done (torch.Tensor): the done flags
            info (list): the info
        """
        obs, reward, done, info = super().step_wait()
        obs = torch.from_numpy(obs).to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info

    def get_obs(self, states):
        """Get the observation from each env based on the state
        
        Args:
            states (list): the list of states
        
        Returns:
            obs (torch.Tensor): the observations
        """
        method_args_list = [(state,) for state in states]
        obs = torch.FloatTensor(
            self.env_method_arg(
                '_get_obs', method_args_list=method_args_list,
                indices=range(self.n_envs)
            )
        )
        return obs.to(self.device)

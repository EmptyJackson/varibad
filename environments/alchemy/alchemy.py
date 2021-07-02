import gym
from gym import spaces

from environments.alchemy.dm_alchemy.dm_alchemy import symbolic_alchemy

class AlchemyEnv(gym.Env):
    def __init__(self, level_name):
        super(AlchemyEnv, self).__init__()

        self.seed()
        self.env = symbolic_alchemy.get_symbolic_alchemy_level(level_name, seed=123)

        self._max_episode_steps = self.env.max_steps_per_trial
        self.step_count = 0

        self.observation_space = spaces.Box(low=-1, high=1, shape=(39,))
        self.action_space = spaces.Discrete(40)

    def step(self, action):
        """
        Execute one step in the environment.
        Should return: state, reward, done, info
        where info has to include a field 'task'.
        """
        timestep = self.env.step(action)
        return timestep.observation['symbolic_obs'], timestep.reward, timestep.last(), {'task': None}

    def reset(self):
        """
        Reset the environment. This should *NOT* automatically reset the task!
        Resetting the task is handled in the varibad wrapper (see wrappers.py).
        """
        return self.env.reset().observation['symbolic_obs']

    def get_task(self):
        """
        Return a task description, such as goal position or target velocity.
        """
        return 0

    def reset_task(self, task=None):
        """
        Reset the task, either at random (if task=None) or the given task.
        Should *not* reset the environment.
        """
        return 0

    # def visualise_behaviour(self,
    #                         env,
    #                         args,
    #                         policy,
    #                         iter_idx,
    #                         encoder=None,
    #                         reward_decoder=None,
    #                         state_decoder=None,
    #                         task_decoder=None,
    #                         image_folder=None,
    #                         **kwargs):
    #     """
    #     Optional. If this is not overwritten, a default visualisation will be used (see utils/evaluation.py).
    #     Should return the following:
    #         episode_latent_means, episode_latent_logvars, episode_prev_obs,
    #         episode_next_obs, episode_actions, episode_rewards, episode_returns
    #     where each element is either a list of length num_episodes,
    #     or "None" if not applicable.
    #     """
    #     pass

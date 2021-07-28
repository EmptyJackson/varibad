import gym
import numpy as np

from gym import spaces
from dm_alchemy import symbolic_alchemy
from dm_alchemy.encode import chemistries_proto_conversion

LEVEL_NAME = 'alchemy/perceptual_mapping_randomized_with_rotation_and_random_bottleneck'
CHEM_NAME = 'chemistries/perceptual_mapping_randomized_with_random_bottleneck/chemistries'

class AlchemyEnv(gym.Env):

    def __init__(self, num_trials=10, num_stones_per_trial=3, num_potions_per_trial=12, max_steps_per_trial=20, fixed=False):
        super(AlchemyEnv, self).__init__()

        # TODO: Use max_rollouts_per_task somewhere (i think)

        self.seed()
        self.fixed = fixed
        if self.fixed:
            chems = chemistries_proto_conversion.load_chemistries_and_items(CHEM_NAME)
            self.env = symbolic_alchemy.get_symbolic_alchemy_fixed(chemistry=chems[0][0], episode_items=chems[0][1])
        else:
            self.env = symbolic_alchemy.get_symbolic_alchemy_level(level_name=LEVEL_NAME, num_trials=num_trials, num_stones_per_trial=num_stones_per_trial, num_potions_per_trial=num_potions_per_trial, max_steps_per_trial=max_steps_per_trial)

        self._max_episode_steps = self.env.max_steps_per_trial
        self.step_count = 0

        # TODO: Values can be outside of [-1, 1], got 2 from manual observation
        self.observation_space = spaces.Box(low=-1, high=2, shape=(39,))
        self.action_space = spaces.Discrete(40)

    def step(self, action):
        """
        Execute one step in the environment.
        Should return: state, reward, done, info
        where info has to include a field 'task'.
        """
        if isinstance(action, np.ndarray) and action.ndim == 1:
            action = action[0]
        self.timestep = self.env.step(action)

        # self.timestep.last()
        return self.timestep.observation['symbolic_obs'], self.timestep.reward, self.env.is_new_trial(), {'task': 0}

    def reset(self):
        """
        Reset the environment. This should *NOT* automatically reset the task!
        Resetting the task is handled in the varibad wrapper (see wrappers.py).

        Completed automatically by Alchemy when trial is done.
        """
        if not self.env.is_new_trial():
            raise Exception("Alchemy reset not on trial boundary.")
        return self.timestep.observation['symbolic_obs']

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
        self.timestep = self.env.reset()
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

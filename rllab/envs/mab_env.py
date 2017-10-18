from rllab.core.serializable import Serializable
import numpy as np

from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.misc import logger
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.envs.base import Env
import numpy as np
from rllab.envs.base import Step
from rllab.core.serializable import Serializable
from rllab.envs.env_spec import EnvSpec
import sandbox.rocky.tf.spaces as spaces



class MABEnv(Env, Serializable):

    def __init__(self, *args, **kwargs):
        self.n_arms = kwargs.get('n_arms')
        self.max_path_length = kwargs.get('max_path_length')
        self.arm_means = np.zeros((self.n_arms))
        self.ts = 0 

        super(MABEnv, self).__init__(*args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)

    def sample_goals(self, num_goals): 
        return np.random.uniform(0.0, 1.0, (num_goals, self.n_arms))
    
    @property
    @overrides
    def action_space(self):
      return spaces.Discrete(self.n_arms)

    @property
    @overrides
    def observation_space(self):
      return spaces.Box(0, 1, shape=(2))

    @overrides
    def reset(self, init_state=None, reset_args=None, **kwargs):
        arm_means = reset_args
        import pdb; pdb.set_trace()
        if arm_means is not None:
            self.arm_means = arm_means
        elif self.arm_means is None:
            self.arm_means = sample_goals(1)[0]
        obs = np.zeros((2))
        return obs


    def step(self, action):
      #  self.forward_dynamics(action)
      #  comvel = self.get_body_comvel("torso")
      #  forward_reward = self.goal_direction*comvel[0]
      #  lb, ub = self.action_bounds
      #  scaling = (ub - lb) * 0.5
      #  ctrl_cost = 0.5 * 1e-2 * np.sum(np.square(action / scaling))
      #  contact_cost = 0.5 * 1e-3 * np.sum(
      #      np.square(np.clip(self.model.data.cfrc_ext, -1, 1))),
      #  survive_reward = 0.05
      #  reward = forward_reward - ctrl_cost - contact_cost + survive_reward
      #  state = self._state
      #  notdone = np.isfinite(state).all() \
      #      and state[2] >= 0.2 and state[2] <= 1.0
      #  done = not notdone
      #  ob = self.get_current_obs()
      #  return Step(ob, float(reward), done)
      #  obs = self.get_current_obs()

        selected_arm_mean = self.arm_means[action]
        reward = float(np.random.random() < selected_arm_mean)
        self.ts += 1
        done = self.ts >= self.max_path_length
        state = np.zeros((2))
        state[0] = reward
        state[1] = 1

        return Step(state, reward, done)


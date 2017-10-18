from rllab.core.serializable import Serializable
import numpy as np

from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.misc import logger
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.envs.base import Env
import numpy as np
from rllab.misc import special
from rllab.envs.base import Step
from rllab.core.serializable import Serializable
from rllab.envs.env_spec import EnvSpec
import sandbox.rocky.tf.spaces as spaces



class MDPEnv(Env, Serializable):

    def __init__(self, *args, **kwargs):
        self.n_states = kwargs.get('n_states')
        self.max_path_length = kwargs.get('max_path_length')
        self.n_actions = kwargs.get('n_actions')
        self.alpha0=1.
        self.mu0=1.
        self.tau0=1.
        self.tau=1.
       # self.arm_means = np.zeros((self.n_arms))
       # self.ts = 0 
        self.state = 0
        self.ts = 0

        super(MDPEnv, self).__init__(*args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)

    def sample_goals(self, num_goals): 
        size = (num_goals, self.n_states, self.n_actions)
        Rs = np.ones(size) * self.mu0 + np.random.normal(size=size) * 1. / np.sqrt(self.tau0)
        Ps = np.random.dirichlet(self.alpha0 * np.ones(self.n_states, dtype=np.float32), size=size)
        return [(Rs[i], Ps[i]) for i in range(num_goals)]
    
    @property
    @overrides
    def action_space(self):
      return spaces.Discrete(self.n_actions)

    @property
    @overrides
    def observation_space(self):
      return spaces.Box(0, 1, shape=(2 + self.n_states))

    @overrides
    def reset(self, init_state=None, reset_args=None, **kwargs):
        Rs, Ps = reset_args
        if Rs is not None and Ps is not None:
            self.Rs, self.Ps = Rs, Ps
        elif Rs is None or Ps is None:
            self.Rs, self.Ps = sample_goals(1)[0]
        obs = np.zeros((2 + self.n_states))
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

      #  selected_arm_mean = self.arm_means[action]
      #  reward = float(np.random.random() < selected_arm_mean)
      #  self.ts += 1
      #  done = self.ts >= self.max_path_length
      #  state = np.zeros((2))
      #  state[0] = reward
      #  state[1] = 1

      #  return Step(state, reward, done)
      ps = self.Ps[self.state, action]
      next_state = special.weighted_sample(ps, np.arange(self.n_states))
      reward_mean = self.Rs[self.state, action]
      reward = reward_mean + np.random.normal() * 1 / np.sqrt(self.tau)
      self.ts += 1
      self.state = next_state
      done = self.ts >= self.max_path_length
      state = np.zeros((2 + self.n_states))
      state[self.state] = 1
      state[self.n_states] = reward
      state[self.n_states + 1] = done
      return Step(state, reward, done)

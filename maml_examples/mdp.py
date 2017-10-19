from sandbox.rocky.tf.algos.maml_trpo import MAMLTRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
#from .rllab.envs.mujoco.ant_env_rand import AntEnvRand
#from .rllab.envs.mujoco.ant_env_rand_goal import AntEnvRandGoal
#from .rllab.envs.mujoco.ant_env_rand_direc import AntEnvRandDirec
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
#from ..sandbox.rocky.tf.policies.maml_minimal_gauss_mlp_policy import MAMLGaussianMLPPolicy
from sandbox.rocky.tf.policies.maml_minimal_categorical_mlp_policy import MAMLCategoricalMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.mdp_env import MDPEnv
import argparse

import tensorflow as tf

stub(globals())

from rllab.misc.instrument import VariantGenerator, variant


class VG(VariantGenerator):

    @variant
    def fast_lr(self):
        return [0.1]

    @variant
    def meta_step_size(self):
        return [0.01] # sometimes 0.02 better

    @variant
    def seed(self):
        return [1]

    @variant
    def n_episodes(self):
      return [10, 26, 50, 76, 100]
    
    @variant
    def fast_batch_size(self, n_episodes):
        return [n_episodes/2]
    
    @variant
    def meta_batch_size(self, n_episodes):
      if n_episodes == 10:
        return [50]
      elif n_episodes == 26:
        return [25]
      elif n_episodes == 50:
        return [12]
      elif n_episodes == 76:
        return [5]
      elif n_episodes == 100:
        return [1]

    @variant
    def n_states(self):
      return [10]

    @variant
    def episode_horizon(self):
      return [10]
    
    @variant
    def n_actions(self):
      return [5]

    @variant
    def discount(self):
        return [0.99]
    
    @variant
    def gpu_frac(self):
        return [1.0]

  #  @variant
  #  def task_var(self):  # fwd/bwd task or goal vel task
  #      # 0 for fwd/bwd, 1 for goal vel (kind of), 2 for goal pose
  #      return [3]


# should also code up alternative KL thing

parser = argparse.ArgumentParser(description="what device to use")
parser.add_argument('--devices', type=int, default=0)
parser.add_argument('--variant', type=int, default=0)

FLAGS = parser.parse_args()

variants = VG().variants()


for v in [variants[FLAGS.variant]]:
    max_path_length = v.n_episodes * v.episode_horizon
    num_grad_updates = 1
    use_maml=True
   # env = TfEnv(MultiEnv(
   #     wrapped_env=MABEnv(n_arms=v.n_arms),
   #     episode_horizon=1,
   #     n_episodes=v.n_episodes, discount=v.discount,
   # ))
   # import functools
   # Env = functools.partial(MABEnv, v.n_arms, max_path_length)
   # env = Env()

    env = MDPEnv(n_states=v.n_states, 
                 n_actions=v.n_actions, 
                 max_path_length=max_path_length)

    policy = MAMLCategoricalMLPPolicy(
        name="policy",
        env_spec=env.spec,
        grad_step_size=v['fast_lr'],
        hidden_nonlinearity=tf.nn.relu,
        hidden_sizes=(100,100),
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)
    algo = MAMLTRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=v['fast_batch_size'], # number of trajs for grad update
        max_path_length=max_path_length,
        meta_batch_size=v['meta_batch_size'],
        num_grad_updates=num_grad_updates,
        n_itr=800,
        use_maml=use_maml,
        step_size=v['meta_step_size'],
        plot=False,
    )

    run_experiment_lite(
        algo.train(frac=v.gpu_frac),
        exp_prefix='maml_mdp', 
        exp_name='N%d_mbs%d' % (v['n_episodes'], v['meta_batch_size']),
        # Number of parallel workers for sampling
        n_parallel=1,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="gap",
        env={'CUDA_VISIBLE_DEVICES' : str(FLAGS.devices)},
        snapshot_gap=25,
        sync_s3_pkl=True,
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=v["seed"],
        mode="local",
        #mode="ec2",
        variant=v,
        # plot=True,
        # terminate_machine=False,
    )

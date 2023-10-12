from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
import highway_env
import torch
import numpy as np
# import the skrl components to build the RL system
from skrl.agents.torch.dqn import DDQN, DDQN_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from skrl.utils.model_instantiators.torch import Shape, deterministic_model
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
import gymnasium as gym
import highway_env
from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
from gymnasium.wrappers.monitoring import video_recorder
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder

from skrl.trainers.torch import ParallelTrainer
import warnings
import os
import gymnasium as gym

import torch.nn as nn
import torch.nn.functional as F

# import the skrl components to build the RL system
from skrl.agents.torch.cem import CEM, CEM_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import CategoricalMixin, Model
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

# seed for reproducibility
set_seed()  # e.g. `set_seed(42)` for fixed seed

environment_name = 'CartPole-v1'


# define model (categorical model) using mixin
class Policy(CategoricalMixin, Model):
    def __init__(self, observation_space, action_space, device, unnormalized_log_prob=True):
        Model.__init__(self, observation_space, action_space, device)
        CategoricalMixin.__init__(self, unnormalized_log_prob)

        self.linear_layer_1 = nn.Linear(self.num_observations, 64)
        self.linear_layer_2 = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, self.num_actions)

    def compute(self, inputs, role):
        x = F.relu(self.linear_layer_1(inputs["states"]))
        x = F.relu(self.linear_layer_2(x))
        return self.output_layer(x), {}


# load and wrap the gymnasium environment.
# note: the environment version may change depending on the gymnasium version
try:
    env = gym.make("CartPole-v1",render_mode = 'rgb_array')
except (gym.error.DeprecatedEnv, gym.error.VersionNotFound) as e:
    env_id = [spec for spec in gym.envs.registry if spec.startswith("CartPole-v")][0]
    print("CartPole-v0 not found. Trying {}".format(env_id))
    env = gym.make(env_id)
env = wrap_env(env)

device = env.device


# instantiate a memory as experience replay
memory = RandomMemory(memory_size=1000, num_envs=env.num_envs, device=device, replacement=False)


# instantiate the agent's model (function approximator).
# CEM requires 1 model, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/cem.html#models
models = {}
models["policy"] = Policy(env.observation_space, env.action_space, device)

# initialize models' parameters (weights and biases)
for model in models.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/cem.html#configuration-and-hyperparameters
cfg = CEM_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 1000
cfg["learning_starts"] = 100
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 1000
cfg["experiment"]["checkpoint_interval"] = 5000
cfg["experiment"]["directory"] = environment_name

agent = CEM(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)

agent.load("/Users/khoavo2003/cs224/Reinforcement_learning_project/Reinforcement_learning_project_1/CartPole-v1/23-10-02_00-52-00-870814_CEM/checkpoints/best_agent.pt")

# Initialize lists to store transitions
obs_list, acts_list, infos_list, next_obs_list, dones_list = [], [], [], [], []

obs, info = env.reset()
counter = 0
episode_len = 500

# Define the number of episodes to collect transitions
num_episodes = 10

while True:
    action = agent.act(obs, 0, 1)[0]
    next_obs, rew, done, truncated, info = env.step(action)

    # Append transitions to the lists
    obs_list.append(obs[0])
    acts_list.append(action)
    infos_list.append(info)
    next_obs_list.append(next_obs[0])
    dones_list.append(done)

    # Update the current observation
    obs = next_obs

    if done or truncated:
        obs, info = env.reset()
        counter = 0
        episode_len = 500
        num_episodes = num_episodes - 1
        if num_episodes == 0:
            break
    counter = counter + 1

# Create a dictionary to store the transitions
my_transitions = {
    "obs": obs_list,
    "acts": acts_list,
    "infos": infos_list,
    "next_obs": next_obs_list,
    "dones": dones_list
}

from imitation.data import types

def load_custom_transitions(my_transitions):
    transitions = types.Transitions(
        obs=np.array(my_transitions["obs"]),
        acts=np.array(my_transitions["acts"], dtype=np.int32),
        infos=my_transitions["infos"],
        next_obs=np.array(my_transitions["next_obs"]),
        dones=np.array(my_transitions["dones"], dtype=np.bool),
    )
    return transitions

# Load your custom transitions
custom_transitions = load_custom_transitions(my_transitions)

env1 = make_vec_env(
    "CartPole-v1",
    rng=np.random.default_rng(42),
    n_envs=8,
    post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # to compute rollouts
)

learner = PPO(
    env=env1,
    policy=MlpPolicy,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.000001,
    n_epochs=1,
    seed=42,
)
reward_net = BasicShapedRewardNet(
    observation_space=env1.observation_space,
    action_space=env1.action_space,
    normalize_input_layer=RunningNorm,
)
gail_trainer = GAIL(
    demonstrations=custom_transitions,
    demo_batch_size=32,
    gen_replay_buffer_capacity=2048,
    n_disc_updates_per_round=4,
    venv=env1,
    gen_algo=learner,
    reward_net=reward_net,
    allow_variable_horizon=True
)

# evaluate the learner before training
env1.seed(42)
learner_rewards_before_training, _ = evaluate_policy(
    learner, env1, 100, return_episode_rewards=True
)

# train the learner and evaluate again
gail_trainer.train(200000)
env1.seed(42)
learner_rewards_after_training, _ = evaluate_policy(
    learner, env1, 100, return_episode_rewards=True
)

print("mean reward after training:", np.mean(learner_rewards_after_training))
print("mean reward before training:", np.mean(learner_rewards_before_training))
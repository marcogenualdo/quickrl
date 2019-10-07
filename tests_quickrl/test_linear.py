import sys
sys.path.append('../')

import quickrl
from quickrl import Agent
from quickrl.interactions import gym_reset, gym_step
from quickrl.value_functions import SparseLinear
from quickrl.policies import MaxValue, TorchNet
import quickrl.learning_algorithms as algos
from quickrl.memories import ListOfTuples
from quickrl.logger import Logger
from quickrl.tile_coding import IHT, tiles

import gym
import torch
from torch import nn

env = gym.make('CartPole-v1')

# processing function
iht = IHT(1024)
tiler = lambda s,a: torch.LongTensor(tiles(iht, 8, tuple(s) + (a,)))

# agent parts
action_value_function = SparseLinear('q', 1024, processing=tiler)
policy = MaxValue([0,1], action_value_function)
memory = ListOfTuples()
action_value_function.optimizer = torch.optim.SGD([action_value_function.weights], lr=0.01)
action_value_function.loss_func = nn.MSELoss()
#action_value_function.loss_func = lambda x,y: - x * y

agent = Agent(
    env, 
    gym_reset,
    gym_step,
    action_value_function=action_value_function,
    policy=policy,
    memory=memory
    )


def eps_gen (episode, episodes):
    return max(0.01, 1. - episode / (episodes - 30))

#algos.n_sarsa(agent, n=5, episodes=300, epsilon_schedule=eps_gen)
#algos.sarsa_lambda(agent, episodes=300, epsilon_schedule=eps_gen)

episodes = 300
for episode in range(episodes):
    eps = eps_gen(episode, episodes)
    state = agent.reset()
    action = agent.epsilon_greedy_action(eps)
    algos.sarsa_lambda_reset(agent)

    done = False
    while not done:
        state, reward, done, info = agent.step(action)
        action = agent.epsilon_greedy_action(eps)

        algos.n_sarsa_step(agent, state, action, n=5)
        #algos.sarsa_lambda_step(agent, state, action)


print('average performace =', torch.tensor(agent.logger.total_rewards[-30:]).mean())
print('best =', max(agent.logger.total_rewards))
env.close()

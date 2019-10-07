import sys
sys.path.append('../')

import quickrl
from quickrl import Agent
from quickrl.interactions import atari_reset, atari_step
from quickrl.value_functions import SparseLinear, TorchNet
from quickrl.policies import MaxValue
import quickrl.learning_algorithms as algos
from quickrl.memories import ListOfTuples
from quickrl.logger import Logger

import gym
import torch
from torch import nn
from torchvision import transforms as T


env = gym.make('Breakout-v0')


class Network (nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.conv_ops = [
            nn.Conv2d(in_channels=2, out_channels=8, kernel_size=5, stride=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU()
        ]
        self.dense = nn.Linear(16*10*7, env.action_space.n)

    
    def forward (self, x):
        if len(x.shape) == 3: 
            x = x.unsqueeze(0)
        batch_size = x.shape[0]

        for op in self.conv_ops:
            x = op(x)
        x = x.view(batch_size, -1)
        x = self.dense(x)
        
        return x.squeeze()


# agent parts
net = Network()
vf = TorchNet('q', net)
policy = MaxValue(list(range(env.action_space.n)), vf)
memory = ListOfTuples(int(1e+5))

vf.loss_func = nn.SmoothL1Loss()
vf.optimizer = torch.optim.SGD(net.parameters(), lr=1e-4, momentum=0.6)

agent = Agent(env,
              atari_reset,
              atari_step,
              gamma=0.99,
              action_value_function=vf,
              policy=policy,
              memory=memory,
              plot_frequency=50
             )


"""
# trying what an image looks like as seen from the nn
from matplotlib import pyplot as plt
img = agent.reset(frames_to_skip=8, state_frames=3)
print(img.shape)
plt.imshow(img)
plt.show()
plt.pause(1)
img, r, d, i  = agent.step(torch.tensor([1]), frames_to_skip=8, state_frames=3)
print(img.shape)
plt.imshow(img)
plt.show()
plt.pause(5)
print(img.max())
sys.exit()
"""


def eps_gen (episode, episodes):
    return max(0.01, 1. - episode / (episodes * 0.8))

algos.uniform_memory_sample(
    agent, 
    batch_size=32,
    update_time=300, 
    episodes=300, 
    epsilon_schedule=eps_gen, 
    frames_to_skip=2
)

print('average performace =', torch.tensor(agent.logger.total_rewards[-100:]).mean())
print('best =', max(agent.logger.total_rewards))

agent.logger.save_rewards('atari_dqn')
torch.save(agent.Q.function.state_dict(), 'atari_qfunction')

env.close()

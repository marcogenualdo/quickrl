import sys
sys.path.append('../')

import quickrl
from quickrl import Agent
from quickrl.interactions import gym_screen_reset, gym_screen_step
from quickrl.value_functions import SparseLinear, TorchNet
from quickrl.policies import MaxValue
import quickrl.learning_algorithms as algos
from quickrl.memories import ListOfTuples, PriorityQueue
from quickrl.logger import Logger

import gym
import torch
from torch import nn
from torchvision import transforms as T


env = gym.make('CartPole-v1')

input_shape = (25, 50)

class Network (nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.conv_ops = [
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        ]
        self.dense = nn.Linear(32*3, 2)

    
    def forward (self, x):
        if len(x.shape) == 3: 
            x = x.unsqueeze(0)

        for op in self.conv_ops:
            x = op(x)
        x = x.view(-1, 32*3)
        x = self.dense(x)
        
        return x.squeeze()


# image processing setup
resize = T.Compose([
    T.ToPILImage(),
    T.Resize(input_shape),
    T.Grayscale(),
    T.ToTensor()
])

def process_image (img):
    img = 1 - img[168:320,:,:]
    return resize(img)

# trying what an image looks like as seen from the nn
from matplotlib import pyplot as plt
env.reset()
img = env.render(mode='rgb_array')
print(img.shape)
img = process_image(img).squeeze()
print(img)
print(img.shape)
plt.imshow(img)
plt.show()

# agent parts
net = Network()
vf = TorchNet('q', net)
policy = MaxValue([0,1], vf)
memory = ListOfTuples()

vf.loss_func = nn.SmoothL1Loss()
vf.optimizer = torch.optim.RMSprop(net.parameters())

agent = Agent(env,
              gym_screen_reset,
              gym_screen_step,
              gamma=0.95,
              action_value_function=vf,
              policy=policy,
              memory=memory,
              plot_frequency=50
             )
agent.process_state = process_image


def eps_gen (episode, episodes):
    return max(0.01, 1. - episode / (episodes - 100))

algos.n_sarsa(agent, 8, 500, eps_gen)

print('average performace =', torch.tensor(agent.logger.total_rewards[-100:]).mean())
print('best =', max(agent.logger.total_rewards))
env.close()

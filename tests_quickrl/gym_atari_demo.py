import random
from matplotlib import pyplot as plt
from time import sleep
import gym

env = gym.make('Breakout-v0')
print(env.action_space)


for episode in range(3):
    state = env.reset()
    done = False
    while not done:
        state, reward, done, info = env.step(random.randint(0,3))
        #plt.imshow(state)
        #plt.show()
        #sleep(0.5)
    print(done)
    state, reward, done, info = env.step(random.randint(0,3))
    plt.imshow(state)
    plt.show()
    sleep(2)
    state, reward, done, info = env.step(random.randint(0,3))
    plt.imshow(state)
    plt.show()
    sleep(2)


env.close()

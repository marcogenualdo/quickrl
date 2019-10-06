from matplotlib import pyplot as plt
import numpy as np
import pickle


class Logger:
    def __init__ (self, agent, plot_frequency=50):
        self.agent = agent

        self.plot_frequency = plot_frequency
        self.setup_plots()

        # metrics
        self.steps_taken = 0
        self.total_rewards = [] 


    def setup_plots(self):
        plt.ion()
        plt.show()

        self.fig = plt.figure()
        self.reward_plot = self.fig.add_subplot(111)


    def update_plots (self):
        n_episodes = self.agent.memory.current_episode
        episodes = np.arange(n_episodes)

        self.reward_plot.clear()
        self.reward_plot.plot(episodes, self.total_rewards)
       
        plt.draw()
        plt.pause(0.001)


    def save_rewards (self, filename):
        n_episodes = self.agent.memory.current_episode

        with open(filename, 'wb') as f:
            pickle.dump(self.total_rewards, f) 


    def reset (self):
        episode = self.agent.memory.current_episode
        if not (episode + 1) % self.plot_frequency:
            self.update_plots()

        if self.total_rewards: 
            print(f'Episode {episode}, Reward = {self.total_rewards[-1]}, Total steps taken = {self.steps_taken}')

        self.total_rewards.append(0.) 
        self.steps_taken += 1

    
    def step (self):
        self.total_rewards[-1] += self.agent.memory.get_last()[0].reward.item()
        self.steps_taken += 1

from random import randint
import torch


class RandomWalk:
    """The state space is a sequence of points. The number of said points is specified by 'cells'. At each time-step the player can move either left or right. No reward is given at any position but the last."""

    def __init__ (self, cells):
        self.cells = cells
        self.current_position = 0
        self.done = True


    def reset (self):
        self.current_position = randint(1, self.cells - 1)
        self.done = False
        return self.current_position


    def step (self, action):
        if not self.done:
            if action == 0:
                self.current_position -= 1
            elif action == 1:
                self.current_position += 1

            if self.current_position == self.cells:
                reward = 1.
            else:
                reward = 0.

            if self.current_position in {0, self.cells}:
                self.done = True

        else:
            reward = 0.

        return self.current_position, reward, self.done, {}


# FUNCTIONS TO USE FOR ENVIRONMENT INTERACTIONS

def random_walk_reset (agent, *args, **kwargs):
    state = agent.env.reset()
    state = torch.tensor([state])

    agent.memory.current_state = state
    agent.logger.reset()


def random_walk_step (agent, action, *args, **kwargs):
    state, reward, done, info = agent.env.step(action.item())
    state = torch.tensor([state])
    reward = torch.tensor([reward])

    agent.memory.remember(state, action, reward, done)
    agent.logger.step()
    return state, reward, done, info



if __name__ == '__main__':
    env = RandomWalk(10)
    print(env.reset())
  
    done = False
    while not done:
        state, reward, done, info = env.step(randint(0,1))
        print(state, reward, done)

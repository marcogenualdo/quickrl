import maze_generator as generator
import random

import numpy as np
from matplotlib import pyplot as plt


class Maze:
    action_space = {'l', 'r', 'u', 'd'}

    move_vector = {'l' : (-1,0), 'r' : (1,0), 'u' : (0,-1), 'd' : (0,1)}
    move_index = {'l' : 0, 'r' : 1, 'u' : 2, 'd' : 3}
    move_name = {v : k for k,v in move_index.items()}


    def __init__ (self, width, height, difficulty, seed = 0):
        self.width = width
        self.height = height
        self.victory = (width-1,0)

        self.obstacles = generator.generate_maze (width, height, difficulty = difficulty, seed = seed) 


    def draw (self):
        generator.repr_maze(self.width, self.height, self.obstacles)


    def reset (self):
        x, y = random.randint(0,self.width-1), random.randint(0,self.height-1)
        while (x,y) in self.obstacles or (x,y) == self.victory:
            x, y = random.randint(0,self.width-1), random.randint(0,self.height-1)

        self.position = (x,y)
        return self.position


    def step (self, action):
        x,y = self.position[0], self.position[1]
        vx, vy = Maze.move_vector[action]

        if vx and x + vx >= 0 and x + vx < self.width and (x+vx,y) not in self.obstacles:
            self.position = (x + vx, y)
        if vy and y + vy >= 0 and y + vy < self.height and (x,y+vy) not in self.obstacles:
            self.position = (x, y + vy) 

        rw, done = self.is_finished()
        return self.position, rw, done, {} # <- no info


    def is_finished (self):
        if self.position == self.victory:
            return 1, True
        return 0, False


def plot_policy (env, policy):
    """Plots a sketch of the maze and each move according to the policy using pyplot.quiver"""
    polx = [[0 for n in range(env.width)] for k in range(env.height)]
    poly = [[0 for n in range(env.width)] for k in range(env.height)]

    for i in range(env.height):
        for j in range(env.width):
            if (j,i) not in env.obstacles and (j,i) != env.victory and (j,i) in policy.favorites:
                (polx[i][j], poly[i][j]) = env.move_vector[policy.favorites[(j,i)]]

    polx[env.victory[1]][env.victory[0]], poly[env.victory[1]][env.victory[0]] = 1,1

    fig = plt.figure()

    px = np.arange (env.width)
    py = np.arange (env.height)
    X, Y = np.meshgrid (px,py)

    ax = fig.add_subplot(111)
    ax.quiver(X, Y, polx, poly)
    plt.show()
   
    # a less pretty terminal print
    #for i in range(maze_height):
    #    for j in range(maze_width):
    #        print(policy[j][i], end = ' ')
    #    print('\n')


if __name__ == '__main__':
    maze = Maze(10,5,20)
    print(maze.reset())
    print(maze.step('d'))
    print(maze) # <- doesn't work!

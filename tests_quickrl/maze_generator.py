import random

directions = {(-1,0) , (1,0) , (0,-1) , (0,1)}


def in_limits (pos, width, height):
    if pos[0] > 0 and pos[0] < width-1 and pos[1] > 0 and pos[1] < height-1:
        return True
    return False


def neighbours (cell, step):
    """Returns the six cells in front and by the side of one cell, depending on which direction it comes from"""
    S = set()
    for i in range(max(-1, step[1]-1), min(1, step[1]+1) + 1):
        for j in range(max(-1, step[0]-1), min(1, step[0]+1) + 1):
            S.add((cell[0] + j, cell[1] + i))
    return S


def generate_maze (width, height, difficulty, seed = None):
    branching_chance = 0.2

    if seed:
        random.seed(seed)

    last1 = (random.randint(1,width-2),0)
    last2 = (random.randint(1,width-2),height-1)

    obstacles = {last1, last2}

    for count in range(difficulty):
        dir1 = random.choice (list(directions))
        dir2 = random.choice (list(directions))

        new_obst = (last1[0] + dir1[0], last1[1] + dir1[1])
        if not (neighbours(new_obst, dir1) - {last1}) & obstacles and in_limits(new_obst, width, height): 
            obstacles.add(new_obst)
            last1 = new_obst
        
        new_obst = (last2[0] + dir2[0], last2[1] + dir2[1])
        if not (neighbours(new_obst, dir2) - {last2}) & obstacles and in_limits(new_obst, width, height): 
            obstacles.add(new_obst)
            last2 = new_obst

        if random.random() < branching_chance:
            last1 = random.choice(list(obstacles))            
        if random.random() < branching_chance:
            last2 = random.choice(list(obstacles))            
    
    return obstacles


def repr_maze (width, height, obstacles):
    for i in range(width+1):
        print('#', end = '')
    print('#\n#', end = '')
    for i in range(height):
        for j in range(width):
            if (j,i) in obstacles:
                print('#', end = '')
            else:
                print(' ', end = '')
        print('#\n#', end = '')
    for i in range(width+1):
        print('#', end = '')
    print('')
    print(len(obstacles))


if __name__ == '__main__':
    width, height = 50, 10
    obstacles = generate_maze(width,height, 1000)

    repr_maze (width, height, obstacles)

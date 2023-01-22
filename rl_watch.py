import pygame
import sys
import numpy as np
from movement import *
import haiku as hk
from save_funcs import restore
from model import MyModel
import jax

BLUE = (0, 0, 255)
RED = (255, 0, 0)
ORANGE = (255, 55, 0)
YELLOW = (255, 255, 0)
WINDOW_HEIGHT = 120
WINDOW_WIDTH = 120
BLOCKSIZE = 20
SCREEN = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
CLOCK = pygame.time.Clock()
EDGE = 10
REWARD = 11

model = hk.transform_with_state(MyModel)
params = restore("./params")
state = restore("./states")

def update_rect(pos, type):
    x = pos[1] * BLOCKSIZE
    y = pos[0] * BLOCKSIZE
    rect = pygame.Rect(x+1, y+1, BLOCKSIZE-2, BLOCKSIZE-2)
    if type == 0:
        pygame.draw.rect(SCREEN, BLUE, rect)
    elif type == 1:
        pygame.draw.rect(SCREEN, YELLOW, rect)
    elif type == 2:
        pygame.draw.rect(SCREEN, RED, rect)


def set_next_reward_pos(grid):
    indices = np.where(grid == 0)
    rand_ind = np.random.choice(np.arange(indices[0].shape[0]))
    pos = (indices[0][rand_ind], indices[1][rand_ind])
    grid[pos] = 11
    update_rect(pos, 2)
    return pos

def create_inititial_grid():
    grid = np.ones((WINDOW_HEIGHT//BLOCKSIZE, WINDOW_WIDTH//BLOCKSIZE)) * 10
    grid[1:-1,1:-1] = 0
    row = np.random.randint(low=1, high=WINDOW_HEIGHT//BLOCKSIZE-1)
    col = np.random.randint(low=1, high=WINDOW_HEIGHT//BLOCKSIZE-1)
    grid[row,col] = 1
    reward_pos = set_next_reward_pos(grid)
    return grid, reward_pos, row, col


def draw_initial_grid(grid):
    SCREEN.fill(BLUE)
    for x in range(0, WINDOW_WIDTH//BLOCKSIZE):
        for y in range(0, WINDOW_HEIGHT//BLOCKSIZE):
            x_pos = x * BLOCKSIZE
            y_pos = y * BLOCKSIZE
            if grid[y,x] == 10:
                rect = pygame.Rect(x_pos, y_pos, BLOCKSIZE, BLOCKSIZE)
                color = ORANGE
            elif grid[y,x] == 11:
                rect = pygame.Rect(x_pos+1, y_pos+1, BLOCKSIZE-2, BLOCKSIZE-2)
                color = RED
            elif grid[y,x] == 1:
                rect = pygame.Rect(x_pos+1, y_pos+1, BLOCKSIZE-2, BLOCKSIZE-2)
                color = YELLOW
            pygame.draw.rect(SCREEN, color, rect)

movement_dict = {1: move_up, 2: move_down, 3: move_left, 4: move_right}

def update_grid(grid, need_to_add, head_pos, reward_pos, tail_pos, direction, at_start):
    grid[head_pos] = direction+1
    original_head_pos = head_pos
    head_pos, restart, hit_reward = movement_dict[direction](grid, head_pos, True)
    if restart:
        grid, reward_pos, row, col = create_inititial_grid()
        draw_initial_grid(grid)
        return grid, 1, (row,col), reward_pos, (row,col), 0, 1
    else:
        update_rect(head_pos, 1)
        if at_start:
            grid[original_head_pos] += 4
        if hit_reward:
            need_to_add += 1
            reward_pos = set_next_reward_pos(grid)
        elif not need_to_add:
            val = int(grid[tail_pos]) - 5
            update_rect(tail_pos, 0)
            tail_pos, _, _ = movement_dict[val](grid, tail_pos, False)

    need_to_add = max(0, need_to_add-1)
    return grid, need_to_add, head_pos, reward_pos, tail_pos, direction, 0

def get_new_key(grid, need_to_add, direction, head_pos, reward_pos):
    preds, _ = model.apply(params, state, None, grid, need_to_add, direction, head_pos, reward_pos)
    preds = preds[0]
    rng = jax.random.PRNGKey(np.random.randint(low=-10000, high=10000))
    key = jax.random.choice(rng, 4, p=preds)
    return key

def main():
    pygame.init()
    
    run = True
    direction = 0
    grid, reward_pos, row, col = create_inititial_grid()
    draw_initial_grid(grid)
    need_to_add = 1
    head_pos = (row,col)
    tail_pos = (row,col)
    at_start = 1
    need_to_execute = 0
    keysup = [True, True, True, True]

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        if not direction:
            keysup = [True, True, True, True]

        if not need_to_execute:
            key = get_new_key(grid, need_to_add, direction, head_pos, reward_pos)
            if key == 0 and keysup[0] and direction != 2:
                direction = 1
                need_to_execute = 1
                keysup = [False, True, True, True]
            elif key == 1 and keysup[1] and direction != 1:
                direction = 2
                need_to_execute = 1
                keysup = [True, False, True, True]
            elif key == 2 and keysup[2] and direction != 4:
                direction = 3
                need_to_execute = 1
                keysup = [True, True, False, True]
            elif key == 3 and keysup[3] and direction != 3:
                direction = 4
                need_to_execute = 1
                keysup = [True, True, True, False]
        if direction:
            grid, need_to_add, head_pos, reward_pos, tail_pos, direction, at_start = update_grid(grid, need_to_add, head_pos, reward_pos, tail_pos, direction, at_start)
            need_to_execute = 0
            if at_start:
                run = True #False

        pygame.display.update()  
        CLOCK.tick(6)

    pygame.quit()
    sys.exit()

main()
import pygame
import sys
import numpy as np
from movement import *

BLUE = (0, 0, 255)
RED = (255, 0, 0)
ORANGE = (255, 55, 0)
YELLOW = (255, 255, 0)
WINDOW_HEIGHT = 320
WINDOW_WIDTH = 320
BLOCKSIZE = 20
SCREEN = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
CLOCK = pygame.time.Clock()
EDGE = 10
REWARD = 11

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

def create_inititial_grid():
    grid = np.ones((WINDOW_HEIGHT//BLOCKSIZE, WINDOW_WIDTH//BLOCKSIZE)) * 10
    grid[1:-1,1:-1] = 0
    grid[2,2] = 2
    set_next_reward_pos(grid)
    return grid


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
            elif grid[y,x] == 2:
                rect = pygame.Rect(x_pos+1, y_pos+1, BLOCKSIZE-2, BLOCKSIZE-2)
                color = YELLOW
            pygame.draw.rect(SCREEN, color, rect)

movement_dict = {1: move_up, 2: move_down, 3: move_left, 4: move_right}

def update_grid(grid, need_to_add, head_pos, tail_pos, direction, at_start):
    grid[head_pos] = direction+1
    head_pos, restart, hit_reward = movement_dict[direction](grid, head_pos, True)
    if restart:
        grid = create_inititial_grid()
        draw_initial_grid(grid)
        return grid, 3, (2,2), (2,2), 0, 1
    else:
        update_rect(head_pos, 1)
        if at_start:
            grid[(2,2)] += 4
        if hit_reward:
            need_to_add += 4
            set_next_reward_pos(grid)
        elif not need_to_add:
            val = int(grid[tail_pos]) - 5
            update_rect(tail_pos, 0)
            tail_pos, _, _ = movement_dict[val](grid, tail_pos, False)

    need_to_add = max(0, need_to_add-1)
    return grid, need_to_add, head_pos, tail_pos, direction, 0


def main():
    pygame.init()
    
    run = True
    direction = 0
    grid = create_inititial_grid()
    draw_initial_grid(grid)
    need_to_add = 3
    head_pos = (2,2)
    tail_pos = (2,2)
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
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP] and keysup[0] and direction != 2:
                direction = 1
                need_to_execute = 1
                keysup = [False, True, True, True]
            elif keys[pygame.K_DOWN] and keysup[1] and direction != 1:
                direction = 2
                need_to_execute = 1
                keysup = [True, False, True, True]
            elif keys[pygame.K_LEFT] and keysup[2] and direction != 4:
                direction = 3
                need_to_execute = 1
                keysup = [True, True, False, True]
            elif keys[pygame.K_RIGHT] and keysup[3] and direction != 3:
                direction = 4
                need_to_execute = 1
                keysup = [True, True, True, False]
        if direction:
            grid, need_to_add, head_pos, tail_pos, direction, at_start = update_grid(grid, need_to_add, head_pos, tail_pos, direction, at_start)
            need_to_execute = 0

        pygame.display.update()
        CLOCK.tick(10)

    pygame.quit()
    sys.exit()

main()
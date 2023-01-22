import pygame
import numpy as np
import time

BLUE = (0, 0, 255)
RED = (255, 0, 0)
ORANGE = (255, 55, 0)
YELLOW = (255, 255, 0)
WINDOW_HEIGHT = 120
WINDOW_WIDTH = 120
BLOCKSIZE = 20
SCREEN = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
CLOCK = pygame.time.Clock()

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
            elif grid[y,x] != 0:
                rect = pygame.Rect(x_pos+1, y_pos+1, BLOCKSIZE-2, BLOCKSIZE-2)
                color = YELLOW
            pygame.draw.rect(SCREEN, color, rect)

pygame.init()
frames = np.load("best_vid2.npy")
for i in range(frames.shape[0]):
    draw_initial_grid(frames[i])
    pygame.display.update()
    time.sleep(0.16)
pygame.quit()
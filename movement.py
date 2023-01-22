def move_up(grid, head_pos, is_head):
    new_pos = (head_pos[0]-1, head_pos[1])
    hit_reward = False
    if is_head:
        if grid[new_pos] != 0 and grid[new_pos] != 11:
            return new_pos, 1, hit_reward
        if grid[new_pos] == 11:
            hit_reward = True
        grid[head_pos] = 2
        grid[new_pos] = 1
    else:
        grid[head_pos] = 0
        grid[new_pos] += 4
    return new_pos, 0, hit_reward

def move_down(grid, head_pos, is_head):
    new_pos = (head_pos[0]+1, head_pos[1])
    hit_reward = False
    if is_head:
        if grid[new_pos] != 0 and grid[new_pos] != 11:
            return new_pos, 1, hit_reward
        if grid[new_pos] == 11:
            hit_reward = True
        grid[head_pos] = 3
        grid[new_pos] = 1
    else:
        grid[head_pos] = 0
        grid[new_pos] += 4
    return new_pos, 0, hit_reward

def move_left(grid, head_pos, is_head):
    new_pos = (head_pos[0], head_pos[1]-1)
    hit_reward = False
    if is_head:
        if grid[new_pos] != 0 and grid[new_pos] != 11:
            return new_pos, 1, hit_reward
        if grid[new_pos] == 11:
            hit_reward = True
        grid[head_pos] = 4
        grid[new_pos] = 1
    else:
        grid[head_pos] = 0
        grid[new_pos] += 4
    return new_pos, 0, hit_reward

def move_right(grid, head_pos, is_head):
    new_pos = (head_pos[0], head_pos[1]+1)
    hit_reward = False
    if is_head:
        if grid[new_pos] != 0 and grid[new_pos] != 11:
            return new_pos, 1, hit_reward
        if grid[new_pos] == 11:
            hit_reward = True
        grid[head_pos] = 5
        grid[new_pos] = 1
    else:
        grid[head_pos] = 0
        grid[new_pos] += 4
    return new_pos, 0, hit_reward
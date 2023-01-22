import os
import sys
import numpy as np
from movement import *
import pickle
import jax
import jax.numpy as jnp
import optax
import haiku as hk
from save_funcs import *
from model import MyModel

WINDOW_HEIGHT = 120 #320
WINDOW_WIDTH = 120 #320
BLOCKSIZE = 20
SCREEN = None
CLOCK = None
B = 16

def set_next_reward_pos(grid):
    indices = np.where(grid == 0)
    rand_ind = np.random.choice(np.arange(indices[0].shape[0]))
    pos = (indices[0][rand_ind], indices[1][rand_ind])
    grid[pos] = 11
    return pos

def create_inititial_grid():
    grid = np.ones((WINDOW_HEIGHT//BLOCKSIZE, WINDOW_WIDTH//BLOCKSIZE)) * 10
    grid[1:-1,1:-1] = 0
    row = np.random.randint(low=1, high=WINDOW_HEIGHT//BLOCKSIZE-1)
    col = np.random.randint(low=1, high=WINDOW_HEIGHT//BLOCKSIZE-1)
    grid[row,col] = 1
    reward_pos =  set_next_reward_pos(grid)
    return grid, reward_pos, row, col

movement_dict = {1: move_up, 2: move_down, 3: move_left, 4: move_right}

def calc_dist_from_reward(head_pos, reward_pos):
    return abs(head_pos[0] - reward_pos[0]) + abs(head_pos[1] - reward_pos[1])

def update_grid(grid, need_to_add, head_pos, tail_pos, reward_pos, direction, at_start):
    grid[head_pos] = direction+1
    original_dist = calc_dist_from_reward(head_pos, reward_pos)
    head_pos, restart, hit_reward = movement_dict[direction](grid, head_pos, True)
    if restart:
        grid, reward_pos, row, col = create_inititial_grid()
        return grid, 1, (row,col), reward_pos, (row,col), 0, 1, -1
    else:
        if at_start:
            grid[tail_pos] += 4
        if hit_reward:
            need_to_add += 1
            reward_pos = set_next_reward_pos(grid)
            hit_reward = 10
        elif calc_dist_from_reward(head_pos, reward_pos) < original_dist:
            hit_reward = -0.025 #0.1
        else:
            hit_reward = -0.1
        if not need_to_add:
            val = int(grid[tail_pos]) - 5
            tail_pos, _, _ = movement_dict[val](grid, tail_pos, False)

    need_to_add = max(0, need_to_add-1)
    return grid, need_to_add, head_pos, reward_pos, tail_pos, direction, 0, hit_reward


def train_model(episodes, model, params, state, opt_state, optimizer, gamma=0.99):
    def get_loss_and_preds(params, state, grid, need_to_add, direction, head_pos, reward_pos):
        preds, state = model.apply(params, state, None, grid, need_to_add, direction, head_pos, reward_pos)
        preds = preds[0]
        rng = jax.random.PRNGKey(np.random.randint(low=-10000, high=10000))
        key = jax.random.choice(rng, 4, p=preds)
        chosen_action = -jnp.log(preds[key])
        return chosen_action, (key, state)

    def train_step(params, state, opt_state):
        avg_fruit_eaten = 0
        avg_steps_taken = 0
        all_grads = []
        all_rewards = jnp.array([])
        for i in range(B):
            direction = 0
            grid, reward_pos, row, col = create_inititial_grid()
            need_to_add = 1
            head_pos = (row,col)
            tail_pos = (row,col)
            at_start = 1
            need_to_execute = 0
            keysup = [True, True, True, True]
            rewards = np.array([])
            
            run = True
            num_moves = 0
            while run and num_moves < 500:
                if not direction:
                    keysup = [True, True, True, True]

                if not need_to_execute:
                    grads, (key, state) = (jax.grad(get_loss_and_preds, has_aux=True)(params, state, grid, need_to_add, direction, head_pos, reward_pos))
                    all_grads.append(grads)
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
                    try:
                        grid, need_to_add, head_pos, reward_pos, tail_pos, direction, at_start, hit_reward = update_grid(grid, need_to_add, head_pos, tail_pos, reward_pos, direction, at_start)
                    except ValueError:
                        hit_reward = 10
                        at_start = 1
                        pass
                    if hit_reward == 10:
                        avg_fruit_eaten += 1
                    rewards = jnp.concatenate([rewards, jnp.array([hit_reward])])
                    need_to_execute = 0
                    num_moves += 1
                    if at_start:
                        run = False

            total_time = rewards.shape[0]
            helper = gamma ** np.arange(total_time)
            for i in range(total_time):
                rewards = rewards.at[i].set(jnp.sum(helper * rewards[i:]))
                helper = helper[:-1]
            rewards -= jnp.mean(rewards)
            rewards /= (jnp.std(rewards) + 1e-5)
            rewards /= B
            all_rewards = jnp.concatenate([all_rewards, rewards])
            avg_steps_taken += len(rewards)

        for i in range(len(all_grads)):
            grad = all_grads[i]
            reward = all_rewards[i]
            for key in grad:
                if type(grad[key]) == dict:
                    for subkey in grad[key]:
                        grad[key][subkey] *= reward
                else:
                    grad[key] *= reward

            updates, opt_state = optimizer.update(grad, opt_state)
            params = optax.apply_updates(params, updates)
        return params, state, opt_state, avg_fruit_eaten/B, avg_steps_taken/B
              
    total_avg_fruit_eaten = 0
    total_avg_steps_take = 0
    for i in range(1, episodes+1):
        params, state, opt_state, avg_fruit_eaten, avg_steps_taken = train_step(params, state, opt_state)
        total_avg_fruit_eaten += avg_fruit_eaten
        total_avg_steps_take += avg_steps_taken
        if not i % 5:
            print("avg fruits:", total_avg_fruit_eaten / 5)
            print("avg steps taken:", total_avg_steps_take / 5)
            total_avg_fruit_eaten = 0
            total_avg_steps_take = 0
        if not i % 20:
            save("./params", jax.device_get(params))
            save("./states", jax.device_get(state))
            with open(os.path.join("./opt_states", "opt_state.pkl"), "wb") as output_file:
                pickle.dump(jax.device_get(opt_state), output_file)
            print("saved!")
        

    return params, state, opt_state

lr = 5e-6
episodes = 10000

optimizer = optax.adam(lr)

#rng = jax.random.PRNGKey(42)
model = hk.transform_with_state(MyModel)
#dummy_grid, _, row, col = create_inititial_grid()
#params, state = model.init(rng, dummy_grid, 1, 0, (1,1), (1,2))
params = restore("./params")
state = restore("./states")
#opt_state = optimizer.init(params)
with open("./opt_states/opt_state.pkl", "rb") as input_file:
  opt_state = pickle.load(input_file)
params, state, opt_state = train_model(episodes, model, params, state, opt_state, optimizer)
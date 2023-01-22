from layers import *
import numpy as np
import haiku as hk

"""
def MyModel(grid, need_to_add, direction, head_pos, reward_pos):
  # These are all the layers used in the model
  linear_layers = hk.Sequential([LinearReluBlock(32) for i in range(3)])
  final_layer = LinearSoftmaxBlock()

  def calc_probs(state_vector):
      return final_layer(linear_layers(state_vector))
  state_vector = np.zeros((12))
  if direction:
      state_vector[direction + 3] = 1
  if grid[(head_pos[0]-1, head_pos[1])] != 0 and grid[(head_pos[0]-1, head_pos[1])] != 11:
      state_vector[8] = 1
  if grid[(head_pos[0]+1, head_pos[1])] != 0 and grid[(head_pos[0]+1, head_pos[1])] != 11:
      state_vector[9] = 1
  if grid[(head_pos[0], head_pos[1]-1)] != 0 and grid[(head_pos[0], head_pos[1]-1)] != 11:
      state_vector[10] = 1
  if grid[(head_pos[0], head_pos[1]+1)] != 0 and grid[(head_pos[0], head_pos[1]+1)] != 11:
      state_vector[11] = 1

  if head_pos[0] < reward_pos[0]:
      state_vector[1] = 1
  elif head_pos[0] > reward_pos[0]:
      state_vector[0] = 1
  if head_pos[1] < reward_pos[1]:
      state_vector[3] = 1
  elif head_pos[1] > reward_pos[1]:
      state_vector[2] = 1
  state_vector = np.reshape(state_vector, (1, -1))

  return calc_probs(state_vector)


"""
def MyModel(grid, need_to_add, direction, head_pos, reward_pos):
  # These are all the layers used in the model
  conv_layers = hk.Sequential([MyConvBlock(32) for _ in range(3)])
  pooling_layer = MaxPoolAndFlatten()
  final_layer = LinearSoftmaxBlock()

  def calc_probs(transformed_grid):
      return final_layer(pooling_layer(conv_layers(transformed_grid)))

  grid = grid / 11
  grid = np.reshape(grid, (1, grid.shape[0], grid.shape[1], 1))
  
  return calc_probs(grid)
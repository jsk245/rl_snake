import haiku as hk
import jax

class MyConvBlock(hk.Module):
  def __init__(self, num_channels, name=None):
    super(MyConvBlock, self).__init__(name=name)
    self.conv = hk.Conv2D(num_channels, 3)
    self.ln = hk.LayerNorm(axis=-1, param_axis=-1, 
    create_scale=True, create_offset=True)

  def __call__(self, x):
    return jax.nn.relu(self.ln(self.conv(x)))

class MaxPoolAndFlatten(hk.Module):
  def __init__(self, name=None):
    super(MaxPoolAndFlatten, self).__init__(name=name)
    self.flat = hk.Flatten()

  def __call__(self, x):
    x = hk.max_pool(x, 2, 2, "VALID")
    return self.flat(x)

class LinearReluBlock(hk.Module):
  def __init__(self, num_features, name=None):
    super(LinearReluBlock, self).__init__(name=name)
    self.linear_layer = hk.Linear(num_features)

  def __call__(self, x):
    return jax.nn.relu(self.linear_layer(x))

class LinearSoftmaxBlock(hk.Module):
  def __init__(self, name=None):
    super(LinearSoftmaxBlock, self).__init__(name=name)
    self.linear_layer = hk.Linear(4)

  def __call__(self, x):
    return jax.nn.softmax(self.linear_layer(x))
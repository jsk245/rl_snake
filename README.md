# Reinforcement Learning Snake
## Game Description
Snake is a one-player game where you control a "snake" using the arrow keys. The goal is to pick up "fruits", which increase the length of the snake. In this repo, I include
a playable version of the game as well as a way to train a model to play snake using reinforcement learning (on a smaller board). Video results from training the model
for 18 hours using Google Colab CPUs are provided below in the "Results" section. All code is written using Python with the NumPy, Pygame, JAX, Optax, and Haiku libraries.
Use `python play.py`to play the game on your own!

## Reinforcement Learning Policy Gradient Details
My model uses the policy gradient reinforcement learning technique, meaning that I optimize the policy that the model follows for how to choose moves. 
In order to train my model, I used the following reward scheme:
- +10 points for eating a fruit (later, I introduced a 50 point reward for eating )
- -1 point for dying
- +.1 points for moving towards the fruit (later turned down to -.025 to stop the snake from looping)
- -.1 points for moving away from the fruit


For training the model, I would let the model play 16 games at a time with a 500 turn limit (I don't think this limit was actually used). During each game, I would use
my model to calculate probabilities for moving in each direction (up, down, left, right). I would then randomly choose a direction using these probabilities, and I would
take the gradient of the -log of the probability for the move chosen as described in the REINFORCE algorithm (watch video in "Sources" section below for more info). 
I would record these gradients at each step along with the reward received. After finishing a game, for each step taken, I would scale the corresponding gradient by a weighted sum of all future rewards and use this to update the parameters with gradient descent.

## Reinforcement Learning Implentation Details
The information given to my model was a 6x6 grid of the game instead of the raw pixel values of the screen. This was done in an attempt to promote faster convergence, 
with the hope being that I would require a smaller model to train. On the grid, I marked each square based on whether the position corresponded to:
- a wall
- the snake's head
- the snake's tail + direction of motion
- the snake's body + direction of motion
- the reward
- empty space

I then passed this grid through 3 convolutional layers, each with 3x3 kernels, strides of 1, padding, and 32 channels. Each convolution was followed by LayerNorms (for 
smoother gradients) and ReLU activations. After this, I used max pooling with 2x2 kernels, strides of 2, and no padding to decrease the size of the output of the 
convolutional layers. I then flattened the output and used a simply linear layer that outputted 4 numbers, which I transformed into probabilities using the Softmax
function, which were used as the probabilities for moving in each direction.

Here are some miscellaneous implentation details:
- I started with a learning rate of 5e-5 and eventually decreased it to 5e-6
- I used batches of size 16
- I didn't use a set number of episodes since I just let my model train until I hit Colab runtime limits
- I used the Adam optimizer

## Results
Here is about 30-60 seconds of footage from an untrained model (top), a partially trained model (middle), and my fully trained model (bottom).

<img src="https://user-images.githubusercontent.com/93054906/213898733-bf6ed8f2-34cd-4c7c-a8c1-d2b8932d0106.gif" width="150" height="150">

<img src="https://user-images.githubusercontent.com/93054906/213898730-724605fa-d5f2-4547-8cbd-df0e2d9448c8.gif" width="150" height="150">

<img src="https://user-images.githubusercontent.com/93054906/213898732-c477b770-536d-4b6a-88a2-0d93a3bd978e.gif" width="150" height="150">

And here are some of the best runs of the model. The top was from before I penalized the model for all steps (instead of rewarding simply approaching the fruit), and the
bottom is from after this adjustment.

<img src="https://user-images.githubusercontent.com/93054906/213898734-f2dc8139-7aee-4f56-8895-acc42384dea1.gif" width="272" height="153">

<img src="https://user-images.githubusercontent.com/93054906/213898739-2059d754-3879-4069-8db6-5c4817ceb585.gif" width="272" height="153">


## Sources
For the game:
- https://www.101computing.net/pong-tutorial-using-pygame-getting-started/ (for how to generally implement a game using Pygame)
- https://patorjk.com/games/snake/ (for the design)

For the deep reinforcement learning:
- https://karpathy.github.io/2016/05/31/rl/ (for learning how policy gradients work)
- https://www.youtube.com/watch?v=Qex3XzcFKP4&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r&index=21 (for learning more about the REINFORCE algorithm)
- https://towardsdatascience.com/snake-with-policy-gradients-deep-reinforcement-learning-5e6e921db054 (for reward shaping)

import logging
from enum import Enum, auto

import matplotlib.pyplot as plt
import numpy as np

import models
from environment.maze import Maze, Render
from models.ppo import PPOModel

logging.basicConfig(format="%(levelname)-8s: %(asctime)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)  # Only show messages *equal to or above* this level


class Test(Enum):
    SHOW_MAZE_ONLY = auto()
    PPO_TRAINING = auto()
    PPO_LOAD = auto()


test = Test.PPO_TRAINING  # which test to run

maze = np.array([
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 1, 1, 0, 1, 0],
    [0, 1, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 1, 1, 1],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0]
])  # 0 = free, 1 = occupied

game = Maze(maze)

# only show the maze
if test == Test.SHOW_MAZE_ONLY:
    game.render(Render.MOVES)
    game.reset()

# train using PPO (Proximal Policy Optimization)
if test == Test.PPO_TRAINING:
    game.render(Render.TRAINING)
    model = PPOModel(game)
    h, w, _, _ = model.train(discount=0.99, episodes=500, update_frequency=5, stop_at_convergence=True)

# load a previously trained PPO model
if test == Test.PPO_LOAD:
    model = PPOModel(game, load=True)

# draw graphs showing development of win rate and cumulative rewards
try:
    h  # force a NameError exception if h does not exist, and thus don't try to show win rate and cumulative reward
    fig, (ax1, ax2) = plt.subplots(2, 1, tight_layout=True)
    fig.suptitle(model.name)
    ax1.plot(*zip(*w))
    ax1.set_xlabel("episode")
    ax1.set_ylabel("win rate")
    ax2.plot(h)
    ax2.set_xlabel("episode")
    ax2.set_ylabel("cumulative reward")
    plt.show()
except NameError:
    pass

# Test the trained model
game.render(Render.MOVES)
game.play(model, start_cell=(4, 1))

plt.show()  # must be placed here else the image disappears immediately at the end of the program
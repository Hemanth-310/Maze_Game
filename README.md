# Reinforcement Learning Maze Solver

A comprehensive implementation of various reinforcement learning algorithms for solving maze navigation problems, including traditional value-based methods and modern policy gradient approaches.

## 🎯 Problem Overview

The project presents a maze navigation challenge where an agent must learn to find the optimal path from a starting position to an exit. The agent receives rewards and penalties based on its actions, creating a Markov Decision Process (MDP) that can be solved using different RL algorithms.

![Maze](https://github.com/Hemanth-310/Maze_Game/blob/main/maze.png)

## 🏆 Reward Structure

The environment defines specific rewards and penalties to guide the learning process:

```python
reward_exit = 10.0          # Reward for reaching the exit
penalty_move = -0.05        # Small penalty for each move
penalty_visited = -0.25     # Penalty for revisiting cells
penalty_impossible_move = -0.75  # Large penalty for invalid moves
```

## 🧠 Available Algorithms

### Value-Based Methods

| Algorithm | Description | Strengths |
|-----------|-------------|-----------|
| **Q-Learning** | Off-policy temporal difference learning | Fast convergence, simple implementation, tabular approach |

### Policy Gradient Methods

| Algorithm | Description | Key Features |
|-----------|-------------|--------------|
| **PPO (Proximal Policy Optimization)** | State-of-the-art policy gradient method | Actor-critic architecture, clipped surrogate loss, stable training |

![Policy Visualization](https://github.com/Hemanth-310/Maze_Game/blob/main/bestmove.png)

## 🚀 Quick Start

### Prerequisites
```bash
pip install matplotlib numpy tensorflow keras
```

### Running the Algorithms

#### Q-Learning (Value-Based)
```bash
python main.py
```
Available options in `main.py`:
- `Test.SHOW_MAZE_ONLY` - Just display the maze
- `Test.RANDOM_MODEL` - Test with random actions
- `Test.Q_LEARNING` - Train Q-Learning model (default)

#### PPO (Policy Gradient)
```bash
python ppo_main.py
```
Available options in `ppo_main.py`:
- `Test.SHOW_MAZE_ONLY` - Just display the maze
- `Test.PPO_TRAINING` - Train PPO model (default)
- `Test.PPO_LOAD` - Load saved PPO model

## 📊 Performance Benchmarks

| Algorithm | Episodes to Converge | Training Time | Convergence Rate |
|-----------|---------------------|---------------|------------------|
| **Q-Learning** | **149.5** | **16.5 sec** | **High** |
| **PPO** | **~100-150** | **~2-3 min** | **High** |

## 🏗️ Project Structure

```
├── environment/
│   ├── __init__.py
│   └── maze.py              # Maze environment implementation
├── models/
│   ├── __init__.py
│   ├── abstractmodel.py     # Base model interface
│   ├── qtable.py           # Q-Learning implementation
│   └── ppo.py              # PPO implementation
├── main.py                 # Q-Learning runner
├── ppo_main.py            # PPO runner
└── README.md
```

## 🎮 Features

- **Visual Training**: Real-time visualization of agent movement and policy
- **Two Algorithm Types**: Value-based (Q-Learning) and Policy Gradient (PPO)
- **Modular Design**: Easy to extend with new algorithms
- **Performance Tracking**: Detailed metrics and convergence analysis
- **Model Persistence**: Save and load trained models (PPO)
- **Interactive Testing**: Test trained models from different starting positions

## 🔬 Algorithm Details

### PPO (Proximal Policy Optimization)
The PPO implementation features:
- **Actor-Critic Architecture**: Separate policy and value networks
- **Clipped Surrogate Objective**: Prevents destructive policy updates
- **Generalized Advantage Estimation**: Improved advantage calculation
- **Entropy Regularization**: Maintains exploration during training
- **Batch Training**: Efficient use of collected experience

### Q-Learning
- **Tabular Approach**: Fast convergence for small state spaces
- **Off-Policy Learning**: Learns optimal policy while following exploratory policy
- **Temporal Difference**: Updates values based on immediate rewards and future estimates

## 📈 Usage Examples

### Training a Q-Learning Agent
```python
from models import QTableModel
from environment.maze import Maze

game = Maze(maze_array)
model = QTableModel(game)
rewards, win_rates, episodes, time = model.train(episodes=200)
```

### Training a PPO Agent
```python
from models.ppo import PPOModel

model = PPOModel(game)
rewards, win_rates, episodes, time = model.train(
    episodes=500, 
    update_frequency=5,
    stop_at_convergence=True
)
```

## 🤝 Contributing

This project serves as an educational resource for understanding different reinforcement learning approaches. Feel free to:
- Add new algorithms
- Improve existing implementations
- Enhance visualization features
- Optimize performance

## 📚 Educational Value

Perfect for learning:
- Reinforcement learning fundamentals
- Algorithm comparison and benchmarking
- Policy gradient methods
- Neural network applications in RL
- Environment design and reward shaping

# RL-powered-Snake-Game
This project demonstrates the power of RL algorithms in a fun, interactive way. Check it out!
# RL-Powered Snake Game 🐍

Welcome to the **Reinforcement Learning-powered Snake Game**! This project showcases a classic Snake game built using **Python** and **PyQt6**, where multiple reinforcement learning (RL) agents, such as **Q-Learning**, **SARSA**, and **Expected SARSA**, are trained to play the game autonomously.

## Key Features 🚀

- **Reinforcement Learning Algorithms**: Implementations of Q-Learning, SARSA, and Expected SARSA.
- **Customizable Settings**: Adjust game speed, grid size, and RL algorithm parameters.
- **Real-Time Monitoring**: Visualizations of Q-values and agent performance.
- **Interactive GUI**: Built with PyQt6, offering an intuitive interface to observe the RL agents' learning process.

## Tech Stack 🛠️

- **Python**: Core programming language.
- **PyQt6**: Used for building the graphical user interface.
- **Numpy**: For efficient computation.
- **Matplotlib**: For visualizing Q-values and rewards.

## How It Works 🤖

The game board is a grid where the snake must navigate and collect rewards (food) while avoiding walls and self-collision. RL agents learn to play the game by receiving rewards for eating food and penalties for hitting walls or colliding with themselves. Through trial and error, the agents gradually improve their performance using different RL strategies.

### RL Agents

- **Q-Learning**: An off-policy RL algorithm where the agent learns the optimal policy by exploring the environment.
- **SARSA**: An on-policy RL algorithm that learns the action-value function based on the current policy.
- **Expected SARSA**: A more advanced variant of SARSA that uses expected values of future rewards.

## Installation ⚙️

To run the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Sherin-SEF-AI/RL-powered-Snake-Game.git
   cd RL-powered-Snake-Game
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

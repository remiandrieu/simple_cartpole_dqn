# DQN Agent for CartPole environment

A simple implementation of a DQN Agent that helped me understand the basics of reinforcement learning.
I documented some of the lines and functions from what I learned.

### How to use:

```shell
# Creating a virtual environment
python -m venv dqn-venv

# Activating the virtual environment
.\dqn-venv\Scripts\activate  # Windows
source dqn-venv/bin/activate # Linux / MacOS

# Installing dependencies
pip install -r requirements.txt

# Starting the learning process
./dqn-venv/Scripts/python.exe ./main.py

# Visualizing the agent's progess
./dqn-venv/Scripts/python.exe ./visualize.py ./checkpoint_20.keras
./dqn-venv/Scripts/python.exe ./visualize.py ./checkpoint_40.keras
...
./dqn-venv/Scripts/python.exe ./visualize.py ./cartpole_solved.keras
```
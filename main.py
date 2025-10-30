import gymnasium as gym
import numpy as np
from DQNAgent import DQNAgent

# Initialize environment
env = gym.make("CartPole-v1")

# Initialize agent
# Neural network structure :
# - 1 input layer (state_size -> 128)
# - 2 hidden layers (128 -> 128)
# - 1 output layer (128 -> action_size)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
hidden_layers_sizes = [128,128]
agent = DQNAgent(state_size, hidden_layers_sizes, action_size)

# Remember past rewards in order to know when to stop
average_rewards=[]

# Number of episodes the agent is training
total_episodes = 500

for episodes in range(total_episodes):
    # Reset the environment and the reward
    # In CartPole, the state is a 4-dimensional vector where
    # state[0] <- cart position
    # state[1] <- cart velocity
    # state[2] <- pole angle
    # state[3] <- pole angular velocity
    current_state, _ = env.reset()
    total_reward = 0

    while True:
        # The agent choose an action (in Cartpole : 0 corresponds to pushing the cart to the left, and 1 to the right)
        action = agent.act(current_state)
        
        # The action is performed
        # next_state <- the state reached after performing the action
        # reward <- the reward obtained by performing the action (in CartPole, the reward is +1 at each step)
        # done <- if the state is a final state (more explainations line 55)
        next_state, reward, done, _, _  = env.step(action)
        total_reward+=reward
        
        # Save the experience in the agent's memory
        # An experience is "In this state, the agent obtained this reward by performing this action and reached this next state which is final or not"
        agent.remember([current_state, reward, action, next_state, done])
        current_state = next_state
        
        # When the agent's memory is filled enough, it can start learning from its previous experiences
        if len(agent.memory) >= agent.batch_size:
            agent.learn()
        
        # If the state is final, save the total reward earned in this episode and go to the next one
        # In CartPole, a state is final if
        # - the pole angle is greater than ±12° (0° corresponds to the pole perfectly vertical)
        # - the cart position is greater than ±2.4 (0 corresponds to the center of the display)
        # - the episode length is greater than 500
        if done:
            average_rewards.append(total_reward)
            break
    
    # Decrease agent's epsilon
    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay
    
    # Only keep the last hundred total rewards
    if len(average_rewards) > 100:
        del average_rewards[0]
    
    # Display agent's performance every 10 episodes 
    if episodes % 10 == 0:
        print(f"Episode: {episodes} | Reward: {total_reward:.0f} | Average: {np.mean(average_rewards):.2f} | Epsilon: {agent.epsilon:.3f}")
    
    # Save agent's model every 100 episodes 
    if episodes % 100 == 0 and episodes > 0:
        agent.save(f'checkpoint_{episodes}.keras')
    
    # After a while, if this agent's performance is good enough, consider the environment solved and save the model
    if len(average_rewards) >= 100 and np.mean(average_rewards) > 200:
        print(f'Solved after {episodes} episodes')
        agent.save('cartpole_solved.keras')
        break

print(f'Training completed after {episodes} episodes')
print(f'Final average reward: {np.mean(average_rewards):.2f}')
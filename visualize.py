import gymnasium as gym
import numpy as np
from DQNAgent import DQNAgent
import time

def visualize_agent(model_path, num_episodes=5, delay=0.02):
    env = gym.make("CartPole-v1", render_mode="human")
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    hidden_layers_sizes = [128, 128]
    agent = DQNAgent(state_size, hidden_layers_sizes, action_size)
    agent.load(model_path)
    agent.epsilon = 0.0
    
    total_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            if total_reward > 200 :
                done = True
            steps += 1
            state = next_state
            
            time.sleep(delay)
            
            if done:
                total_rewards.append(total_reward)
                print(f"Épisode {episode + 1}/{num_episodes} - Reward: {total_reward:.0f} - Steps: {steps}")
                break
    
    env.close()
    
    print(f"Average reward: {np.mean(total_rewards):.2f}")
    print(f"Reward min: {np.min(total_rewards):.0f}")
    print(f"Reward max: {np.max(total_rewards):.0f}")
    print(f"Standard deviation: {np.std(total_rewards):.2f}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        visualize_agent(model_path)
    else:
        print("How to use:")
        print(f"  python visualize.py [chemin_modele.keras]")
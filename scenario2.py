from FourRooms import FourRooms
import numpy as np
import random
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt

class QLearningAgent:
    def __init__(self, stochastic=False):
        self.stochastic = stochastic
        self.alpha = 0.15  # Increased learning rate
        self.gamma = 0.95   # Adjusted discount factor
        self.epsilon = 1.0
        self.epsilon_decay = 0.997
        self.min_epsilon = 0.05
        self.temp = 5.0
        self.temp_decay = 0.99
        self.min_temp = 0.1
        self.Q = defaultdict(lambda: np.zeros(4))

    def get_state_key(self, position, packages_left):
        return (*position, packages_left)

    def epsilon_greedy(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        return np.argmax(self.Q[state])

    def softmax(self, state):
        q_vals = self.Q[state] - np.max(self.Q[state])
        exp_vals = np.exp(q_vals / self.temp)
        return np.random.choice(4, p=exp_vals/exp_vals.sum())

    def update_parameters(self, strategy):
        if strategy == 'epsilon_greedy':
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        elif strategy == 'softmax':
            self.temp = max(self.min_temp, self.temp * self.temp_decay)

    def update_q_value(self, state, action, reward, next_state):
        current_q = self.Q[state][action]
        max_next_q = np.max(self.Q[next_state])
        self.Q[state][action] = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)

def train_agent(agent, strategy, num_episodes=2000, max_steps=400):
    env = FourRooms('multi', stochastic=agent.stochastic)
    episode_rewards = []
    
    for episode in range(num_episodes):
        env.newEpoch()
        position = env.getPosition()
        packages_left = env.getPackagesRemaining()
        state = agent.get_state_key(position, packages_left)
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < max_steps:
            if strategy == 'epsilon_greedy':
                action = agent.epsilon_greedy(state)
            else:
                action = agent.softmax(state)
            
            cell_type, new_pos, new_packages, done = env.takeAction(action)
            reward = 10 * (packages_left - new_packages) - 1  # Reward per package collected
            next_state = agent.get_state_key(new_pos, new_packages)
            
            agent.update_q_value(state, action, reward, next_state)
            
            total_reward += reward
            state = next_state
            packages_left = new_packages
            steps += 1
        
        agent.update_parameters(strategy)
        episode_rewards.append(total_reward)
        
        if (episode+1) % 200 == 0:
            avg_reward = np.mean(episode_rewards[-200:])
            print(f"Ep {episode+1:4d} | {strategy:12s} | Avg Reward: {avg_reward:6.1f}")

    return episode_rewards

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-stochastic', action='store_true')
    args = parser.parse_args()
    
    strategies = ['epsilon_greedy', 'softmax']
    results = {}
    
    for strategy in strategies:
        print(f"\n=== Training {strategy.upper()} Strategy ===")
        agent = QLearningAgent(stochastic=args.stochastic)
        results[strategy] = train_agent(agent, strategy)
    
    plt.figure(figsize=(12, 7))
    for strategy in strategies:
        smoothed = np.convolve(results[strategy], np.ones(200)/200, mode='valid')
        plt.plot(smoothed, label=f"{strategy} Strategy")
    
    plt.title("Scenario 2 Learning Progress (200-Episode MA)", fontsize=14)
    plt.xlabel("Training Episodes", fontsize=12)
    plt.ylabel("Average Total Reward", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('scenario2_learning_curve.png', dpi=120)
    
    env = FourRooms('multi', stochastic=args.stochastic)
    env.showPath(-1, savefig='scenario2_final_path.png')

if __name__ == "__main__":
    main()

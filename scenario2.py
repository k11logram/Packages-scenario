from FourRooms import FourRooms
import numpy as np
import random
import argparse
from collections import defaultdict, deque
import matplotlib.pyplot as plt

class QLearningAgent:
    def __init__(self, stochastic=False):
        self.stochastic = stochastic
        self.alpha = 0.3  # Learning rate
        self.gamma = 0.90  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.998
        self.min_epsilon = 0.1
        self.temp = 2.0  # Softmax temperature
        self.temp_decay = 0.998
        self.min_temp = 0.5
        self.Q = defaultdict(lambda: np.zeros(4))
        self.replay_buffer = deque(maxlen=1000)
        self.batch_size = 32

    def get_state(self, pos, pkgs):
        """State includes (x, y) position and packages remaining"""
        return (pos[0], pos[1], pkgs)

    def calculate_reward(self, old_pos, new_pos, old_pkgs, new_pkgs):
        """Custom reward function for multi-package scenario"""
        if new_pkgs < old_pkgs:
            return 100  # Package collected
        elif new_pos != old_pos:
            return -0.5  # Movement penalty
        else:
            return -2  # Staying penalty

    def epsilon_greedy(self, state):
        if random.random() < self.epsilon:
            return random.choice([FourRooms.UP, FourRooms.DOWN, FourRooms.LEFT, FourRooms.RIGHT])
        return np.argmax(self.Q[state])

    def softmax(self, state):
        q_vals = self.Q[state]
        exp_vals = np.exp((q_vals - np.max(q_vals)) / max(self.temp, 1e-8))
        probs = exp_vals / np.sum(exp_vals)
        return np.random.choice(4, p=probs)

    def update_parameters(self, strategy):
        if strategy == 'epsilon_greedy':
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        elif strategy == 'softmax':
            self.temp = max(self.min_temp, self.temp * self.temp_decay)

    def update_q(self, state, action, reward, next_state):
        self.replay_buffer.append((state, action, reward, next_state))
        if len(self.replay_buffer) >= self.batch_size:
            batch = random.sample(self.replay_buffer, self.batch_size)
            for s, a, r, ns in batch:
                max_next_q = np.max(self.Q[ns]) if ns else 0
                self.Q[s][a] += self.alpha * (r + self.gamma * max_next_q - self.Q[s][a])

def train(agent, strategy, num_episodes=100000, max_steps=500):
    env = FourRooms('multi', stochastic=agent.stochastic)
    rewards = []
    successes = 0
    
    for episode in range(num_episodes):
        env.newEpoch()
        pos = env.getPosition()
        pkgs = env.getPackagesRemaining()
        state = agent.get_state(pos, pkgs)
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < max_steps:
            action = agent.epsilon_greedy(state) if strategy == 'epsilon_greedy' else agent.softmax(state)
            old_pos = pos
            old_pkgs = pkgs
            
            try:
                _, pos, pkgs, done = env.takeAction(action)
            except Exception as e:
                break
            
            reward = agent.calculate_reward(old_pos, pos, old_pkgs, pkgs)
            next_state = agent.get_state(pos, pkgs)
            agent.update_q(state, action, reward, next_state)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done and pkgs == 0:
                successes += 1
                
        agent.update_parameters(strategy)
        rewards.append(total_reward)
        
        if (episode + 1) % 1000 == 0:
            avg_reward = np.mean(rewards[-1000:])
            success_rate = (successes / 1000) * 100
            print(f"Ep {episode+1:6d} | {strategy:12s} | Avg Reward: {avg_reward:6.1f} | Success: {success_rate:5.1f}%")
            successes = 0
            
    return rewards, agent

def visualize_path(agent, strategy, stochastic=False):
    env = FourRooms('multi', stochastic=stochastic)
    env.newEpoch()
    pos = env.getPosition()
    pkgs = env.getPackagesRemaining()
    state = agent.get_state(pos, pkgs)
    done = False
    steps = 0
    
    while not done and steps < 500:
        action = agent.epsilon_greedy(state) if strategy == 'epsilon_greedy' else agent.softmax(state)
        _, pos, pkgs, done = env.takeAction(action)
        state = agent.get_state(pos, pkgs)
        steps += 1
    
    env.showPath(-1, savefig=f'scenario2_{strategy}_path.png')
    print(f"Saved path visualization: scenario2_{strategy}_path.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-stochastic', action='store_true', help='Enable stochastic actions')
    args = parser.parse_args()
    
    strategies = ['epsilon_greedy', 'softmax']
    results = {}
    
    for strategy in strategies:
        print(f"\n=== Training {strategy.upper()} Strategy ===")
        agent = QLearningAgent(stochastic=args.stochastic)
        rewards, trained_agent = train(agent, strategy)
        results[strategy] = rewards
        visualize_path(trained_agent, strategy, args.stochastic)
    
    # Plot learning curves
    plt.figure(figsize=(12, 6))
    for strategy, rewards in results.items():
        smoothed = np.convolve(rewards, np.ones(100)/100, mode='valid')
        plt.plot(smoothed, label=strategy)
    
    plt.xlabel("Training Episodes")
    plt.ylabel("Average Reward (100-episode MA)")
    plt.title("Scenario 2: Multi-Package Collection Learning Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('scenario2_learning_curves.png', dpi=120)
    print("\nSaved learning curve: scenario2_learning_curves.png")

if __name__ == "__main__":
    main()

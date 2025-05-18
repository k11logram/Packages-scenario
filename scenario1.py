from FourRooms import FourRooms
import numpy as np
import random
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt

class QLearningAgent:
    def __init__(self, stochastic=False):
        self.stochastic = stochastic
        self.alpha = 0.1
        self.gamma = 0.99
        # Epsilon-greedy parameters
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01
        # Softmax parameters
        self.temp = 2.0
        self.temp_decay = 0.995
        self.min_temp = 0.1
        # Q-table: state (x, y, packages_left) -> 4 actions
        self.Q = defaultdict(lambda: np.zeros(4))

    def get_state(self, pos, pkgs):
        return (pos[0], pos[1], pkgs)

    def epsilon_greedy(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        return np.argmax(self.Q[state])

    def softmax(self, state):
        q_vals = self.Q[state]
        q_vals = q_vals - np.max(q_vals)  # numerical stability
        exp_vals = np.exp(q_vals / self.temp)
        probs = exp_vals / np.sum(exp_vals)
        return np.random.choice(4, p=probs)

    def update_parameters(self, strategy):
        if strategy == 'epsilon_greedy':
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        elif strategy == 'softmax':
            self.temp = max(self.min_temp, self.temp * self.temp_decay)

    def update_q(self, state, action, reward, next_state):
        max_next_q = np.max(self.Q[next_state])
        self.Q[state][action] += self.alpha * (reward + self.gamma * max_next_q - self.Q[state][action])

def train(agent, strategy, num_episodes=1000, max_steps=200):
    env = FourRooms('simple', stochastic=agent.stochastic)
    rewards = []
    for episode in range(num_episodes):
        env.newEpoch()
        pos = env.getPosition()
        pkgs = env.getPackagesRemaining()
        state = agent.get_state(pos, pkgs)
        total_reward = 0
        steps = 0
        done = False
        while not done and steps < max_steps:
            if strategy == 'epsilon_greedy':
                action = agent.epsilon_greedy(state)
            else:
                action = agent.softmax(state)
            _, new_pos, new_pkgs, done = env.takeAction(action)
            reward = 10 if new_pkgs == 0 else -1
            next_state = agent.get_state(new_pos, new_pkgs)
            agent.update_q(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            steps += 1
        agent.update_parameters(strategy)
        rewards.append(total_reward)
        if (episode+1) % 100 == 0:
            avg = np.mean(rewards[-100:])
            print(f"Ep {episode+1:4d} | {strategy:12s} | Avg Reward: {avg:5.1f}")
    return rewards

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-stochastic', action='store_true', help='Enable stochastic actions')
    args = parser.parse_args()
    strategies = ['epsilon_greedy', 'softmax']
    results = {}
    for strategy in strategies:
        print(f"\n=== Training {strategy.upper()} Strategy ===")
        agent = QLearningAgent(stochastic=args.stochastic)
        results[strategy] = train(agent, strategy)
    # Plot learning curves
    plt.figure(figsize=(12, 6))
    for strategy in strategies:
        smoothed = np.convolve(results[strategy], np.ones(100)/100, mode='valid')
        plt.plot(smoothed, label=f"{strategy}")
    plt.xlabel("Episode")
    plt.ylabel("Average Total Reward (100-episode MA)")
    plt.title("Learning Progress: Epsilon-Greedy vs Softmax")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('learning_curve.png', dpi=120)
    # Show final path
    env = FourRooms('simple', stochastic=args.stochastic)
    env.showPath(-1, savefig='final_path.png')

if __name__ == "__main__":
    main()

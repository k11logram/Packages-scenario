from FourRooms import FourRooms
import numpy as np
import random
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt

class QLearningAgent:
    def __init__(self, stochastic=False):
        self.stochastic = stochastic
        self.alpha = 0.12
        self.gamma = 0.97
        # Epsilon-greedy parameters
        self.epsilon = 1.0
        self.epsilon_decay = 0.996
        self.min_epsilon = 0.05
        # Softmax parameters
        self.temp = 3.0
        self.temp_decay = 0.995
        self.min_temp = 0.1
        # Q-table: state (x, y, pkgs_left) -> 4 actions
        self.Q = defaultdict(lambda: np.zeros(4))

    def get_state(self, pos, pkgs_left):
        return (pos[0], pos[1], pkgs_left)

    def epsilon_greedy(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        return np.argmax(self.Q[state])

    def softmax(self, state):
        q_vals = self.Q[state] - np.max(self.Q[state])
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

def train(agent, strategy, num_episodes=2000, max_steps=400):
    env = FourRooms('rgb', stochastic=agent.stochastic)
    rewards = []
    for episode in range(num_episodes):
        env.newEpoch()
        pos = env.getPosition()
        pkgs_left = env.getPackagesRemaining()
        state = agent.get_state(pos, pkgs_left)
        total_reward = 0
        steps = 0
        done = False
        while not done and steps < max_steps:
            if strategy == 'epsilon_greedy':
                action = agent.epsilon_greedy(state)
            else:
                action = agent.softmax(state)
            cell_type, new_pos, new_pkgs_left, done = env.takeAction(action)
            if new_pkgs_left < pkgs_left:
                reward = 10
            elif done and new_pkgs_left == pkgs_left:
                reward = -20  # Penalty for wrong order
            else:
                reward = -1
            next_state = agent.get_state(new_pos, new_pkgs_left)
            agent.update_q(state, action, reward, next_state)
            state = next_state
            pkgs_left = new_pkgs_left
            total_reward += reward
            steps += 1
        agent.update_parameters(strategy)
        rewards.append(total_reward)
        if (episode+1) % 200 == 0:
            avg = np.mean(rewards[-200:])
            print(f"Ep {episode+1:4d} | {strategy:12s} | Avg Reward: {avg:5.1f}")
    return rewards, agent

def run_final_policy(agent, env, max_steps=400):
    env.newEpoch()
    pos = env.getPosition()
    pkgs_left = env.getPackagesRemaining()
    state = agent.get_state(pos, pkgs_left)
    done = False
    steps = 0
    while not done and steps < max_steps:
        # Always use greedy policy for final path
        action = np.argmax(agent.Q[state])
        _, new_pos, new_pkgs_left, done = env.takeAction(action)
        state = agent.get_state(new_pos, new_pkgs_left)
        pkgs_left = new_pkgs_left
        steps += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-stochastic', action='store_true', help='Enable stochastic actions')
    args = parser.parse_args()

    strategies = ['epsilon_greedy', 'softmax']
    results = {}
    agents = {}

    for strategy in strategies:
        print(f"\n=== Training {strategy.upper()} Strategy ===")
        agent = QLearningAgent(stochastic=args.stochastic)
        rewards, trained_agent = train(agent, strategy)
        results[strategy] = rewards
        agents[strategy] = trained_agent

    # Plot learning curves
    plt.figure(figsize=(12, 6))
    for strategy, color in zip(strategies, ['blue', 'orange']):
        smoothed = np.convolve(results[strategy], np.ones(200)/200, mode='valid')
        plt.plot(smoothed, label=f"{strategy}", color=color)
    plt.xlabel("Episode")
    plt.ylabel("Average Total Reward (200-episode MA)")
    plt.title("Scenario 3: Ordered RGB Package Collection")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('scenario3_learning_curve.png', dpi=120)

    # Show final path for each strategy
    for strategy in strategies:
        env = FourRooms('rgb', stochastic=args.stochastic)
        run_final_policy(agents[strategy], env)
        env.showPath(-1, savefig=f'scenario3_final_path_{strategy}.png')

if __name__ == "__main__":
    main()

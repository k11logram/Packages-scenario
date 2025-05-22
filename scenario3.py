from FourRooms import FourRooms
import numpy as np
import random
import argparse
from collections import defaultdict, deque
import matplotlib.pyplot as plt

class QLearningAgent:
    def __init__(self, stochastic=False):
        self.stochastic = stochastic
        self.alpha = 0.25  # Lower learning rate for ordered collection
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.05
        self.temp = 1.5
        self.temp_decay = 0.995
        self.min_temp = 0.3
        self.Q = defaultdict(lambda: np.zeros(4))
        self.replay_buffer = deque(maxlen=1000)
        self.batch_size = 32
        self.color_order = [FourRooms.RED, FourRooms.GREEN, FourRooms.BLUE]

    def get_state(self, pos, target_idx):
        """State includes (x, y) position and current target color index"""
        return (pos[0], pos[1], target_idx)

    def calculate_reward(self, collected_color, current_target_idx):
        """Custom reward function for ordered collection"""
        if collected_color == self.color_order[current_target_idx]:
            return 100  # Correct package collected
        elif collected_color > 0:
            return -50  # Wrong order penalty
        else:
            return -0.5  # Movement penalty

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

def train(agent, strategy, num_episodes=150000, max_steps=500):
    env = FourRooms('rgb', stochastic=agent.stochastic)
    rewards = []
    successes = 0
    
    for episode in range(num_episodes):
        env.newEpoch()
        pos = env.getPosition()
        current_target = 0  # Start with RED package
        state = agent.get_state(pos, current_target)
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < max_steps and current_target < 3:
            action = agent.epsilon_greedy(state) if strategy == 'epsilon_greedy' else agent.softmax(state)
            old_pos = pos
            
            try:
                grid_type, pos, _, done = env.takeAction(action)
            except Exception as e:
                break
            
            reward = agent.calculate_reward(grid_type, current_target)
            
            if grid_type == agent.color_order[current_target]:
                current_target += 1  # Move to next target color
                next_state = agent.get_state(pos, current_target) if current_target < 3 else None
            else:
                next_state = agent.get_state(pos, current_target)
            
            agent.update_q(state, action, reward, next_state)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if current_target >= 3:
                successes += 1
                done = True
                
        agent.update_parameters(strategy)
        rewards.append(total_reward)
        
        if (episode + 1) % 1000 == 0:
            avg_reward = np.mean(rewards[-1000:])
            success_rate = (successes / 1000) * 100
            print(f"Ep {episode+1:6d} | {strategy:12s} | Avg Reward: {avg_reward:6.1f} | Success: {success_rate:5.1f}%")
            successes = 0
            
    return rewards, agent

def visualize_path(agent, strategy, stochastic=False):
    env = FourRooms('rgb', stochastic=stochastic)
    env.newEpoch()
    pos = env.getPosition()
    current_target = 0
    state = agent.get_state(pos, current_target)
    done = False
    steps = 0
    
    while not done and steps < 500 and current_target < 3:
        action = agent.epsilon_greedy(state) if strategy == 'epsilon_greedy' else agent.softmax(state)
        grid_type, pos, _, done = env.takeAction(action)
        
        if grid_type == agent.color_order[current_target]:
            current_target += 1
            state = agent.get_state(pos, current_target) if current_target < 3 else None
        else:
            state = agent.get_state(pos, current_target)
        
        steps += 1
    
    env.showPath(-1, savefig=f'scenario3_{strategy}_path.png')
    print(f"Saved path visualization: scenario3_{strategy}_path.png")

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
    plt.title("Scenario 3: Ordered Package Collection Learning Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('scenario3_learning_curves.png', dpi=120)
    print("\nSaved learning curve: scenario3_learning_curves.png")

if __name__ == "__main__":
    main()

from FourRooms import FourRooms
import numpy as np
import random
import argparse
from collections import defaultdict, deque
import matplotlib.pyplot as plt

class QLearningAgent:
    def __init__(self, stochastic=False):
        self.stochastic = stochastic
        # Enhanced learning parameters
        self.alpha = 0.3  # Increased learning rate
        self.gamma = 0.90
        # Improved exploration parameters
        self.epsilon = 1.0
        self.epsilon_decay = 0.998  # Slower decay
        self.min_epsilon = 0.1
        # Softmax adjustments
        self.temp = 2.0
        self.temp_decay = 0.998
        self.min_temp = 0.5
        # Q-table with directional momentum
        self.Q = defaultdict(lambda: np.zeros(4))
        self.replay_buffer = deque(maxlen=1000)  # Experience replay
        self.batch_size = 32

    def get_state(self, pos, pkgs, last_action=None):
        """Enhanced state with directional momentum"""
        return (pos[0], pos[1], pkgs, last_action)

    def calculate_reward(self, old_pos, new_pos, old_pkgs, new_pkgs):
        """Improved reward structure"""
        if new_pkgs < old_pkgs:
            return 100
        elif new_pos != old_pos:
            return -0.5  # Small penalty for movement without progress
        else:
            return -2    # Larger penalty for staying put

    def epsilon_greedy(self, state, exploration_factor=None):
        eps = exploration_factor if exploration_factor is not None else self.epsilon
        if random.random() < eps:
            return random.choice([FourRooms.UP, FourRooms.DOWN, FourRooms.LEFT, FourRooms.RIGHT])
        return np.argmax(self.Q[state])

    def softmax(self, state, temp_factor=None):
        temp = temp_factor if temp_factor is not None else self.temp
        q_vals = self.Q[state]
        q_vals = q_vals - np.max(q_vals)
        exp_vals = np.exp(q_vals / max(temp, 1e-8))
        probs = exp_vals / np.sum(exp_vals)
        return np.random.choice([FourRooms.UP, FourRooms.DOWN, FourRooms.LEFT, FourRooms.RIGHT], p=probs)

    def update_parameters(self, strategy):
        if strategy == 'epsilon_greedy':
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        elif strategy == 'softmax':
            self.temp = max(self.min_temp, self.temp * self.temp_decay)

    def update_q(self, state, action, reward, next_state):
        # Experience replay implementation
        self.replay_buffer.append((state, action, reward, next_state))
        
        if len(self.replay_buffer) >= self.batch_size:
            batch = random.sample(self.replay_buffer, self.batch_size)
            for s, a, r, ns in batch:
                max_next_q = np.max(self.Q[ns])
                self.Q[s][a] += self.alpha * (r + self.gamma * max_next_q - self.Q[s][a])

def train(agent, strategy, num_episodes=50000, max_steps=200):  # Increased episodes
    env = FourRooms('simple', stochastic=agent.stochastic)
    rewards = []
    successful_episodes = 0

    for episode in range(num_episodes):
        env.newEpoch()
        pos = env.getPosition()
        pkgs = env.getPackagesRemaining()
        state = agent.get_state(pos, pkgs, None)
        total_reward = 0
        steps = 0
        done = False
        last_action = None

        while not done and steps < max_steps:
            if strategy == 'epsilon_greedy':
                action = agent.epsilon_greedy(state)
            else:
                action = agent.softmax(state)

            old_pos = pos
            old_pkgs = pkgs

            try:
                _, new_pos, new_pkgs, done = env.takeAction(action)
            except Exception as e:
                print(f"Error taking action: {e}")
                break

            reward = agent.calculate_reward(old_pos, new_pos, old_pkgs, new_pkgs)
            next_state = agent.get_state(new_pos, new_pkgs, action)
            agent.update_q(state, action, reward, next_state)

            state = next_state
            pos = new_pos
            pkgs = new_pkgs
            last_action = action
            total_reward += reward
            steps += 1

            if done and pkgs == 0:
                successful_episodes += 1

        agent.update_parameters(strategy)
        rewards.append(total_reward)

        if (episode + 1) % 100 == 0:
            avg = np.mean(rewards[-100:])
            success_rate = (successful_episodes / 100) * 100
            print(f"Ep {episode+1:4d} | {strategy:12s} | Avg Reward: {avg:6.1f} | Success: {success_rate:4.0f}%")
            successful_episodes = 0

    return rewards, agent

# Rest of the code remains the same as previous version for visualization and execution


def run_final_policy(agent, strategy, max_steps=200):
    env = FourRooms('simple', stochastic=agent.stochastic)
    env.newEpoch()
    pos = env.getPosition()
    pkgs = env.getPackagesRemaining()
    state = agent.get_state(pos, pkgs)
    done = False
    steps = 0
    initial_packages = pkgs

    print(f"Starting position: {pos}")
    print(f"Initial packages: {initial_packages}")

    while not done and steps < max_steps:
        if strategy == 'epsilon_greedy':
            action = agent.epsilon_greedy(state, exploration_factor=0.0)
        else:
            action = agent.softmax(state, temp_factor=0.01)

        try:
            _, new_pos, new_pkgs, done = env.takeAction(action)
        except Exception as e:
            print(f"Error in final policy: {e}")
            break

        state = agent.get_state(new_pos, new_pkgs)
        pos = new_pos
        pkgs = new_pkgs
        steps += 1

        if steps % 50 == 0:
            print(f"Step {steps}: Position {pos}, Packages remaining: {pkgs}")

    print(f"Final position: {pos}")
    print(f"Steps taken: {steps}")
    print(f"Packages collected: {initial_packages - pkgs}/{initial_packages}")
    print(f"All packages collected: {pkgs == 0}")
    print(f"Terminal state reached: {done}")

    return done and pkgs == 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-stochastic', action='store_true', help='Enable stochastic actions')
    args = parser.parse_args()

    strategies = ['epsilon_greedy', 'softmax']
    results = {}
    agents = {}

    # Test environment
    print("Testing basic environment functionality...")
    test_env = FourRooms('simple', stochastic=args.stochastic)
    test_env.newEpoch()
    print(f"Initial position: {test_env.getPosition()}")
    print(f"Initial packages: {test_env.getPackagesRemaining()}")
    print(f"Environment created successfully!")

    for strategy in strategies:
        print(f"\n=== Training {strategy.upper()} Strategy ===")
        agent = QLearningAgent(stochastic=args.stochastic)
        try:
            rewards, trained_agent = train(agent, strategy)
            results[strategy] = rewards
            agents[strategy] = trained_agent
            print(f"Training completed for {strategy}")
        except Exception as e:
            print(f"Error during training {strategy}: {e}")
            continue

    if not results:
        print("No successful training runs. Check environment setup.")
        return

    # Plotting learning curves
    plt.figure(figsize=(12, 6))
    colors = {'epsilon_greedy': 'blue', 'softmax': 'orange'}
    for strategy in results:
        if len(results[strategy]) > 100:
            smoothed = np.convolve(results[strategy], np.ones(100)/100, mode='valid')
            plt.plot(smoothed, label=strategy, color=colors[strategy])
    plt.xlabel("Episode")
    plt.ylabel("Average Total Reward (100-episode MA)")
    plt.title("Scenario 1: Simple Package Collection")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('scenario1_learning_curve.png', dpi=120)
    print("Learning curve saved to scenario1_learning_curve.png")

    # Show final path for each strategy
    for strategy in agents:
        print(f"\n=== Generating Final Path for {strategy} ===")
        try:
            success = run_final_policy(agents[strategy], strategy)
            path_filename = f'scenario1_final_path_{strategy}.png'
            # Create a fresh environment for path visualization
            env = FourRooms('simple', stochastic=args.stochastic)
            env.newEpoch()
            pos = env.getPosition()
            pkgs = env.getPackagesRemaining()
            state = agents[strategy].get_state(pos, pkgs)
            done = False
            steps = 0
            while not done and steps < 200:
                if strategy == 'epsilon_greedy':
                    action = agents[strategy].epsilon_greedy(state, exploration_factor=0.0)
                else:
                    action = agents[strategy].softmax(state, temp_factor=0.01)
                _, new_pos, new_pkgs, done = env.takeAction(action)
                state = agents[strategy].get_state(new_pos, new_pkgs)
                pos = new_pos
                pkgs = new_pkgs
                steps += 1
            env.showPath(-1, savefig=path_filename)
            print(f"Final path saved to {path_filename}")
            print(f"Success: {success}")
        except Exception as e:
            print(f"Error generating final path for {strategy}: {e}")

if __name__ == "__main__":
    main()
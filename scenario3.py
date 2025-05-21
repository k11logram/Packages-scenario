from FourRooms import FourRooms
import numpy as np
import random
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt

class QLearningAgent:
    def __init__(self, stochastic=False):
        self.stochastic = stochastic
        
        # Learning parameters - improved
        self.alpha = 0.15  # Increased learning rate
        self.gamma = 0.99  # High discount factor for long-term planning
        
        # Epsilon-greedy parameters
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.02
        
        # Softmax parameters
        self.temp = 1.0
        self.temp_decay = 0.99
        self.min_temp = 0.05
        
        # Q-table: state -> actions
        self.Q = defaultdict(lambda: np.zeros(4))
    
    def get_state(self, pos, pkgs_left, collection_status):
        """Enhanced state representation that includes package collection status"""
        # Include position, packages left, and collection status in state
        return (pos[0], pos[1], pkgs_left, collection_status)
    
    def epsilon_greedy(self, state, exploration_factor=None):
        # Use provided exploration factor or default to self.epsilon
        eps = exploration_factor if exploration_factor is not None else self.epsilon
        
        if random.random() < eps:
            return random.randint(0, 3)
        return np.argmax(self.Q[state])
    
    def softmax(self, state, temp_factor=None):
        # Use provided temperature or default to self.temp
        temp = temp_factor if temp_factor is not None else self.temp
        
        q_vals = self.Q[state]
        # For numerical stability
        q_vals = q_vals - np.max(q_vals)
        exp_vals = np.exp(q_vals / max(temp, 1e-8))
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

def determine_package_color(cell_type):
    """
    Map cell type to package color
    Using FourRooms constants: RED = 1, GREEN = 2, BLUE = 3
    """
    if cell_type == FourRooms.RED:
        return "R"
    elif cell_type == FourRooms.GREEN:
        return "G"
    elif cell_type == FourRooms.BLUE:
        return "B"
    return None

def check_valid_collection(collection_status, color):
    """Check if package was collected in the correct RGB order"""
    if len(collection_status) == 0 and color == "R":
        return True
    if collection_status == "R" and color == "G":
        return True
    if collection_status == "RG" and color == "B":
        return True
    return False

def next_expected_color(collection_status):
    """Determine the next expected color based on current collection status"""
    if len(collection_status) == 0:
        return "R"
    elif collection_status == "R":
        return "G"
    elif collection_status == "RG":
        return "B"
    return None

def train(agent, strategy, num_episodes=2000, max_steps=400):
    """Train the Q-learning agent"""
    env = FourRooms('rgb', stochastic=agent.stochastic)
    rewards = []
    
    # For debugging/analysis
    correct_collections = 0
    incorrect_collections = 0
    
    for episode in range(num_episodes):
        env.newEpoch()
        pos = env.getPosition()
        
        # Initialize tracking variables
        pkgs_left = env.getPackagesRemaining()
        collection_status = ""
        
        state = agent.get_state(pos, pkgs_left, collection_status)
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < max_steps:
            # Select action based on strategy
            if strategy == 'epsilon_greedy':
                action = agent.epsilon_greedy(state)
            else:
                action = agent.softmax(state)
            
            # Take action and observe result
            cell_type, new_pos, new_pkgs_left, done = env.takeAction(action)
            
            # Check if a package was collected
            if new_pkgs_left < pkgs_left:
                collected_color = determine_package_color(cell_type)
                valid_collection = check_valid_collection(collection_status, collected_color)
                
                if valid_collection:
                    # Package collected in correct order
                    collection_status += collected_color
                    reward = 80  # Increased reward for correct collection
                    correct_collections += 1
                else:
                    # Package collected in wrong order (simulation will be terminated by env)
                    reward = -80  # Stronger penalty for incorrect order
                    incorrect_collections += 1
            elif done and new_pkgs_left > 0:
                # Failed to collect all packages
                reward = -30
            else:
                # Regular step penalty
                reward = -1
            
            next_state = agent.get_state(new_pos, new_pkgs_left, collection_status)
            
            # Update Q-table
            agent.update_q(state, action, reward, next_state)
            
            state = next_state
            pos = new_pos
            pkgs_left = new_pkgs_left
            total_reward += reward
            steps += 1
        
        agent.update_parameters(strategy)
        rewards.append(total_reward)
        
        if (episode + 1) % 200 == 0:
            avg = np.mean(rewards[-200:])
            correct_rate = correct_collections / max(correct_collections + incorrect_collections, 1) * 100
            print(f"Ep {episode+1:4d} | {strategy:12s} | Avg Reward: {avg:5.1f} | Correct collections: {correct_rate:.1f}%")
            
            # Reset collection counters
            correct_collections = 0
            incorrect_collections = 0
    
    return rewards, agent

def run_final_policy(agent, strategy, max_steps=600):  # Increased max_steps
    """Run the final learned policy with anti-stuck mechanisms"""
    env = FourRooms('rgb', stochastic=agent.stochastic)
    env.newEpoch()
    pos = env.getPosition()
    
    # Initialize tracking variables
    pkgs_left = env.getPackagesRemaining()
    collection_status = ""
    
    state = agent.get_state(pos, pkgs_left, collection_status)
    done = False
    steps = 0
    
    # Anti-stuck mechanisms
    visited_positions = set()
    position_counts = defaultdict(int)
    stuck_threshold = 10
    no_progress_count = 0
    
    # For analysis
    packages_collected = []
    
    while not done and steps < max_steps:
        # Track position frequency
        position_counts[pos] += 1
        visited_positions.add(pos)
        
        # Determine if agent is stuck
        is_stuck = position_counts[pos] > stuck_threshold
        
        # Adaptive exploration based on stuckness
        if is_stuck or no_progress_count > 50:
            # Take completely random action if seriously stuck
            action = random.randint(0, 3)
            no_progress_count = 0  # Reset counter after random action
        else:
            # Use policy with increased exploration in evaluation
            exploration_factor = 0.15  # Higher exploration during evaluation
            
            if strategy == 'epsilon_greedy':
                action = agent.epsilon_greedy(state, exploration_factor)
            else:
                action = agent.softmax(state, exploration_factor)
        
        # Take action
        cell_type, new_pos, new_pkgs_left, done = env.takeAction(action)
        
        # Check if a package was collected
        if new_pkgs_left < pkgs_left:
            collected_color = determine_package_color(cell_type)
            valid = check_valid_collection(collection_status, collected_color)
            packages_collected.append((collected_color, valid))
            
            if valid:
                collection_status += collected_color
                # Reset anti-stuck mechanisms after successful collection
                visited_positions.clear()
                position_counts.clear()
                no_progress_count = 0
            else:
                # Simulation will terminate if wrong order
                pass
        else:
            # No package collected, increment no-progress counter
            no_progress_count += 1
        
        state = agent.get_state(new_pos, new_pkgs_left, collection_status)
        pos = new_pos
        pkgs_left = new_pkgs_left
        steps += 1
    
    print(f"Steps taken: {steps}")
    print(f"Packages collected: {packages_collected}")
    print(f"All packages collected: {pkgs_left == 0}")
    print(f"Final collection status: {collection_status}")
    
    # Verify correct RGB order
    if collection_status == "RGB":
        print("SUCCESS: Packages collected in correct RGB order!")
    else:
        print(f"FAILED: Packages collected in incorrect order: {collection_status}")
    
    return packages_collected

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
    colors = {'epsilon_greedy': 'blue', 'softmax': 'orange'}
    
    for strategy in strategies:
        smoothed = np.convolve(results[strategy], np.ones(200)/200, mode='valid')
        plt.plot(smoothed, label=strategy, color=colors[strategy])
    
    plt.xlabel("Episode")
    plt.ylabel("Average Total Reward (200-episode MA)")
    plt.title("Scenario 3: Ordered RGB Package Collection")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('scenario3_learning_curve.png', dpi=120)
    print("Learning curve saved to scenario3_learning_curve.png")
    
    # Show final path for each strategy
    for strategy in strategies:
        print(f"\n=== Generating Final Path for {strategy} ===")
        collected = run_final_policy(agents[strategy], strategy)
        path_filename = f'scenario3_final_path_{strategy}.png'
        env = FourRooms('rgb', stochastic=args.stochastic)
        env.showPath(-1, savefig=path_filename)
        print(f"Final path saved to {path_filename}")
        print(f"Collection order: {collected}")

if __name__ == "__main__":
    main()

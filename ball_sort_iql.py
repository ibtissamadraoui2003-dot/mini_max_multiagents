"""
Ball Sort Puzzle - Cooperative Multi-Agent Reinforcement Learning
Using Independent Q-Learning (IQL)

EDUCATIONAL DEMONSTRATION:
- Multiple agents cooperating toward the same goal
- Each agent learns independently with its own Q-table
- Cooperation emerges through shared rewards
- No explicit communication between agents
"""

import pygame
import numpy as np
import random
from collections import defaultdict
import time

# =============================================================================
# GAME CONFIGURATION
# =============================================================================

# Colors for balls (RGB)
COLORS = {
    0: (255, 0, 0),      # Red
    1: (0, 255, 0),      # Green
    2: (0, 0, 255),      # Blue
    3: (255, 255, 0),    # Yellow
}
COLOR_NAMES = {0: "Red", 1: "Green", 2: "Blue", 3: "Yellow"}

# Game settings
NUM_TUBES = 4           # Total number of tubes
TUBE_CAPACITY = 4       # Max balls per tube
NUM_COLORS = 3          # Number of different colors (one tube will be empty)
NUM_AGENTS = 2          # Number of agents

# Visualization settings
SCREEN_WIDTH = 900
SCREEN_HEIGHT = 700
TUBE_WIDTH = 60
TUBE_HEIGHT = 200
BALL_RADIUS = 25
TUBE_SPACING = 120

# Colors for UI
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
DARK_GRAY = (50, 50, 50)
TUBE_COLOR = (180, 180, 180)
AGENT_COLORS = [(255, 100, 100), (100, 100, 255)]  # Red-ish, Blue-ish for agents

# Learning parameters
ALPHA = 0.15         # Learning rate (slightly higher for faster learning)
GAMMA = 0.95         # Discount factor
EPSILON_START = 1.0  # Initial exploration rate
EPSILON_MIN = 0.02   # Minimum exploration rate
EPSILON_DECAY = 0.998  # Slower decay for longer exploration (~500 episodes to reach min)


# =============================================================================
# BALL SORT ENVIRONMENT
# =============================================================================

class BallSortEnv:
    def __init__(self):
        self.num_tubes = NUM_TUBES
        self.tube_capacity = TUBE_CAPACITY
        self.num_colors = NUM_COLORS
        self.tubes = None
        self.steps = 0
        self.max_steps = 50  # Fewer steps per episode for faster learning cycles
        self.reset()
    
    def reset(self):
        """Reset environment with a new shuffled puzzle"""
        self.steps = 0
        
        # Create balls: TUBE_CAPACITY balls of each color
        all_balls = []
        for color in range(self.num_colors):
            all_balls.extend([color] * self.tube_capacity)
        
        # Shuffle balls
        random.shuffle(all_balls)
        
        # Distribute balls into tubes (leave last tube empty)
        self.tubes = [[] for _ in range(self.num_tubes)]
        ball_idx = 0
        for tube_idx in range(self.num_colors):  # Fill only first NUM_COLORS tubes
            for _ in range(self.tube_capacity):
                self.tubes[tube_idx].append(all_balls[ball_idx])
                ball_idx += 1
        
        # Ensure puzzle is not already solved
        while self.is_solved():
            random.shuffle(all_balls)
            ball_idx = 0
            for tube_idx in range(self.num_colors):
                self.tubes[tube_idx] = []
                for _ in range(self.tube_capacity):
                    self.tubes[tube_idx].append(all_balls[ball_idx])
                    ball_idx += 1
        
        return self.get_state()
    
    def get_state(self):
        top_colors = []
        tube_sizes = []
        
        for tube in self.tubes:
            if len(tube) > 0:
                top_colors.append(tube[-1])  # Top ball color
            else:
                top_colors.append(-1)  # Empty tube marker
            tube_sizes.append(len(tube))
        
        return (tuple(top_colors), tuple(tube_sizes))
    
    def is_valid_move(self, from_tube, to_tube):
        """Check if moving top ball from from_tube to to_tube is valid"""
        # Can't move to same tube
        if from_tube == to_tube:
            return False
        
        # Source tube must have balls
        if len(self.tubes[from_tube]) == 0:
            return False
        
        # Destination tube must not be full
        if len(self.tubes[to_tube]) >= self.tube_capacity:
            return False
        
        # Destination must be empty OR have same color on top
        if len(self.tubes[to_tube]) > 0:
            src_color = self.tubes[from_tube][-1]
            dst_color = self.tubes[to_tube][-1]
            if src_color != dst_color:
                return False
        
        return True
    
    def execute_move(self, from_tube, to_tube):
        """Execute a move (assumes move is valid)"""
        ball = self.tubes[from_tube].pop()
        self.tubes[to_tube].append(ball)
    
    def is_solved(self):
        colors_found = set()
        
        for tube in self.tubes:
            if len(tube) == 0:
                continue  # Empty tube is fine
            
            # Check if all balls in tube are same color
            if len(set(tube)) > 1:
                return False  # Mixed colors = not solved
            
            # Check if tube is full (has all balls of this color)
            if len(tube) != self.tube_capacity:
                return False  # Partial tube = not solved
            
            # Check we haven't seen this color in another tube
            color = tube[0]
            if color in colors_found:
                return False  # Same color in multiple tubes = not solved
            colors_found.add(color)
        
        # Verify we have all colors properly sorted
        return len(colors_found) == self.num_colors
    
    def count_sorted_balls(self):
        count = 0
        for tube in self.tubes:
            if len(tube) > 0:
                # Only count if ALL balls in this tube are same color
                if len(set(tube)) == 1:
                    count += len(tube)  # All balls in this tube count
        return count
    
    def get_sorting_score(self):
        pure_tubes = 0  # Tubes with only one color
        max_same_color = 0  # Largest group of same color in each tube
        
        for tube in self.tubes:
            if len(tube) == 0:
                continue
            if len(set(tube)) == 1:
                pure_tubes += 1
                max_same_color += len(tube)
            else:
                # Count largest same-color group from bottom
                first_color = tube[0]
                group_size = 0
                for ball in tube:
                    if ball == first_color:
                        group_size += 1
                    else:
                        break
                max_same_color += group_size
        
        return pure_tubes, max_same_color
    
    def step(self, agent_actions):
        self.steps += 1
        prev_sorted = self.count_sorted_balls()
        
        valid_moves = []
        invalid_count = 0
        
        # Collect valid moves (check for conflicts)
        occupied_sources = set()
        occupied_destinations = set()
        
        for agent_id, (from_tube, to_tube) in enumerate(agent_actions):
            # STAY action - always valid (indicated by (-1, -1))
            if from_tube == -1 and to_tube == -1:
                valid_moves.append(('STAY', -1))  # Mark as valid STAY
                continue
            
            # Check basic validity
            if not self.is_valid_move(from_tube, to_tube):
                valid_moves.append(None)
                invalid_count += 1
                continue
            
            # Check for conflicts with other agents
            if from_tube in occupied_sources or from_tube in occupied_destinations:
                valid_moves.append(None)
                invalid_count += 1
                continue
            if to_tube in occupied_sources or to_tube in occupied_destinations:
                valid_moves.append(None)
                invalid_count += 1
                continue
            
            # Mark as valid
            valid_moves.append((from_tube, to_tube))
            occupied_sources.add(from_tube)
            occupied_destinations.add(to_tube)
        
        # Execute all valid moves (skip STAY actions)
        moves_made = 0
        stay_count = 0
        for move in valid_moves:
            if move is not None and move[0] == 'STAY':
                stay_count += 1
            elif move is not None:
                self.execute_move(move[0], move[1])
                moves_made += 1
        
        # Calculate reward
        new_sorted = self.count_sorted_balls()
        
        if self.is_solved():
            reward = 100  # Big reward for solving
        else:
            # Reward for improving sorting
            improvement = new_sorted - prev_sorted
            reward = improvement * 2
            
            # Penalty for invalid moves
            reward -= invalid_count * 1
            
            # Small penalty for STAY when moves exist (discourages passivity)
            if stay_count > 0 and len(self.get_all_valid_actions()) > 1:
                reward -= stay_count * 0.3
            
            # Small step penalty to encourage efficiency
            reward -= 0.1
        
        # Check termination
        done = self.is_solved() or self.steps >= self.max_steps
        
        info = {
            'valid_moves': valid_moves,
            'moves_made': moves_made,
            'invalid_count': invalid_count,
            'sorted_balls': new_sorted,
            'solved': self.is_solved()
        }
        
        return self.get_state(), reward, done, info
    
    def get_all_valid_actions(self):
        valid = [(-1, -1)]  # STAY is always valid
        for from_tube in range(self.num_tubes):
            for to_tube in range(self.num_tubes):
                if self.is_valid_move(from_tube, to_tube):
                    valid.append((from_tube, to_tube))
        return valid


# =============================================================================
# INDEPENDENT Q-LEARNING AGENT
# =============================================================================

class IQLAgent:

    
    def __init__(self, agent_id, num_tubes, alpha=ALPHA, gamma=GAMMA):
        self.agent_id = agent_id
        self.num_tubes = num_tubes
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        
        # Q-table: maps (state, action) -> value
        # Action is now (from_tube, to_tube) tuple, or (-1, -1) for STAY
        self.q_table = defaultdict(float)
        
        # Statistics
        self.total_updates = 0
        
        # Track last action for visualization
        self.last_action = (-1, -1)
    
    def get_action_space(self):
        """Get all possible actions as (from_tube, to_tube) pairs"""
        actions = [(-1, -1)]  # STAY
        for from_tube in range(self.num_tubes):
            for to_tube in range(self.num_tubes):
                if from_tube != to_tube:
                    actions.append((from_tube, to_tube))
        return actions
    
    def state_to_key(self, state):
        """Convert state to hashable key"""
        return state  # Already a tuple
    
    def get_q_value(self, state, action):
        """Get Q-value for state-action pair"""
        key = (self.state_to_key(state), action)
        return self.q_table[key]
    
    def get_best_action(self, state, valid_actions=None):
        """Get action with highest Q-value"""
        if valid_actions is None:
            valid_actions = self.get_action_space()
        
        best_action = valid_actions[0]
        best_value = self.get_q_value(state, best_action)
        
        for action in valid_actions[1:]:
            value = self.get_q_value(state, action)
            if value > best_value:
                best_value = value
                best_action = action
        
        return best_action, best_value
    
    def select_action(self, state, epsilon, valid_actions=None):
        """Select action using epsilon-greedy policy"""
        if valid_actions is None:
            valid_actions = self.get_action_space()
        
        if random.random() < epsilon:
            # Exploration: random action
            action = random.choice(valid_actions)
        else:
            # Exploitation: best action
            action = self.get_best_action(state, valid_actions)[0]
        
        self.last_action = action
        return action
    
    def update(self, state, action, reward, next_state, done):
        """
        Update Q-value using Q-learning update rule:
        Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') - Q(s,a) ]
        """
        current_q = self.get_q_value(state, action)
        
        if done:
            target = reward
        else:
            # Get max Q-value for next state
            _, max_next_q = self.get_best_action(next_state)
            target = reward + self.gamma * max_next_q
        
        # Q-learning update
        new_q = current_q + self.alpha * (target - current_q)
        
        key = (self.state_to_key(state), action)
        self.q_table[key] = new_q
        self.total_updates += 1
        
        return current_q, new_q
    
    def get_q_table_size(self):
        """Get number of entries in Q-table"""
        return len(self.q_table)


# =============================================================================
# VISUALIZATION
# =============================================================================

class BallSortVisualizer:
    """Pygame-based visualization for Ball Sort puzzle"""
    
    def __init__(self, env, agents):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Ball Sort - Cooperative Multi-Agent IQL")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 22)
        self.title_font = pygame.font.Font(None, 40)
        
        self.env = env
        self.agents = agents
        
        # Calculate tube positions (adjusted for stats panel)
        total_width = (NUM_TUBES - 1) * TUBE_SPACING + TUBE_WIDTH
        start_x = (SCREEN_WIDTH - total_width) // 2
        self.tube_positions = []
        for i in range(NUM_TUBES):
            x = start_x + i * TUBE_SPACING
            y = 180  # Moved down to accommodate expanded stats
            self.tube_positions.append((x, y))
    
    def draw_tube(self, tube_idx, highlight_agent=None):
        """Draw a single tube with its balls"""
        x, y = self.tube_positions[tube_idx]
        tube = self.env.tubes[tube_idx]
        
        # Draw tube outline
        tube_rect = pygame.Rect(x, y, TUBE_WIDTH, TUBE_HEIGHT)
        
        # Highlight if agent controls this tube
        if highlight_agent is not None:
            highlight_color = AGENT_COLORS[highlight_agent]
            pygame.draw.rect(self.screen, highlight_color, 
                           tube_rect.inflate(8, 8), 3, border_radius=5)
        
        # Draw tube background
        pygame.draw.rect(self.screen, TUBE_COLOR, tube_rect, border_radius=5)
        pygame.draw.rect(self.screen, DARK_GRAY, tube_rect, 2, border_radius=5)
        
        # Draw balls (from bottom to top)
        ball_y = y + TUBE_HEIGHT - BALL_RADIUS - 5
        for i, ball_color in enumerate(tube):
            ball_x = x + TUBE_WIDTH // 2
            pygame.draw.circle(self.screen, COLORS[ball_color], 
                             (ball_x, ball_y), BALL_RADIUS)
            pygame.draw.circle(self.screen, BLACK, 
                             (ball_x, ball_y), BALL_RADIUS, 2)
            ball_y -= BALL_RADIUS * 2 + 2
        
        # Draw tube label
        label = self.font.render(f"Tube {tube_idx}", True, BLACK)
        label_rect = label.get_rect(center=(x + TUBE_WIDTH // 2, y + TUBE_HEIGHT + 25))
        self.screen.blit(label, label_rect)
    
    def draw_agent_info(self, agent_idx, action, q_before, q_after, valid_move):
        """Draw information about an agent's action"""
        x = 50 + agent_idx * 420
        y = 430
        
        # Agent box
        box_rect = pygame.Rect(x, y, 380, 160)
        pygame.draw.rect(self.screen, AGENT_COLORS[agent_idx], box_rect, 0, border_radius=10)
        pygame.draw.rect(self.screen, BLACK, box_rect, 2, border_radius=10)
        
        # Agent title
        title = self.title_font.render(f"Agent {agent_idx + 1}", True, BLACK)
        self.screen.blit(title, (x + 10, y + 10))
        
        # Flexibility note
        flex_text = self.small_font.render("Can move from ANY tube", True, DARK_GRAY)
        self.screen.blit(flex_text, (x + 10, y + 45))
        
        # Action taken - now shows (from_tube, to_tube)
        from_tube, to_tube = action
        if from_tube == -1 and to_tube == -1:
            action_text = "STAY (no move)"
            valid = True  # STAY is always valid
        else:
            action_text = f"Tube {from_tube} → Tube {to_tube}"
            valid = valid_move is not None and valid_move[0] != 'STAY'
        
        status = "✓ Valid" if valid else "✗ Invalid"
        status_color = (0, 120, 0) if valid else (200, 0, 0)
        
        action_label = self.font.render("Action:", True, BLACK)
        self.screen.blit(action_label, (x + 10, y + 70))
        
        action_str = self.font.render(f"{action_text}", True, BLACK)
        self.screen.blit(action_str, (x + 90, y + 70))
        
        status_str = self.font.render(f"({status})", True, status_color)
        self.screen.blit(status_str, (x + 10, y + 100))
        
        # Q-value update
        q_text = self.small_font.render(f"Q-value: {q_before:.2f} → {q_after:.2f}", True, BLACK)
        self.screen.blit(q_text, (x + 10, y + 130))
    
    def draw_stats(self, episode, step, reward, epsilon, solved, elapsed_time=0, solve_rate=0, fast_mode=False):
        """Draw episode statistics"""
        # Title
        title = self.title_font.render("Ball Sort - Cooperative Multi-Agent IQL", True, BLACK)
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 30))
        self.screen.blit(title, title_rect)
        
        # Stats panel - Row 1
        stats_y = 70
        
        episode_text = self.font.render(f"Episode: {episode}", True, BLACK)
        self.screen.blit(episode_text, (50, stats_y))
        
        step_text = self.font.render(f"Step: {step}", True, BLACK)
        self.screen.blit(step_text, (200, stats_y))
        
        reward_text = self.font.render(f"Reward: {reward:.2f}", True, BLACK)
        self.screen.blit(reward_text, (320, stats_y))
        
        epsilon_text = self.font.render(f"Epsilon: {epsilon:.3f}", True, BLACK)
        self.screen.blit(epsilon_text, (480, stats_y))
        
        sorted_text = self.font.render(f"Sorted: {self.env.count_sorted_balls()}/{NUM_COLORS * TUBE_CAPACITY}", True, BLACK)
        self.screen.blit(sorted_text, (650, stats_y))
        
        # Stats panel - Row 2 (Time and Performance)
        stats_y2 = 100
        
        time_min = elapsed_time / 60
        time_text = self.font.render(f"Time: {time_min:.1f} min", True, DARK_GRAY)
        self.screen.blit(time_text, (50, stats_y2))
        
        solve_text = self.font.render(f"Solve Rate: {solve_rate:.1f}%", True, DARK_GRAY)
        self.screen.blit(solve_text, (200, stats_y2))

        # Learning phase indicator
        if episode <= 30:
            phase = "Phase 1: Exploration"
            phase_color = (200, 100, 0)  # Orange
        elif episode <= 100:
            phase = "Phase 2: Learning"
            phase_color = (0, 100, 200)  # Blue
        elif episode <= 300:
            phase = "Phase 3: Coordination"
            phase_color = (100, 150, 0)  # Green-ish
        else:
            phase = "Phase 4: Mastery"
            phase_color = (0, 150, 0)  # Green
        
        phase_text = self.font.render(phase, True, phase_color)
        self.screen.blit(phase_text, (380, stats_y2))
        
        # Fast mode indicator
        if fast_mode:
            fast_text = self.font.render("[FAST MODE]", True, (200, 0, 0))
            self.screen.blit(fast_text, (580, stats_y2))
        
        # Controls reminder
        controls = self.small_font.render("SPACE=Pause  F=Fast  ESC=Quit", True, GRAY)
        self.screen.blit(controls, (700, stats_y2))
        
        if solved:
            solved_text = self.title_font.render("🎉 SOLVED! 🎉", True, (0, 150, 0))
            solved_rect = solved_text.get_rect(center=(SCREEN_WIDTH // 2, 140))
            self.screen.blit(solved_text, solved_rect)
    
    def draw_legend(self):
        """Draw color legend"""
        x = SCREEN_WIDTH - 150
        y = 150
        
        legend_title = self.font.render("Colors:", True, BLACK)
        self.screen.blit(legend_title, (x, y))
        
        for color_id, color_rgb in COLORS.items():
            if color_id < NUM_COLORS:
                y += 30
                pygame.draw.circle(self.screen, color_rgb, (x + 15, y + 10), 12)
                pygame.draw.circle(self.screen, BLACK, (x + 15, y + 10), 12, 2)
                name = self.small_font.render(COLOR_NAMES[color_id], True, BLACK)
                self.screen.blit(name, (x + 35, y))
    
    def draw_q_table_info(self):
        """Draw Q-table statistics"""
        y = 650
        for i, agent in enumerate(self.agents):
            text = self.small_font.render(
                f"Agent {i+1} Q-table: {agent.get_q_table_size()} entries, "
                f"{agent.total_updates} updates", True, DARK_GRAY)
            self.screen.blit(text, (50 + i * 400, y))
    
    def render(self, episode, step, reward, epsilon, actions, q_values, valid_moves, 
                solved=False, elapsed_time=0, solve_rate=0, fast_mode=False):
        """Render the complete game state"""
        self.screen.fill(WHITE)
        
        # Draw statistics
        self.draw_stats(episode, step, reward, epsilon, solved, elapsed_time, solve_rate, fast_mode)
        
        # Draw tubes (highlight based on current actions)
        action_sources = set()
        action_dests = set()
        for action in actions:
            if action[0] != -1:  # Not STAY
                action_sources.add(action[0])
                action_dests.add(action[1])
        
        for i in range(NUM_TUBES):
            # Highlight tubes being used in current actions
            highlight = None
            if i in action_sources:
                highlight = 0  # Source = Agent 1 color
            elif i in action_dests:
                highlight = 1  # Dest = Agent 2 color
            self.draw_tube(i, highlight)
        
        # Draw agent info with full action tuple
        for i in range(NUM_AGENTS):
            action = actions[i] if actions[i] else (-1, -1)
            q_before, q_after = q_values[i]
            valid_move = valid_moves[i] if valid_moves else None
            self.draw_agent_info(i, action, q_before, q_after, valid_move)
        
        # Draw legend
        self.draw_legend()
        
        # Draw Q-table info
        self.draw_q_table_info()
        
        pygame.display.flip()
    
    def close(self):
        pygame.quit()


# =============================================================================
# CONSOLE LOGGING
# =============================================================================

def print_state(env, step):
    """Print current state to console"""
    print(f"\n{'='*60}")
    print(f"Step {step} - Current State:")
    print(f"{'='*60}")
    
    state = env.get_state()
    top_colors, tube_sizes = state
    
    print("\nTubes:")
    for i, tube in enumerate(env.tubes):
        color_names = [COLOR_NAMES[c] for c in tube]
        print(f"  Tube {i}: {color_names} (size: {len(tube)})")
    
    print(f"\nState representation: top_colors={top_colors}, sizes={tube_sizes}")
    print(f"Sorted balls: {env.count_sorted_balls()}/{NUM_COLORS * TUBE_CAPACITY}")


def print_actions(agents, actions, valid_moves):
    """Print actions chosen by each agent"""
    print("\nActions:")
    for i, (agent, action, valid) in enumerate(zip(agents, actions, valid_moves)):
        from_tube, to_tube = action  # Now action is (from_tube, to_tube)
        
        if from_tube == -1 and to_tube == -1:
            action_str = "STAY"
            status = "VALID"
        else:
            action_str = f"Move from Tube {from_tube} to Tube {to_tube}"
            status = "VALID" if valid is not None and valid[0] != 'STAY' else "INVALID"
        
        print(f"  Agent {i+1}: {action_str} [{status}]")


def print_q_updates(agents, q_values, reward):
    """Print Q-value updates"""
    print(f"\nQ-Learning Updates (Shared Reward: {reward:.2f}):")
    for i, (agent, (q_before, q_after)) in enumerate(zip(agents, q_values)):
        delta = q_after - q_before
        print(f"  Agent {i+1}: Q = {q_before:.3f} → {q_after:.3f} (Δ = {delta:+.3f})")
        print(f"           Q-table size: {agent.get_q_table_size()} entries")


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def train_iql():
    """Main training loop for Independent Q-Learning"""
    
    print("\n" + "="*70)
    print("BALL SORT - COOPERATIVE MULTI-AGENT REINFORCEMENT LEARNING")
    print("Using Independent Q-Learning (IQL)")
    print("="*70)
    print("""
EDUCATIONAL CONCEPTS:
          explica:
1. Multiple agents cooperating toward the same goal
2. Each agent has its own Q-table (Independent Learning)
3. Agents do NOT communicate directly
4. Cooperation emerges through shared rewards
5. This demonstrates a key challenge: non-stationarity
   (each agent's optimal policy depends on other agents)
   
NEW: Each agent can now move balls from ANY tube!
     This allows true cooperation and flexible problem-solving.
    """)
    print("="*70)
    
    # Initialize environment and agents
    env = BallSortEnv()
    
    # Create agents - they can now move from ANY tube (no assigned tube restriction)
    agents = []
    for i in range(NUM_AGENTS):
        agent = IQLAgent(
            agent_id=i,
            num_tubes=NUM_TUBES
        )
        agents.append(agent)
    
    # Initialize visualizer
    viz = BallSortVisualizer(env, agents)
    
    # Training parameters
    epsilon = EPSILON_START
    num_episodes = 5000  # Many more episodes for extended learning
    render_delay = 0.5   # Start slow so students can observe
    fast_mode = False    # Toggle with 'F' key
    
    # Training phases (for 5+ minute session)
    # Phase 1 (0-50): Slow visualization, high exploration
    # Phase 2 (50-200): Medium speed, learning patterns
    # Phase 3 (200-500): Faster, coordination emerging
    # Phase 4 (500+): Fast, exploitation and mastery
    
    # Statistics
    episode_rewards = []
    episode_steps = []
    solved_count = 0
    total_start_time = time.time()
    
    running = True
    episode = 0
    
    try:
        while running and episode < num_episodes:
            # Reset environment
            state = env.reset()
            episode_reward = 0
            step = 0
            
            print(f"\n{'#'*70}")
            print(f"EPISODE {episode + 1}")
            print(f"{'#'*70}")
            print(f"Epsilon (exploration rate): {epsilon:.3f}")
            
            done = False
            
            while not done and running:
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_SPACE:
                            # Pause/resume
                            paused = True
                            while paused:
                                for e in pygame.event.get():
                                    if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                                        paused = False
                                    elif e.type == pygame.QUIT:
                                        running = False
                                        paused = False
                        elif event.key == pygame.K_f:
                            # Toggle fast mode
                            fast_mode = not fast_mode
                            print(f"\n>>> FAST MODE: {'ON' if fast_mode else 'OFF'} <<<\n")
                
                if not running:
                    break
                
                print_state(env, step)
                
                # Each agent selects an action independently
                # Action is now (from_tube, to_tube) - agents can pick ANY tube!
                actions = []
                valid_actions = env.get_all_valid_actions()  # Same valid actions for all
                for agent in agents:
                    action = agent.select_action(state, epsilon, valid_actions)
                    actions.append(action)  # action is already (from_tube, to_tube)
                
                # Execute actions in environment
                next_state, reward, done, info = env.step(actions)
                episode_reward += reward
                
                # Print actions
                print_actions(agents, actions, info['valid_moves'])
                
                # Update each agent's Q-table (shared reward)
                q_values = []
                for i, agent in enumerate(agents):
                    action = actions[i]  # Full (from_tube, to_tube) action
                    q_before, q_after = agent.update(state, action, reward, next_state, done)
                    q_values.append((q_before, q_after))
                
                # Print Q-value updates
                print_q_updates(agents, q_values, reward)
                
                # Calculate current stats
                elapsed_time = time.time() - total_start_time
                solve_rate = 100 * solved_count / max(1, episode) if episode > 0 else 0
                
                # Render visualization
                viz.render(
                    episode=episode + 1,
                    step=step,
                    reward=reward,
                    epsilon=epsilon,
                    actions=actions,
                    q_values=q_values,
                    valid_moves=info['valid_moves'],
                    solved=info['solved'],
                    elapsed_time=elapsed_time,
                    solve_rate=solve_rate,
                    fast_mode=fast_mode
                )
                
                # Delay for visualization (respects fast mode)
                if fast_mode:
                    time.sleep(0.01)  # Minimal delay in fast mode
                else:
                    time.sleep(render_delay)
                
                state = next_state
                step += 1
            
            # Episode complete
            elapsed_time = time.time() - total_start_time
            elapsed_min = elapsed_time / 60
            
            if info['solved']:
                solved_count += 1
                print(f"\n🎉 PUZZLE SOLVED in {step} steps!")
                print("Final tube state:")
                for i, tube in enumerate(env.tubes):
                    color_names = [COLOR_NAMES[c] for c in tube]
                    print(f"  Tube {i}: {color_names} (size: {len(tube)})")
                
                # Re-render to make sure we show the solved state
                viz.render(
                    episode=episode + 1,
                    step=step,
                    reward=reward,
                    epsilon=epsilon,
                    actions=actions,
                    q_values=q_values,
                    valid_moves=info['valid_moves'],
                    solved=True,
                    elapsed_time=elapsed_time,
                    solve_rate=100 * solved_count / max(1, episode + 1),
                    fast_mode=fast_mode
                )
                
                if not fast_mode:
                    time.sleep(1.5)  # Longer pause to see the solved state
                else:
                    time.sleep(0.3)
            
            episode_rewards.append(episode_reward)
            episode_steps.append(step)
            
            # Decay epsilon
            epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
            
            episode += 1
            
            # Print episode summary with timing
            print(f"\nEpisode {episode} Summary:")
            print(f"  Total Reward: {episode_reward:.2f}")
            print(f"  Steps: {step}")
            print(f"  Solved: {info['solved']}")
            print(f"  Episodes Solved So Far: {solved_count}/{episode} ({100*solved_count/episode:.1f}%)")
            print(f"  Training Time: {elapsed_min:.1f} minutes")
            
            # Gradual speed up over 5+ minutes of training
            # Phase 1: Episodes 0-30 (slow, ~0.5s delay)
            # Phase 2: Episodes 30-100 (medium, ~0.2s delay)
            # Phase 3: Episodes 100-300 (faster, ~0.1s delay)
            # Phase 4: Episodes 300+ (fast, ~0.05s delay)
            if not fast_mode:
                if episode <= 30:
                    render_delay = 0.4  # Slow - observe early exploration
                elif episode <= 100:
                    render_delay = 0.2  # Medium - see patterns emerging
                elif episode <= 300:
                    render_delay = 0.1  # Faster - coordination visible
                else:
                    render_delay = 0.05  # Fast - watch exploitation
            
            # Print learning phase info periodically
            if episode in [30, 100, 300, 500]:
                print(f"\n{'='*60}")
                print(f"LEARNING PHASE UPDATE (Episode {episode})")
                print(f"{'='*60}")
                print(f"Epsilon: {epsilon:.3f}")
                print(f"Solve Rate: {100*solved_count/episode:.1f}%")
                if len(episode_rewards) >= 10:
                    print(f"Avg Reward (last 10): {np.mean(episode_rewards[-10:]):.2f}")
                print(f"Training Time: {elapsed_min:.1f} minutes")
                print(f"{'='*60}\n")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    finally:
        # Print final statistics
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"Total Episodes: {episode}")
        print(f"Episodes Solved: {solved_count} ({100*solved_count/max(1,episode):.1f}%)")
        
        if episode_rewards:
            print(f"Average Reward (last 10): {np.mean(episode_rewards[-10:]):.2f}")
            print(f"Average Steps (last 10): {np.mean(episode_steps[-10:]):.1f}")
        
        print("\nAgent Q-Table Statistics:")
        for i, agent in enumerate(agents):
            print(f"  Agent {i+1}: {agent.get_q_table_size()} entries, {agent.total_updates} updates")
        
        print("\n" + "="*70)
        print("KEY OBSERVATIONS FOR STUDENTS:")
        print("="*70)
        print("""
1. INDEPENDENT LEARNING: Each agent maintains its own Q-table
   Q_i(s, a_i) for agent i

2. SHARED REWARD: All agents receive the same reward signal
   This creates implicit coordination

3. NON-STATIONARITY: From each agent's perspective, the environment
   appears non-stationary because other agents are also learning

4. EMERGENT COOPERATION: Despite no explicit communication,
   agents learn to avoid conflicts and coordinate moves

5. EXPLORATION vs EXPLOITATION: Epsilon-greedy ensures agents
   explore different strategies before converging
        """)
        
        viz.close()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("BALL SORT - COOPERATIVE MULTI-AGENT IQL")
    print("="*60)
    print("\nControls:")
    print("  SPACE - Pause/Resume training")
    print("  F     - Toggle Fast Mode (speed up learning)")
    print("  ESC   - Quit")
    print("\nTraining will run for 5+ minutes through 4 phases:")
    print("  Phase 1 (Ep 1-30):   Slow - Observe exploration")
    print("  Phase 2 (Ep 30-100): Medium - See patterns emerge")
    print("  Phase 3 (Ep 100-300): Faster - Watch coordination")
    print("  Phase 4 (Ep 300+):   Fast - Exploitation & mastery")
    print("\nPress F at any time to speed up or slow down.")
    print("="*60)
    print("\nStarting training...\n")
    
    train_iql()

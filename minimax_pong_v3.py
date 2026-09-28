"""
Minimax Q-Learning Pong v3 - FAST LEARNING VERSION
===================================================
"""

import pygame
import random
from collections import defaultdict

# =============================================================================
# CONFIGURATION
# =============================================================================

# Layout: [LEFT PANEL | GAME AREA | RIGHT PANEL]
LEFT_PANEL_WIDTH = 200
RIGHT_PANEL_WIDTH = 200
GAME_WIDTH = 500
GAME_HEIGHT = 450
TOP_BAR_HEIGHT = 50

SCREEN_WIDTH = LEFT_PANEL_WIDTH + GAME_WIDTH + RIGHT_PANEL_WIDTH  # 900
SCREEN_HEIGHT = GAME_HEIGHT + TOP_BAR_HEIGHT  # 500
FPS = 60

# Game area boundaries (for ball/paddle logic)
GAME_LEFT = LEFT_PANEL_WIDTH
GAME_RIGHT = LEFT_PANEL_WIDTH + GAME_WIDTH
GAME_TOP = TOP_BAR_HEIGHT
GAME_BOTTOM = SCREEN_HEIGHT

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 100, 100)
BLUE = (100, 100, 255)
GREEN = (100, 255, 100)
YELLOW = (255, 255, 100)
GRAY = (80, 80, 80)
DARK_GRAY = (40, 40, 40)
PANEL_BG = (25, 25, 35)

PADDLE_WIDTH = 12
PADDLE_HEIGHT = 70
PADDLE_SPEED = 5
PADDLE_MARGIN = 30

BALL_SIZE = 12
BALL_SPEED_X = 5
BALL_SPEED_Y = 4

ACTIONS = ['UP', 'STAY', 'DOWN']
ACTION_TO_DY = {'UP': -PADDLE_SPEED, 'STAY': 0, 'DOWN': PADDLE_SPEED}

# Learning parameters
ALPHA = 0.5       
GAMMA = 0.95
EPSILON_START = 0.3
EPSILON_MIN = 0.02
EPSILON_DECAY = 0.998


# =============================================================================
# SIMPLE AND EFFECTIVE STATE
# =============================================================================

def get_agent_state(ball_y, ball_vx, paddle_y, is_agent1):
    # Is ball coming toward this agent?
    if is_agent1:
        ball_coming = ball_vx < 0  # Ball going left toward agent 1
    else:
        ball_coming = ball_vx > 0  # Ball going right toward agent 2
    
    # Ball position relative to paddle (works regardless of coordinate system)
    diff = ball_y - paddle_y
    threshold = PADDLE_HEIGHT // 3
    
    if diff < -threshold:
        relative_pos = 'ABOVE'
    elif diff > threshold:
        relative_pos = 'BELOW'
    else:
        relative_pos = 'AT'
    
    return (ball_coming, relative_pos)


# =============================================================================
# MINIMAX Q-LEARNING
# =============================================================================

class MinimaxAgent:
    """
    Minimax Q-Learning with joint state-action Q-table.
    
    Q(s, a1, a2) = Expected value when both agents take these actions
    V(s) = max_a1 min_a2 Q(s, a1, a2)  (from Agent 1's perspective)
    """
    
    def __init__(self):
        # Q[state1][state2][(action1, action2)] = value
        self.Q = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.epsilon = EPSILON_START
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.update_count = 0
    
    def get_joint_state(self, game):
        """Get states for both agents"""
        s1 = get_agent_state(game.ball_y, game.ball_vx, game.paddle1_y, True)
        s2 = get_agent_state(game.ball_y, game.ball_vx, game.paddle2_y, False)
        return (s1, s2)
    
    def get_state_value(self, s1, s2):
        """V(s) = max_a1 min_a2 Q(s, a1, a2)"""
        q_table = self.Q[s1][s2]
        
        best_value = float('-inf')
        for a1 in ACTIONS:
            worst_for_a1 = min(q_table[(a1, a2)] for a2 in ACTIONS)
            best_value = max(best_value, worst_for_a1)
        
        return best_value if best_value != float('-inf') else 0.0
    
    def select_actions(self, s1, s2):
        """Select actions using epsilon-greedy Minimax"""
        if random.random() < self.epsilon:
            # Explore
            a1 = random.choice(ACTIONS)
            a2 = random.choice(ACTIONS)
            exploring = True
        else:
            # Exploit using Minimax
            q_table = self.Q[s1][s2]
            
            # Agent 1: find action that maximizes minimum
            best_a1 = ACTIONS[0]
            best_min = float('-inf')
            for a1 in ACTIONS:
                min_val = min(q_table[(a1, a2)] for a2 in ACTIONS)
                if min_val > best_min:
                    best_min = min_val
                    best_a1 = a1
            
            # Agent 2: find action that minimizes Q given Agent 1's choice
            best_a2 = ACTIONS[0]
            best_val = float('inf')
            for a2 in ACTIONS:
                val = q_table[(best_a1, a2)]
                if val < best_val:
                    best_val = val
                    best_a2 = a2
            
            a1, a2 = best_a1, best_a2
            exploring = False
        
        return a1, a2, exploring
    
    def get_optimal_action_for_state(self, ball_coming, relative_pos):
        if ball_coming:
            if relative_pos == 'ABOVE':
                return 'UP'
            elif relative_pos == 'BELOW':
                return 'DOWN'
            else:
                return 'STAY'
        else:
            # Ball going away - optimal is often STAY or return to center
            return 'STAY'
    
    def update(self, s1, s2, a1, a2, reward, next_s1, next_s2):
        """Q-learning update with Minimax value"""
        current_q = self.Q[s1][s2][(a1, a2)]
        next_v = self.get_state_value(next_s1, next_s2)
        
        target = reward + self.gamma * next_v
        self.Q[s1][s2][(a1, a2)] += self.alpha * (target - current_q)
        self.update_count += 1
    
    def decay_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
    
    def get_q(self, s1, s2, a1, a2):
        return self.Q[s1][s2][(a1, a2)]
    
    def count_learned_states(self):
        """Count how many state-action pairs have been visited"""
        count = 0
        for s1 in self.Q:
            for s2 in self.Q[s1]:
                for (a1, a2), val in self.Q[s1][s2].items():
                    if val != 0:
                        count += 1
        return count


# =============================================================================
# PONG GAME
# =============================================================================

class PongGame:
    def __init__(self):
        self.reset()
    
    def reset(self):
        # Paddles positioned within game area
        self.paddle1_y = GAME_TOP + GAME_HEIGHT // 2
        self.paddle2_y = GAME_TOP + GAME_HEIGHT // 2
        self.reset_ball()
        self.score1 = 0
        self.score2 = 0
        self.rally = 0
        self.max_rally = 0
        self.total_hits = 0
    
    def reset_ball(self):
        self.ball_x = GAME_LEFT + GAME_WIDTH // 2
        self.ball_y = GAME_TOP + GAME_HEIGHT // 2
        self.ball_vx = BALL_SPEED_X * random.choice([-1, 1])
        self.ball_vy = BALL_SPEED_Y * random.uniform(-1, 1)
        self.rally = 0
    
    def apply_actions(self, a1, a2):
        self.paddle1_y += ACTION_TO_DY[a1]
        self.paddle2_y += ACTION_TO_DY[a2]
        
        # Clamp within game area
        h = PADDLE_HEIGHT // 2
        self.paddle1_y = max(GAME_TOP + h, min(GAME_BOTTOM - h, self.paddle1_y))
        self.paddle2_y = max(GAME_TOP + h, min(GAME_BOTTOM - h, self.paddle2_y))
    
    def update(self):
        """Update ball, return reward for Agent 1"""
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy
        
        # Wall bounce (top and bottom of game area)
        if self.ball_y <= GAME_TOP + BALL_SIZE // 2:
            self.ball_y = GAME_TOP + BALL_SIZE // 2
            self.ball_vy = abs(self.ball_vy)
        elif self.ball_y >= GAME_BOTTOM - BALL_SIZE // 2:
            self.ball_y = GAME_BOTTOM - BALL_SIZE // 2
            self.ball_vy = -abs(self.ball_vy)
        
        # Paddle 1 (left side of game area)
        p1_x = GAME_LEFT + PADDLE_MARGIN
        p1_right = p1_x + PADDLE_WIDTH
        p1_top = self.paddle1_y - PADDLE_HEIGHT // 2
        p1_bot = self.paddle1_y + PADDLE_HEIGHT // 2
        
        if (self.ball_x - BALL_SIZE//2 <= p1_right and 
            self.ball_x > p1_x and
            p1_top <= self.ball_y <= p1_bot and 
            self.ball_vx < 0):
            self.ball_vx = abs(self.ball_vx)
            self.ball_x = p1_right + BALL_SIZE // 2
            self.rally += 1
            self.total_hits += 1
            offset = (self.ball_y - self.paddle1_y) / (PADDLE_HEIGHT / 2)
            self.ball_vy = BALL_SPEED_Y * offset
        
        # Paddle 2 (right side of game area)
        p2_x = GAME_RIGHT - PADDLE_MARGIN - PADDLE_WIDTH
        p2_right = GAME_RIGHT - PADDLE_MARGIN
        p2_top = self.paddle2_y - PADDLE_HEIGHT // 2
        p2_bot = self.paddle2_y + PADDLE_HEIGHT // 2
        
        if (self.ball_x + BALL_SIZE//2 >= p2_x and 
            self.ball_x < p2_right and
            p2_top <= self.ball_y <= p2_bot and 
            self.ball_vx > 0):
            self.ball_vx = -abs(self.ball_vx)
            self.ball_x = p2_x - BALL_SIZE // 2
            self.rally += 1
            self.total_hits += 1
            offset = (self.ball_y - self.paddle2_y) / (PADDLE_HEIGHT / 2)
            self.ball_vy = BALL_SPEED_Y * offset
        
        # Scoring
        reward = 0
        scored = False
        
        if self.ball_x <= GAME_LEFT:
            # Agent 2 scores
            reward = -1
            self.score2 += 1
            self.max_rally = max(self.max_rally, self.rally)
            self.reset_ball()
            scored = True
        elif self.ball_x >= GAME_RIGHT:
            # Agent 1 scores
            reward = +1
            self.score1 += 1
            self.max_rally = max(self.max_rally, self.rally)
            self.reset_ball()
            scored = True
        
        return reward, scored


# =============================================================================
# RENDERER - Separate panels from game area
# =============================================================================

class Renderer:
    def __init__(self, screen):
        self.screen = screen
        self.font_big = pygame.font.Font(None, 42)
        self.font_med = pygame.font.Font(None, 26)
        self.font_small = pygame.font.Font(None, 20)
        self.font_tiny = pygame.font.Font(None, 17)
    
    def render(self, game, agent, s1, s2, a1, a2, exploring, step, speed):
        self.screen.fill(DARK_GRAY)
        
        # Draw the three sections
        self._draw_left_panel(game, agent, s1, s2, a1, a2, exploring)
        self._draw_game_area(game, a1, a2)
        self._draw_right_panel(game, agent, s1, s2, a1, a2, step, speed)
        self._draw_top_bar(game, step, speed)
    
    def _draw_top_bar(self, game, step, speed):
        """Top bar with score and rally"""
        pygame.draw.rect(self.screen, (30, 30, 50), (0, 0, SCREEN_WIDTH, TOP_BAR_HEIGHT))
        pygame.draw.line(self.screen, GRAY, (0, TOP_BAR_HEIGHT), (SCREEN_WIDTH, TOP_BAR_HEIGHT), 2)
        
        # Title
        title = self.font_med.render("MINIMAX Q-LEARNING PONG", True, YELLOW)
        self.screen.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, 5))
        
        # Score in center
        score = self.font_big.render(f"{game.score1}  -  {game.score2}", True, WHITE)
        self.screen.blit(score, (SCREEN_WIDTH // 2 - score.get_width() // 2, 22))
        
        # Rally on left
        rally_color = GREEN if game.rally >= 5 else YELLOW if game.rally >= 2 else WHITE
        rally = self.font_med.render(f"Échanges: {game.rally}", True, rally_color)
        self.screen.blit(rally, (LEFT_PANEL_WIDTH + 10, 15))
        
        # Speed on right
        speed_text = self.font_med.render(f"Vitesse: {speed}x", True, GRAY)
        self.screen.blit(speed_text, (GAME_RIGHT - 110, 15))
    
    def _draw_left_panel(self, game, agent, s1, s2, a1, a2, exploring):
        """Left panel: Agent States"""
        # Panel background
        pygame.draw.rect(self.screen, PANEL_BG, (0, TOP_BAR_HEIGHT, LEFT_PANEL_WIDTH, GAME_HEIGHT))
        pygame.draw.line(self.screen, GRAY, (LEFT_PANEL_WIDTH, TOP_BAR_HEIGHT), 
                        (LEFT_PANEL_WIDTH, SCREEN_HEIGHT), 2)
        
        x, y = 10, TOP_BAR_HEIGHT + 15
        lh = 22
        
        # Section: États des Agents
        self.screen.blit(self.font_med.render("ÉTATS DES AGENTS", True, YELLOW), (x, y))
        y += lh + 10
        
        # Agent 1 state box
        pygame.draw.rect(self.screen, (30, 30, 60), (x, y, 180, 80), border_radius=5)
        pygame.draw.rect(self.screen, BLUE, (x, y, 180, 80), 2, border_radius=5)
        
        # Translate state values to French
        ball_coming_1 = "Oui" if s1[0] else "Non"
        ball_pos_1 = {"ABOVE": "AU-DESSUS", "AT": "ALIGNÉ", "BELOW": "EN-DESSOUS"}.get(s1[1], s1[1])
        action_1 = {"UP": "HAUT", "DOWN": "BAS", "STAY": "RESTE"}.get(a1, a1)
        
        self.screen.blit(self.font_small.render("AGENT 1 (Bleu)", True, BLUE), (x + 10, y + 8))
        self.screen.blit(self.font_tiny.render(f"Balle arrive: {ball_coming_1}", True, WHITE), (x + 10, y + 28))
        self.screen.blit(self.font_tiny.render(f"Position: {ball_pos_1}", True, WHITE), (x + 10, y + 44))
        a1_color = GREEN if a1 == "STAY" else YELLOW
        self.screen.blit(self.font_tiny.render(f"Action: {action_1}", True, a1_color), (x + 10, y + 60))
        y += 95
        
        # Agent 2 state box
        pygame.draw.rect(self.screen, (60, 30, 30), (x, y, 180, 80), border_radius=5)
        pygame.draw.rect(self.screen, RED, (x, y, 180, 80), 2, border_radius=5)
        
        # Translate state values to French
        ball_coming_2 = "Oui" if s2[0] else "Non"
        ball_pos_2 = {"ABOVE": "AU-DESSUS", "AT": "ALIGNÉ", "BELOW": "EN-DESSOUS"}.get(s2[1], s2[1])
        action_2 = {"UP": "HAUT", "DOWN": "BAS", "STAY": "RESTE"}.get(a2, a2)
        
        self.screen.blit(self.font_small.render("AGENT 2 (Rouge)", True, RED), (x + 10, y + 8))
        self.screen.blit(self.font_tiny.render(f"Balle arrive: {ball_coming_2}", True, WHITE), (x + 10, y + 28))
        self.screen.blit(self.font_tiny.render(f"Position: {ball_pos_2}", True, WHITE), (x + 10, y + 44))
        a2_color = GREEN if a2 == "STAY" else YELLOW
        self.screen.blit(self.font_tiny.render(f"Action: {action_2}", True, a2_color), (x + 10, y + 60))
        y += 100
        
        # Section: Explication Minimax
        self.screen.blit(self.font_med.render("RÈGLE MINIMAX", True, GREEN), (x, y))
        y += lh + 5
        
        pygame.draw.rect(self.screen, (25, 40, 25), (x, y, 180, 100), border_radius=5)
        pygame.draw.rect(self.screen, GREEN, (x, y, 180, 100), 2, border_radius=5)
        
        explanations = [
            "Jeu à Somme Nulle:",
            "  Agent1 gagne → Agent2 perd",
            "",
            "Agent 1: Joueur MAX",
            "Agent 2: Joueur MIN",
            "",
            "V(s) = max min Q(s,a1,a2)"
        ]
        for i, line in enumerate(explanations):
            color = BLUE if "Agent 1" in line or "Agent1" in line else (RED if "Agent 2" in line or "Agent2" in line else WHITE)
            self.screen.blit(self.font_tiny.render(line, True, color), (x + 8, y + 5 + i * 13))
        y += 115
        
        # Mode indicator
        mode = "EXPLORATION" if exploring else "EXPLOITATION"
        mode_color = GREEN if exploring else YELLOW
        self.screen.blit(self.font_small.render(f"Mode: {mode}", True, mode_color), (x, y))
    
    def _draw_game_area(self, game, a1, a2):
        """Center: Clean game view"""
        # Game background
        pygame.draw.rect(self.screen, BLACK, 
                        (GAME_LEFT, GAME_TOP, GAME_WIDTH, GAME_HEIGHT))
        
        # Center line
        for y in range(GAME_TOP, GAME_BOTTOM, 20):
            pygame.draw.rect(self.screen, GRAY, 
                           (GAME_LEFT + GAME_WIDTH // 2 - 1, y, 2, 10))
        
        # Paddle 1 (left)
        p1_x = GAME_LEFT + PADDLE_MARGIN
        p1_rect = pygame.Rect(p1_x, game.paddle1_y - PADDLE_HEIGHT // 2,
                              PADDLE_WIDTH, PADDLE_HEIGHT)
        pygame.draw.rect(self.screen, BLUE, p1_rect, border_radius=3)
        
        # Paddle 2 (right)
        p2_x = GAME_RIGHT - PADDLE_MARGIN - PADDLE_WIDTH
        p2_rect = pygame.Rect(p2_x, game.paddle2_y - PADDLE_HEIGHT // 2,
                              PADDLE_WIDTH, PADDLE_HEIGHT)
        pygame.draw.rect(self.screen, RED, p2_rect, border_radius=3)
        
        # Ball trail
        trail_x = game.ball_x - game.ball_vx * 2
        trail_y = game.ball_y - game.ball_vy * 2
        pygame.draw.circle(self.screen, (60, 60, 60), (int(trail_x), int(trail_y)), BALL_SIZE // 3)
        
        # Ball
        pygame.draw.circle(self.screen, WHITE, (int(game.ball_x), int(game.ball_y)), BALL_SIZE // 2)
        
        # Direction arrow
        arrow_end_x = game.ball_x + game.ball_vx * 8
        arrow_end_y = game.ball_y + game.ball_vy * 8
        pygame.draw.line(self.screen, YELLOW, (game.ball_x, game.ball_y),
                        (arrow_end_x, arrow_end_y), 2)
        
        # Agent labels at bottom of game area
        self.screen.blit(self.font_small.render("Agent 1", True, BLUE), 
                        (GAME_LEFT + 10, GAME_BOTTOM - 25))
        t2 = self.font_small.render("Agent 2", True, RED)
        self.screen.blit(t2, (GAME_RIGHT - 10 - t2.get_width(), GAME_BOTTOM - 25))
        
        # Action arrows next to paddles
        a1_arrow = "↑" if a1 == "UP" else ("↓" if a1 == "DOWN" else "•")
        a2_arrow = "↑" if a2 == "UP" else ("↓" if a2 == "DOWN" else "•")
        
        arrow_font = pygame.font.Font(None, 36)
        self.screen.blit(arrow_font.render(a1_arrow, True, BLUE), 
                        (p1_x + PADDLE_WIDTH + 5, game.paddle1_y - 12))
        self.screen.blit(arrow_font.render(a2_arrow, True, RED), 
                        (p2_x - 20, game.paddle2_y - 12))
    
    def _draw_right_panel(self, game, agent, s1, s2, a1, a2, step, speed):
        """Right panel: Learning Statistics"""
        # Panel background
        pygame.draw.rect(self.screen, PANEL_BG, 
                        (GAME_RIGHT, TOP_BAR_HEIGHT, RIGHT_PANEL_WIDTH, GAME_HEIGHT))
        pygame.draw.line(self.screen, GRAY, (GAME_RIGHT, TOP_BAR_HEIGHT), 
                        (GAME_RIGHT, SCREEN_HEIGHT), 2)
        
        x, y = GAME_RIGHT + 10, TOP_BAR_HEIGHT + 15
        lh = 22
        
        # Section: Apprentissage Q
        self.screen.blit(self.font_med.render("APPRENTISSAGE Q", True, YELLOW), (x, y))
        y += lh + 10
        
        # Q-value box
        pygame.draw.rect(self.screen, (30, 35, 45), (x, y, 180, 75), border_radius=5)
        pygame.draw.rect(self.screen, WHITE, (x, y, 180, 75), 1, border_radius=5)
        
        q_val = agent.get_q(s1, s2, a1, a2)
        v_val = agent.get_state_value(s1, s2)
        
        self.screen.blit(self.font_tiny.render("Valeur Q actuelle:", True, GRAY), (x + 8, y + 8))
        q_color = GREEN if q_val > 0 else (RED if q_val < 0 else WHITE)
        self.screen.blit(self.font_med.render(f"Q = {q_val:+.3f}", True, q_color), (x + 8, y + 24))
        
        self.screen.blit(self.font_tiny.render("Valeur d'état V(s):", True, GRAY), (x + 8, y + 48))
        v_color = GREEN if v_val > 0 else (RED if v_val < 0 else WHITE)
        self.screen.blit(self.font_small.render(f"V = {v_val:+.3f}", True, v_color), (x + 8, y + 62))
        y += 90
        
        # Section: Statistiques
        self.screen.blit(self.font_med.render("STATISTIQUES", True, YELLOW), (x, y))
        y += lh + 5
        
        stats = [
            (f"Étape: {step}", WHITE),
            (f"Epsilon: {agent.epsilon:.3f}", WHITE),
            (f"", WHITE),
            (f"Échanges max: {game.max_rally}", GREEN if game.max_rally >= 10 else WHITE),
            (f"Frappes totales: {game.total_hits}", WHITE),
            (f"", WHITE),
            (f"Entrées Q: {agent.count_learned_states()}", GRAY),
        ]
        
        for text, color in stats:
            self.screen.blit(self.font_small.render(text, True, color), (x, y))
            y += lh - 2
        y += 15
        
        # Section: Contrôles
        self.screen.blit(self.font_med.render("CONTRÔLES", True, YELLOW), (x, y))
        y += lh + 5
        
        controls = [
            "1-9: Vitesse",
            "ESPACE: Pause",
            "R: Réinitialiser",
            "ESC: Quitter",
        ]
        for ctrl in controls:
            self.screen.blit(self.font_tiny.render(ctrl, True, GRAY), (x, y))
            y += 16


# =============================================================================
# MAIN
# =============================================================================

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Minimax Q-Learning Pong v3 - FAST LEARNING")
    clock = pygame.time.Clock()
    
    game = PongGame()
    agent = MinimaxAgent()
    renderer = Renderer(screen)
    
    running = True
    paused = False
    step = 0
    speed = 1
    
    print("\n" + "="*60)
    print("MINIMAX Q-LEARNING PONG v3 - FAST LEARNING")
    print("="*60)
    print("\nKEY INSIGHT:")
    print("  Simple state = (ball_coming?, ball_above/at/below?)")
    print("  This lets agents learn the RIGHT thing quickly!")
    print("\nCONTROLS:")
    print("  1-9: Speed (1=1x, 5=20x, 9=100x)")
    print("  SPACE: Pause")
    print("  R: Reset")
    print("\nWATCH: Rally counter should increase within 1 minute!")
    print("="*60 + "\n")
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    game = PongGame()
                    agent = MinimaxAgent()
                    step = 0
                    print("RESET!")
                elif event.key >= pygame.K_1 and event.key <= pygame.K_9:
                    speeds = [1, 2, 5, 10, 20, 30, 50, 75, 100]
                    speed = speeds[event.key - pygame.K_1]
                    print(f"Speed: {speed}x")
        
        if paused:
            s1, s2 = agent.get_joint_state(game)
            renderer.render(game, agent, s1, s2, 'STAY', 'STAY', False, step, speed)
            pause_text = pygame.font.Font(None, 72).render("PAUSED", True, YELLOW)
            screen.blit(pause_text, (SCREEN_WIDTH//2 - pause_text.get_width()//2, 
                                     SCREEN_HEIGHT//2 - 36))
            pygame.display.flip()
            clock.tick(FPS)
            continue
        
        for _ in range(speed):
            step += 1
            
            # 1. Get current state
            s1, s2 = agent.get_joint_state(game)
            
            # 2. Select actions
            a1, a2, exploring = agent.select_actions(s1, s2)
            
            # 3. Apply actions
            game.apply_actions(a1, a2)
            
            # 4-5. Update ball, get reward
            reward, scored = game.update()
            
            # 6. Get next state
            next_s1, next_s2 = agent.get_joint_state(game)
            
            # 7. Update Q-table
            agent.update(s1, s2, a1, a2, reward, next_s1, next_s2)
            agent.decay_epsilon()
            
            # Logging
            if step % 500 == 0:
                print(f"Step {step:5d} | Score: {game.score1}-{game.score2} | "
                      f"Max Rally: {game.max_rally:2d} | Total Hits: {game.total_hits} | "
                      f"ε: {agent.epsilon:.3f}")
            
            if scored and game.max_rally > 0 and game.rally == 0:
                # Just scored, rally was > 0
                pass
        
        # 8. Render
        s1, s2 = agent.get_joint_state(game)
        a1, a2, exploring = agent.select_actions(s1, s2)
        renderer.render(game, agent, s1, s2, a1, a2, exploring, step, speed)
        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()
    print(f"\nFINAL: Steps={step}, Score={game.score1}-{game.score2}, "
          f"MaxRally={game.max_rally}, TotalHits={game.total_hits}")


if __name__ == "__main__":
    main()

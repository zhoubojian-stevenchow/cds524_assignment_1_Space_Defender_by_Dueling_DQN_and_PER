"""
=============================================================================
SPACE DEFENDER - LOCAL SCRIPT WITH VIDEO RECORDING
=============================================================================
Run this on your laptop to record gameplay videos.

SETUP:
1. pip install pygame torch numpy imageio imageio-ffmpeg
2. Download best_eval_model.pth from Google Drive
3. Put it in the same folder as this script
4. Run: python space_defender_local.py

MODES:
- Interactive: Watch and control playback
- Auto-record: Automatically record N games to video file

=============================================================================
"""

import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import os

# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    SCREEN_WIDTH = 600
    SCREEN_HEIGHT = 800
    PLAYER_WIDTH = 50
    PLAYER_HEIGHT = 40
    PLAYER_SPEED = 8
    PLAYER_BULLET_SPEED = 12
    PLAYER_SHOOT_COOLDOWN = 15
    PLAYER_MAX_HEALTH = 3
    ENEMY_WIDTH = 40
    ENEMY_HEIGHT = 35
    ENEMY_SPEED = 3
    ENEMY_BULLET_SPEED = 6
    ENEMY_SPAWN_RATE = 60
    MAX_ENEMIES = 8
    ENEMY_SHOOT_CHANCE = 0.02
    STATE_SIZE = 88
    ACTION_SIZE = 6

# =============================================================================
# GAME OBJECTS
# =============================================================================
class Player:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.width, self.height = Config.PLAYER_WIDTH, Config.PLAYER_HEIGHT
        self.speed = Config.PLAYER_SPEED
        self.health = Config.PLAYER_MAX_HEALTH
        self.shoot_cooldown = 0
        self.score = 0
        self.alive = True

    def move(self, direction):
        self.x = max(0, min(self.x + direction * self.speed, Config.SCREEN_WIDTH - self.width))

    def update(self):
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1

    def can_shoot(self):
        return self.shoot_cooldown == 0

    def shoot(self):
        self.shoot_cooldown = Config.PLAYER_SHOOT_COOLDOWN
        return Bullet(self.x + self.width // 2 - 3, self.y - 15, -Config.PLAYER_BULLET_SPEED, True)

    def take_damage(self):
        self.health -= 1
        if self.health <= 0:
            self.alive = False


class Enemy:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.width, self.height = Config.ENEMY_WIDTH, Config.ENEMY_HEIGHT
        self.speed = Config.ENEMY_SPEED + random.uniform(-1, 1)
        self.health, self.alive = 1, True
        self.direction = random.choice([-1, 0, 1])
        self.change_dir_timer = random.randint(30, 90)

    def update(self):
        self.y += self.speed
        self.x = max(0, min(self.x + self.direction * 2, Config.SCREEN_WIDTH - self.width))
        self.change_dir_timer -= 1
        if self.change_dir_timer <= 0:
            self.direction = random.choice([-1, 0, 1])
            self.change_dir_timer = random.randint(30, 90)
        if self.y > Config.SCREEN_HEIGHT:
            self.alive = False

    def should_shoot(self):
        return random.random() < Config.ENEMY_SHOOT_CHANCE

    def shoot(self):
        return Bullet(self.x + self.width // 2 - 3, self.y + self.height, Config.ENEMY_BULLET_SPEED, False)


class Bullet:
    def __init__(self, x, y, speed, is_player_bullet):
        self.x, self.y, self.speed = x, y, speed
        self.width, self.height = 6, 15
        self.is_player_bullet = is_player_bullet
        self.alive = True

    def update(self):
        self.y += self.speed
        if self.y < -self.height or self.y > Config.SCREEN_HEIGHT:
            self.alive = False

# =============================================================================
# ENVIRONMENT
# =============================================================================
class SpaceDefenderEnv:
    def __init__(self):
        self.state_buffer = deque(maxlen=4)
        self.reset()

    def reset(self):
        self.player = Player(
            Config.SCREEN_WIDTH // 2 - Config.PLAYER_WIDTH // 2,
            Config.SCREEN_HEIGHT - 100
        )
        self.enemies = []
        self.player_bullets = []
        self.enemy_bullets = []
        self.frame_count = 0
        self.spawn_timer = 0
        self.game_over = False
        self.enemies_destroyed = 0
        self.survival_time = 0

        initial_frame = self._get_single_frame()
        for _ in range(4):
            self.state_buffer.append(initial_frame)
        return self._get_stacked_state()

    def _get_stacked_state(self):
        return np.concatenate(self.state_buffer)

    def _get_single_frame(self):
        state = []
        state.append(self.player.x / Config.SCREEN_WIDTH)
        state.append(1.0 if self.player.can_shoot() else 0.0)
        state.append(self.player.health / Config.PLAYER_MAX_HEALTH)

        enemies_sorted = sorted(
            self.enemies,
            key=lambda e: abs(e.x - self.player.x) + abs(e.y - self.player.y)
        )[:3]

        for i in range(3):
            if i < len(enemies_sorted):
                enemy = enemies_sorted[i]
                rel_x = (enemy.x - self.player.x) / Config.SCREEN_WIDTH
                rel_y = (enemy.y - self.player.y) / Config.SCREEN_HEIGHT
                threat = 1.0 if abs(enemy.x - self.player.x) < 100 else 0.0
                state.extend([rel_x, rel_y, threat])
            else:
                state.extend([0.0, -1.0, 0.0])

        bullets_sorted = sorted(
            self.enemy_bullets,
            key=lambda b: abs(b.x - self.player.x) + abs(b.y - self.player.y)
        )[:5]

        for i in range(5):
            if i < len(bullets_sorted):
                bullet = bullets_sorted[i]
                rel_x = (bullet.x - self.player.x) / Config.SCREEN_WIDTH
                rel_y = (bullet.y - self.player.y) / Config.SCREEN_HEIGHT
                state.extend([rel_x, rel_y])
            else:
                state.extend([0.0, -1.0])

        return np.array(state, dtype=np.float32)

    def _check_collision(self, obj1, obj2):
        return (obj1.x < obj2.x + obj2.width and
                obj1.x + obj1.width > obj2.x and
                obj1.y < obj2.y + obj2.height and
                obj1.y + obj1.height > obj2.y)

    def step(self, action):
        self.frame_count += 1
        self.survival_time += 1
        reward = 0.01

        move_dir = 0
        should_shoot = False

        if action == 0: move_dir = -1
        elif action == 1: move_dir = 1
        elif action == 2: move_dir = 0
        elif action == 3: should_shoot = True
        elif action == 4: move_dir = -1; should_shoot = True
        elif action == 5: move_dir = 1; should_shoot = True

        self.player.move(move_dir)
        self.player.update()

        if should_shoot and self.player.can_shoot():
            self.player_bullets.append(self.player.shoot())

        self.spawn_timer += 1
        if self.spawn_timer >= Config.ENEMY_SPAWN_RATE and len(self.enemies) < Config.MAX_ENEMIES:
            self.spawn_timer = 0
            spawn_x = random.randint(0, Config.SCREEN_WIDTH - Config.ENEMY_WIDTH)
            self.enemies.append(Enemy(spawn_x, -Config.ENEMY_HEIGHT))

        for enemy in self.enemies[:]:
            enemy.update()
            if enemy.should_shoot():
                self.enemy_bullets.append(enemy.shoot())
            if not enemy.alive:
                self.enemies.remove(enemy)

        for bullet in self.player_bullets[:]:
            bullet.update()
            if not bullet.alive:
                self.player_bullets.remove(bullet)

        for bullet in self.enemy_bullets[:]:
            bullet.update()
            if not bullet.alive:
                self.enemy_bullets.remove(bullet)

        for bullet in self.player_bullets[:]:
            for enemy in self.enemies[:]:
                if self._check_collision(bullet, enemy):
                    bullet.alive = False
                    enemy.health -= 1
                    reward += 10
                    if enemy.health <= 0:
                        enemy.alive = False
                        self.enemies.remove(enemy)
                        self.enemies_destroyed += 1
                        self.player.score += 100
                        reward += 50
                    if bullet in self.player_bullets:
                        self.player_bullets.remove(bullet)
                    break

        for bullet in self.enemy_bullets[:]:
            if self._check_collision(bullet, self.player):
                bullet.alive = False
                self.enemy_bullets.remove(bullet)
                self.player.take_damage()
                reward -= 30
                if not self.player.alive:
                    self.game_over = True
                    reward -= 100

        for enemy in self.enemies[:]:
            if self._check_collision(enemy, self.player):
                enemy.alive = False
                self.enemies.remove(enemy)
                self.player.take_damage()
                reward -= 30
                if not self.player.alive:
                    self.game_over = True
                    reward -= 100

        done = self.game_over
        info = {
            'score': self.player.score,
            'enemies_destroyed': self.enemies_destroyed,
            'survival_time': self.survival_time,
            'health': self.player.health
        }

        self.state_buffer.append(self._get_single_frame())
        return self._get_stacked_state(), reward, done, info

# =============================================================================
# NEURAL NETWORK - DYNAMIC ARCHITECTURE (Works with ANY model!)
# =============================================================================

def inspect_model_file(model_path):
    """
    Print all keys in a .pth file to understand its structure.
    """
    checkpoint = torch.load(model_path, map_location='cpu')
    
    print(f"\n   === MODEL STRUCTURE ===")
    for key, value in checkpoint.items():
        if hasattr(value, 'shape'):
            print(f"   {key}: {list(value.shape)}")
        else:
            print(f"   {key}: {type(value)}")
    print()
    
    return checkpoint


def inspect_model_architecture(model_path):
    """
    Inspect a .pth file and determine the network architecture.
    Returns layer sizes for feature, value, and advantage streams.
    Supports multiple naming conventions.
    """
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Try to detect the naming pattern
    keys = list(checkpoint.keys())
    
    # Check for different possible naming conventions
    has_feature = any('feature' in k for k in keys)
    has_fc = any('fc' in k.lower() for k in keys)
    has_layers = any('layers' in k for k in keys)
    has_hidden = any('hidden' in k for k in keys)
    
    architecture = {
        'feature': [],
        'value': [],
        'advantage': [],
        'type': 'unknown'
    }
    
    # Pattern 1: Dueling DQN with feature/value_stream/advantage_stream
    if has_feature and any('value_stream' in k for k in keys):
        architecture['type'] = 'dueling_dqn'
        
        # Find feature layers
        i = 0
        while f'feature.{i}.weight' in checkpoint:
            weight_shape = checkpoint[f'feature.{i}.weight'].shape
            architecture['feature'].append((weight_shape[1], weight_shape[0]))
            i += 2  # Skip ReLU (no weights)
        
        # Find value stream layers
        i = 0
        while f'value_stream.{i}.weight' in checkpoint:
            weight_shape = checkpoint[f'value_stream.{i}.weight'].shape
            architecture['value'].append((weight_shape[1], weight_shape[0]))
            i += 2
        
        # Find advantage stream layers
        i = 0
        while f'advantage_stream.{i}.weight' in checkpoint:
            weight_shape = checkpoint[f'advantage_stream.{i}.weight'].shape
            architecture['advantage'].append((weight_shape[1], weight_shape[0]))
            i += 2
    
    # Pattern 2: Simple sequential with fc1, fc2, fc3, etc.
    elif has_fc:
        architecture['type'] = 'simple_dqn'
        i = 1
        while f'fc{i}.weight' in checkpoint:
            weight_shape = checkpoint[f'fc{i}.weight'].shape
            architecture['feature'].append((weight_shape[1], weight_shape[0]))
            i += 1
    
    # Pattern 3: layers.0, layers.1, etc.
    elif has_layers:
        architecture['type'] = 'sequential'
        i = 0
        while f'layers.{i}.weight' in checkpoint:
            weight_shape = checkpoint[f'layers.{i}.weight'].shape
            architecture['feature'].append((weight_shape[1], weight_shape[0]))
            i += 1
    
    # Pattern 4: Try to find any Linear layer patterns
    else:
        architecture['type'] = 'auto_detect'
        weight_keys = sorted([k for k in keys if 'weight' in k and 'bias' not in k])
        for key in weight_keys:
            weight_shape = checkpoint[key].shape
            if len(weight_shape) == 2:  # Linear layer
                architecture['feature'].append((weight_shape[1], weight_shape[0]))
    
    return architecture


class DynamicDuelingDQN(nn.Module):
    """
    Dynamically builds a Dueling DQN that matches ANY saved model architecture.
    """
    def __init__(self, architecture):
        super().__init__()
        
        # Build feature layers dynamically
        feature_modules = []
        for in_size, out_size in architecture['feature']:
            feature_modules.append(nn.Linear(in_size, out_size))
            feature_modules.append(nn.ReLU())
        self.feature = nn.Sequential(*feature_modules)
        
        # Build value stream dynamically
        value_modules = []
        for i, (in_size, out_size) in enumerate(architecture['value']):
            value_modules.append(nn.Linear(in_size, out_size))
            if i < len(architecture['value']) - 1:  # Don't add ReLU after final layer
                value_modules.append(nn.ReLU())
        self.value_stream = nn.Sequential(*value_modules)
        
        # Build advantage stream dynamically
        advantage_modules = []
        for i, (in_size, out_size) in enumerate(architecture['advantage']):
            advantage_modules.append(nn.Linear(in_size, out_size))
            if i < len(architecture['advantage']) - 1:  # Don't add ReLU after final layer
                advantage_modules.append(nn.ReLU())
        self.advantage_stream = nn.Sequential(*advantage_modules)

    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


class SimpleDQN(nn.Module):
    """
    Simple DQN for models that don't use dueling architecture.
    """
    def __init__(self, layer_sizes):
        super().__init__()
        
        layers = []
        for i, (in_size, out_size) in enumerate(layer_sizes):
            layers.append(nn.Linear(in_size, out_size))
            if i < len(layer_sizes) - 1:  # Don't add ReLU after final layer
                layers.append(nn.ReLU())
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)


def load_any_model(model_path):
    """
    Load ANY .pth model file by automatically detecting its architecture.
    """
    print(f"   Inspecting model architecture...")
    
    # First, show the model structure for debugging
    checkpoint = inspect_model_file(model_path)
    
    # Inspect the saved weights to determine architecture
    architecture = inspect_model_architecture(model_path)
    
    print(f"   Model type: {architecture['type']}")
    
    if architecture['type'] == 'dueling_dqn':
        if architecture['feature']:
            feature_sizes = [str(l[1]) for l in architecture['feature']]
            print(f"   Feature: {architecture['feature'][0][0]} → {' → '.join(feature_sizes)}")
        if architecture['value']:
            value_sizes = [str(l[1]) for l in architecture['value']]
            print(f"   Value: {' → '.join(value_sizes)}")
        if architecture['advantage']:
            adv_sizes = [str(l[1]) for l in architecture['advantage']]
            print(f"   Advantage: {' → '.join(adv_sizes)}")
        
        model = DynamicDuelingDQN(architecture)
        model.load_state_dict(checkpoint)
        return model
    
    elif architecture['type'] in ['simple_dqn', 'sequential', 'auto_detect']:
        if architecture['feature']:
            sizes = [str(l[0]) for l in architecture['feature']] + [str(architecture['feature'][-1][1])]
            print(f"   Layers: {' → '.join(sizes)}")
        
        model = SimpleDQN(architecture['feature'])
        
        # Try to load with different key mappings
        try:
            model.load_state_dict(checkpoint)
        except:
            # Manual loading for fc1, fc2, etc. naming
            new_state_dict = {}
            for i, (old_key, value) in enumerate(checkpoint.items()):
                if 'weight' in old_key:
                    layer_idx = (i // 2) * 2  # Account for weight/bias pairs
                    new_key = f'layers.{layer_idx}.weight'
                elif 'bias' in old_key:
                    layer_idx = ((i - 1) // 2) * 2
                    new_key = f'layers.{layer_idx}.bias'
                else:
                    new_key = old_key
                new_state_dict[new_key] = value
            
            try:
                model.load_state_dict(new_state_dict)
            except Exception as e:
                print(f"   ⚠️ Could not auto-load. Error: {e}")
                print(f"   Trying direct assignment...")
                # Last resort: just use the checkpoint directly
                return DirectLoadModel(checkpoint)
        
        return model
    
    else:
        print(f"   ⚠️ Unknown architecture, attempting direct load...")
        return DirectLoadModel(checkpoint)


class DirectLoadModel(nn.Module):
    """
    Fallback: Build model directly from checkpoint keys.
    """
    def __init__(self, checkpoint):
        super().__init__()
        
        # Find all weight matrices and build layers
        weight_keys = [k for k in checkpoint.keys() if 'weight' in k]
        weight_keys.sort()
        
        self.layers = nn.ModuleList()
        for key in weight_keys:
            weight = checkpoint[key]
            if len(weight.shape) == 2:
                in_f, out_f = weight.shape[1], weight.shape[0]
                layer = nn.Linear(in_f, out_f)
                
                # Load weights
                layer.weight.data = weight
                bias_key = key.replace('weight', 'bias')
                if bias_key in checkpoint:
                    layer.bias.data = checkpoint[bias_key]
                
                self.layers.append(layer)
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x

# =============================================================================
# AGENT - Automatically loads ANY model architecture
# =============================================================================
class Agent:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if model_path and os.path.exists(model_path):
            # Dynamically load ANY model architecture
            self.policy_net = load_any_model(model_path)
            self.policy_net.to(self.device)
            self.policy_net.eval()
            print(f"✅ Model loaded: {model_path}")
            print(f"   Device: {self.device}")
        else:
            print(f"⚠️ No model found at: {model_path}")
            print(f"   Agent will act randomly!")
            # Default architecture for random play
            default_arch = {
                'feature': [(88, 256), (256, 256)],
                'value': [(256, 128), (128, 1)],
                'advantage': [(256, 128), (128, 6)]
            }
            self.policy_net = DynamicDuelingDQN(default_arch).to(self.device)

    def get_action(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item(), q_values.cpu().numpy()[0]

# =============================================================================
# RENDER FRAME FUNCTION
# =============================================================================
def render_frame(pygame, screen, fonts, env, action, q_values, reward, cumulative_reward, 
                 game_number, damage_flash=0, paused=False, speed=1.0, recording=False):
    
    font_large, font_medium, font_small = fonts
    
    # Colors
    DARK_BLUE = (10, 10, 40)
    WHITE = (255, 255, 255)
    BLUE = (50, 150, 255)
    CYAN = (0, 255, 255)
    RED = (255, 50, 50)
    ORANGE = (255, 165, 0)
    GREEN = (50, 255, 50)
    YELLOW = (255, 255, 0)
    PURPLE = (180, 100, 255)
    GRAY = (100, 100, 100)
    DARK_GRAY = (50, 50, 50)
    LIGHT_BLUE = (100, 180, 255)
    
    ACTION_NAMES = ["← LEFT", "→ RIGHT", "• STAY", "↑ SHOOT", "←↑ L+SHOOT", "→↑ R+SHOOT"]
    ACTION_COLORS = [LIGHT_BLUE, LIGHT_BLUE, YELLOW, ORANGE, PURPLE, PURPLE]
    SHORT_NAMES = ["←", "→", "•", "↑", "←↑", "→↑"]
    
    info = {
        'score': env.player.score,
        'enemies_destroyed': env.enemies_destroyed
    }
    
    # Background
    screen.fill(DARK_BLUE)
    
    # Damage flash
    if damage_flash > 0:
        flash_surface = pygame.Surface((Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT))
        flash_surface.fill(RED)
        flash_surface.set_alpha(damage_flash * 8)
        screen.blit(flash_surface, (0, 0))
    
    # Stars
    for i in range(50):
        x = (i * 37 + env.frame_count) % Config.SCREEN_WIDTH
        y = (i * 73 + env.frame_count * 2) % Config.SCREEN_HEIGHT
        pygame.draw.circle(screen, WHITE, (x, y), 1)
    
    # Player
    p = env.player
    pygame.draw.polygon(screen, BLUE, [
        (p.x + p.width // 2, p.y),
        (p.x + p.width, p.y + p.height),
        (p.x + p.width // 2, p.y + p.height - 10),
        (p.x, p.y + p.height)
    ])
    if env.frame_count % 4 < 2:
        pygame.draw.polygon(screen, ORANGE, [
            (p.x + 10, p.y + p.height),
            (p.x + p.width // 2, p.y + p.height + 15),
            (p.x + p.width - 10, p.y + p.height)
        ])
    
    # Enemies
    for e in env.enemies:
        pygame.draw.polygon(screen, RED, [
            (e.x, e.y),
            (e.x + e.width, e.y),
            (e.x + e.width // 2, e.y + e.height)
        ])
    
    # Bullets
    for b in env.player_bullets:
        pygame.draw.rect(screen, CYAN, (b.x, b.y, b.width, b.height))
    for b in env.enemy_bullets:
        pygame.draw.rect(screen, RED, (b.x, b.y, b.width, b.height))
    
    # ===== UI: TOP LEFT =====
    screen.blit(font_large.render(f"SCORE: {info['score']}", True, WHITE), (10, 10))
    screen.blit(font_medium.render(f"Kills: {info['enemies_destroyed']}", True, WHITE), (10, 45))
    
    # Health bar
    screen.blit(font_small.render("HEALTH:", True, WHITE), (10, 75))
    for i in range(3):
        color = GREEN if i < env.player.health else DARK_GRAY
        pygame.draw.rect(screen, color, (80 + i * 30, 75, 25, 20))
        pygame.draw.rect(screen, WHITE, (80 + i * 30, 75, 25, 20), 1)
    
    # ===== UI: TOP RIGHT =====
    #screen.blit(font_medium.render(f"Game: {game_number}", True, YELLOW), (Config.SCREEN_WIDTH - 100, 10))
    screen.blit(font_small.render(f"Frame: {env.frame_count}", True, GRAY), (Config.SCREEN_WIDTH - 100, 35))
    
    if recording:
        pygame.draw.circle(screen, RED, (Config.SCREEN_WIDTH - 110, 62), 8)
        screen.blit(font_small.render("REC", True, RED), (Config.SCREEN_WIDTH - 95, 55))
    else:
        screen.blit(font_small.render(f"Speed: {speed:.1f}x", True, GRAY), (Config.SCREEN_WIDTH - 100, 55))
        status_color = ORANGE if paused else GREEN
        status_text = "PAUSED" if paused else "RUNNING"
        screen.blit(font_small.render(status_text, True, status_color), (Config.SCREEN_WIDTH - 100, 75))
    
    # ===== UI: BOTTOM LEFT - ACTION =====
    action_box_y = Config.SCREEN_HEIGHT - 120
    pygame.draw.rect(screen, DARK_GRAY, (10, action_box_y, 200, 50), border_radius=5)
    pygame.draw.rect(screen, ACTION_COLORS[action], (10, action_box_y, 200, 50), 2, border_radius=5)
    screen.blit(font_small.render("CURRENT ACTION:", True, GRAY), (20, action_box_y + 5))
    screen.blit(font_medium.render(f"[{action}] {ACTION_NAMES[action]}", True, ACTION_COLORS[action]), (20, action_box_y + 25))
    
    # ===== UI: BOTTOM LEFT - REWARD =====
    reward_box_y = Config.SCREEN_HEIGHT - 60
    pygame.draw.rect(screen, DARK_GRAY, (10, reward_box_y, 200, 50), border_radius=5)
    screen.blit(font_small.render("REWARD:", True, GRAY), (20, reward_box_y + 5))
    
    reward_color = GREEN if reward > 0 else (RED if reward < 0 else WHITE)
    reward_sign = "+" if reward > 0 else ""
    screen.blit(font_medium.render(f"Frame: {reward_sign}{reward:.2f}", True, reward_color), (20, reward_box_y + 25))
    
    cum_color = GREEN if cumulative_reward > 0 else (RED if cumulative_reward < 0 else WHITE)
    cum_sign = "+" if cumulative_reward > 0 else ""
    screen.blit(font_small.render(f"Total: {cum_sign}{cumulative_reward:.1f}", True, cum_color), (120, reward_box_y + 28))
    
    # ===== UI: BOTTOM RIGHT - Q-VALUES =====
    qval_box_x = Config.SCREEN_WIDTH - 220
    qval_box_y = Config.SCREEN_HEIGHT - 180
    pygame.draw.rect(screen, DARK_GRAY, (qval_box_x - 5, qval_box_y - 5, 220, 175), border_radius=5)
    screen.blit(font_small.render("Q-VALUES", True, YELLOW), (qval_box_x + 70, qval_box_y))
    
    q_min, q_max = q_values.min(), q_values.max()
    q_range = max(q_max - q_min, 0.01)
    
    bar_height = 18
    bar_max_width = 120
    
    for i, q_val in enumerate(q_values):
        y_pos = qval_box_y + 25 + i * (bar_height + 5)
        norm_q = (q_val - q_min) / q_range
        bar_width = int(norm_q * bar_max_width)
        
        is_selected = (i == action)
        bar_color = CYAN if is_selected else BLUE
        border_color = WHITE if is_selected else GRAY
        
        screen.blit(font_small.render(SHORT_NAMES[i], True, WHITE), (qval_box_x, y_pos))
        pygame.draw.rect(screen, DARK_GRAY, (qval_box_x + 25, y_pos, bar_max_width, bar_height))
        pygame.draw.rect(screen, bar_color, (qval_box_x + 25, y_pos, bar_width, bar_height))
        pygame.draw.rect(screen, border_color, (qval_box_x + 25, y_pos, bar_max_width, bar_height), 1)
        screen.blit(font_small.render(f"{q_val:.1f}", True, WHITE), (qval_box_x + 150, y_pos))
    
    # Version label
    screen.blit(font_medium.render("v7 DUELING DQN + PER", True, PURPLE), 
               (Config.SCREEN_WIDTH // 2 - 100, Config.SCREEN_HEIGHT - 30))
    
    pygame.display.flip()

# =============================================================================
# RECORD VIDEO FUNCTION
# =============================================================================
def record_video(model_path, output_path="gameplay.mp4", num_games=5, fps=60):
    """
    Record gameplay to MP4 video file.
    Writes frames directly to file to avoid memory issues.
    """
    try:
        import imageio.v2 as imageio
    except ImportError:
        try:
            import imageio
        except ImportError:
            print("❌ imageio not installed!")
            print("   Run: pip install imageio imageio-ffmpeg")
            return None
    
    import pygame
    
    print("\n" + "="*60)
    print("🎬 RECORDING VIDEO")
    print("="*60)
    print(f"   Model: {model_path}")
    print(f"   Output: {output_path}")
    print(f"   Games: {num_games}")
    print(f"   FPS: {fps}")
    print("="*60 + "\n")
    
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT))
    pygame.display.set_caption("Space Defender - Recording...")
    
    font_large = pygame.font.SysFont('arial', 28, bold=True)
    font_medium = pygame.font.SysFont('arial', 20)
    font_small = pygame.font.SysFont('arial', 16)
    fonts = (font_large, font_medium, font_small)
    
    # Load agent
    agent = Agent(model_path)
    
    game_scores = []
    total_frames = 0
    
    # Open video writer - writes frames directly to file (no memory buildup!)
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8)
    
    try:
        for game_num in range(1, num_games + 1):
            print(f"Recording Game {game_num}/{num_games}...", end=" ", flush=True)
            
            env = SpaceDefenderEnv()
            state = env.reset()
            cumulative_reward = 0
            damage_flash = 0
            last_health = env.player.health
            last_frame = None
            
            while not env.game_over:
                # Handle pygame events (prevents freezing)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        writer.close()
                        pygame.quit()
                        return None
                
                # Get action
                action, q_values = agent.get_action(state)
                
                # Step
                state, reward, done, info = env.step(action)
                cumulative_reward += reward
                
                # Damage flash
                if env.player.health < last_health:
                    damage_flash = 15
                last_health = env.player.health
                if damage_flash > 0:
                    damage_flash -= 1
                
                # Render
                render_frame(pygame, screen, fonts, env, action, q_values, 
                            reward, cumulative_reward, game_num, damage_flash, recording=True)
                
                # Capture and write frame directly to file
                frame = np.transpose(pygame.surfarray.array3d(screen), (1, 0, 2))
                writer.append_data(frame)
                last_frame = frame
                total_frames += 1
                
                # Safety limit
                if env.frame_count > 5000:
                    break
            
            game_scores.append(info['score'])
            print(f"Score: {info['score']}, Kills: {info['enemies_destroyed']}")
            
            # Add pause between games (1 second) - use last frame
            if last_frame is not None:
                for _ in range(fps):
                    writer.append_data(last_frame)
                    total_frames += 1
        
        writer.close()
        pygame.quit()
        
        print(f"\n✅ Video saved: {output_path}")
        
    except Exception as e:
        writer.close()
        pygame.quit()
        print(f"\n❌ Error during recording: {e}")
        print("   Try: pip install imageio-ffmpeg")
        return None
    
    # Summary
    print("\n" + "="*60)
    print("📊 RECORDING COMPLETE")
    print("="*60)
    print(f"   Games recorded: {num_games}")
    print(f"   Total frames: {total_frames}")
    print(f"   Scores: {game_scores}")
    print(f"   Mean score: {np.mean(game_scores):.1f}")
    print(f"   Max score: {max(game_scores)}")
    print(f"   Video saved to: {output_path}")
    print("="*60)
    
    return output_path, game_scores

# =============================================================================
# INTERACTIVE MODE
# =============================================================================
def run_interactive(model_path):
    """Run game interactively with keyboard controls."""
    import pygame
    
    print("\n" + "="*60)
    print("🎮 INTERACTIVE MODE")
    print("="*60)
    print("Controls:")
    print("  SPACE  - Pause/Resume")
    print("  UP     - Speed up")
    print("  DOWN   - Slow down")
    print("  R      - Reset game")
    print("  Q      - Quit")
    print("="*60 + "\n")
    
    pygame.init()
    screen = pygame.display.set_mode((Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT))
    pygame.display.set_caption("Space Defender - Q-Learning Agent")
    clock = pygame.time.Clock()
    
    font_large = pygame.font.SysFont('arial', 28, bold=True)
    font_medium = pygame.font.SysFont('arial', 20)
    font_small = pygame.font.SysFont('arial', 16)
    fonts = (font_large, font_medium, font_small)
    
    agent = Agent(model_path)
    env = SpaceDefenderEnv()
    state = env.reset()
    
    running = True
    paused = False
    speed = 1.0
    game_number = 1
    cumulative_reward = 0
    damage_flash = 0
    total_scores = []
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                    print(f"{'⏸️ PAUSED' if paused else '▶️ RESUMED'}")
                elif event.key == pygame.K_UP:
                    speed = min(speed * 1.5, 4.0)
                    print(f"Speed: {speed:.1f}x")
                elif event.key == pygame.K_DOWN:
                    speed = max(speed / 1.5, 0.25)
                    print(f"Speed: {speed:.1f}x")
                elif event.key == pygame.K_r:
                    env = SpaceDefenderEnv()
                    state = env.reset()
                    cumulative_reward = 0
                    damage_flash = 0
                    print("🔄 Game Reset")
                elif event.key == pygame.K_q:
                    running = False
        
        if not paused:
            action, q_values = agent.get_action(state)
            last_health = env.player.health
            state, reward, done, info = env.step(action)
            cumulative_reward += reward
            
            if env.player.health < last_health:
                damage_flash = 15
            if damage_flash > 0:
                damage_flash -= 1
            
            if done:
                total_scores.append(info['score'])
                print(f"Game {game_number} Over! Score: {info['score']}, Kills: {info['enemies_destroyed']}")
                game_number += 1
                env = SpaceDefenderEnv()
                state = env.reset()
                cumulative_reward = 0
        else:
            action, q_values = agent.get_action(state)
            reward = 0
        
        render_frame(pygame, screen, fonts, env, action, q_values,
                    reward, cumulative_reward, game_number, damage_flash, paused, speed)
        
        clock.tick(int(60 * speed))
    
    pygame.quit()
    
    if total_scores:
        print("\n" + "="*50)
        print("📊 SESSION SUMMARY")
        print(f"   Games: {len(total_scores)}")
        print(f"   Scores: {total_scores}")
        print(f"   Mean: {np.mean(total_scores):.1f}")
        print("="*50)

# =============================================================================
# MAIN
# =============================================================================
def main():
    model_path = input("Type in your model path:")
    
    if not os.path.exists(model_path):
        print("\n" + "="*60)
        print("❌ MODEL NOT FOUND")
        print("="*60)
        print(f"   Expected file: {model_path}")
        print("")
        print("   Please download best_eval_model.pth from your")
        print("   Google Drive and place it in this folder:")
        print(f"   {os.getcwd()}")
        print("="*60)
        return
    
    print("\n" + "="*60)
    print("🚀 SPACE DEFENDER - LOCAL PLAYER & RECORDER")
    print("="*60)
    print("")
    print("Choose mode:")
    print("  [1] Interactive - Watch agent play (with controls)")
    print("  [2] Record Video - Save gameplay to MP4 file")
    print("")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        num_games = input("Number of games to record (default=5): ").strip()
        num_games = int(num_games) if num_games else 5
        
        output_file = input("Output filename (default=gameplay.mp4): ").strip()
        output_file = output_file if output_file else "gameplay.mp4"
        
        record_video(model_path, output_file, num_games)
    else:
        run_interactive(model_path)

if __name__ == "__main__":
    main()

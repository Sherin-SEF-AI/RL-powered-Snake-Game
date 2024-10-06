import sys
import random
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QSlider, QTabWidget, QGridLayout, QLineEdit, QComboBox)
from PyQt6.QtGui import QPainter, QColor, QPen, QFont
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Game constants
GRID_SIZE = 20
CELL_SIZE = 25
WINDOW_WIDTH = GRID_SIZE * CELL_SIZE + 600
WINDOW_HEIGHT = GRID_SIZE * CELL_SIZE + 200

# Colors
BACKGROUND_COLOR = "#2C3E50"
SNAKE_COLOR = "#2ECC71"
FOOD_COLOR = "#E74C3C"
TEXT_COLOR = "#ECF0F1"

# Directions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class SnakeGame:
    def __init__(self, grid_size=GRID_SIZE):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        self.snake = deque([(self.grid_size // 2, self.grid_size // 2)])
        self.direction = random.choice([UP, RIGHT, DOWN, LEFT])
        self.food = self.spawn_food()
        self.score = 0
        self.steps = 0
        self.game_over = False

    def spawn_food(self):
        while True:
            food = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            if food not in self.snake:
                return food

    def move(self, action):
        self.direction = action
        head = self.snake[0]
        if self.direction == UP:
            new_head = ((head[0] - 1) % self.grid_size, head[1])
        elif self.direction == RIGHT:
            new_head = (head[0], (head[1] + 1) % self.grid_size)
        elif self.direction == DOWN:
            new_head = ((head[0] + 1) % self.grid_size, head[1])
        else:  # LEFT
            new_head = (head[0], (head[1] - 1) % self.grid_size)

        self.steps += 1
        reward = -0.01  # Small negative reward for each step

        if new_head in self.snake:
            self.game_over = True
            reward = -10
        elif new_head == self.food:
            self.score += 1
            reward = 10
            self.snake.appendleft(new_head)
            self.food = self.spawn_food()
        else:
            self.snake.appendleft(new_head)
            self.snake.pop()

        return reward

    def get_state(self):
        head = self.snake[0]
        point_l = (head[0], (head[1] - 1) % self.grid_size)
        point_r = (head[0], (head[1] + 1) % self.grid_size)
        point_u = ((head[0] - 1) % self.grid_size, head[1])
        point_d = ((head[0] + 1) % self.grid_size, head[1])

        state = [
            # Danger straight
            (self.direction == UP and self.is_collision(point_u)) or
            (self.direction == DOWN and self.is_collision(point_d)) or
            (self.direction == LEFT and self.is_collision(point_l)) or
            (self.direction == RIGHT and self.is_collision(point_r)),

            # Danger right
            (self.direction == UP and self.is_collision(point_r)) or
            (self.direction == DOWN and self.is_collision(point_l)) or
            (self.direction == LEFT and self.is_collision(point_u)) or
            (self.direction == RIGHT and self.is_collision(point_d)),

            # Danger left
            (self.direction == UP and self.is_collision(point_l)) or
            (self.direction == DOWN and self.is_collision(point_r)) or
            (self.direction == LEFT and self.is_collision(point_d)) or
            (self.direction == RIGHT and self.is_collision(point_u)),

            # Move direction
            self.direction == LEFT,
            self.direction == RIGHT,
            self.direction == UP,
            self.direction == DOWN,

            # Food location 
            self.food[1] < head[1],  # food left
            self.food[1] > head[1],  # food right
            self.food[0] < head[0],  # food up
            self.food[0] > head[0]   # food down
        ]
        return np.array(state, dtype=int)

    def is_collision(self, point):
        return point in self.snake or point[0] < 0 or point[0] >= self.grid_size or point[1] < 0 or point[1] >= self.grid_size

class RLAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = {}
        self.epsilon = 0.9
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.1
        self.gamma = 0.9

    def get_action(self, state):
        state = tuple(state)
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size)
        
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        return np.argmax(self.q_table[state])

    def train(self, state, action, reward, next_state, next_action, done):
        raise NotImplementedError("Subclasses must implement this method")

class QLearningAgent(RLAgent):
    def train(self, state, action, reward, next_state, next_action, done):
        state = tuple(state)
        next_state = tuple(next_state)
        
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.action_size)
        
        current_q = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state])
        new_q = current_q + self.learning_rate * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q

        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

class SARSAAgent(RLAgent):
    def train(self, state, action, reward, next_state, next_action, done):
        state = tuple(state)
        next_state = tuple(next_state)
        
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.action_size)
        
        current_q = self.q_table[state][action]
        next_q = self.q_table[next_state][next_action]
        new_q = current_q + self.learning_rate * (reward + self.gamma * next_q - current_q)
        self.q_table[state][action] = new_q

        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

class ExpectedSARSAAgent(RLAgent):
    def train(self, state, action, reward, next_state, next_action, done):
        state = tuple(state)
        next_state = tuple(next_state)
        
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.action_size)
        
        current_q = self.q_table[state][action]
        next_q_values = self.q_table[next_state]
        expected_q = np.sum(next_q_values * (self.epsilon / self.action_size + (1 - self.epsilon) * (next_q_values == np.max(next_q_values))))
        new_q = current_q + self.learning_rate * (reward + self.gamma * expected_q - current_q)
        self.q_table[state][action] = new_q

        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

class GameWidget(QWidget):
    def __init__(self, game):
        super().__init__()
        self.game = game
        self.setFixedSize(GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE)

    def paintEvent(self, event):
        painter = QPainter(self)
        self.draw_grid(painter)
        self.draw_snake(painter)
        self.draw_food(painter)

    def draw_grid(self, painter):
        painter.setPen(QPen(QColor("#34495E"), 1, Qt.PenStyle.SolidLine))
        for i in range(GRID_SIZE):
            painter.drawLine(i * CELL_SIZE, 0, i * CELL_SIZE, GRID_SIZE * CELL_SIZE)
            painter.drawLine(0, i * CELL_SIZE, GRID_SIZE * CELL_SIZE, i * CELL_SIZE)

    def draw_snake(self, painter):
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(SNAKE_COLOR))
        for segment in self.game.snake:
            painter.drawRect(segment[1] * CELL_SIZE, segment[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)

    def draw_food(self, painter):
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(FOOD_COLOR))
        painter.drawRect(self.game.food[1] * CELL_SIZE, self.game.food[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)

class SnakeGameGUI(QMainWindow):
    updateSignal = pyqtSignal()

    def __init__(self, game):
        super().__init__()
        self.game = game
        self.agents = {
            "Q-Learning": QLearningAgent(11, 4),
            "SARSA": SARSAAgent(11, 4),
            "Expected SARSA": ExpectedSARSAAgent(11, 4)
        }
        self.current_agent = self.agents["Q-Learning"]
        self.manual_mode = False
        self.game_speed = 100
        self.episode_rewards = []
        self.episode_lengths = []
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Snake Game RL')
        self.setGeometry(100, 100, WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setStyleSheet(f"background-color: {BACKGROUND_COLOR}; color: {TEXT_COLOR};")

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # Game area
        game_layout = QVBoxLayout()
        main_layout.addLayout(game_layout)

        self.game_widget = GameWidget(self.game)
        game_layout.addWidget(self.game_widget)

        info_layout = QHBoxLayout()
        game_layout.addLayout(info_layout)

        self.score_label = QLabel('Score: 0')
        info_layout.addWidget(self.score_label)

        self.epsilon_label = QLabel(f'Epsilon: {self.current_agent.epsilon:.2f}')
        info_layout.addWidget(self.epsilon_label)

        # Control and visualization area
        control_viz_layout = QVBoxLayout()
        main_layout.addLayout(control_viz_layout)

        # Control buttons
        button_layout = QHBoxLayout()
        control_viz_layout.addLayout(button_layout)

        self.start_button = QPushButton('Start')
        self.start_button.clicked.connect(self.toggle_game)
        button_layout.addWidget(self.start_button)

        self.reset_button = QPushButton('Reset')
        self.reset_button.clicked.connect(self.reset_game)
        button_layout.addWidget(self.reset_button)

        self.mode_button = QPushButton('Switch to Manual')
        self.mode_button.clicked.connect(self.toggle_mode)
        button_layout.addWidget(self.mode_button)

        # Agent selection
        agent_layout = QHBoxLayout()
        control_viz_layout.addLayout(agent_layout)

        agent_layout.addWidget(QLabel('RL Algorithm:'))
        self.agent_combo = QComboBox()
        self.agent_combo.addItems(self.agents.keys())
        self.agent_combo.currentTextChanged.connect(self.change_agent)
        agent_layout.addWidget(self.agent_combo)

        # Speed slider
        speed_layout = QHBoxLayout()
        control_viz_layout.addLayout(speed_layout)

        speed_layout.addWidget(QLabel('Game Speed:'))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(50)
        self.speed_slider.setMaximum(500)
        self.speed_slider.setValue(self.game_speed)
        self.speed_slider.valueChanged.connect(self.change_speed)
        speed_layout.addWidget(self.speed_slider)

        # Visualization tabs
        self.tab_widget = QTabWidget()
        control_viz_layout.addWidget(self.tab_widget)

        # Performance plot
        self.performance_fig, self.performance_ax = plt.subplots(2, 1, figsize=(5, 6))
        self.performance_canvas = FigureCanvas(self.performance_fig)
        self.tab_widget.addTab(self.performance_canvas, "Performance")

        # Q-values display
        self.q_values_widget = QWidget()
        self.q_values_layout = QGridLayout()
        self.q_values_widget.setLayout(self.q_values_layout)
        self.tab_widget.addTab(self.q_values_widget, "Q-Values")

        # Initialize Q-values display
        for i in range(self.current_agent.action_size):
            self.q_values_layout.addWidget(QLabel(f"Action {i}:"), i, 0)
            self.q_values_layout.addWidget(QLabel("N/A"), i, 1)

        self.update_q_values_display()

        # Research tools
        research_layout = QGridLayout()
        control_viz_layout.addLayout(research_layout)

        research_layout.addWidget(QLabel("Learning Rate:"), 0, 0)
        self.lr_input = QLineEdit(str(self.current_agent.learning_rate))
        self.lr_input.returnPressed.connect(self.update_learning_rate)
        research_layout.addWidget(self.lr_input, 0, 1)

        research_layout.addWidget(QLabel("Gamma:"), 1, 0)
        self.gamma_input = QLineEdit(str(self.current_agent.gamma))
        self.gamma_input.returnPressed.connect(self.update_gamma)
        research_layout.addWidget(self.gamma_input, 1, 1)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_game)

        self.updateSignal.connect(self.update_plots)

    def toggle_game(self):
        if self.timer.isActive():
            self.timer.stop()
            self.start_button.setText('Start')
        else:
            self.timer.start(self.game_speed)
            self.start_button.setText('Pause')

    def reset_game(self):
        self.game.reset()
        self.game_widget.update()
        self.update_labels()

    def toggle_mode(self):
        self.manual_mode = not self.manual_mode
        self.mode_button.setText('Switch to AI' if self.manual_mode else 'Switch to Manual')

    def change_speed(self, value):
        self.game_speed = value
        if self.timer.isActive():
            self.timer.start(self.game_speed)

    def change_agent(self, agent_name):
        self.current_agent = self.agents[agent_name]
        self.update_labels()
        self.update_q_values_display()

    def update_game(self):
        if not self.manual_mode:
            state = self.game.get_state()
            action = self.current_agent.get_action(state)
            reward = self.game.move(action)
            next_state = self.game.get_state()
            next_action = self.current_agent.get_action(next_state)
            self.current_agent.train(state, action, reward, next_state, next_action, self.game.game_over)
        
        self.update_labels()
        self.game_widget.update()
        self.update_q_values_display()

        if self.game.game_over:
            self.episode_rewards.append(self.game.score)
            self.episode_lengths.append(self.game.steps)
            self.updateSignal.emit()
            self.game.reset()

    def update_labels(self):
        self.score_label.setText(f'Score: {self.game.score}')
        self.epsilon_label.setText(f'Epsilon: {self.current_agent.epsilon:.2f}')

    def update_plots(self):
        self.performance_ax[0].clear()
        self.performance_ax[1].clear()

        self.performance_ax[0].plot(self.episode_rewards)
        self.performance_ax[0].set_title('Episode Rewards')
        self.performance_ax[0].set_xlabel('Episode')
        self.performance_ax[0].set_ylabel('Reward')

        self.performance_ax[1].plot(self.episode_lengths)
        self.performance_ax[1].set_title('Episode Lengths')
        self.performance_ax[1].set_xlabel('Episode')
        self.performance_ax[1].set_ylabel('Length')

        self.performance_fig.tight_layout()
        self.performance_canvas.draw()

    def update_q_values_display(self):
        state = tuple(self.game.get_state())
        if state in self.current_agent.q_table:
            q_values = self.current_agent.q_table[state]
            for i, q_value in enumerate(q_values):
                self.q_values_layout.itemAtPosition(i, 1).widget().setText(f"{q_value:.2f}")
        else:
            for i in range(self.current_agent.action_size):
                self.q_values_layout.itemAtPosition(i, 1).widget().setText("N/A")

    def update_learning_rate(self):
        try:
            new_lr = float(self.lr_input.text())
            if 0 < new_lr < 1:
                self.current_agent.learning_rate = new_lr
            else:
                raise ValueError
        except ValueError:
            self.lr_input.setText(str(self.current_agent.learning_rate))

    def update_gamma(self):
        try:
            new_gamma = float(self.gamma_input.text())
            if 0 < new_gamma < 1:
                self.current_agent.gamma = new_gamma
            else:
                raise ValueError
        except ValueError:
            self.gamma_input.setText(str(self.current_agent.gamma))

    def keyPressEvent(self, event):
        if self.manual_mode:
            key = event.key()
            if key == Qt.Key.Key_Up:
                self.game.move(UP)
            elif key == Qt.Key.Key_Right:
                self.game.move(RIGHT)
            elif key == Qt.Key.Key_Down:
                self.game.move(DOWN)
            elif key == Qt.Key.Key_Left:
                self.game.move(LEFT)
            self.update_labels()
            self.game_widget.update()
            self.update_q_values_display()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    game = SnakeGame()
    gui = SnakeGameGUI(game)
    gui.show()
    sys.exit(app.exec())

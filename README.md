# Go Neural Network

A full implementation of a Go-playing AI that combines a neural network with Monte Carlo Tree Search (MCTS), inspired by AlphaGo.

## Features

- **Complete Go Game Engine**: Full implementation of Go rules including move validation, capture detection, ko rule, and elementary territory scoring
- **Deep Neural Network AI**: Dual-head Convolutional Neural Network with policy and value networks using residual blocks
- **Monte Carlo Tree Search (MCTS)**: Tree search algorithm with neural network guidance for gameplay
- **Self-Play Training**: Automatic data generation through self-play games for continuous improvement
- **Command-Line Interface**: Play against the trained AI with real-time board visualization
- **Territory Scoring**: Automatic dead stone removal and territory calculation for accurate game outcomes

## Quick Start

**Clone and install:**
```bash
git clone <repository-url>
cd Go-Neural-Network
pip install -r requirements.txt
```

**Train a model:**
```bash
python scripts/train.py --board-size 9 --iterations 10 --games 10
```

**Play against the AI:**
```bash
python scripts/play.py --model models/model_iter_10.pth --board-size 9
```

**Run tests:**
```bash
python scripts/test_basic.py
```

## Requirements

- Python 3.8+
- PyTorch 1.9.0+
- NumPy 1.21.0+

## Training

### Basic Training
Train a new model from scratch:
```bash
python scripts/train.py --board-size 9 --iterations 10 --games 10
```

### Advanced Training
For better performance, use more iterations and games:
```bash
python scripts/train.py --board-size 9 --iterations 20 --games 30
```

### Continue Training
Continue training from an existing model:
```bash
python scripts/train.py --iterations 20 --games 30 --load-model models/model_iter_2.pth
```

### Training Arguments
- `--board-size`: Board size (9 for 9x9, 19 for 19x19 standard Go)
- `--iterations`: Number of training iterations (default: 10, use 20-50 for better results)
- `--games`: Number of self-play games per iteration (default: 10, use 20-50 for better results)
- `--device`: Device to train on (`cpu` or `cuda`)
- `--save-dir`: Directory to save models (default: `models`)
- `--eval-games`: Number of games for final evaluation (default: 10)
- `--load-model`: Path to existing model to continue training from (optional)

## Playing Against the AI

Play an interactive game against the trained model:
```bash
python scripts/play.py --model models/model_iter_10.pth --board-size 9
```

### Play Arguments
- `--model`: Path to trained model file
- `--board-size`: Board size (default: 9, use 19 for full size)
- `--simulations`: Number of MCTS simulations (higher = stronger but slower, default: 400)
- `--ai-first`: AI plays first (black)

### Move Input Format
When playing, enter moves as:
- `row col` (e.g., `3 3` to place a stone at row 3, column 3)
- `pass` to pass your turn

## Project Structure

```
Go-Neural-Network/
├── go_game.py              # Go game engine with rules and board logic
├── neural_network.py        # Neural network architecture (CNN with residual blocks)
├── mcts.py                  # Monte Carlo Tree Search implementation
├── self_play.py             # Self-play data generation and replay buffer
├── scripts/
│   ├── train.py             # Training pipeline with self-play and supervised learning
│   ├── play.py              # Interactive game player
│   └── test_basic.py        # Basic test suite
├── models/                  # Elementarily trained model files (.pth)
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## AI Architecture

### Neural Network
- **Input**: 7 feature planes representing board state (player stones, enemy stones, empty spaces, liberties, turn indicator, move count)
- **Processing**: 
  - Residual CNN architecture with configurable residual blocks (default: 5)
  - Batch normalization and ReLU activations
  - Dual-head output architecture
- **Output**: 
  - **Policy Head**: Move probability distribution over all legal moves (board_size² + 1 for pass)
  - **Value Head**: Position evaluation (-1 to +1, predicting game outcome)
- **Training**: Self-play data generation with MCTS visit probabilities as training targets

### Monte Carlo Tree Search (MCTS)
- **Algorithm**: UCT (Upper Confidence Bound applied to Trees) variant
- **Neural Network Guidance**: Uses policy network for move priors and value network for position evaluation
- **Exploration**: UCB formula balances exploitation (Q-value) and exploration (prior probability)
- **Simulations**: Configurable number of simulations per move (default: 400-800)
- **Temperature**: Controls move selection randomness during training vs. deterministic play

### Training Process
1. **Self-Play Generation**: Current model plays against itself using MCTS
2. **Data Collection**: Store game positions with MCTS visit probabilities and final game outcomes
3. **Supervised Learning**: Train network on collected data with policy and value losses
4. **Iteration**: Repeat process to gradually improve model strength

## Game Rules Implementation

- **Move Validation**: Complete rule set including ko rule, suicide prevention, and capture detection
- **Capture Mechanics**: Automatic removal of groups with no liberties
- **Passing**: Players may pass at any time (standard Go rule)
- **Game Ending**: Game ends after two consecutive passes
- **Scoring**: 
  - Automatic dead stone removal after game end
  - Territory calculation (enclosed empty spaces)
  - Final score = stones on board + territory + captured stones + komi (6.5 for white)

## Performance Tips

- **Board Size**: Start with 9x9 for faster training and testing (minutes to hours)
- **GPU**: Use CUDA if available for significantly faster training (5-10x speedup)
- **Simulations**: More MCTS simulations = stronger play but slower (400-800 recommended)
- **Training Time**: 
  - Quick test: 10 iterations × 10 games = ~30 minutes
  - Good results: 20 iterations × 30 games = ~2-3 hours
  - Strong model: 50 iterations × 50 games = ~8-12 hours

## Example Training Session

```bash
# Quick test (30 minutes)
python scripts/train.py --board-size 9 --iterations 10 --games 10

# Better results (2-3 hours)
python scripts/train.py --board-size 9 --iterations 20 --games 30

# Strong model (8-12 hours)
python scripts/train.py --board-size 9 --iterations 50 --games 50

# Production (days)
python scripts/train.py --board-size 19 --iterations 100 --games 100 --device cuda
```

## Evaluation

After training, evaluate model performance:
```bash
python scripts/train.py --iterations 20 --games 20 --eval-games 20
```

The evaluation plays the model against a random player and reports win rate statistics.

## License

See LICENSE file for details.
import numpy as np
from typing import List, Tuple, Dict
from go_game import GoGame, Player
from mcts import MCTS
from neural_network import GoNeuralNetworkWrapper


class SelfPlayGenerator:
    def __init__(self, neural_network: GoNeuralNetworkWrapper, 
                 num_simulations: int = 800, temperature: float = 1.0):
        self.neural_network = neural_network
        self.mcts = MCTS(neural_network, num_simulations=num_simulations, 
                        temperature=temperature)
    
    def generate_game(self, board_size: int = 19) -> List[Dict]:
        game = GoGame(board_size=board_size)
        examples = []
        
        move_count = 0
        max_moves = board_size * board_size * 2  # Reasonable limit
        
        while not game.game_over and move_count < max_moves:
            current_player = game.get_current_player()
            
            # Get MCTS policy
            policy = self.mcts.get_policy(game)
            
            # Sample move from policy
            move = self.mcts.select_move(game, training=True)
            
            # Store training example
            features = game.get_board_features(current_player)
            examples.append({
                'features': features.copy(),
                'policy': policy.copy(),
                'player': current_player,
                'move': move
            })
            
            # Make move
            if move == (-1, -1):
                game.pass_move()
            else:
                game.make_move(move[0], move[1])
            
            move_count += 1
        
        # Determine final outcome
        winner = game.get_winner()
        
        # Assign values to all examples
        for example in examples:
            if winner is None:
                value = 0.0  # Draw
            elif winner == example['player']:
                value = 1.0  # Win
            else:
                value = -1.0  # Loss
            
            example['value'] = value
        
        return examples
    
    def generate_batch(self, num_games: int, board_size: int = 19) -> Dict:
        all_features = []
        all_policies = []
        all_values = []
        
        print(f"Generating {num_games} self-play games...")
        
        for game_idx in range(num_games):
            if (game_idx + 1) % 10 == 0:
                print(f"  Generated {game_idx + 1}/{num_games} games")
            
            examples = self.generate_game(board_size)
            
            for example in examples:
                all_features.append(example['features'])
                
                # Convert policy dict to array
                policy_array = np.zeros(board_size * board_size + 1)
                for move, prob in example['policy'].items():
                    row, col = move
                    if row == -1 and col == -1:
                        idx = board_size * board_size
                    else:
                        idx = row * board_size + col
                    policy_array[idx] = prob
                
                all_policies.append(policy_array)
                all_values.append(example['value'])
        
        return {
            'features': np.array(all_features),
            'policies': np.array(all_policies),
            'values': np.array(all_values)
        }


class DataBuffer:
    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self.features = []
        self.policies = []
        self.values = []
    
    def add(self, features: np.ndarray, policies: np.ndarray, values: np.ndarray):
        self.features.extend(features)
        self.policies.extend(policies)
        self.values.extend(values)
        
        # Trim if exceeding max size
        if len(self.features) > self.max_size:
            excess = len(self.features) - self.max_size
            self.features = self.features[excess:]
            self.policies = self.policies[excess:]
            self.values = self.values[excess:]
    
    def get_batch(self, batch_size: int) -> Dict:
        if len(self.features) < batch_size:
            batch_size = len(self.features)
        
        indices = np.random.choice(len(self.features), batch_size, replace=False)
        
        batch_features = np.array([self.features[i] for i in indices])
        batch_policies = np.array([self.policies[i] for i in indices])
        batch_values = np.array([self.values[i] for i in indices])
        
        return {
            'features': batch_features,
            'policies': batch_policies,
            'values': batch_values
        }
    
    def size(self) -> int:
        return len(self.features)
    
    def clear(self):
        self.features = []
        self.policies = []
        self.values = []
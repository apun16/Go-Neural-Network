import numpy as np
import math
from typing import Optional, List, Tuple
from go_game import GoGame, Player


class MCTSNode:
    def __init__(self, game_state: GoGame, parent: Optional['MCTSNode'] = None, 
                 move: Optional[Tuple[int, int]] = None):
        self.game_state = game_state.copy()
        self.parent = parent
        self.move = move
        self.children: List['MCTSNode'] = []
        self.visit_count = 0
        self.total_value = 0.0
        self.mean_value = 0.0
        self.prior_prob = 0.0  # Prior probability from neural network
        
    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def is_terminal(self) -> bool:
        return self.game_state.game_over
    
    def get_ucb_score(self, c_puct: float = 5.0) -> float:
        if self.visit_count == 0:
            return float('inf')
        
        # UCB formula: Q + U
        # Q = mean value, U = exploration bonus
        ucb_value = self.mean_value + c_puct * self.prior_prob * \
                   (math.sqrt(self.parent.visit_count) / (1 + self.visit_count))
        
        return ucb_value
    
    def select_child(self, c_puct: float = 5.0) -> 'MCTSNode':
        return max(self.children, key=lambda child: child.get_ucb_score(c_puct))
    
    def expand(self, move_probs: dict):
        valid_moves = self.game_state.get_valid_moves()
        current_player = self.game_state.get_current_player()
        
        for move in valid_moves:
            child_game = self.game_state.copy()
            if move == (-1, -1):
                child_game.pass_move()
            else:
                child_game.make_move(move[0], move[1])
            
            child_node = MCTSNode(child_game, parent=self, move=move)
            
            # Set prior probability
            move_idx = child_node._move_to_index(move)
            if move_idx in move_probs:
                child_node.prior_prob = move_probs[move_idx]
            else:
                child_node.prior_prob = 1e-6  # Small default prior
            
            self.children.append(child_node)
    
    def _move_to_index(self, move: Tuple[int, int]) -> int:
        row, col = move
        if row == -1 and col == -1:
            return self.game_state.board_size * self.game_state.board_size
        return row * self.game_state.board_size + col
    
    def backup(self, value: float):
        self.visit_count += 1
        self.total_value += value
        self.mean_value = self.total_value / self.visit_count
        
        if self.parent is not None:
            # Value is from perspective of child's player, so negate for parent
            self.parent.backup(-value)


class MCTS:
    def __init__(self, neural_network, num_simulations: int = 800, 
                 c_puct: float = 5.0, temperature: float = 1.0):
        self.neural_network = neural_network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
    
    def search(self, game: GoGame) -> Tuple[dict, MCTSNode]:
        root = MCTSNode(game)
        
        # Get initial policy from neural network
        current_player = game.get_current_player()
        features = game.get_board_features(current_player)
        policy_probs, value = self.neural_network.model.predict(features)
        
        # Convert policy to move dictionary
        move_probs_dict = {}
        valid_moves = game.get_valid_moves(current_player)
        for move in valid_moves:
            row, col = move
            if row == -1 and col == -1:
                idx = game.board_size * game.board_size
            else:
                idx = row * game.board_size + col
            if idx < len(policy_probs):
                move_probs_dict[idx] = policy_probs[idx]
            else:
                move_probs_dict[idx] = 1e-6
        
        # Expand root with initial policy
        root.expand(move_probs_dict)
        
        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            
            # Selection: traverse tree until leaf
            while not node.is_leaf():
                node = node.select_child(self.c_puct)
            
            # Expansion and evaluation
            if not node.is_terminal():
                # Get policy and value from neural network
                node_player = node.game_state.get_current_player()
                features = node.game_state.get_board_features(node_player)
                policy_probs, value = self.neural_network.model.predict(features)
                
                # Convert to move dictionary
                move_probs_dict = {}
                valid_moves = node.game_state.get_valid_moves(node_player)
                for move in valid_moves:
                    row, col = move
                    if row == -1 and col == -1:
                        idx = node.game_state.board_size * node.game_state.board_size
                    else:
                        idx = row * node.game_state.board_size + col
                    if idx < len(policy_probs):
                        move_probs_dict[idx] = policy_probs[idx]
                    else:
                        move_probs_dict[idx] = 1e-6
                
                node.expand(move_probs_dict)
                
                # Value is from perspective of node's player
                # But we need to consider it from root's perspective
                value = self._get_value_from_perspective(value, root.game_state.get_current_player(), 
                                                        node_player)
            else:
                # Terminal node: get actual game outcome
                winner = node.game_state.get_winner()
                if winner is None:
                    value = 0.0
                elif winner == root.game_state.get_current_player():
                    value = 1.0
                else:
                    value = -1.0
            
            # Backup
            node.backup(value)
        
        # Get visit probabilities
        visit_counts = {}
        total_visits = sum(child.visit_count for child in root.children)
        
        for child in root.children:
            if total_visits > 0:
                visit_counts[child.move] = child.visit_count / total_visits
            else:
                visit_counts[child.move] = 1.0 / len(root.children)
        
        return visit_counts, root
    
    def _get_value_from_perspective(self, value: float, root_player: Player, 
                                    node_player: Player) -> float:
        if root_player == node_player:
            return value
        else:
            return -value
    
    def select_move(self, game: GoGame, training: bool = False) -> Tuple[int, int]:
        visit_probs, root = self.search(game)
        
        moves = list(visit_probs.keys())
        probs = np.array([visit_probs[move] for move in moves])
        
        # Heuristic: Penalize pass moves early in training (optional heuristic, not a rule)
        # This helps training but doesn't enforce a rule - passing is always legal
        from go_game import Player
        total_stones = np.sum(game.board != Player.EMPTY.value)
        if training and total_stones < 10:  # Only in training mode, as a heuristic
            for i, move in enumerate(moves):
                if move == (-1, -1):
                    probs[i] *= 0.1  # Reduce probability of pass early in training
            # Renormalize
            probs = probs / probs.sum()
        
        if training and self.temperature > 0:
            # Apply temperature
            probs = probs ** (1.0 / self.temperature)
            probs = probs / probs.sum()
            
            # Sample from distribution
            idx = np.random.choice(len(moves), p=probs)
            return moves[idx]
        else:
            # Select move with highest probability
            idx = np.argmax(probs)
            return moves[idx]
    
    def get_policy(self, game: GoGame) -> dict:
        visit_probs, _ = self.search(game)
        return visit_probs
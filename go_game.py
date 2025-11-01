import numpy as np
from enum import Enum
from typing import Optional, List, Tuple, Set
from copy import deepcopy


class Player(Enum):
    BLACK = 1
    WHITE = 2
    EMPTY = 0
    
    def other(self):
        if self == Player.BLACK:
            return Player.WHITE
        elif self == Player.WHITE:
            return Player.BLACK
        return Player.EMPTY


class GoGame:
    def __init__(self, board_size: int = 19):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.current_player = Player.BLACK
        self.move_history: List[Tuple[int, int]] = []
        self.pass_count = 0
        self.game_over = False
        self.captured_stones = {Player.BLACK: 0, Player.WHITE: 0}
        self.ko_point: Optional[Tuple[int, int]] = None
        self.board_history: List[np.ndarray] = []
        
    def get_board(self) -> np.ndarray:
        return self.board.copy()
    
    def get_current_player(self) -> Player:
        return self.current_player
    
    def is_valid_move(self, row: int, col: int, player: Optional[Player] = None) -> bool:
        if player is None:
            player = self.current_player
            
        # Check bounds
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            return False
        
        # Check if position is empty
        if self.board[row, col] != Player.EMPTY.value:
            return False
        
        # Check ko rule
        if self.ko_point == (row, col):
            return False
        
        # Try the move on a temporary board
        temp_board = self.board.copy()
        temp_board[row, col] = player.value
        
        # Check if this move would capture enemy stones first
        # Note: _capture_stones removes the stone at the end, so we need to work with a copy
        captured = self._capture_stones(row, col, player, temp_board.copy())
        
        # If capturing, apply captures to temp board
        if captured > 0:
            enemy = player.other()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                    if temp_board[nr, nc] == enemy.value:
                        group = self._get_group(nr, nc, temp_board)
                        if not self._has_liberty(nr, nc, temp_board):
                            for r, c in group:
                                temp_board[r, c] = Player.EMPTY.value
        
        # Suicide rule: After captures, the placed stone's group must have liberties
        # Make sure the stone is still on temp_board
        temp_board[row, col] = player.value
        if self._has_liberty(row, col, temp_board):
            return True
        
        # If no captures and no liberties, it's suicide - invalid
        return False
    
    def make_move(self, row: int, col: int) -> bool:
        if self.game_over:
            return False
        
        # Check if move is pass
        if row == -1 and col == -1:
            return self.pass_move()
        
        if not self.is_valid_move(row, col):
            return False
        
        # Save board state for ko detection
        previous_board = self.board.copy()
        
        # Place stone
        self.board[row, col] = self.current_player.value
        
        # Capture enemy stones
        captured = self._capture_adjacent_groups(row, col, self.current_player)
        self.captured_stones[self.current_player.other()] += captured
        
        # Check for ko
        if captured == 1:
            # Check if board state matches previous state (simple ko detection)
            for r in range(self.board_size):
                for c in range(self.board_size):
                    if self.board[r, c] != previous_board[r, c] and self.board[r, c] == Player.EMPTY.value:
                        self.ko_point = (r, c)
                        break
        else:
            self.ko_point = None
        
        # Add to history
        self.move_history.append((row, col))
        self.board_history.append(self.board.copy())
        
        # Reset pass count
        self.pass_count = 0
        
        # Switch player
        self.current_player = self.current_player.other()
        
        return True
    
    def pass_move(self) -> bool:
        if self.game_over:
            return False
        
        self.move_history.append((-1, -1))
        self.pass_count += 1
        self.current_player = self.current_player.other()
        
        # Game ends after two consecutive passes
        if self.pass_count >= 2:
            self.game_over = True
            # Remove dead stones before scoring
            self._remove_dead_stones()
        
        return True
    
    def _get_group(self, row: int, col: int, board: Optional[np.ndarray] = None) -> Set[Tuple[int, int]]:
        if board is None:
            board = self.board
        
        color = board[row, col]
        if color == Player.EMPTY.value:
            return set()
        
        group = set()
        stack = [(row, col)]
        
        while stack:
            r, c = stack.pop()
            if (r, c) in group:
                continue
            
            if not (0 <= r < self.board_size and 0 <= c < self.board_size):
                continue
            
            if board[r, c] != color:
                continue
            
            group.add((r, c))
            
            # Add neighbors
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                    if (nr, nc) not in group:
                        stack.append((nr, nc))
        
        return group
    
    def _has_liberty(self, row: int, col: int, board: Optional[np.ndarray] = None) -> bool:
        if board is None:
            board = self.board
        
        group = self._get_group(row, col, board)
        
        for r, c in group:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                    if board[nr, nc] == Player.EMPTY.value:
                        return True
        
        return False
    
    def _capture_adjacent_groups(self, row: int, col: int, player: Player) -> int:
        captured = 0
        enemy = player.other()
        
        # Check all adjacent positions
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            
            if not (0 <= nr < self.board_size and 0 <= nc < self.board_size):
                continue
            
            if self.board[nr, nc] == enemy.value:
                group = self._get_group(nr, nc)
                if not self._has_liberty(nr, nc):
                    # Capture this group
                    for r, c in group:
                        self.board[r, c] = Player.EMPTY.value
                        captured += 1
        
        return captured
    
    def _capture_stones(self, row: int, col: int, player: Player, board: np.ndarray) -> int:
        captured = 0
        enemy = player.other()
        
        # Temporarily place the stone
        board[row, col] = player.value
        
        # Check adjacent enemy groups
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            
            if not (0 <= nr < self.board_size and 0 <= nc < self.board_size):
                continue
            
            if board[nr, nc] == enemy.value:
                group = self._get_group(nr, nc, board)
                if not self._has_liberty(nr, nc, board):
                    captured += len(group)
        
        # Remove temporary stone
        board[row, col] = Player.EMPTY.value
        
        return captured
    
    def get_valid_moves(self, player: Optional[Player] = None) -> List[Tuple[int, int]]:
        if player is None:
            player = self.current_player
        
        valid_moves = [(-1, -1)]  # Pass is always valid in standard Go rules
        
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.is_valid_move(row, col, player):
                    valid_moves.append((row, col))
        
        return valid_moves
    
    def _remove_dead_stones(self):
        dead_groups = set()
        visited_groups = set()
        
        # Check each stone group to see if it has liberties
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row, col] != Player.EMPTY.value:
                    group = self._get_group(row, col)
                    group_id = tuple(sorted(group))
                    
                    if group_id in visited_groups:
                        continue
                    
                    visited_groups.add(group_id)
                    
                    # Check if this group has any liberties
                    has_liberty = False
                    for r, c in group:
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = r + dr, c + dc
                            if (0 <= nr < self.board_size and 0 <= nc < self.board_size and
                                self.board[nr, nc] == Player.EMPTY.value):
                                has_liberty = True
                                break
                        if has_liberty:
                            break
                    
                    # If group has no liberties, it's dead
                    if not has_liberty:
                        dead_groups.add(group_id)
        
        # Remove dead stones and count them as captured
        for group_id in dead_groups:
            group = set(group_id)
            if len(group) > 0:
                # Determine which player's stones these are
                r, c = list(group)[0]
                stone_color = self.board[r, c]
                
                if stone_color == Player.BLACK.value:
                    self.captured_stones[Player.BLACK] += len(group)
                elif stone_color == Player.WHITE.value:
                    self.captured_stones[Player.WHITE] += len(group)
                
                # Remove the stones
                for r, c in group:
                    self.board[r, c] = Player.EMPTY.value
    
    def _calculate_territory(self) -> dict:
        territory = {Player.BLACK: 0, Player.WHITE: 0}
        visited = set()
        
        for row in range(self.board_size):
            for col in range(self.board_size):
                if (self.board[row, col] == Player.EMPTY.value and 
                    (row, col) not in visited):
                    # Find connected empty space
                    region = self._get_empty_region(row, col)
                    
                    # Check what color surrounds this region
                    surrounding_color = self._get_surrounding_color(region)
                    
                    if surrounding_color == Player.BLACK:
                        territory[Player.BLACK] += len(region)
                        visited.update(region)
                    elif surrounding_color == Player.WHITE:
                        territory[Player.WHITE] += len(region)
                        visited.update(region)
                    # If surrounded by both or neither, it's neutral (no territory)
        
        return territory
    
    def _get_empty_region(self, row: int, col: int) -> Set[Tuple[int, int]]:
        if self.board[row, col] != Player.EMPTY.value:
            return set()
        
        region = set()
        stack = [(row, col)]
        
        while stack:
            r, c = stack.pop()
            if (r, c) in region:
                continue
            
            if not (0 <= r < self.board_size and 0 <= c < self.board_size):
                continue
            
            if self.board[r, c] != Player.EMPTY.value:
                continue
            
            region.add((r, c))
            
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.board_size and 0 <= nc < self.board_size and
                    (nr, nc) not in region):
                    stack.append((nr, nc))
        
        return region
    
    def _get_surrounding_color(self, region: Set[Tuple[int, int]]) -> Optional[Player]:
        if len(region) == 0:
            return None
        
        # Check all neighbors of all points in the region
        surrounding_colors = set()
        
        for r, c in region:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                
                # If neighbor is out of bounds, region touches edge - not surrounded
                if not (0 <= nr < self.board_size and 0 <= nc < self.board_size):
                    return None
                
                # If neighbor is not empty, note its color
                if self.board[nr, nc] != Player.EMPTY.value:
                    if self.board[nr, nc] == Player.BLACK.value:
                        surrounding_colors.add(Player.BLACK)
                    elif self.board[nr, nc] == Player.WHITE.value:
                        surrounding_colors.add(Player.WHITE)
        
        # If surrounded by exactly one color, return it
        if len(surrounding_colors) == 1:
            return list(surrounding_colors)[0]
        
        # Otherwise, it's neutral (surrounded by both or neither)
        return None
    
    def calculate_score(self) -> dict:
        # Count stones remaining on board
        black_stones = np.sum(self.board == Player.BLACK.value)
        white_stones = np.sum(self.board == Player.WHITE.value)
        
        # Calculate territory (only if game is over)
        territory = {Player.BLACK: 0, Player.WHITE: 0}
        if self.game_over:
            territory = self._calculate_territory()
        
        # Total score = stones on board + territory + captured stones
        black_score = black_stones + territory[Player.BLACK] + self.captured_stones[Player.BLACK]
        white_score = white_stones + territory[Player.WHITE] + self.captured_stones[Player.WHITE]
        
        # Add komi for white (only if game has meaningful moves)
        komi = 6.5  # Standard komi for white
        if black_stones + white_stones > 0:
            white_score += komi
        
        return {
            Player.BLACK: black_score,
            Player.WHITE: white_score
        }
    
    def get_winner(self) -> Optional[Player]:
        if not self.game_over:
            return None
        
        scores = self.calculate_score()
        if scores[Player.BLACK] > scores[Player.WHITE]:
            return Player.BLACK
        elif scores[Player.WHITE] > scores[Player.BLACK]:
            return Player.WHITE
        return None  # Tie
    
    def copy(self) -> 'GoGame':
        return deepcopy(self)
    
    def get_board_features(self, player: Player) -> np.ndarray:
        features = []
        
        # Current board state from player's perspective
        player_board = (self.board == player.value).astype(float)
        enemy_board = (self.board == player.other().value).astype(float)
        empty_board = (self.board == Player.EMPTY.value).astype(float)
        
        features.extend([player_board, enemy_board, empty_board])
        
        # Add ones plane
        ones = np.ones((self.board_size, self.board_size), dtype=float)
        features.append(ones)
        
        # Add turn indicator
        turn_plane = np.full((self.board_size, self.board_size), 
                           1.0 if self.current_player == player else 0.0, 
                           dtype=float)
        features.append(turn_plane)
        
        # Add liberties (simplified - could be enhanced)
        liberties = np.zeros((self.board_size, self.board_size), dtype=float)
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board[r, c] != Player.EMPTY.value:
                    liberties[r, c] = self._count_liberties(r, c)
        features.append(liberties)
        
        # Add move count feature (normalized)
        move_count_feature = np.full((self.board_size, self.board_size), 
                                     len(self.move_history) / (self.board_size * self.board_size),
                                     dtype=float)
        features.append(move_count_feature)
        
        return np.stack(features, axis=0)
    
    def _count_liberties(self, row: int, col: int) -> float:
        count = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                if self.board[nr, nc] == Player.EMPTY.value:
                    count += 1
        return float(count) / 4.0  # Normalize
    
    def __str__(self) -> str:
        symbols = {Player.EMPTY.value: '.', Player.BLACK.value: 'X', Player.WHITE.value: 'O'}
        lines = []
        for row in self.board:
            lines.append(' '.join(symbols[val] for val in row))
        return '\n'.join(lines)
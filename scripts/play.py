import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from go_game import GoGame, Player
from neural_network import GoNeuralNetworkWrapper, GoNeuralNetwork
from mcts import MCTS
import argparse


def print_board(game: GoGame):
    board = game.board
    board_size = game.board_size
    
    # Print column numbers
    print("   ", end="")
    for col in range(min(board_size, 10)):
        print(f" {col}", end="")
    for col in range(10, min(board_size, 19)):
        print(f"{col}", end="")
    if board_size >= 19:
        print(" 9", end="")
    print()
    
    # Print board with row numbers
    symbols = {0: '.', 1: 'X', 2: 'O'}
    colors = {0: '', 1: '\033[94m', 2: '\033[91m'}  # Blue for black, red for white
    reset = '\033[0m'
    
    for row in range(board_size):
        print(f"{row:2d} ", end="")
        for col in range(board_size):
            val = board[row, col]
            symbol = symbols[val]
            color = colors[val]
            print(f"{color}{symbol}{reset} ", end="")
        print()


def get_human_move(game: GoGame) -> tuple:
    while True:
        try:
            move_str = input("Enter your move (row col) or 'pass' to pass: ").strip().lower()
            
            if move_str == 'pass':
                return (-1, -1)
            
            parts = move_str.split()
            if len(parts) != 2:
                print("Invalid format. Please enter 'row col' or 'pass'")
                continue
            
            row = int(parts[0])
            col = int(parts[1])
            
            if game.is_valid_move(row, col):
                return (row, col)
            else:
                print("Invalid move. Please try again.")
        except ValueError:
            print("Invalid input. Please enter numbers.")
        except KeyboardInterrupt:
            print("\nGame cancelled.")
            return None


def play_game(model_path: str = None, board_size: int = 9, 
              num_simulations: int = 400, human_first: bool = True):
    # Initialize game
    game = GoGame(board_size=board_size)
    
    # Load or create model
    # Handle both absolute and relative paths
    if model_path:
        if not os.path.isabs(model_path):
            # If relative path, check in models directory
            models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
            model_path_abs = os.path.join(models_dir, model_path)
            if os.path.exists(model_path_abs):
                model_path = model_path_abs
            else:
                model_path_abs = os.path.join(os.path.dirname(os.path.dirname(__file__)), model_path)
                if os.path.exists(model_path_abs):
                    model_path = model_path_abs
    
    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        wrapper = GoNeuralNetworkWrapper.load(model_path)
    else:
        print("Using untrained network (random play)...")
        model = GoNeuralNetwork(
            board_size=board_size,
            feature_planes=7,
            num_res_blocks=5,
            num_filters=64 if board_size == 9 else 128
        )
        wrapper = GoNeuralNetworkWrapper(model)
    
    # Initialize MCTS
    mcts = MCTS(wrapper, num_simulations=num_simulations, temperature=0.0)
    
    print("\n" + "="*50)
    print("Go Game - Play against AI")
    print("="*50)
    print(f"Board size: {board_size}x{board_size}")
    print("You are playing as:", "BLACK (X)" if human_first else "WHITE (O)")
    print("AI is playing as:", "WHITE (O)" if human_first else "BLACK (X)")
    print("\nEnter moves as 'row col' (e.g., '3 3') or 'pass' to pass")
    print("="*50 + "\n")
    
    move_count = 0
    max_moves = board_size * board_size * 2
    
    while not game.game_over and move_count < max_moves:
        current_player = game.get_current_player()
        is_human_turn = (current_player == Player.BLACK and human_first) or \
                       (current_player == Player.WHITE and not human_first)
        
        print_board(game)
        print(f"\nCurrent player: {current_player.name}")
        print(f"Move: {move_count + 1}\n")
        
        if is_human_turn:
            # Human move
            move = get_human_move(game)
            if move is None:
                return
        else:
            # AI move
            print("AI is thinking...")
            move = mcts.select_move(game, training=False)
            if move == (-1, -1):
                print("AI passes")
            else:
                print(f"AI plays: ({move[0]}, {move[1]})")
        
        # Make move
        if move == (-1, -1):
            game.pass_move()
        else:
            if not game.make_move(move[0], move[1]):
                print("Invalid move! AI will try a different move.")
                # Fallback: try a random valid move
                valid_moves = game.get_valid_moves()
                if len(valid_moves) > 0:
                    move = valid_moves[0]
                    if move == (-1, -1):
                        game.pass_move()
                    else:
                        game.make_move(move[0], move[1])
                else:
                    print("No valid moves available!")
                    break
                continue
        
        move_count += 1
        
        # Check for game end
        if game.pass_count >= 2:
            game.game_over = True
    
    # Game over
    print("\n" + "="*50)
    print("Game Over!")
    print("="*50)
    print_board(game)
    
    winner = game.get_winner()
    scores = game.calculate_score()
    
    print(f"\nFinal Scores:")
    print(f"  BLACK: {scores[Player.BLACK]:.1f}")
    print(f"  WHITE: {scores[Player.WHITE]:.1f}")
    
    if winner is None:
        print("Result: Draw!")
    elif (winner == Player.BLACK and human_first) or (winner == Player.WHITE and not human_first):
        print("Result: You Win!")
    else:
        print("Result: AI Wins!")


def main():
    parser = argparse.ArgumentParser(description='Play Go against AI')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained model file')
    parser.add_argument('--board-size', type=int, default=9,
                       help='Board size (default: 9, use 19 for full size)')
    parser.add_argument('--simulations', type=int, default=400,
                       help='Number of MCTS simulations for AI')
    parser.add_argument('--ai-first', action='store_true',
                       help='AI plays first (black)')
    
    args = parser.parse_args()
    
    play_game(
        model_path=args.model,
        board_size=args.board_size,
        num_simulations=args.simulations,
        human_first=not args.ai_first
    )


if __name__ == '__main__':
    main()
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from go_game import GoGame, Player
from neural_network import GoNeuralNetwork, GoNeuralNetworkWrapper


def test_game_engine():
    print("Testing Go game engine...")
    
    game = GoGame(board_size=9)
    
    # Test basic moves
    assert game.make_move(3, 3), "Failed to make valid move"
    assert game.board[3, 3] == Player.BLACK.value, "Stone not placed correctly"
    
    # Test move validation
    assert not game.is_valid_move(3, 3), "Invalid move allowed"
    assert game.is_valid_move(3, 4), "Valid move rejected"
    
    # Test game progression
    game2 = GoGame(board_size=9)
    assert game2.make_move(3, 3), "Failed to make move"
    assert game2.make_move(3, 4), "Failed to make move"
    assert game2.make_move(4, 3), "Failed to make move"
    assert len(game2.get_valid_moves()) > 0, "No valid moves available"
    
    print("Game engine tests passed!")


def test_neural_network():
    print("Testing neural network...")
    
    model = GoNeuralNetwork(board_size=9, num_res_blocks=2, num_filters=32)
    wrapper = GoNeuralNetworkWrapper(model)
    
    # Test prediction
    game = GoGame(board_size=9)
    features = game.get_board_features(Player.BLACK)
    
    policy, value = wrapper.model.predict(features)
    
    assert len(policy) == 9 * 9 + 1, f"Policy size incorrect: {len(policy)}"
    assert -1 <= value <= 1, f"Value out of range: {value}"
    assert abs(sum(policy) - 1.0) < 0.01, "Policy not normalized"
    
    print("✓ Neural network tests passed!")


def test_integration():
    print("Testing component integration...")
    
    game = GoGame(board_size=9)
    model = GoNeuralNetwork(board_size=9, num_res_blocks=2, num_filters=32)
    wrapper = GoNeuralNetworkWrapper(model)
    
    # Get move probabilities
    move_probs = wrapper.get_move_probabilities(game, Player.BLACK)
    
    assert len(move_probs) > 0, "No valid moves found"
    # Pass move is always available in standard Go rules
    assert (-1, -1) in move_probs, "Pass move not in probabilities"
    
    # Make a few moves
    for _ in range(5):
        valid_moves = game.get_valid_moves()
        if len(valid_moves) > 0:
            # Take first non-pass move
            for move in valid_moves:
                if move != (-1, -1):
                    game.make_move(move[0], move[1])
                    break
    
    print("Integration tests passed!")


if __name__ == '__main__':
    print("="*50)
    print("Running Go Neural Network Tests")
    print("="*50 + "\n")
    
    try:
        test_game_engine()
        test_neural_network()
        test_integration()
        
        print("\n" + "="*50)
        print("All tests passed!")
        print("="*50)
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


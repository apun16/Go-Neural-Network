import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict
import time

from go_game import GoGame, Player
from neural_network import GoNeuralNetwork, GoNeuralNetworkWrapper
from self_play import SelfPlayGenerator, DataBuffer
from mcts import MCTS


class GoTrainer:    
    def __init__(self, board_size: int = 19, num_res_blocks: int = 5,
                 num_filters: int = 128, device: str = 'cpu',
                 learning_rate: float = 0.001, buffer_size: int = 50000):
        self.board_size = board_size
        self.device = device
        
        # Initialize network
        self.model = GoNeuralNetwork(
            board_size=board_size,
            feature_planes=7,
            num_res_blocks=num_res_blocks,
            num_filters=num_filters
        ).to(device)
        
        self.wrapper = GoNeuralNetworkWrapper(self.model, device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Loss functions
        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()
        
        # Data buffer
        self.buffer = DataBuffer(max_size=buffer_size)
        
        # Training statistics
        self.training_history = {
            'policy_loss': [],
            'value_loss': [],
            'total_loss': []
        }
    
    def train_step(self, batch: Dict) -> Dict:
        self.model.train()
        
        features = torch.FloatTensor(batch['features']).to(self.device)
        target_policies = torch.FloatTensor(batch['policies']).to(self.device)
        target_values = torch.FloatTensor(batch['values']).to(self.device)
        
        # Forward pass
        policy_logits, predicted_values = self.model(features)
        
        # Policy loss (cross-entropy with soft targets)
        policy_loss = -torch.sum(target_policies * policy_logits, dim=1).mean()
        
        # Value loss (MSE)
        predicted_values = predicted_values.squeeze()
        value_loss = self.value_loss_fn(predicted_values, target_values)
        
        # Total loss
        total_loss = policy_loss + value_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def train_epoch(self, batch_size: int = 32, num_batches: int = 100) -> Dict:
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_loss = 0.0
        
        for _ in range(num_batches):
            batch = self.buffer.get_batch(batch_size)
            losses = self.train_step(batch)
            
            total_policy_loss += losses['policy_loss']
            total_value_loss += losses['value_loss']
            total_loss += losses['total_loss']
        
        avg_losses = {
            'policy_loss': total_policy_loss / num_batches,
            'value_loss': total_value_loss / num_batches,
            'total_loss': total_loss / num_batches
        }
        
        self.training_history['policy_loss'].append(avg_losses['policy_loss'])
        self.training_history['value_loss'].append(avg_losses['value_loss'])
        self.training_history['total_loss'].append(avg_losses['total_loss'])
        
        return avg_losses
    
    def generate_data(self, num_games: int, num_simulations: int = 800):
        generator = SelfPlayGenerator(
            self.wrapper,
            num_simulations=num_simulations,
            temperature=1.0
        )
        
        batch_data = generator.generate_batch(num_games, self.board_size)
        self.buffer.add(batch_data['features'], 
                       batch_data['policies'], 
                       batch_data['values'])
        
        print(f"Generated {len(batch_data['features'])} training examples")
        print(f"Buffer size: {self.buffer.size()}")
    
    def train(self, num_iterations: int = 10, games_per_iteration: int = 10,
              num_simulations: int = 800, batches_per_epoch: int = 100,
              batch_size: int = 32, save_path: str = 'models'):
        os.makedirs(save_path, exist_ok=True)
        
        print(f"Starting training for {num_iterations} iterations")
        print(f"Device: {self.device}")
        print(f"Board size: {self.board_size}")
        
        for iteration in range(num_iterations):
            print(f"\n{'='*50}")
            print(f"Iteration {iteration + 1}/{num_iterations}")
            print(f"{'='*50}")
            
            # Generate self-play data
            print("\n1. Generating self-play data...")
            self.generate_data(games_per_iteration, num_simulations)
            
            # Train on buffer
            print("\n2. Training on buffer...")
            for epoch in range(5):  # Multiple epochs per iteration
                losses = self.train_epoch(batch_size=batch_size, 
                                        num_batches=batches_per_epoch)
                
                if (epoch + 1) % 5 == 0:
                    print(f"  Epoch {epoch + 1}/5: "
                          f"Policy Loss: {losses['policy_loss']:.4f}, "
                          f"Value Loss: {losses['value_loss']:.4f}, "
                          f"Total Loss: {losses['total_loss']:.4f}")
            
            # Save model
            model_path = os.path.join(save_path, f'model_iter_{iteration + 1}.pth')
            self.wrapper.save(model_path)
            print(f"\n3. Saved model to {model_path}")
        
        print(f"\n{'='*50}")
        print("Training completed!")
        print(f"{'='*50}")
    
    def evaluate_model(self, num_games: int = 10) -> Dict:
        wins = 0
        losses = 0
        draws = 0
        
        mcts = MCTS(self.wrapper, num_simulations=400, temperature=0.0)
        
        for game_idx in range(num_games):
            game = GoGame(board_size=self.board_size)
            
            while not game.game_over:
                current_player = game.get_current_player()
                
                if current_player == Player.BLACK:
                    # Neural network plays black
                    move = mcts.select_move(game, training=False)
                else:
                    # Random player plays white
                    valid_moves = game.get_valid_moves()
                    move = valid_moves[np.random.randint(len(valid_moves))]
                
                if move == (-1, -1):
                    game.pass_move()
                else:
                    game.make_move(move[0], move[1])
            
            winner = game.get_winner()
            if winner == Player.BLACK:
                wins += 1
            elif winner == Player.WHITE:
                losses += 1
            else:
                draws += 1
        
        return {
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'win_rate': wins / num_games
        }
    
    def save_checkpoint(self, path: str):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'board_size': self.board_size
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint.get('training_history', {
            'policy_loss': [],
            'value_loss': [],
            'total_loss': []
        })


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Go Neural Network')
    parser.add_argument('--board-size', type=int, default=9, 
                       help='Board size (default: 9, use 19 for full size)')
    parser.add_argument('--iterations', type=int, default=10,
                       help='Number of training iterations (default: 10, use 20-50 for better results)')
    parser.add_argument('--games', type=int, default=10,
                       help='Number of self-play games per iteration (default: 10, use 20-50 for better results)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to train on (cpu or cuda)')
    parser.add_argument('--save-dir', type=str, default='models',
                       help='Directory to save models')
    parser.add_argument('--eval-games', type=int, default=10,
                       help='Number of games for final evaluation (default: 10)')
    parser.add_argument('--load-model', type=str, default=None,
                       help='Path to existing model to continue training from (optional)')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = GoTrainer(
        board_size=args.board_size,
        num_res_blocks=5,
        num_filters=64 if args.board_size == 9 else 128,
        device=args.device,
        learning_rate=0.001
    )
    
    # Load existing model if specified
    if args.load_model and os.path.exists(args.load_model):
        print(f"\nLoading existing model from {args.load_model}...")
        try:
            trainer.wrapper = GoNeuralNetworkWrapper.load(args.load_model, device=args.device)
            print("Successfully loaded model. Continuing training...")
        except Exception as e:
            print(f"Warning: Could not load model ({e}). Starting fresh training.")
    else:
        print("\nStarting fresh training...")
    
    # Print training summary
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Board size: {args.board_size}x{args.board_size}")
    print(f"Iterations: {args.iterations}")
    print(f"Games per iteration: {args.games}")
    print(f"Total self-play games: {args.iterations * args.games}")
    print(f"Device: {args.device}")
    print(f"Save directory: {args.save_dir}")
    print("="*60 + "\n")
    
    # Train
    trainer.train(
        num_iterations=args.iterations,
        games_per_iteration=args.games,
        num_simulations=400 if args.board_size == 9 else 800,
        batches_per_epoch=50,
        batch_size=32,
        save_path=args.save_dir
    )
    
    # Evaluate
    print("\n" + "="*60)
    print("EVALUATING FINAL MODEL")
    print("="*60)
    results = trainer.evaluate_model(num_games=args.eval_games)
    print(f"\nResults:")
    print(f"  Win Rate: {results['win_rate']:.1%}")
    print(f"  Wins: {results['wins']}, Losses: {results['losses']}, Draws: {results['draws']}")
    print("="*60)


if __name__ == '__main__':
    main()
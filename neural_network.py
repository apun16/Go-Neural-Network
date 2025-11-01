import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GoResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super(GoResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class GoNeuralNetwork(nn.Module):
    def __init__(self, board_size: int = 19, feature_planes: int = 7, 
                 num_res_blocks: int = 5, num_filters: int = 128):
        super(GoNeuralNetwork, self).__init__()
        
        self.board_size = board_size
        self.feature_planes = feature_planes
        self.num_res_blocks = num_res_blocks
        self.num_filters = num_filters
        
        # Initial convolution
        self.conv_input = nn.Conv2d(feature_planes, num_filters, 
                                    kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_filters)
        
        # Residual tower
        self.res_blocks = nn.ModuleList([
            GoResidualBlock(num_filters) for _ in range(num_res_blocks)
        ])
        
        # Policy head
        self.conv_policy = nn.Conv2d(num_filters, 32, kernel_size=1)
        self.bn_policy = nn.BatchNorm2d(32)
        self.fc_policy = nn.Linear(32 * board_size * board_size, 
                                   board_size * board_size + 1)  # +1 for pass
        
        # Value head
        self.conv_value = nn.Conv2d(num_filters, 32, kernel_size=1)
        self.bn_value = nn.BatchNorm2d(32)
        self.fc_value1 = nn.Linear(32 * board_size * board_size, 256)
        self.fc_value2 = nn.Linear(256, 1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Input convolution
        out = F.relu(self.bn_input(self.conv_input(x)))
        
        # Residual blocks
        for res_block in self.res_blocks:
            out = res_block(out)
        
        # Policy head
        policy_out = F.relu(self.bn_policy(self.conv_policy(out)))
        policy_out = policy_out.view(policy_out.size(0), -1)
        policy = self.fc_policy(policy_out)
        policy = F.log_softmax(policy, dim=1)
        
        # Value head
        value_out = F.relu(self.bn_value(self.conv_value(out)))
        value_out = value_out.view(value_out.size(0), -1)
        value_out = F.relu(self.fc_value1(value_out))
        value = torch.tanh(self.fc_value2(value_out))
        
        return policy, value
    
    def predict(self, board_features: np.ndarray) -> tuple:
        self.eval()
        with torch.no_grad():
            # Add batch dimension
            if len(board_features.shape) == 3:
                board_features = board_features[np.newaxis, :]
            
            x = torch.FloatTensor(board_features)
            
            policy_logits, value = self.forward(x)
            policy_probs = torch.exp(policy_logits).numpy()[0]
            value = value.numpy()[0, 0]
            
            return policy_probs, value


class GoNeuralNetworkWrapper:
    def __init__(self, model: GoNeuralNetwork, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.board_size = model.board_size
    
    def move_to_index(self, row: int, col: int) -> int:
        if row == -1 and col == -1:  # Pass
            return self.board_size * self.board_size
        return row * self.board_size + col
    
    def index_to_move(self, index: int) -> tuple:
        if index == self.board_size * self.board_size:
            return (-1, -1)
        row = index // self.board_size
        col = index % self.board_size
        return (row, col)
    
    def get_move_probabilities(self, game, player) -> np.ndarray:
        features = game.get_board_features(player)
        policy_probs, _ = self.model.predict(features)
        
        # Create move probability dictionary
        move_probs = {}
        valid_moves = game.get_valid_moves(player)
        
        for move in valid_moves:
            idx = self.move_to_index(move[0], move[1])
            move_probs[move] = policy_probs[idx]
        
        # Normalize probabilities
        total_prob = sum(move_probs.values())
        if total_prob > 0:
            move_probs = {move: prob / total_prob for move, prob in move_probs.items()}
        
        return move_probs
    
    def evaluate(self, game, player) -> tuple:
        features = game.get_board_features(player)
        policy_probs, value = self.model.predict(features)
        return policy_probs, value
    
    def save(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'board_size': self.board_size,
            'model_config': {
                'feature_planes': self.model.feature_planes,
                'num_res_blocks': self.model.num_res_blocks,
                'num_filters': self.model.num_filters
            }
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu'):
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['model_config']
        board_size = checkpoint['board_size']
        
        model = GoNeuralNetwork(
            board_size=board_size,
            feature_planes=config['feature_planes'],
            num_res_blocks=config['num_res_blocks'],
            num_filters=config['num_filters']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        wrapper = cls(model, device)
        return wrapper


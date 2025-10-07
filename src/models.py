"""
Neural network architectures for quantum state tomography
"""

import torch
import torch.nn as nn


class TomographyNet(nn.Module):
    """Neural network for quantum state tomography"""
    
    def __init__(self, input_dim, hidden_dims=[256, 128, 64, 32]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 3))
        layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        bloch = self.network(x)
        # Enforce Bloch sphere constraint: ||r|| <= 1
        norm = torch.norm(bloch, dim=1, keepdim=True)
        bloch = bloch / torch.clamp(norm, min=1.0)
        return bloch
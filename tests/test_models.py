"""
Tests for neural network models
"""

import pytest
import torch
from src.models import TomographyNet


def test_model_initialization():
    """Test model can be initialized"""
    model = TomographyNet(input_dim=3, hidden_dims=[256, 128, 64, 32])
    assert model is not None


def test_model_forward():
    """Test forward pass"""
    model = TomographyNet(input_dim=3)
    x = torch.randn(10, 3)
    output = model(x)
    
    assert output.shape == (10, 3)
    # Check Bloch sphere constraint
    norms = torch.norm(output, dim=1)
    assert torch.all(norms <= 1.0 + 1e-6)


def test_model_gradient():
    """Test backward pass"""
    model = TomographyNet(input_dim=3)
    x = torch.randn(10, 3)
    target = torch.randn(10, 3)
    
    output = model(x)
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()
    
    # Check gradients exist
    for param in model.parameters():
        assert param.grad is not None
"""
Training utilities for quantum state tomography neural networks
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class QuantumDataset(Dataset):
    """PyTorch dataset for quantum measurements"""
    
    def __init__(self, measurements, bloch_vectors):
        self.measurements = torch.FloatTensor(measurements)
        self.bloch_vectors = torch.FloatTensor(bloch_vectors)
    
    def __len__(self):
        return len(self.measurements)
    
    def __getitem__(self, idx):
        return self.measurements[idx], self.bloch_vectors[idx]


def fidelity_metric(pred_bloch, true_bloch):
    """Calculate fidelity between predicted and true states"""
    dot_product = np.sum(pred_bloch * true_bloch, axis=1)
    fidelity = (1 + dot_product) / 2
    return fidelity


def train_model(model, train_loader, val_loader, epochs=1000, lr=1e-3, 
                patience=100, device='cpu'):
    """Train the tomography model"""
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    val_fidelities = []
    
    model.to(device)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for measurements, bloch in train_loader:
            measurements, bloch = measurements.to(device), bloch.to(device)
            
            optimizer.zero_grad()
            pred = model(measurements)
            loss = criterion(pred, bloch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_true = []
        
        with torch.no_grad():
            for measurements, bloch in val_loader:
                measurements, bloch = measurements.to(device), bloch.to(device)
                pred = model(measurements)
                loss = criterion(pred, bloch)
                val_loss += loss.item()
                
                all_preds.append(pred.cpu().numpy())
                all_true.append(bloch.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Calculate fidelity
        all_preds = np.vstack(all_preds)
        all_true = np.vstack(all_true)
        fidelities = fidelity_metric(all_preds, all_true)
        mean_fidelity = np.mean(fidelities)
        val_fidelities.append(mean_fidelity)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}, Fidelity={mean_fidelity:.4f}")
    
    # Restore best model
    model.load_state_dict(best_model_state)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_fidelities': val_fidelities,
        'best_epoch': len(train_losses) - patience,
        'final_fidelity': val_fidelities[-patience] if len(val_fidelities) > patience else val_fidelities[-1]
    }
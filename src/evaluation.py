"""
Evaluation metrics and utilities
"""

import numpy as np
import torch


def calculate_metrics(predictions, true_values):
    """Calculate all evaluation metrics"""
    
    # Fidelity
    fidelities = fidelity_metric(predictions, true_values)
    
    # RMSE per component
    rmse = np.sqrt(np.mean((predictions - true_values)**2, axis=0))
    
    # High-fidelity fraction
    frac_above_95 = np.mean(fidelities > 0.95)
    
    return {
        'mean_fidelity': np.mean(fidelities),
        'std_fidelity': np.std(fidelities),
        'fidelity_distribution': fidelities,
        'rmse_x': rmse[0],
        'rmse_y': rmse[1],
        'rmse_z': rmse[2],
        'frac_above_95': frac_above_95
    }


def fidelity_metric(pred_bloch, true_bloch):
    """Calculate fidelity between predicted and true states"""
    dot_product = np.sum(pred_bloch * true_bloch, axis=1)
    fidelity = (1 + dot_product) / 2
    return fidelity


def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate model on test set"""
    
    model.eval()
    all_preds = []
    all_true = []
    
    with torch.no_grad():
        for measurements, bloch in test_loader:
            measurements = measurements.to(device)
            pred = model(measurements)
            all_preds.append(pred.cpu().numpy())
            all_true.append(bloch.numpy())
    
    all_preds = np.vstack(all_preds)
    all_true = np.vstack(all_true)
    
    metrics = calculate_metrics(all_preds, all_true)
    metrics['predictions'] = all_preds
    metrics['true_values'] = all_true
    
    return metrics
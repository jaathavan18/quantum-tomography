"""
Visualization utilities for quantum tomography results
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_training_curves(history, save_path):
    """Plot training and validation curves"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history['train_losses'], label='Train Loss', alpha=0.7)
    axes[0].plot(history['val_losses'], label='Val Loss', alpha=0.7)
    axes[0].axvline(history['best_epoch'], color='r', linestyle='--', label='Best Epoch')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Training Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['val_fidelities'], label='Val Fidelity', color='green', alpha=0.7)
    axes[1].axvline(history['best_epoch'], color='r', linestyle='--', label='Best Epoch')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Mean Fidelity')
    axes[1].set_title('Validation Fidelity')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_fidelity_cdf(results_list, save_path, title="Fidelity CDF"):
    """Plot CDF of fidelities for multiple experiments"""
    
    plt.figure(figsize=(10, 6))
    
    for r in results_list:
        fidelities = np.sort(r['fidelity_distribution'])
        cdf = np.arange(1, len(fidelities) + 1) / len(fidelities)
        label = r.get('label', 'Experiment')
        plt.plot(fidelities, cdf, label=label, alpha=0.7)
    
    plt.xlabel('Fidelity')
    plt.ylabel('CDF')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_bloch_sphere_failures(predictions, true_values, fidelities, save_path, n_examples=5):
    """Plot worst predictions on Bloch sphere"""
    
    worst_indices = np.argsort(fidelities)[:n_examples]
    
    fig = plt.figure(figsize=(15, 3))
    
    for i, idx in enumerate(worst_indices):
        ax = fig.add_subplot(1, n_examples, i+1, projection='3d')
        
        true_vec = true_values[idx]
        pred_vec = predictions[idx]
        
        # Draw Bloch sphere
        u = np.linspace(0, 2*np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, alpha=0.1, color='gray')
        
        # Plot vectors
        ax.quiver(0, 0, 0, true_vec[0], true_vec[1], true_vec[2], 
                 color='blue', arrow_length_ratio=0.1, linewidth=2, label='True')
        ax.quiver(0, 0, 0, pred_vec[0], pred_vec[1], pred_vec[2], 
                 color='red', arrow_length_ratio=0.1, linewidth=2, label='Predicted')
        
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_title(f'F={fidelities[idx]:.3f}')
        
        if i == 0:
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
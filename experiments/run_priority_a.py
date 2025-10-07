"""
Priority A: Finite Shots + Readout Noise Experiments
"""

import sys
from pathlib import Path
import pickle
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generation import generate_dataset
from src.models import TomographyNet
from src.training import QuantumDataset, train_model
from src.evaluation import evaluate_model
from torch.utils.data import DataLoader
import torch

OUTPUT_DIR = Path("results/expt_3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_single_experiment(config, seed):
    """Run a single experiment with given configuration"""
    
    print(f"\n{'='*60}")
    print(f"Running: {config['name']}")
    print(f"Seed: {seed}")
    print(f"{'='*60}")
    
    # Generate data
    print("Generating training data...")
    train_meas, train_bloch = generate_dataset(
        n_states=config['n_train'],
        ensemble_type=config.get('ensemble_type', 'general'),
        mixing_p=config.get('mixing_p', 0.25),
        measurement_type=config['measurement_type'],
        shots=config.get('shots', None),
        noise_level=config.get('noise_level', 0),
        seed=seed
    )
    
    print("Generating validation data...")
    val_meas, val_bloch = generate_dataset(
        n_states=config['n_val'],
        ensemble_type=config.get('ensemble_type', 'general'),
        mixing_p=config.get('mixing_p', 0.25),
        measurement_type=config['measurement_type'],
        shots=config.get('shots', None),
        noise_level=config.get('noise_level', 0),
        seed=seed + 1000
    )
    
    print("Generating test data...")
    test_meas, test_bloch = generate_dataset(
        n_states=config['n_test'],
        ensemble_type=config.get('ensemble_type', 'general'),
        mixing_p=config.get('mixing_p', 0.25),
        measurement_type=config['measurement_type'],
        shots=config.get('shots', None),
        noise_level=config.get('noise_level', 0),
        seed=seed + 2000
    )
    
    # Create dataloaders
    train_dataset = QuantumDataset(train_meas, train_bloch)
    val_dataset = QuantumDataset(val_meas, val_bloch)
    test_dataset = QuantumDataset(test_meas, test_bloch)
    
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512)
    test_loader = DataLoader(test_dataset, batch_size=512)
    
    # Create model
    input_dim = train_meas.shape[1]
    model = TomographyNet(input_dim=input_dim, hidden_dims=[256, 128, 64, 32])
    
    # Train
    print("Training model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    history = train_model(
        model, train_loader, val_loader,
        epochs=1000, lr=1e-3, patience=100,
        device=device
    )
    
    # Evaluate on test set
    print("Evaluating on test set...")
    metrics = evaluate_model(model, test_loader, device=device)
    
    results = {
        'config': config,
        'seed': seed,
        'history': history,
        'test_fidelity_mean': metrics['mean_fidelity'],
        'test_fidelity_std': metrics['std_fidelity'],
        'test_fidelity_distribution': metrics['fidelity_distribution'],
        'rmse_x': metrics['rmse_x'],
        'rmse_y': metrics['rmse_y'],
        'rmse_z': metrics['rmse_z'],
        'frac_above_95': metrics['frac_above_95'],
        'predictions': metrics['predictions'],
        'true_values': metrics['true_values']
    }
    
    print(f"\nResults:")
    print(f"  Mean Fidelity: {results['test_fidelity_mean']:.4f} ± {results['test_fidelity_std']:.4f}")
    print(f"  RMSE (x,y,z): ({metrics['rmse_x']:.4f}, {metrics['rmse_y']:.4f}, {metrics['rmse_z']:.4f})")
    print(f"  Frac > 0.95: {results['frac_above_95']:.4f}")
    
    return results


def run_priority_a():
    """Priority A: Finite shots + readout noise"""
    
    print("\n" + "="*80)
    print("PRIORITY A: Finite Shots + Readout Noise")
    print("="*80)
    
    all_results = []
    seeds = [48, 49, 50]
    
    for shots in [10, 100, 1000]:
        for noise in [0, 0.01, 0.05]:
            for measurement_type in ['baseline', 'XZ']:
                for seed in seeds:
                    config = {
                        'name': f'PriorityA_shots{shots}_noise{int(noise*100)}pct_{measurement_type}_seed{seed}',
                        'n_train': 80000,
                        'n_val': 10000,
                        'n_test': 10000,
                        'measurement_type': measurement_type,
                        'shots': shots,
                        'noise_level': noise
                    }
                    
                    results = run_single_experiment(config, seed)
                    all_results.append(results)
                    
                    # Save individual result
                    save_path = OUTPUT_DIR / f"{config['name']}.pkl"
                    with open(save_path, 'wb') as f:
                        pickle.dump(results, f)
    
    # Save summary
    summary_data = []
    for r in all_results:
        summary_data.append({
            'Experiment': r['config']['name'],
            'Mean Fidelity': r['test_fidelity_mean'],
            'Std Fidelity': r['test_fidelity_std'],
            'RMSE_x': r['rmse_x'],
            'RMSE_y': r['rmse_y'],
            'RMSE_z': r['rmse_z'],
            'Frac > 0.95': r['frac_above_95'],
        })
    
    df = pd.DataFrame(summary_data)
    df.to_csv(OUTPUT_DIR / "priority_a_summary.csv", index=False)
    print(f"\nPriority A completed: {len(all_results)} experiments")
    
    return all_results


if __name__ == "__main__":
    run_priority_a()
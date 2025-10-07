"""
Priority B: Pure vs Mixed vs Near-Pure Ensembles
"""

import sys
from pathlib import Path
import pickle
import pandas as pd

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
    """Run a single experiment - same as Priority A"""
    # [Same implementation as in run_priority_a.py]
    # Copy the entire run_single_experiment function from above
    pass


def run_priority_b():
    """Priority B: Pure vs mixed vs near-pure ensembles"""
    
    print("\n" + "="*80)
    print("PRIORITY B: Pure vs Mixed vs Near-Pure")
    print("="*80)
    
    all_results = []
    seeds = [48, 49, 50]
    
    ensembles = [
        ('pure', None),
        ('near_pure', None),
        ('mixed', 0.1),
        ('mixed', 0.25),
        ('mixed', 0.5)
    ]
    
    for ensemble_type, mixing_p in ensembles:
        for measurement_type in ['baseline', 'XY', 'XZ', 'YZ']:
            for seed in seeds:
                ensemble_name = f"{ensemble_type}_p{mixing_p}" if mixing_p else ensemble_type
                config = {
                    'name': f'PriorityB_{ensemble_name}_{measurement_type}_seed{seed}',
                    'n_train': 80000,
                    'n_val': 10000,
                    'n_test': 10000,
                    'ensemble_type': ensemble_type,
                    'mixing_p': mixing_p,
                    'measurement_type': measurement_type
                }
                
                results = run_single_experiment(config, seed)
                all_results.append(results)
                
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
    df.to_csv(OUTPUT_DIR / "priority_b_summary.csv", index=False)
    print(f"\nPriority B completed: {len(all_results)} experiments")
    
    return all_results


if __name__ == "__main__":
    run_priority_b()
"""
Priority D: Adaptive vs non-adaptive measurement
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
    # [Copy from run_priority_a.py]
    pass


def run_priority_d():
    """Priority D: Adaptive vs non-adaptive measurement"""
    
    print("\n" + "="*80)
    print("PRIORITY D: Adaptive Measurement Protocol")
    print("="*80)
    
    all_results = []
    seeds = [48, 49, 50]
    
    for measurement_type in ['baseline', 'XZ']:
        for seed in seeds:
            config = {
                'name': f'PriorityD_adaptive_{measurement_type}_seed{seed}',
                'n_train': 80000,
                'n_val': 10000,
                'n_test': 10000,
                'measurement_type': measurement_type
            }
            
            results = run_single_experiment(config, seed)
            all_results.append(results)
            
            save_path = OUTPUT_DIR / f"{config['name']}.pkl"
            with open(save_path, 'wb') as f:
                pickle.dump(results, f)
    
    print("\nNOTE: Full adaptive measurement would require sequential decision-making.")
    print("This experiment compares measurement efficiency as a proxy.")
    
    # Save summary
    summary_data = []
    for r in all_results:
        summary_data.append({
            'Experiment': r['config']['name'],
            'Mean Fidelity': r['test_fidelity_mean'],
            'Std Fidelity': r['test_fidelity_std'],
        })
    
    df = pd.DataFrame(summary_data)
    df.to_csv(OUTPUT_DIR / "priority_d_summary.csv", index=False)
    print(f"\nPriority D completed: {len(all_results)} experiments")
    
    return all_results


if __name__ == "__main__":
    run_priority_d()
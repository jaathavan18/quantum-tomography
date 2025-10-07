"""
Aggregate results across seeds
"""

import sys
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path("results/expt_3")


def load_all_results():
    """Load all experiment results"""
    results = []
    for pkl_file in RESULTS_DIR.glob("*.pkl"):
        with open(pkl_file, 'rb') as f:
            results.append(pickle.load(f))
    return results


def aggregate_by_config():
    """Aggregate results across seeds for same configuration"""
    
    results = load_all_results()
    
    grouped = defaultdict(list)
    
    for r in results:
        # Extract config key (everything except seed)
        name_parts = r['config']['name'].rsplit('_seed', 1)
        config_key = name_parts[0]
        grouped[config_key].append(r)
    
    summary = []
    for config_key, result_list in grouped.items():
        fidelities = [r['test_fidelity_mean'] for r in result_list]
        
        summary.append({
            'Configuration': config_key,
            'Mean Fidelity': np.mean(fidelities),
            'Std across seeds': np.std(fidelities),
            'Min Fidelity': np.min(fidelities),
            'Max Fidelity': np.max(fidelities),
            'N Seeds': len(result_list)
        })
    
    df = pd.DataFrame(summary)
    df = df.sort_values('Mean Fidelity', ascending=False)
    
    return df


def main():
    """Main aggregation function"""
    
    print("Aggregating results across seeds...")
    
    df = aggregate_by_config()
    
    # Save
    output_path = RESULTS_DIR / "aggregated_results.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\nAggregated {len(df)} configurations")
    print(f"Saved to: {output_path}")
    
    # Print top 10
    print("\nTop 10 Configurations:")
    print(df.head(10).to_string(index=False))
    
    # Print bottom 10
    print("\nBottom 10 Configurations:")
    print(df.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
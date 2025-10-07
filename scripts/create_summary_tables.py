"""
Create summary tables for the paper
"""

import sys
from pathlib import Path
import pandas as pd
import pickle
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path("results/expt_3")
TABLES_DIR = Path("paper/tables")
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def load_all_results():
    """Load all experiment results"""
    results = []
    for pkl_file in RESULTS_DIR.glob("*.pkl"):
        with open(pkl_file, 'rb') as f:
            results.append(pickle.load(f))
    return results


def create_priority_summary_table():
    """Create overall priority summary table"""
    
    results = load_all_results()
    
    priorities = {}
    for r in results:
        if 'PriorityA' in r['config']['name']:
            priorities.setdefault('A', []).append(r)
        elif 'PriorityB' in r['config']['name']:
            priorities.setdefault('B', []).append(r)
        elif 'PriorityC' in r['config']['name']:
            priorities.setdefault('C', []).append(r)
        elif 'PriorityD' in r['config']['name']:
            priorities.setdefault('D', []).append(r)
    
    summary = []
    for priority, res_list in sorted(priorities.items()):
        fids = [r['test_fidelity_mean'] for r in res_list]
        summary.append({
            'Priority': priority,
            'N Experiments': len(res_list),
            'Mean Fidelity': f"{np.mean(fids):.3f} ± {np.std(fids):.3f}",
            'Min': f"{np.min(fids):.3f}",
            'Max': f"{np.max(fids):.3f}"
        })
    
    df = pd.DataFrame(summary)
    
    # Save as CSV
    df.to_csv(TABLES_DIR / "priority_summary.csv", index=False)
    
    # Save as LaTeX
    latex = df.to_latex(index=False, escape=False)
    with open(TABLES_DIR / "priority_summary.tex", 'w') as f:
        f.write(latex)
    
    print("Priority summary table created")
    return df


def create_measurement_comparison_table():
    """Create measurement scheme comparison table"""
    
    results = load_all_results()
    
    # Group by measurement type
    by_measurement = {}
    for r in results:
        meas_type = r['config']['measurement_type']
        by_measurement.setdefault(meas_type, []).append(r['test_fidelity_mean'])
    
    summary = []
    for meas_type, fids in sorted(by_measurement.items()):
        summary.append({
            'Measurement': meas_type,
            'Mean Fidelity': f"{np.mean(fids):.3f}",
            'Std': f"{np.std(fids):.3f}",
            'N': len(fids)
        })
    
    df = pd.DataFrame(summary)
    
    df.to_csv(TABLES_DIR / "measurement_comparison.csv", index=False)
    
    latex = df.to_latex(index=False, escape=False)
    with open(TABLES_DIR / "measurement_comparison.tex", 'w') as f:
        f.write(latex)
    
    print("Measurement comparison table created")
    return df


def create_all_tables():
    """Create all paper tables"""
    
    print("Creating summary tables...")
    
    create_priority_summary_table()
    create_measurement_comparison_table()
    
    print(f"\nAll tables saved to: {TABLES_DIR}")


if __name__ == "__main__":
    create_all_tables()
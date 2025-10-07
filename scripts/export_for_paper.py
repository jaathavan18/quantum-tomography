"""
Export selected results for paper figures and tables
"""

import sys
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path("results/expt_3")
PAPER_DIR = Path("paper")
PAPER_DIR.mkdir(exist_ok=True)


def load_results():
    """Load all results"""
    results = []
    for pkl_file in RESULTS_DIR.glob("*.pkl"):
        with open(pkl_file, 'rb') as f:
            results.append(pickle.load(f))
    return results


def export_main_results_table():
    """Export Table 1: Main results summary"""
    
    results = load_results()
    
    print(f"Loaded {len(results)} results")
    
    # Separate by priority
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
    
    print(f"Found priorities: {list(priorities.keys())}")
    
    # Check if we have any results
    if not priorities:
        print("ERROR: No results found! Check your results/expt_3/ directory.")
        return
    
    rows = []
    for priority in ['A', 'B', 'C', 'D']:
        if priority not in priorities:
            print(f"WARNING: No results for Priority {priority}, skipping...")
            continue
            
        res_list = priorities[priority]
        fids = [r['test_fidelity_mean'] for r in res_list]
        
        rows.append({
            'Priority': priority,
            'Experiments': len(res_list),
            'Mean Fidelity': f"{np.mean(fids):.3f} $\\pm$ {np.std(fids):.3f}",
            'Range': f"[{np.min(fids):.3f}, {np.max(fids):.3f}]"
        })
    
    if not rows:
        print("ERROR: No data to export!")
        return
    
    df = pd.DataFrame(rows)
    
    # Save as LaTeX
    latex = df.to_latex(index=False, escape=False, column_format='lccc')
    with open(PAPER_DIR / "table1_main_results.tex", 'w') as f:
        f.write(latex)
    
    print("Exported Table 1: Main results")


def export_measurement_comparison_table():
    """Export Table 2: Measurement scheme comparison"""
    
    results = load_results()
    
    # Group by measurement type
    by_meas = {}
    for r in results:
        meas = r['config']['measurement_type']
        by_meas.setdefault(meas, []).append(r['test_fidelity_mean'])
    
    # Calculate baseline for relative performance
    baseline_mean = np.mean(by_meas.get('baseline', []))
    
    rows = []
    for meas in ['baseline', 'XZ', 'XY', 'YZ', 'sic']:
        if meas in by_meas:
            fids = by_meas[meas]
            mean_fid = np.mean(fids)
            relative = 100 * mean_fid / baseline_mean if baseline_mean > 0 else 0
            
            rows.append({
                'Measurement': meas,
                'Mean Fidelity': f"{mean_fid:.3f} $\\pm$ {np.std(fids):.3f}",
                'Relative': f"{relative:.1f}\\%"
            })
    
    df = pd.DataFrame(rows)
    
    latex = df.to_latex(index=False, escape=False, column_format='lcc')
    with open(PAPER_DIR / "table2_measurement_comparison.tex", 'w') as f:
        f.write(latex)
    
    print("Exported Table 2: Measurement comparison")


def export_shots_noise_table():
    """Export Table 3: Shots vs Noise performance"""
    
    results = load_results()
    
    # Filter Priority A
    priority_a = [r for r in results if 'PriorityA' in r['config']['name']]
    
    # Organize by shots, noise, measurement
    data = {}
    for r in priority_a:
        shots = r['config']['shots']
        noise = int(r['config']['noise_level'] * 100)
        meas = r['config']['measurement_type']
        
        key = (meas, noise, shots)
        data.setdefault(key, []).append(r['test_fidelity_mean'])
    
    # Create table
    rows = []
    for meas in ['baseline', 'XZ']:
        for noise in [0, 1, 5]:
            row = {'Measurement': meas, 'Noise': f"{noise}\\%"}
            for shots in [10, 100, 1000]:
                key = (meas, noise, shots)
                if key in data:
                    fids = data[key]
                    row[f'{shots} shots'] = f"{np.mean(fids):.3f}"
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    latex = df.to_latex(index=False, escape=False)
    with open(PAPER_DIR / "table3_shots_noise.tex", 'w') as f:
        f.write(latex)
    
    print("Exported Table 3: Shots vs Noise")


def main():
    """Export all paper materials"""
    
    print("Exporting materials for paper...")
    
    export_main_results_table()
    export_measurement_comparison_table()
    export_shots_noise_table()
    
    print("\nAll paper materials exported to:", PAPER_DIR)


if __name__ == "__main__":
    main()
"""
Generate all figures for the paper
"""

import sys
from pathlib import Path
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization import plot_training_curves, plot_fidelity_cdf, plot_bloch_sphere_failures

RESULTS_DIR = Path("results/expt_3")
FIGURES_DIR = Path("paper/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_all_results():
    """Load all experiment results"""
    results = []
    for pkl_file in RESULTS_DIR.glob("*.pkl"):
        with open(pkl_file, 'rb') as f:
            results.append(pickle.load(f))
    return results


def normalize_result_keys(result):
    """Normalize keys to match expected format"""
    # Create a normalized copy
    normalized = result.copy()
    
    # Map old keys to new keys if needed
    if 'test_fidelity_distribution' in result and 'fidelity_distribution' not in result:
        normalized['fidelity_distribution'] = result['test_fidelity_distribution']
    
    return normalized


def generate_priority_a_figures(results_a):
    """Generate Priority A figures"""
    
    # Training curves
    if len(results_a) > 0:
        plot_training_curves(
            results_a[0]['history'],
            FIGURES_DIR / "training_curves.pdf"
        )
        print("  - Training curves saved")
    
    # Fidelity CDF - normalize results first
    results_with_labels = []
    for r in results_a[:9]:
        normalized = normalize_result_keys(r)
        normalized['label'] = f"{r['config']['shots']} shots, {int(r['config']['noise_level']*100)}% noise"
        results_with_labels.append(normalized)
    
    plot_fidelity_cdf(
        results_with_labels,
        FIGURES_DIR / "priority_a_cdf.pdf",
        title="Priority A: Fidelity CDF"
    )
    print("  - Priority A CDF saved")


def generate_priority_b_figures(results_b):
    """Generate Priority B figures"""
    
    # Fidelity CDF by ensemble
    results_with_labels = []
    for r in results_b[:12]:  # First 12 for variety
        normalized = normalize_result_keys(r)
        ensemble = r['config']['ensemble_type']
        mixing = r['config'].get('mixing_p')
        label = f"{ensemble}_p{mixing}" if mixing else ensemble
        normalized['label'] = label
        results_with_labels.append(normalized)
    
    plot_fidelity_cdf(
        results_with_labels,
        FIGURES_DIR / "priority_b_cdf.pdf",
        title="Priority B: Fidelity CDF by Ensemble"
    )
    print("  - Priority B CDF saved")


def generate_comparison_figures(all_results):
    """Generate comparison figures across experiments"""
    
    # By measurement type
    by_measurement = {}
    for r in all_results:
        meas = r['config']['measurement_type']
        if meas not in by_measurement:
            by_measurement[meas] = []
        normalized = normalize_result_keys(r)
        normalized['label'] = meas
        by_measurement[meas].append(normalized)
    
    # Plot each measurement type
    for meas, results in by_measurement.items():
        if len(results) > 0:
            plot_fidelity_cdf(
                results[:20],  # Limit to 20 for clarity
                FIGURES_DIR / f"cdf_{meas}.pdf",
                title=f"Fidelity CDF: {meas}"
            )
    
    print("  - Measurement comparison CDFs saved")


def generate_failure_analysis(results):
    """Generate failure analysis figures"""
    
    # Find experiment with widest fidelity range
    max_std = 0
    best_result = None
    
    for r in results:
        if 'test_fidelity_std' in r and r['test_fidelity_std'] > max_std:
            max_std = r['test_fidelity_std']
            best_result = r
    
    if best_result and 'predictions' in best_result and 'true_values' in best_result:
        # Calculate fidelities
        predictions = best_result['predictions']
        true_values = best_result['true_values']
        
        # Calculate fidelity
        dot_product = np.sum(predictions * true_values, axis=1)
        fidelities = (1 + dot_product) / 2
        
        plot_bloch_sphere_failures(
            predictions,
            true_values,
            fidelities,
            FIGURES_DIR / "failure_analysis.pdf",
            n_examples=5
        )
        print("  - Failure analysis saved")


def generate_all_figures():
    """Generate all paper figures"""
    
    print("Loading results...")
    all_results = load_all_results()
    print(f"Loaded {len(all_results)} results")
    
    # Separate by priority
    results_a = [r for r in all_results if 'PriorityA' in r['config']['name']]
    results_b = [r for r in all_results if 'PriorityB' in r['config']['name']]
    results_c = [r for r in all_results if 'PriorityC' in r['config']['name']]
    results_d = [r for r in all_results if 'PriorityD' in r['config']['name']]
    
    print(f"\nGenerating figures...")
    print(f"Priority A: {len(results_a)} results")
    if results_a:
        generate_priority_a_figures(results_a)
    
    print(f"Priority B: {len(results_b)} results")
    if results_b:
        generate_priority_b_figures(results_b)
    
    print(f"\nGenerating comparison figures...")
    generate_comparison_figures(all_results)
    
    print(f"\nGenerating failure analysis...")
    generate_failure_analysis(all_results)
    
    print(f"\nAll figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    generate_all_figures()
"""
Master script to run all experiments
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.run_priority_a import run_priority_a
from experiments.run_priority_b import run_priority_b
from experiments.run_priority_c import run_priority_c
from experiments.run_priority_d import run_priority_d


def main():
    """Run all experimental priorities"""
    
    print("="*80)
    print("QUANTUM STATE TOMOGRAPHY - COMPLETE EXPERIMENTAL SUITE")
    print("="*80)
    
    # Run all priorities
    results_a = run_priority_a()
    results_b = run_priority_b()
    results_c = run_priority_c()
    results_d = run_priority_d()
    
    # Combine and generate final report
    all_results = results_a + results_b + results_c + results_d
    
    print(f"\nAll experiments complete! Total: {len(all_results)} experiments")
    print(f"Results saved to: results/expt_3/")


if __name__ == "__main__":
    main()
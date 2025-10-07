"""
Tests for measurement simulations
"""

import pytest
import numpy as np
from src.data_generation import MeasurementSimulator, QuantumStateGenerator


def test_sic_povm():
    """Test SIC-POVM measurements"""
    gen = QuantumStateGenerator(seed=42)
    sim = MeasurementSimulator()
    
    rho, _ = gen.generate_pure_state()
    probs = sim.measure_sic_povm(rho)
    
    # Check 4 outcomes
    assert probs.shape == (4,)
    
    # Should sum to 1 (probabilities)
    assert np.isclose(np.sum(probs), 1.0)
    
    # All non-negative
    assert np.all(probs >= -1e-10)


def test_two_basis_measurement():
    """Test two-basis measurements"""
    gen = QuantumStateGenerator(seed=42)
    sim = MeasurementSimulator()
    
    rho, _ = gen.generate_pure_state()
    
    for basis in ['XY', 'XZ', 'YZ']:
        meas = sim.measure_two_basis(rho, basis)
        assert meas.shape == (2,)
        assert np.all(meas >= -1.0) and np.all(meas <= 1.0)
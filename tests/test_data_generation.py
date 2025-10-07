"""
Tests for data generation module
"""

import pytest
import numpy as np
from src.data_generation import QuantumStateGenerator, MeasurementSimulator, generate_dataset


def test_pure_state_generation():
    """Test pure state generation"""
    gen = QuantumStateGenerator(seed=42)
    rho, bloch = gen.generate_pure_state()
    
    # Check density matrix properties
    assert rho.shape == (2, 2)
    assert np.isclose(np.trace(rho), 1.0)
    assert np.allclose(rho, rho.conj().T)  # Hermitian
    
    # Check Bloch vector
    assert bloch.shape == (3,)
    assert np.linalg.norm(bloch) <= 1.0 + 1e-10


def test_mixed_state_generation():
    """Test mixed state generation"""
    gen = QuantumStateGenerator(seed=42)
    rho, bloch = gen.generate_mixed_state(p=0.5)
    
    # Check density matrix
    assert np.isclose(np.trace(rho), 1.0)
    assert np.allclose(rho, rho.conj().T)
    
    # Check Bloch vector is inside sphere
    assert np.linalg.norm(bloch) <= 1.0 + 1e-10


def test_pauli_measurements():
    """Test Pauli measurements"""
    gen = QuantumStateGenerator(seed=42)
    sim = MeasurementSimulator()
    
    rho, bloch = gen.generate_pure_state()
    meas = sim.measure_xyz(rho)
    
    # Check measurement outcomes
    assert meas.shape == (3,)
    assert np.all(meas >= -1.0) and np.all(meas <= 1.0)
    
    # Check consistency with Bloch vector
    assert np.allclose(meas, bloch, atol=1e-10)


def test_shot_noise():
    """Test finite-shot noise"""
    sim = MeasurementSimulator()
    
    # Create perfect +1 eigenstate
    probs = np.array([1.0, 1.0, 1.0])
    
    # Apply shot noise
    noisy = sim.apply_shot_noise(probs, shots=100)
    
    # Should be close but not exact
    assert np.all(noisy <= 1.0)
    assert np.all(noisy >= -1.0)


def test_readout_noise():
    """Test readout noise"""
    sim = MeasurementSimulator()
    
    probs = np.array([1.0, 0.0, -1.0])
    noisy = sim.apply_readout_noise(probs, noise_level=0.1)
    
    # Noise should modify values
    assert not np.allclose(probs, noisy)
    assert np.all(noisy >= -1.0) and np.all(noisy <= 1.0)


def test_dataset_generation():
    """Test full dataset generation"""
    meas, bloch = generate_dataset(
        n_states=100,
        ensemble_type='pure',
        measurement_type='baseline',
        seed=42
    )
    
    assert meas.shape == (100, 3)
    assert bloch.shape == (100, 3)
    assert np.all(np.linalg.norm(bloch, axis=1) <= 1.0 + 1e-10)
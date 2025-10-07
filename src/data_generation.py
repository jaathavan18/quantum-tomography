"""
Data generation for quantum state tomography.
Includes state generation, measurement simulation, and noise models.
"""

import numpy as np
from scipy.stats import unitary_group


class QuantumStateGenerator:
    """Generate random quantum states and density matrices"""
    
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
    
    def generate_pure_state(self):
        """Generate random pure state"""
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2*np.pi)
        
        state = np.array([
            np.cos(theta/2),
            np.exp(1j*phi) * np.sin(theta/2)
        ], dtype=complex)
        
        rho = np.outer(state, state.conj())
        return rho, np.array([np.sin(theta)*np.cos(phi), 
                              np.sin(theta)*np.sin(phi), 
                              np.cos(theta)])
    
    def generate_mixed_state(self, p):
        """Generate mixed state with mixing parameter p"""
        rho_pure, _ = self.generate_pure_state()
        rho_mixed = np.eye(2) / 2
        rho = (1-p) * rho_pure + p * rho_mixed
        bloch = self.density_to_bloch(rho)
        return rho, bloch
    
    def generate_near_pure_state(self, eigenvalues=[0.99, 0.01]):
        """Generate near-pure state with specified eigenvalues"""
        U = unitary_group.rvs(2)
        rho = U @ np.diag(eigenvalues) @ U.conj().T
        bloch = self.density_to_bloch(rho)
        return rho, bloch
    
    def density_to_bloch(self, rho):
        """Convert density matrix to Bloch vector"""
        pauli_x = np.array([[0, 1], [1, 0]])
        pauli_y = np.array([[0, -1j], [1j, 0]])
        pauli_z = np.array([[1, 0], [0, -1]])
        
        x = np.real(np.trace(rho @ pauli_x))
        y = np.real(np.trace(rho @ pauli_y))
        z = np.real(np.trace(rho @ pauli_z))
        
        return np.array([x, y, z])
    
    def bloch_to_density(self, bloch):
        """Convert Bloch vector to density matrix"""
        x, y, z = bloch
        rho = 0.5 * np.array([
            [1 + z, x - 1j*y],
            [x + 1j*y, 1 - z]
        ], dtype=complex)
        return rho


class MeasurementSimulator:
    """Simulate quantum measurements with noise"""
    
    def __init__(self):
        self.pauli_x = np.array([[0, 1], [1, 0]])
        self.pauli_y = np.array([[0, -1j], [1j, 0]])
        self.pauli_z = np.array([[1, 0], [0, -1]])
    
    def measure_pauli(self, rho, pauli):
        """Measure expectation value of Pauli operator"""
        return np.real(np.trace(rho @ pauli))
    
    def measure_xyz(self, rho):
        """Measure all three Pauli operators"""
        return np.array([
            self.measure_pauli(rho, self.pauli_x),
            self.measure_pauli(rho, self.pauli_y),
            self.measure_pauli(rho, self.pauli_z)
        ])
    
    def measure_two_basis(self, rho, basis_pair):
        """Measure two Pauli bases"""
        paulis = {
            'X': self.pauli_x,
            'Y': self.pauli_y,
            'Z': self.pauli_z
        }
        
        return np.array([
            self.measure_pauli(rho, paulis[basis_pair[0]]),
            self.measure_pauli(rho, paulis[basis_pair[1]])
        ])
    
    def measure_sic_povm(self, rho):
        """Simulate SIC-POVM measurement (4 outcomes)"""
        sic_elements = [
            np.array([[1, 0], [0, 0]]) / 2,
            np.array([[1, 1], [1, 1]]) / 4,
            np.array([[1, -1j], [1j, 1]]) / 4,
            np.array([[1, 1j], [-1j, 1]]) / 4
        ]
        
        probs = [np.real(np.trace(rho @ elem)) for elem in sic_elements]
        return np.array(probs)
    
    def apply_shot_noise(self, probabilities, shots):
        """Convert probabilities to finite-shot counts"""
        prob_plus = (probabilities + 1) / 2
        counts_plus = np.random.binomial(shots, prob_plus)
        return 2 * counts_plus / shots - 1
    
    def apply_readout_noise(self, probabilities, noise_level):
        """Apply readout flip noise"""
        prob_plus = (probabilities + 1) / 2
        noisy_prob = prob_plus * (1 - noise_level) + (1 - prob_plus) * noise_level
        return 2 * noisy_prob - 1
    
    def measure_with_noise(self, rho, measurement_type, shots=None, noise_level=0):
        """Measure with optional shot noise and readout noise"""
        
        if measurement_type == 'baseline':
            probs = self.measure_xyz(rho)
        elif measurement_type in ['XY', 'XZ', 'YZ']:
            probs = self.measure_two_basis(rho, measurement_type)
        elif measurement_type == 'sic':
            return self.measure_sic_povm(rho)
        else:
            raise ValueError(f"Unknown measurement type: {measurement_type}")
        
        if shots is not None:
            probs = self.apply_shot_noise(probs, shots)
        
        if noise_level > 0:
            probs = self.apply_readout_noise(probs, noise_level)
        
        return probs


def generate_dataset(n_states, ensemble_type='general', mixing_p=0.25, 
                     measurement_type='baseline', shots=None, noise_level=0, seed=None):
    """
    Generate complete dataset
    
    Args:
        n_states: number of states to generate
        ensemble_type: 'general', 'pure', 'near_pure', 'mixed'
        mixing_p: mixing parameter for mixed states
        measurement_type: 'baseline', 'XY', 'XZ', 'YZ', 'sic'
        shots: number of shots (None for infinite)
        noise_level: readout noise level (0-1)
        seed: random seed
    
    Returns:
        measurements: array of measurement outcomes
        bloch_vectors: array of true Bloch vectors
    """
    gen = QuantumStateGenerator(seed=seed)
    sim = MeasurementSimulator()
    
    measurements = []
    bloch_vectors = []
    
    for _ in range(n_states):
        if ensemble_type == 'pure':
            rho, bloch = gen.generate_pure_state()
        elif ensemble_type == 'near_pure':
            rho, bloch = gen.generate_near_pure_state([0.99, 0.01])
        elif ensemble_type == 'mixed':
            rho, bloch = gen.generate_mixed_state(mixing_p)
        else:  # general ensemble
            rand = np.random.rand()
            if rand < 0.7:
                rho, bloch = gen.generate_pure_state()
            else:
                rho, bloch = gen.generate_mixed_state(np.random.uniform(0.1, 0.5))
        
        meas = sim.measure_with_noise(rho, measurement_type, shots, noise_level)
        
        measurements.append(meas)
        bloch_vectors.append(bloch)
    
    return np.array(measurements), np.array(bloch_vectors)
import numpy as np
from itertools import combinations


def NoiseEstPauli(outcomes: list,
                  subsystems: list = None
                  ) -> dict:
    """
    Calibration estimator for f_I in the Pauli shadows channel.
    """
    n_qubits = len(outcomes[0][1])
    
    if subsystems is None:
        """Defaults to 2-RDMs."""
        subsystems = []
        for j in range(1, 3):
            for I in combinations(range(n_qubits), j):
                subsystems.append(frozenset(I))
    
    calibration_coeffs = {I : 0.0 for I in subsystems}
    for pauli_basis, sample in outcomes:
        for I in calibration_coeffs:
            subsystem_pauli = [pauli_basis[i][1] for i in I]
            if all(W == "Z" for W in subsystem_pauli):
                sign = 1
                for i in I:
                    sign *= int(pauli_basis[i][0] + "1")
                    if sample[i] == 1:
                        sign = -sign
                calibration_coeffs[I] += sign
    
    n_samples = len(outcomes)
    for I in calibration_coeffs:
        calibration_coeffs[I] /= n_samples
    
    return calibration_coeffs


def NoiseEstPauliSym(outcomes: list,
                     k: int = 2
                     ) -> dict:
    """
    Calibration estimator for f_k in the subsystem-symmetrized
    Pauli shadows channel.
    """
    n_qubits = len(outcomes[0][1])
    
    calibration_coeffs = {j : 0.0 for j in range(1, k + 1)}
    for measurement, sample in outcomes:
        
        pauli_basis, permutation = measurement
        
        for j in calibration_coeffs:
            for I in combinations(range(n_qubits), j):
                permuted_I = [permutation[i] for i in I]
                subsystem_pauli = [pauli_basis[i][1] for i in permuted_I]
                if all(W == "Z" for W in subsystem_pauli):
                    sign = 1
                    for i in permuted_I:
                        sign *= int(pauli_basis[i][0] + "1")
                        if sample[i] == 1:
                            sign = -sign
                    calibration_coeffs[j] += sign
    
    n_samples = len(outcomes)
    for j in calibration_coeffs:
        calibration_coeffs[j] /= n_samples
    
    return calibration_coeffs


def RShadow(qubit_rdm: dict, calibration_data) -> dict:
    
    if isinstance(calibration_data, list):
        calibration_coeffs = NoiseEstPauli(calibration_data)
    elif isinstance(calibration_data, dict):
        calibration_coeffs = calibration_data
    else:
        raise TypeError(
            '`calibration_data` must be either list (raw shadow outcomes)'
            ' or dict (coefficients computes from NoiseEstPauli).'
        )
    
    robust_rdm = {}
    for term, val in qubit_rdm.items():
        z = tuple([pauli[0] for pauli in term])
        w = len(z)
        noiseless_shadow_coeff = 1.0 / 3.0**w
        
        try:
            noisy_shadow_coeff = calibration_coeffs[z]
        except KeyError:
            noisy_shadow_coeff = noiseless_shadow_coeff
            print('Subsystem {} not calibrated, '
                  'returning noisy expectation value for {}'.format(z, term)
                  )

        robust_rdm[term] = val * (noiseless_shadow_coeff / noisy_shadow_coeff)
    
    return robust_rdm


def invert_permutation(permutation):
    
    return np.arange(len(permutation))[np.argsort(permutation)].tolist()


if __name__ == '__main__':
    import sys
    pauli_shadows_dir = '../pauli_shadows'
    if pauli_shadows_dir not in sys.path:
        sys.path.append(pauli_shadows_dir)

    from pauli_shadows.simulation.simple_noise_models import (
        bit_flip,
        depolarize_local,
        depolarize_global,
        amplitude_damp
    )


    def calibration_samples(n_qubits: int,
                            n_samples: int,
                            noise_model,
                            p: float
                            ) -> list:
        """
        Function for testing the calibration process. Generates `n_samples` of
        bit strings from Pauli-shadow sampling |0^n>, affected by `noise_model`.
        """

        if isinstance(noise_model, str):
            if noise_model == 'bf':
                noise_model = bit_flip
            elif noise_model == 'dl':
                noise_model = depolarize_local
            elif noise_model == 'dg':
                noise_model = depolarize_global
            elif noise_model == 'ad':
                noise_model = amplitude_damp
            else:
                raise ValueError('No valid noise model specified.')

        """Random samples according to the distribution P(0) = 1/2, P(1) = 1/2, which
        is the result of measuring |0> in the \pm X/Y/Z basis uniformly at random"""
        noiseless_samples = np.random.randint(2, size=(n_samples, n_qubits))

        """We back out a choice of Pauli measurement basis using the conditional distributions:
        P(X|0) = 1/6, P(Y|0) = 1/6, P(Z|0) = 1/3, P(-X|0) = 1/6, P(-Y|0) = 1/6, P(-Z|0) = 0
        P(X|1) = 1/6, P(Y|1) = 1/6, P(Z|1) = 0,   P(-X|1) = 1/6, P(-Y|1) = 1/6, P(-Z|1) = 1/3
        These conditional probabilities can be determined using Bayes' theorem."""
        pauli_measurements_inferred = np.random.randint(6, size=(n_samples, n_qubits))

        calibration_outcomes = []
        for j in range(n_samples):
            sample = noiseless_samples[j]

            pauli_basis = []
            for i in range(n_qubits):
                if pauli_measurements_inferred[j, i] == 0:
                    pauli_basis.append("+X")
                elif pauli_measurements_inferred[j, i] == 1:
                    pauli_basis.append("-X")
                elif pauli_measurements_inferred[j, i] == 2:
                    pauli_basis.append("+Y")
                elif pauli_measurements_inferred[j, i] == 3:
                    pauli_basis.append("-Y")
                else:
                    if sample[i] == 0:
                        pauli_basis.append("+Z")
                    else:
                        pauli_basis.append("-Z")

            sample = sample.tolist()

            noise_model(sample, p)
            classical_shadow = [pauli_basis, sample]

            calibration_outcomes.append(classical_shadow)

        return calibration_outcomes


    n_qubits = 8
    n_samples = int(1e4)
    noise_model = 'dl'
    p = 0.1
    samples = calibration_samples(n_qubits, n_samples, noise_model, p=p)
    calib_coeffs = NoiseEstPauli(samples)

import numpy as np
from itertools import combinations
from sys import argv
from time import time
import pickle as pkl
import json

from simple_noise_models import bit_flip, depolarize_local, depolarize_global, amplitude_damp


def calibration_samples(n_qubits: int, n_samples: int, noise_model, p: float, symmetrize: bool = False) -> list:
    
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
#         if symmetrize:
#             permutation = np.random.permutation(n_qubits)
#             sample = sample[permutation].tolist()
        
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
        
#         if symmetrize:
# #             permutation = np.random.permutation(n_qubits)
# #             sample = sample[permutation].tolist()
            
#             noise_model(sample, p)
#             classical_shadow = [[pauli_basis, permutation.tolist()], sample]
#         else:
        sample = sample.tolist()

        noise_model(sample, p)
        classical_shadow = [pauli_basis, sample]
        
        calibration_outcomes.append(classical_shadow)
    
    return calibration_outcomes


def NoiseEstPauli(outcomes: list, subsystems: list = None, symmetrize: bool = False) -> dict:
    
    n_qubits = len(outcomes[0][1])
    
    if subsystems is None:
        subsystems = []
        for j in range(1, 3):
            for z in combinations(range(n_qubits), j):
                subsystems.append(frozenset(z))
    
    calibration_coeffs = {z : 0.0 for z in subsystems}
    for pauli_basis, sample in outcomes:
        
        if symmetrize:
#             pauli_basis, permutation = pauli_basis
#             inverse_permutation = invert_permutation(permutation)
            permutation = np.random.permutation(n_qubits)
        
        for z in calibration_coeffs:
            if symmetrize:
                permutation = np.random.permutation(n_qubits)
                z_permuted = frozenset([permutation[i] for i in z])
            else:
                z_permuted = z
            
            subsystem_pauli = [pauli_basis[i][1] for i in z_permuted]
            if all(W == "Z" for W in subsystem_pauli):
                sign = 1
                for i in z_permuted:
                    sign *= int(pauli_basis[i][0] + "1")
                    if sample[i] == 1:
                        sign = -sign
                calibration_coeffs[z] += sign
    
    n_samples = len(outcomes)
    for z in calibration_coeffs:
        calibration_coeffs[z] /= n_samples
    
    return calibration_coeffs


def NoiseEstPauliSym(outcomes: list, k: int = 2) -> dict:
    
    n_qubits = len(outcomes[0][1])
    
    calibration_coeffs = {j : 0.0 for j in range(1, k + 1)}
    for measurement, sample in outcomes:
        
        pauli_basis, permutation = measurement
        
        for j in calibration_coeffs:
            subsystem_pauli = [pauli_basis[i][1] for i in z]
            if all(W == "Z" for W in subsystem_pauli):
                sign = 1
                for i in z:
                    sign *= int(pauli_basis[i][0] + "1")
                    if sample[i] == 1:
                        sign = -sign
                calibration_coeffs[z] += sign
    
    n_samples = len(outcomes)
    for z in calibration_coeffs:
        calibration_coeffs[z] /= n_samples
    
    return calibration_coeffs


def RShadow(qubit_rdm: dict, calibration_data) -> dict:
    
    if isinstance(calibration_data, list):
        calibration_coeffs = NoiseEstPauli(calibration_data)
    elif isinstance(calibration_data, dict):
        calibration_coeffs = calibration_data
    else:
        raise TypeError('`calibration_data` must be either list (raw shadow outcomes) or dict (coefficients computes from NoiseEstPauli).')
    
    robust_rdm = {}
    for term, val in qubit_rdm.items():
        z = tuple([pauli[0] for pauli in term])
        w = len(z)
        noiseless_shadow_coeff = 1.0 / 3.0**w
        
        try:
            noisy_shadow_coeff = calibration_coeffs[z]
        except KeyError:
            noisy_shadow_coeff = noiseless_shadow_coeff
            print('Subsystem {} not calibrated, returning noisy expectation value for {}'.format(z, term))
        
        
        robust_rdm[term] = val * (noiseless_shadow_coeff / noisy_shadow_coeff)
    
    return robust_rdm


def combine_rdms(rdm1: dict, num_samples1: int, rdm2: dict, num_samples2: int) -> dict:

    combined_rdm = {}
    total_num_samples = num_samples1 + num_samples2
    for op in rdm1:
        val = rdm1[op] * num_samples1
        try:
            val += rdm2[op] * num_samples2
        except KeyError:
            pass

        combined_rdm[op] = val / total_num_samples

    return combined_rdm


def invert_permutation(permutation):
    
    return np.arange(len(permutation))[np.argsort(permutation)].tolist()


if __name__ == '__main__':

    n_qubits = int(argv[1])
    min_num_samples = int(float(argv[2]))
    max_num_samples = int(float(argv[3]))
    num_points = int(argv[4])
    noise_parameter = float(argv[5])

    scratch_dir = '/wheeler/scratch/azhao/mps_shadows/calibration/'

    if min_num_samples == max_num_samples:
        x = [max_num_samples]
        num_points = 1
    else:
        x0 = np.log10(min_num_samples)
        x1 = np.log10(max_num_samples)
        x = np.logspace(x0, x1, num=num_points, dtype=int)
        x[0] = min_num_samples
        x[-1] = max_num_samples

    t1 = time()
    subsystems = []
    for k in range(1, 2 + 1):
        for z in combinations(range(n_qubits), k):
            subsystems.append(z)

    for noise_model, noise_model_str in zip([depolarize_local, amplitude_damp, bit_flip], ['dl', 'ad', 'bf']): 

        calibration_outcomes = calibration_samples(n_qubits, max_num_samples, noise_model, noise_parameter)

        staggered_estimates = [NoiseEstPauli(calibration_outcomes[:x[0]], subsystems)]
        for i in range(1, len(x)):

            start = x[i - 1]
            end = x[i]

            calibration_coeffs = NoiseEstPauli(calibration_outcomes[start:end], subsystems)
            calibration_coeffs = combine_rdms(staggered_estimates[i - 1], start, calibration_coeffs, end - start)

            staggered_estimates.append(calibration_coeffs)

        data_fn = scratch_dir + 'samples/' + 'n={}_calibration_samples_'.format(n_qubits)
        data_fn += '_samples={}_noise={:.5f}{}'.format(max_num_samples, noise_parameter * 100, noise_model_str) + '.json'
        with open(data_fn, 'w') as f:
            json.dump(calibration_outcomes, f)

        calibration_fn = scratch_dir + 'estimates/' + 'n={}_calibration_estimates_'.format(n_qubits)
        data_fn += '_samples={}_noise={:.5f}{}'.format(max_num_samples, noise_parameter * 100, noise_model_str) + '.pkl'
        with open(calibration_fn, 'wb') as f:
            pkl.dump(staggered_estimates, f)

    dt = time() - t1
    print('Time elapsed:', dt)

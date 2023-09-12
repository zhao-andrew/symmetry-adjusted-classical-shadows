import numpy as np
from scipy.special import binom
from itertools import combinations
from sys import argv
from time import time
import pickle as pkl
import json

from simple_noise_models import bit_flip, depolarize_local, depolarize_global, amplitude_damp

from free_fermion_simulator import fock_cov_matrix
from sample_gaussian_shadows import shadow_sampling
from fermion_shadows_prediction import (diagonal_majoranas,
                                        majorana_ops_measured_by_permutation,
                                        invert_permutation,
                                        diagonal_majorana_matrix_element,
                                        permutation_parity)


def calibration_samples(n_modes: int, n_samples: int, noise_model, p: float) -> list:

    cov_matrix = fock_cov_matrix(n_modes, 0)

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
    
    calibration_outcomes = shadow_sampling(cov_matrix, repetitions=n_samples)
    for i in range(n_samples):
        noise_model(calibration_outcomes[i][1], p)
    
    return calibration_outcomes


def NoiseEstFermi(outcomes: list, k: int = 2) -> dict:
    
    n_modes = len(outcomes[0][1])
    N = 2 * n_modes

    k_body_diagonal_majoranas = diagonal_majoranas(n_modes, k)
    nCk = [binom(n_modes, j) for j in range(k + 1)]
    
    calibration_coeffs = {j : 0.0 for j in range(1, k + 1)}
    for permutation, bit_string in outcomes:
        inverse_permutation = invert_permutation(permutation)
        UZUdag = majorana_ops_measured_by_permutation(inverse_permutation, k_body_diagonal_majoranas)

        for _, op in UZUdag:
            if op in k_body_diagonal_majoranas:
                j = len(op) // 2

                permutation_on_op = [permutation[i] for i in op]
                sign = (-1)**permutation_parity(permutation_on_op)
                val = sign * diagonal_majorana_matrix_element(op, bit_string)

                calibration_coeffs[j] += val / nCk[j]

    n_samples = len(outcomes)
    for j in calibration_coeffs:
        calibration_coeffs[j] /= n_samples
    
    return calibration_coeffs


def RShadowFermi(majorana_rdm: dict, n_modes, calibration_data, k=2) -> dict:
    
    if isinstance(calibration_data, list):
        calibration_coeffs = NoiseEstFermi(calibration_data, k)
    elif isinstance(calibration_data, dict):
        calibration_coeffs = calibration_data
    else:
        raise TypeError('`calibration_data` must be either list (raw shadow outcomes) or dict (coefficients computes from NoiseEstFermi).')

    noiseless_shadow_coeff = [binom(n_modes, j) / binom(2 * n_modes, 2 * j) for j in range(k + 1)]

    robust_rdm = {}
    for term, val in majorana_rdm.items():
        j = len(term) // 2
        f = noiseless_shadow_coeff[j]
        
        noisy_shadow_coeff = calibration_coeffs[j]

        robust_rdm[term] = val * (f / noisy_shadow_coeff)
    
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

if __name__ == '__main__':

    n_qubits = 10
    min_num_samples = int(1e3)
    max_num_samples = int(5e4)
    num_points = 7
    noise_model = 'ad'
    noise_parameter = 0.

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

    outcomes = calibration_samples(n_qubits, max_num_samples, noise_model, noise_parameter)
    calib_coeff = NoiseEstFermi(outcomes, 2)

    dt = time() - t1
    print('Time elapsed:', dt)

    exact_coeff = {j : binom(n_qubits, j) / binom(2 * n_qubits, 2 * j) for j in range(1, 2 + 1)}

    print(calib_coeff)
    print(exact_coeff)

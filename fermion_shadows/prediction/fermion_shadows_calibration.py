"""
NoiseEst for matchgate shadows. Theory follows straightforwardly from that
described in arXiv:2011.09636, but application to fermions explicitly written
down in arXiv:[].
"""
from scipy.special import binom


def NoiseEstFermi(outcomes: list,
                  k: int = 2
                  ) -> dict:
    """
    Calibration estimator for f_{2k} in the matchgate shadows channel.
    """
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


def RShadowFermi(majorana_rdm: dict,
                 n_modes,
                 calibration_data,
                 k: int = 2
                 ) -> dict:
    
    if isinstance(calibration_data, list):
        calibration_coeffs = NoiseEstFermi(calibration_data, k)
    elif isinstance(calibration_data, dict):
        calibration_coeffs = calibration_data
    else:
        raise TypeError(
            '`calibration_data` must be either list (raw shadow outcomes)'
            ' or dict (coefficients computes from NoiseEstFermi).'
        )

    noiseless_shadow_coeff = [binom(n_modes, j) / binom(2 * n_modes, 2 * j) for j in range(k + 1)]

    robust_rdm = {}
    for term, val in majorana_rdm.items():
        j = len(term) // 2
        f = noiseless_shadow_coeff[j]
        
        noisy_shadow_coeff = calibration_coeffs[j]

        robust_rdm[term] = val * (f / noisy_shadow_coeff)
    
    return robust_rdm


if __name__ == '__main__':
    import sys

    fermion_shadows_dir = '../fermion_shadows'
    if fermion_shadows_dir not in sys.path:
        sys.path.append(fermion_shadows_dir)
    from fermion_shadows.simulation.free_fermion_simulator import fock_cov_matrix
    from fermion_shadows.simulation.sample_gaussian_state_shadows import shadow_sampling
    from fermion_shadows_prediction import (
        diagonal_majoranas,
        majorana_ops_measured_by_permutation,
        invert_permutation,
        diagonal_majorana_matrix_element,
        permutation_parity
    )

    pauli_shadows_dir = '../pauli_shadows'
    if pauli_shadows_dir not in sys.path:
        sys.path.append(pauli_shadows_dir)
    from pauli_shadows.simulation.simple_noise_models import (
        bit_flip,
        depolarize_local,
        depolarize_global,
        amplitude_damp
    )


    def calibration_samples(n_modes: int,
                            n_samples: int,
                            noise_model,
                            p: float
                            ) -> list:
        """
        Function for testing the calibration process. Generates `n_samples` of
        bit strings from matchgate-shadow sampling |0^n>, affected by `noise_model`.
        """

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


    n_qubits = 8
    num_samples = int(1e4)
    noise_model = 'dl'
    noise_parameter = 0.1
    outcomes = calibration_samples(n_qubits, num_samples, noise_model, noise_parameter)
    calib_coeff = NoiseEstFermi(outcomes, k=2)

    noiseless_coeff = {j : binom(n_qubits, j) / binom(2 * n_qubits, 2 * j) for j in range(1, 2 + 1)}

    print(calib_coeff)
    print(noiseless_coeff)
    # They should be different, as noise corrupts the calib_coeff from the noiseless values

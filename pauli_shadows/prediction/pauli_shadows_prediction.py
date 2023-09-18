import numpy as np

import sys
pauli_shadows_dir = '../pauli_shadows'
if pauli_shadows_dir not in sys.path:
    sys.path.append(pauli_shadows_dir)
from pauli_shadows.qubit_rdm_tools import QubitRDM


def pauli_shadow_estimates(outcomes,
                           k: int = 2,
                           subsystems=None
                           ) -> QubitRDM:
    """
    k-local Pauli observable estimation from random Pauli measurements.
    Average estimates by the standard mean.

    Example:
    pauli_measurement_basis = ["+X", "-Z", "-Y", "+Y"]
    bit_string = [1, 0, 1, 1]
    outcomes[0] = (pauli_measurement_basis, bit_string)
    """

    num_samples = len(outcomes)
    n_qubits = len(outcomes[0][1])
    estimated_rdm = QubitRDM(n_qubits, k, subsystems=subsystems)

    for pauli_measurement_basis, bit_string in outcomes:
        for subsystem in estimated_rdm.subsystems:
            sign = 1
            estimated_pauli_string = []
            for i in subsystem:
                sign *= int(pauli_measurement_basis[i][0] + '1')
                estimated_pauli_string.append((i, pauli_measurement_basis[i][1]))
            
            sign *= bit_string_subsystem_parity_sign(bit_string, subsystem)
            estimated_pauli_string = tuple(estimated_pauli_string)

            estimated_rdm.add_pauli_expectation(estimated_pauli_string, sign)

    for term in estimated_rdm._pauli_expectations:
        w = len(term)
        estimated_rdm._pauli_expectations[term] *= 3**w / num_samples

    return estimated_rdm


def bit_string_subsystem_parity_sign(bit_string,
                                     subsystem
                                     ) -> int:
    sign = 1
    for i in subsystem:
        if bit_string[i] == 1:
            sign = -sign

    return sign


def pauli_shadow_median_of_means_estimates(outcomes,
                                           num_batches: int,
                                           k: int = 2
                                           ) -> QubitRDM:
    n_qubits = len(outcomes[0][1])

    num_samples = len(outcomes)
    if num_samples % num_batches != 0:
        raise ValueError(
            '`num_samples` ({}) must be evenly divisible by `num_batches` ({})'.format(num_samples, num_batches))
    num_samples_per_batch = num_samples // num_batches

    rdm_batches = []
    for j in range(num_batches):
        estimated_rdm_batch_j = pauli_shadow_estimates(
            outcomes[num_samples_per_batch * j:num_samples_per_batch * (j + 1)], k=k)
        rdm_batches.append(estimated_rdm_batch_j)

    median_of_means_rdm = QubitRDM(n_qubits, k)
    for pauli_observable in median_of_means_rdm.get_all_pauli_expectations().keys():
        observable_batches = []
        for j in range(num_batches):
            mean_j = rdm_batches[j].get_pauli_expectation(pauli_observable)
            observable_batches.append(mean_j)

        median_of_means_observable = np.median(observable_batches)
        median_of_means_rdm.add_pauli_expectation(pauli_observable, median_of_means_observable, overwrite=True)

    return median_of_means_rdm

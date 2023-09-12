import numpy as np
from itertools import combinations, product
from typing import Sequence, Union
from scipy.special import binom


def majorana_shadow_estimates_full_system(outcomes,
                                          k: int = 2) -> dict:
    """
    Estimates the expectation values of all Majorana operators up to degree 2k from classical shadows `outcomes`.
    """

    # Initialize
    n_modes = len(outcomes[0][1])
    expectations = initialize_majorana_ops_dict(n_modes, k)
    diagonal_ops = diagonal_majoranas(n_modes, k)

    # For each sample, compute and add the shadow estimate
    for permutation, bit_string in outcomes:

        # Get the Majorana operators which are diagonal in the basis of the Majorana swap unitary defined by permutation
        estimated_ops = majorana_ops_measured_by_permutation(
            permutation, diagonal_ops=diagonal_ops
        )
        inverse_permutation = invert_permutation(permutation)

        for diag_op, target_op in estimated_ops:

            # Compute the estimate of the arbitrary Majorana operator estimated_op by evaluating the matrix element
            # (with sign, as determined by the permutation on Majorana indices) of the associated diagonal operator
            inverse_permutation_on_predicted_op = [inverse_permutation[i] for i in target_op]
            sign = (-1)**permutation_parity(inverse_permutation_on_predicted_op)

            val = sign * diagonal_majorana_matrix_element(diag_op, bit_string)

            expectations[target_op] += val

    # Compute the shadow coefficients for the estimates
    shadow_coefficient = {}
    for j in range(1, k + 1):
        f = binom(2 * n_modes, 2 * j) / binom(n_modes, j)
        shadow_coefficient[j] = f

    # Complete the estimates by multiplying by the appropriate shadow coefficients and then dividing by the total number
    # of samples to obtain a standard mean estimator (recall that we can rigorously show that median-of-means estimation
    # is not required for this protocol).
    num_samples = len(outcomes)
    for op in expectations:
        j = len(op) // 2
        expectations[op] *= shadow_coefficient[j] / num_samples

    return expectations


def majorana_shadow_estimates_spin_adapted(outcomes,
                                           up_modes: Union[int, Sequence[int]],
                                           down_modes: Union[int, Sequence[int]],
                                           k: int = 2) -> dict:
    """
    Estimates the expectation values of all Majorana operators up to degree 2k from classical shadows `outcomes`,
    where there is always an even number of indices in each of the up and down spin supports.
    """

    # Initialize
    n_modes = len(outcomes[0][1])

    if isinstance(up_modes, int):
        n_up_modes = up_modes
        up_modes = list(range(up_modes))
    else:
        n_up_modes = len(up_modes)

    if isinstance(down_modes, int):
        n_down_modes = down_modes
        down_modes = list(range(n_up_modes, down_modes))
    else:
        n_down_modes = len(down_modes)

    expectations = initialize_majorana_ops_dict(n_modes, k)
    diagonal_ops = diagonal_majoranas(n_modes, k)

    # For each sample, compute and add the shadow estimate
    for permutation, bit_string in outcomes:

        # Get the Majorana operators which are diagonal in the basis of the Majorana swap unitary defined by permutation
        estimated_ops = majorana_ops_measured_by_permutation(
            permutation, diagonal_ops=diagonal_ops
        )
        inverse_permutation = invert_permutation(permutation)

        for diag_op, target_op in estimated_ops:

            # Compute the estimate of the arbitrary Majorana operator estimated_op by evaluating the matrix element
            # (with sign, as determined by the permutation on Majorana indices) of the associated diagonal operator
            inverse_permutation_on_predicted_op = [inverse_permutation[i] for i in target_op]
            sign = (-1)**permutation_parity(inverse_permutation_on_predicted_op)

            val = sign * diagonal_majorana_matrix_element(diag_op, bit_string)

            expectations[target_op] += val

    # Compute the shadow coefficients for the estimates. Note that the spin-adapted estimators are different from the
    # full-system estimators.
    shadow_coefficient = {}
    for i, j in product(range(k + 1), repeat=2):
        if (i, j) == (0, 0):
            continue
        f_i = binom(2 * n_up_modes, 2 * i) / binom(n_up_modes, i)
        f_j = binom(2 * n_down_modes, 2 * j) / binom(n_down_modes, j)
        shadow_coefficient[(i, j)] = f_i * f_j

    # Complete the estimates by multiplying by the appropriate shadow coefficients and then dividing by the total number
    # of samples to obtain a standard mean estimator (recall that we can rigorously show that median-of-means estimation
    # is not required for this protocol).
    # If `outcomes` was not produced by the spin-adapted shadows protocol, then this will produce nonsense estimates!
    up_majorana_modes = [2 * p + i for p in up_modes for i in range(2)]
    down_majorana_modes = [2 * p + i for p in down_modes for i in range(2)]
    num_samples = len(outcomes)
    for op in expectations:
        up_indices_in_op = [p for p in op if p in up_majorana_modes]
        down_indices_in_op = [p for p in op if p in down_majorana_modes]
        i = len(up_indices_in_op) // 2
        j = len(down_indices_in_op) // 2
        try:
            expectations[op] *= shadow_coefficient[(i, j)] / num_samples
        except KeyError:
            pass

    return expectations


def compute_expectation_values_from_majorana_observables(
        majorana_expectations: dict,
        observables: Union[dict, Sequence[dict]]) -> Union[float, np.ndarray]:
    if isinstance(observables, dict):
        observables = [observables]

    observable_expectations = []
    for obs in observables:
        expectation_value = 0.0
        for term, val in obs.items():
            try:
                expectation_value += val * majorana_expectations[term]
            except KeyError:
                if term == ():
                    expectation_value += val
                    
        observable_expectations.append(expectation_value)

    if len(observable_expectations) == 1:
        observable_expectations = observable_expectations[0]
    else:
        observable_expectations = np.array(observable_expectations)

    return observable_expectations


def initialize_majorana_ops_dict(n_modes: int, k: int) -> dict:

    majorana_ops = {}
    for j in range(1, k + 1):
        for mu in combinations(range(2 * n_modes), 2 * j):
            majorana_ops[mu] = 0

    return majorana_ops


def diagonal_majoranas(n_modes: int, k: int) -> list:

    list_of_diagonal_majoranas = []
    for j in range(1, k + 1):
        for P in combinations(range(n_modes), j):
            diag_index = tuple([2 * p + i for p in P for i in range(2)])
            list_of_diagonal_majoranas.append(diag_index)

    return list_of_diagonal_majoranas


def diagonal_majorana_matrix_element(diagonal_majorana, bit_string) -> int:
    """
    Computes <b|\Gamma|b> where b is a bit string and \Gamma is a diagonal Majorana operator. Assumes that
    `diagonal_majorana` indeed corresponds to indices of diagonal Majorana operators (in the standard basis) and does
    not verify that this actually holds.
    """

    m = 1
    for i in diagonal_majorana[::2]:
        if bit_string[i // 2] == 1:
            m = -m

    return m


def majorana_ops_measured_by_permutation(permutation, diagonal_ops) -> list:
    """
    Determines which Majorana operators are diagonal in the basis of the permutation (Gaussian Majorana swap circuit).
    Does this by working backwards: we enumerate all diagonal Majorana operators, then perform the inverse of the
    permutation to figure out which (in general nondiagonal) operators they came from.
    """

    measured_ops = []

    for diag_op in diagonal_ops:
        permuted_op = tuple(sorted([permutation[i] for i in diag_op]))
        measured_ops.append((diag_op, permuted_op))

    return measured_ops


def permute_majorana(majorana_indices, permutation):
    """
    Permutes indices according to the permutation Q. Rather than returning the resulting indices as-is, it returns them
    sorted along with the sign of the permutation which was required to do so (note that this is NOT the sign of Q).
    Mathematically, for example with a 2-Majorana operator, we have

    U \gamma_i \gamma_j U^* = \gamma_{\pi^{-1}(i)} \gamma_{\pi^{-1}(j)} = sign * \gamma_{i'} \gamma_{j'}

    where U is the Majorana swap circuit generated by the permutation matrix Q whose action by permutation \pi is

    Q e_i = e_{\pi(i)},

    and i' < j' is always such that sign = -1 if \pi^{-1}(i) > \pi^{-1}(j).

    Parameters
    ----------
    indices : iterable of int
        Input (Majorana operator) indices.
    Q : iterable of int
        Length 2n permutation.

    Returns
    -------
    tuple
        The permuted indices, sorted.
    sign : int
        The sign required to sort the permuted indices.

    """

    inverse_permutation = invert_permutation(permutation)

    l = [inverse_permutation[i] for i in majorana_indices]
    sign = (-1)**permutation_parity(l)
    l.sort()

    return tuple(l), sign


def permutation_parity(input_list) -> int:
    """
    Determines the parity of the permutation required to sort the list.
    Outputs 0 (even) or 1 (odd).
    """

    parity = 0
    for i, j in combinations(range(len(input_list)), 2):
        if input_list[i] > input_list[j]:
            parity += 1

    return parity % 2


def invert_permutation(permutation):
    """
    Given input permutation Q, returns Q^{-1}.

    Parameters
    ----------
    permutation : iterable
        A permutation as an iterable.

    Returns
    -------
    tuple
        The inverse permutation.

    """

    return tuple(np.arange(len(permutation))[np.argsort(permutation)])


if __name__ == '__main__':

    from scipy.stats import ortho_group
    import cirq
    from optimal_gaussian_circuit import  optimal_gaussian_circuit
    from fermion_rdm_tools import exact_majorana_rdm
    from fermion_shadows_cirq_simulation import sample_shadows_full_system, sample_shadows_spin_adapted

    n_qubits = 4
    k = 2

    up_qubits = list(range(n_qubits // 2))
    down_qubits = list(range(n_qubits // 2, n_qubits))
    qubits = cirq.GridQubit.rect(rows=2, cols=n_qubits // 2)

    # state_Q = ortho_group.rvs(2 * n_qubits)
    state_Q_up = ortho_group.rvs(n_qubits)
    state_Q_down = ortho_group.rvs(n_qubits)
    state_Q = np.block([[state_Q_up, np.zeros(state_Q_up.shape)],
                        [np.zeros(state_Q_down.shape), state_Q_down]])
    state_circuit = cirq.Circuit(optimal_gaussian_circuit(qubits, state_Q))
    state = state_circuit.final_state_vector(qubit_order=qubits)
    exact_rdm = exact_majorana_rdm(qubits, state_vector=state)

    repetitions_exponent = 3
    repetitions = int(10**repetitions_exponent)
    # outcomes = sample_shadows_full_system(qubits, state_circuit, repetitions=repetitions)
    outcomes = sample_shadows_spin_adapted(qubits[:n_qubits // 2], qubits[n_qubits // 2:],
                                           state_circuit, repetitions=repetitions)

    exact_rdm_vals = np.array(list(exact_rdm.values())).real

    n_samples = np.logspace(repetitions_exponent - 1, repetitions_exponent, num=9, dtype=int)
    rdm_estimates = []
    errors = []
    for T in n_samples:
        # sampled_rdm = majorana_shadow_estimates_full_system(outcomes[:T], k)
        sampled_rdm = majorana_shadow_estimates_spin_adapted(outcomes[:T], up_qubits, down_qubits, k)
        sampled_rdm_vals = np.array(list(sampled_rdm.values()))
        rdm_estimates.append(sampled_rdm_vals)
        errors.append(np.linalg.norm(exact_rdm_vals - sampled_rdm_vals))
    print('Majorana expectations RMSE:', errors)
    fit = np.polyfit(np.log(n_samples), np.log(errors), 1)
    print(fit[0])

"""
Given a fermionic Gaussian state, sample matchgate shadows from it.
"""
import numpy as np
from scipy.special import binom
from itertools import combinations, product
from typing import Sequence, Tuple, List, Dict, Union

from free_fermion_simulator import sample_gaussian_state


def shadow_sampling(cov_matrix: np.ndarray,
                    repetitions: int = 1
                    ) -> List[List[List[int]]]:

    N = cov_matrix.shape[0]

    outcomes = []
    for _ in range(repetitions):
        permutation = np.random.permutation(N)
        permutation = [i.item() for i in permutation] # cast np.int64 to native Python int
        permuted_cov_mat = cov_matrix[permutation][:, permutation] # P.T @ M @ P where P = np.eye(N)[:, permutation]
        # permuted_cov_mat = cov_matrix[np.ix_(permutation, permutation)] # faster at around N ~ 1e6

        bit_string = sample_gaussian_state(permuted_cov_mat)

        outcomes.append([permutation, bit_string])

    return outcomes


def rdm_shadow_estimates(outcomes: Sequence[Sequence[Sequence[int]]], k: int = 2) -> Dict:
    """
    Estimates the expectation values of all Majorana operators up to degree 2k from classical shadows `outcomes`.
    """

    # Initialize
    n_modes = len(outcomes[0][1])
    expectations = rdm_majorana_ops(n_modes, k)
    diagonal_ops = diagonal_majoranas(n_modes, k)

    # For each sample, compute and add the shadow estimate
    for permutation, bit_string in outcomes:

        # Get the Majorana operators which are diagonal in the basis of the Majorana swap unitary defined by permutation
        estimated_ops = measured_majorana_ops(permutation, diagonal_ops=diagonal_ops)

        for estimated_op, diag_op, sign in estimated_ops:

            # Compute the estimate of the arbitrary Majorana operator estimated_op by evaluating the matrix element
            # (with sign, as determined by the permutation on Majorana indices) of the associated diagonal operator
            val = sign * diagonal_majorana_matrix_element(diag_op, bit_string)
            expectations[estimated_op] += val

    # Compute the shadow coefficients for the estimates
    shadow_coefficient = []
    for j in range(1, k + 1):
        f = binom(2 * n_modes, 2 * j) / binom(n_modes, j)
        shadow_coefficient.append(f)

    # Complete the estimates by multiplying by the appropriate shadow coefficients and then dividing by the total number
    # of samples to obtain a standard mean estimator (recall that we can rigorously show that median-of-means estimation
    # is not required for this protocol).
    num_samples = len(outcomes)
    for op in expectations:
        k = len(op) // 2
        expectations[op] *= shadow_coefficient[k - 1] / num_samples

    return expectations


def rdm_shadow_estimates_spin_adapted(outcomes: Sequence[Sequence[Sequence[int]]],
                                      up_modes: Union[int, Sequence[int]],
                                      down_modes: Union[int, Sequence[int]],
                                      k: int = 2) -> Dict:
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
        down_modes = list(range(n_up_modes, n_up_modes + n_down_modes))
    else:
        n_down_modes = len(down_modes)

    assert n_modes == n_up_modes + n_down_modes
    expectations = rdm_majorana_ops(n_modes, k)
    diagonal_ops = diagonal_majoranas(n_modes, k)

    # For each sample, compute and add the shadow estimate
    for permutation, bit_string in outcomes:

        # Get the Majorana operators which are diagonal in the basis of the Majorana swap unitary defined by permutation
        estimated_ops = measured_majorana_ops(
            permutation, diagonal_ops=diagonal_ops
        )

        for estimated_op, diag_op, sign in estimated_ops:

            # Compute the estimate of the arbitrary Majorana operator estimated_op by evaluating the matrix element
            # (with sign, as determined by the permutation on Majorana indices) of the associated diagonal operator
            val = sign * diagonal_majorana_matrix_element(diag_op, bit_string)
            expectations[estimated_op] += val

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


def rdm_majorana_ops(n_modes: int, k: int) -> Dict:

    majorana_ops = {}
    for j in range(1, k + 1):
        for mu in combinations(range(2 * n_modes), 2 * j):
            majorana_ops[mu] = 0

    return majorana_ops


def diagonal_majoranas(n_modes: int, k: int) -> List[List[int]]:

    list_of_diagonal_majoranas = []
    for j in range(1, k + 1):
        for P in combinations(range(n_modes), j):
            diag_index = [2 * p + i for p in P for i in range(2)]
            list_of_diagonal_majoranas.append(diag_index)

    return list_of_diagonal_majoranas


def diagonal_majorana_matrix_element(diagonal_majorana: Sequence[int],
                                     bit_string: Sequence[int]
                                     ) -> int:
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


def measured_majorana_ops(permutation: Sequence[int],
                          k: Union[int, None] = None,
                          diagonal_ops: Union[Sequence[Sequence[int]], None] = None) -> List:
    """
    Determines which Majorana operators are diagonal in the basis of the permutation (Gaussian Majorana swap circuit).
    Does this by working backwards: we enumerate all diagonal Majorana operators, then perform the inverse of the
    permutation to figure out which (nondiagonal) operators they came from.
    """

    n_modes = len(permutation) // 2

    # Normally we need to invert the permutation. However our shadow sampler code actually implements permutation**(-1)
    # by virtue of how permuting numpy arrays works with slicing. Therefore, `permutation` is already the inverse of the
    # simulated operation.
    # permutation_inverse = invert_permutation(permutation)
    permutation_inverse = permutation

    measured_ops = []

    # If for some reason you want to keep re-evaluating the diagonal Majoranas.
    if diagonal_ops is None:
        diagonal_ops = diagonal_majoranas(n_modes, k)

    for op in diagonal_ops:
        permuted_op, sign = permute_majorana(op, permutation_inverse)
        measured_ops.append((permuted_op, tuple(op), sign))

    return measured_ops


def permute_majorana(majorana_indices: Sequence[int], permutation: Sequence[int]) -> Tuple[Tuple[int], int]:
    """
    Permutes indices according to the permutation Q. Rather than returning the resulting indices as-is, it returns them
    sorted along with the sign of the permutation which was required to do so (note that this is NOT the sign of Q).
    Mathematically, for example with a 2-Majorana operator, we have

    U \gamma_i \gamma_j U^* = \gamma_{\pi(i)} \gamma_{\pi(j)} = sign * \gamma_{i'} \gamma_{j'}

    where U is the Majorana swap circuit permuting by \pi, and i' < j' always such that sign = -1 if \pi(i) > \pi(j).

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

    l = [permutation[i] for i in majorana_indices]
    sign = (-1)**permutation_parity(l)
    l.sort()

    return tuple(l), sign


def permutation_parity(input_list: Sequence[int]) -> int:
    """
    Determines the parity of the permutation required to sort the list.
    Outputs 0 (even) or 1 (odd).
    """

    parity = 0
    for i, j in combinations(range(len(input_list)), 2):
        if input_list[i] > input_list[j]:
            parity += 1

    return parity % 2


if __name__ == '__main__':
    from time import time
    from scipy.stats import unitary_group, ortho_group
    from free_fermion_simulator import (one_body_majorana_expectations,
                                        two_body_majorana_expectations,
                                        embed_passive_into_active_transformation,
                                        gaussian_cov_matrix)


    n = 8
    N = 2 * n
    eta = 2

    u = unitary_group.rvs(n)
    Q = embed_passive_into_active_transformation(u)
    M = gaussian_cov_matrix(Q, eta)
    exact_rdm = two_body_majorana_expectations(M)

    M = gaussian_cov_matrix(Q, eta)

    n_reps = int(1e5)
    ti = time()
    outcomes = shadow_sampling(M, n_reps)
    tf = time()
    dt = tf - ti
    print('Time elapsed to sample:', dt)

    estimated_rdm = rdm_shadow_estimates(outcomes, k=2)

    rmse = 0.0
    for op in exact_rdm:
        rmse += np.abs(exact_rdm[op] - estimated_rdm[op])**2
    rmse /= len(exact_rdm)
    rmse **= 0.5
    print('RMSE (Majorana expectations):', rmse)

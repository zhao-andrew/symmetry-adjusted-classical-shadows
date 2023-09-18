"""
Efficient classical simulation of free fermions using
their O(2n) (fermionic Gaussian) representation.
"""
import numpy as np
from scipy.linalg import schur
from itertools import combinations


def fock_cov_matrix(n_modes, occ_modes):
    """
    Fermionic Gaussian covariance matrix of a Fock
    basis state.
    """

    if isinstance(occ_modes, (int, np.integer)):
        occ_modes_list = list(range(occ_modes))
    else:
        occ_modes_list = occ_modes

    M = np.zeros((2 * n_modes, 2 * n_modes))
    for p in range(n_modes):
        M[2 * p, 2 * p + 1] = -1.0 if p in occ_modes_list else 1.0
    M -= M.T

    return M


def gaussian_cov_matrix(ortho_matrix, init_occ_modes):
    """
    Covariance matrix of a Gaussian state U_Q |init>,
    where Q is a 2n x 2n orthogonal matrix and the
    initial state is a Fock state with occupied modes
    given by `init_occ_modes`
    """
    n_modes = ortho_matrix.shape[0] // 2
    M = ortho_matrix.T @ fock_cov_matrix(n_modes, init_occ_modes) @ ortho_matrix

    return M


def one_body_majorana_expectations(cov_matrix):
    """
    Expectation values of one-body Majorana operators

    \Gamma_{(p, q)} = -i \gamma_p \gamma_q,

    where p < q, from a covariance matrix.
    """
    N = cov_matrix.shape[0]

    majorana_expectations = {}
    for p, q in combinations(range(N), 2):
        majorana_expectations[(p, q)] = cov_matrix[p, q]

    return majorana_expectations


def two_body_majorana_expectations(cov_matrix):
    """
    Expectation values of two-body Majorana operators

    \Gamma_S = -\gamma_{S[0]} \gamma_{S[1]} \gamma_{S[2]} \gamma_{S[3]}

    from a covariance matrix.
    """
    N = cov_matrix.shape[0]

    majorana_expectations = one_body_majorana_expectations(cov_matrix)

    for S in combinations(range(N), 4):
        submatrix = cov_matrix[np.ix_(S, S)]
        wick_expectation = pfaffian(submatrix)

        majorana_expectations[S] = wick_expectation

    return majorana_expectations


def pfaffian(A):
    """
    Computes the Pfaffian of a real antisymmetric 2n x 2n matrix A.
    Does not check these assumptions.
    """
    N = A.shape[0]

    # noinspection PyTupleAssignmentBalance
    T, Z = schur(A, output='real')
    det_Z = np.linalg.det(Z)
    pf_T = 1.0
    for i in range(N // 2):
        pf_T *= T[2 * i, 2 * i + 1]

    return det_Z * pf_T


def slater_opdm(unitary, occ_modes):
    """
    One-particle density matrix

    opdm[p, q] = <p^ q>

    of a Slater determinant described by the first `occ_modes`
    columns of an input n x n unitary matrix.
    """
    if isinstance(occ_modes, (int, np.integer)):
        occ_modes_list = list(range(occ_modes))
    else:
        occ_modes_list = occ_modes

    config_matrix = unitary[:, occ_modes_list]
    opdm = (config_matrix @ config_matrix.conj().T).T

    return opdm


def slater_tpdm(unitary, occ_modes):
    """
    Two-particle density matrix

    tpdm[(p, q), (r, s)] = <p^ q^ s r>

    of a Slater determinant described by the first `occ_modes`
    columns of an input n x n unitary matrix, flattened along
    the pairs of indices (p, q) and (r, s)
    """
    n_orbitals = unitary.shape[0]

    opdm = slater_opdm(unitary, occ_modes)

    L = n_orbitals * (n_orbitals - 1)
    L //= 2
    tpdm = np.zeros((L, L), dtype=unitary.dtype)
    i = 0
    for p, q in combinations(range(n_orbitals), 2):

        j = 0
        for r, s in combinations(range(n_orbitals), 2):
            if i <= j:
                D_pqsr = opdm[p, r] * opdm[q, s] - opdm[q, r] * opdm[p, s]
                tpdm[i, j] = D_pqsr
                tpdm[j, i] = D_pqsr.conjugate()

            j += 1

        i += 1

    return tpdm


def sample_gaussian_state(cov_matrix):
    """
    Sample a bit string from measuring `cov_matrix` in the
    Fock basis. Algorithm described in arXiv:1112.2184.
    """
    n_modes = cov_matrix.shape[0] // 2
    M = cov_matrix.copy()

    outcome = []
    for j in range(n_modes):
        prob_j_1 = 0.5 * (1.0 - M[2 * j, 2 * j + 1])
        r = np.random.random()
        if r < prob_j_1:
            n_j = 1
            prob_n_j = prob_j_1
        else:
            n_j = 0
            prob_n_j = 1.0 - prob_j_1
        outcome.append(n_j)

        m_2j = M[2 * j, :]
        m_2j1 = M[2 * j + 1, :]
        M_update = np.outer(m_2j1, m_2j)
        M_update -= M_update.T
        M_update *= (-1)**n_j / (2 * prob_n_j)

        M += M_update

    return outcome


# Some misc helper functions
def bit_list_to_int(b):

    x = 0
    i = 0
    for b_i in list(reversed(b)):
        if b_i == 1:
            x += 2**i
        i += 1

    return x


def embed_passive_into_active_transformation(unitary_matrix):

    n = unitary_matrix.shape[0]

    ortho_matrix = np.zeros((2 * n, 2 * n))

    for p in range(n):
        real_part = unitary_matrix[p, p].real
        imag_part = unitary_matrix[p, p].imag
        ortho_matrix[2 * p:2 * p + 2, 2 * p:2 * p + 2] = np.array(
            [[real_part, imag_part],
             [-imag_part, real_part]]
        )

    for p, q in combinations(range(n), 2):
        real_part = unitary_matrix[q, p].real
        imag_part = unitary_matrix[q, p].imag
        ortho_matrix[2 * p:2 * p + 2, 2 * q:2 * q + 2] = np.array(
            [[real_part, imag_part],
             [-imag_part, real_part]]
        )

        real_part = unitary_matrix[p, q].real
        imag_part = unitary_matrix[p, q].imag
        ortho_matrix[2 * q:2 * q + 2, 2 * p:2 * p + 2] = np.array(
            [[real_part, imag_part],
             [-imag_part, real_part]]
        )

    return ortho_matrix


def embed_passive_into_active_observable(coefficient_matrix: np.ndarray) -> dict:

    n = coefficient_matrix.shape[0]

    majorana_observable = {() : 0.5 * np.trace(coefficient_matrix)}
    for p in range(n):
        term = (2 * p, 2 * p + 1)
        majorana_observable[term] = -0.5 * coefficient_matrix[p, p]

    for p, q in combinations(range(n), 2):
        terms = [(2 * p, 2 * q),
                 (2 * p + 1, 2 * q + 1),
                 (2 * p, 2 * q + 1),
                 (2 * p + 1, 2 * q)]
        coeffs = [-coefficient_matrix[p, q].imag,
                  -coefficient_matrix[p, q].imag,
                  -coefficient_matrix[p, q].real,
                  coefficient_matrix[p, q].real]

        for term, coeff in zip(terms, coeffs):
            majorana_observable[term] = 0.5 * coeff

    return majorana_observable


if __name__ == '__main__':
    from scipy.stats import ortho_group, unitary_group

    import sys
    fermion_shadows_dir = '../fermion_shadows'
    if fermion_shadows_dir not in sys.path:
        sys.path.append(fermion_shadows_dir)
    from fermion_shadows.prediction.fermion_shadows_prediction import (
        compute_expectation_values_from_majorana_observables)


    """Test the transformation of number-conserving observables written
    in the ladder operators into a linear combination of Majorana operators"""
    n = 7
    eta = n // 2

    u = unitary_group.rvs(n)
    d = np.random.random(size=n) - 0.5
    d = np.diag(d)
    h = u @ d @ u.conj().T
    U = embed_passive_into_active_transformation(u)
    M = gaussian_cov_matrix(U, eta)
    majorana_expectations = one_body_majorana_expectations(M)

    hamiltonian = embed_passive_into_active_observable(h)

    energy = compute_expectation_values_from_majorana_observables(majorana_expectations, hamiltonian)
    energy_check = np.sum(d[:eta])

    print('Energy from Majorana representation:', energy)
    print('Energy from ladder representation:', energy_check)

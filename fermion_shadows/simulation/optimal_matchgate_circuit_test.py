from optimal_matchgate_circuit import optimal_gaussian_circuit, embed_unitary_into_orthogonal_matrix

import numpy as np
import cirq
import openfermion as of
from scipy.stats import ortho_group, unitary_group


def test_random_adjoint_action(n_qubits: int):
    N = 2 * n_qubits

    orthogonal_matrix = ortho_group.rvs(N)

    qubits = cirq.LineQubit.range(n_qubits)
    U = cirq.Circuit(optimal_gaussian_circuit(qubits, orthogonal_matrix)).unitary()

    majorana_op_list = [
        of.get_sparse_operator(
            of.jordan_wigner(
                of.MajoranaOperator((i, ))),
            n_qubits=n_qubits)
        for i in range(N)
    ]
    total_error = 0.0
    for i in range(N):
        majorana_op_rotated_by_circuit = U.conj().T @ majorana_op_list[i] @ U

        majorana_op_rotated_exact = 0.0
        for j in range(N):
            majorana_op_rotated_exact += orthogonal_matrix[i, j] * majorana_op_list[j]

        error = np.linalg.norm(majorana_op_rotated_exact - majorana_op_rotated_by_circuit)
        total_error += error**2
    total_error **= 0.5

    if np.isclose(total_error, 0.0):
        print('General Gaussian test passed.')
    else:
        print('Test failed, error = {}'.format(total_error))

    return

def test_random_adjoint_action_number_conserving(n_qubits: int):
    unitary_matrix = unitary_group.rvs(n_qubits)
    orthogonal_matrix = embed_unitary_into_orthogonal_matrix(unitary_matrix)

    qubits = cirq.LineQubit.range(n_qubits)
    U = cirq.Circuit(optimal_gaussian_circuit(qubits, orthogonal_matrix)).unitary()
    U_openfermion = cirq.Circuit(of.optimal_givens_decomposition(qubits, unitary_matrix.copy())).unitary()

    ladder_op_list = [
        of.get_sparse_operator(
            of.jordan_wigner(
                of.FermionOperator('{}^'.format(i))),
            n_qubits=n_qubits)
        for i in range(n_qubits)
    ]
    total_error = 0.0
    for i in range(n_qubits):
        ladder_op_rotated_by_circuit = U.conj().T @ ladder_op_list[i] @ U

        ladder_op_rotated_exact = 0.0
        for j in range(n_qubits):
            ladder_op_rotated_exact += unitary_matrix[i, j].conj() * ladder_op_list[j]

        error = np.linalg.norm(ladder_op_rotated_exact - ladder_op_rotated_by_circuit)
        total_error += error**2
    total_error **= 0.5

    close_to_I = U @ U_openfermion.conj().T
    infidelity_with_openfermion_circuit = 1.0 - np.abs(np.trace(close_to_I) / 2**n_qubits)**2

    if np.isclose(total_error, 0.0) and np.isclose(infidelity_with_openfermion_circuit, 0.0):
        print('Number-conserving test passed.')
    else:
        print('Test failed, total error = {} and infidelity with OpenFermion circuit = {}'.format(
            total_error, infidelity_with_openfermion_circuit))

    return


if __name__ == '__main__':

    for n_qubits in [8, 9]:
        print('Testing random {}-qubit circuits...'.format(n_qubits))
        test_random_adjoint_action(n_qubits)
        test_random_adjoint_action_number_conserving(n_qubits)

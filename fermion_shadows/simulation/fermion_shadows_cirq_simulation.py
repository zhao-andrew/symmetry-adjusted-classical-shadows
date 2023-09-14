import numpy as np
from itertools import combinations

import cirq
import cirq_google
import qsimcirq

from optimal_matchgate_circuit import optimal_gaussian_circuit, optimal_gaussian_circuit_spin_block_diag


def sample_shadows_full_system(qubits,
                               circuit,
                               simulator=None,
                               repetitions=1):

    qubit_ordering = get_cirq_qubit_order(qubits)

    n_qubits = len(qubits)
    N = 2 * n_qubits

    if simulator is None:
        simulator = qsimcirq.QSimSimulator()
    
    qvm_simulator = isinstance(simulator, cirq_google.ProcessorSampler)
    if qvm_simulator:
        circuit = optimize_circuit_for_SqrtIswapTargetGateset(circuit, align=True)

    measurement_circuit = cirq.Circuit(cirq.measure(qubit) for qubit in qubits)

    outcomes = []
    for _ in range(repetitions):
        permutation = np.random.permutation(N)
        permutation = [i.item() for i in permutation]
        Q = np.eye(N)
        Q = Q[:, permutation]

        shadow_circuit = cirq.Circuit(optimal_gaussian_circuit(qubits, Q))
        if qvm_simulator:
            shadow_circuit = optimize_circuit_for_SqrtIswapTargetGateset(shadow_circuit, align=True)
            full_circuit = merge_circuits_phxz_layers(circuit, shadow_circuit)
        else:
            shadow_circuit = cirq.align_left(shadow_circuit)
            full_circuit = circuit + shadow_circuit
        full_circuit += measurement_circuit

        results = simulator.run(full_circuit, repetitions=1)
        bit_string = results.data[qubit_ordering].values
        bit_string = [i.item() for i in bit_string[0]]

        outcomes.append([permutation, bit_string])

    return outcomes


def sample_shadows_spin_adapted(up_qubits,
                                down_qubits,
                                circuit,
                                simulator=None,
                                repetitions=1):

    qubits = up_qubits + down_qubits
    qubit_ordering = get_cirq_qubit_order(qubits)

    n_up_sites = len(up_qubits)
    N_up = 2 * n_up_sites
    n_down_sites = len(down_qubits)
    N_down = 2 * n_down_sites

    if simulator is None:
        simulator = qsimcirq.QSimSimulator()
    qvm_simulator = isinstance(simulator, cirq_google.ProcessorSampler)
    if qvm_simulator:
        circuit = optimize_circuit_for_SqrtIswapTargetGateset(circuit, align=True)

    measurement_circuit = cirq.Circuit(cirq.measure(qubit) for qubit in qubits)

    outcomes = []
    for _ in range(repetitions):
        up_permutation = np.random.permutation(N_up)
        up_permutation = [i.item() for i in up_permutation]
        Q_up = np.eye(N_up)
        Q_up = Q_up[:, up_permutation]

        down_permutation = np.random.permutation(N_down)
        down_permutation = [i.item() for i in down_permutation]
        Q_down = np.eye(N_down)
        Q_down = Q_down[:, down_permutation]

        Q_up_det = (-1)**permutation_parity(up_permutation)
        shadow_circuit = cirq.Circuit(
            optimal_gaussian_circuit_spin_block_diag(up_qubits, down_qubits,
                                                     Q_up, Q_down,
                                                     up_matrix_det=Q_up_det)
        )

        if qvm_simulator:
            shadow_circuit = optimize_circuit_for_SqrtIswapTargetGateset(shadow_circuit, align=True)
            full_circuit = merge_circuits_phxz_layers(circuit, shadow_circuit)
        else:
            shadow_circuit = cirq.align_left(shadow_circuit)
            full_circuit = circuit + shadow_circuit
        full_circuit += measurement_circuit

        results = simulator.run(full_circuit, repetitions=1)
        bit_string = results.data[qubit_ordering].values
        bit_string = [i.item() for i in bit_string[0]]

        down_permutation_shifted = [i + N_up for i in down_permutation]
        permutation = up_permutation + down_permutation_shifted

        outcomes.append([permutation, bit_string])

    return outcomes


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


def optimize_circuit_for_SqrtIswapTargetGateset(circuit, align=True):

    optimized_circuit = cirq.optimize_for_target_gateset(circuit, gateset=cirq.SqrtIswapTargetGateset())
    optimized_circuit = cirq.merge_single_qubit_gates_to_phxz(optimized_circuit)

    if align:
        optimized_circuit = cirq.align_left(optimized_circuit)

    return optimized_circuit


def merge_circuits_phxz_layers(circuit1, circuit2, align=True):

    final_layer_of_circuit1 = cirq.Circuit(circuit1[-1])
    first_layer_of_circuit2 = cirq.Circuit(circuit2[0])
    merged_layer = cirq.merge_single_qubit_gates_to_phxz(final_layer_of_circuit1 + first_layer_of_circuit2)
    try:
        circuit = circuit1[:-1] + merged_layer + circuit2[1:]
    except IndexError:
        circuit = circuit1[:-1] + merged_layer
    
    if align:
        circuit = cirq.align_left(circuit)

    return circuit


def get_cirq_qubit_order(qubits):

    ordering = []
    for qubit in qubits:
        if isinstance(qubit, cirq.GridQubit):
            row, col = qubit.row, qubit.col
            label = 'q({}, {})'.format(row, col)
        elif isinstance(qubit, cirq.LineQubit):
            x = qubit.x
            label = 'q({})'.format(x)
        else:
            raise TypeError('Qubits must be either LineQubit or GridQubit.')

        ordering.append(label)

    return ordering

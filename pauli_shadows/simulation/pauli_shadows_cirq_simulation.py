import numpy as np
from itertools import product
from copy import deepcopy

import cirq
import cirq_google
import qsimcirq


# Construct the 24-element single-qubit Clifford group
I = np.eye(2)
X = np.array([[0.0, 1.0],
              [1.0, 0.0]])
Y = np.array([[0.0, -1.j],
              [1.j, 0.0]])
Z = np.array([[1.0, 0.0],
              [0.0, -1.0]])

H = np.array([[1, 1],
              [1, -1]]) / 2.0**0.5
S = np.array([[1, 0],
              [0, 1j]])
G = H @ S.conj()

V = H @ S @ H @ S
W = V.conj().T

S3 = [I, V, W, H, H @ V, H @ W]
P1 = [I, X, Y, Z]

def identify_pauli_matrix(matrix: np.ndarray) -> str:
    
    if np.isclose(matrix, X).all():
        return "+X"
    elif np.isclose(matrix, -X).all():
        return "-X"
    elif np.isclose(matrix, Y).all():
        return "+Y"
    elif np.isclose(matrix, -Y).all():
        return "-Y"
    elif np.isclose(matrix, Z).all():
        return "+Z"
    elif np.isclose(matrix, -Z).all():
        return "-Z"
    else:
        raise ValueError('Matrix is not any signed Pauli matrix.')

CliffordGroup1 = {}
i = 0
for A, B in product(S3, P1):
    U = A @ B
    rotated_pauli_Z = U.conj().T @ Z @ U
    pauli_label = identify_pauli_matrix(rotated_pauli_Z)
    CliffordGroup1[i] = (U, pauli_label)
    i += 1


# Define Cirq gates for each Clifford gate
CliffordGroup1_Cirq = {}
for i in range(24):
    class CliffordGate(cirq.Gate):

        def __init__(self):
            super(CliffordGate, self)
            self.unitary = CliffordGroup1[i][0]
            self.circuit_diagram_info = "Cl_{}".format(i)

        def _num_qubits_(self):
            return 1

        def _unitary_(self):
            return self.unitary

        def _circuit_diagram_info_(self, args):
            return self.circuit_diagram_info
    
    CliffordGroup1_Cirq[i] = deepcopy(CliffordGate())


# Shadow sampling functions
def pauli_shadow_sampling(qubits, circuit, simulator=None, repetitions=1):

    n_qubits = len(qubits)
    qubit_ordering = get_cirq_qubit_order(qubits)

    if simulator is None:
        simulator = qsimcirq.QSimSimulator()
    
    qvm_simulator = isinstance(simulator, cirq_google.ProcessorSampler)
    if qvm_simulator:
        circuit = optimize_circuit_for_SqrtIswapTargetGateset(circuit, align=True)
    
    measurement_circuit = cirq.Circuit(cirq.measure(qubit) for qubit in qubits)

    random_local_clifford_labels = np.random.randint(24, size=(repetitions, n_qubits))
    
    outcomes = []
    for j in range(repetitions):
        clifford_labels = random_local_clifford_labels[j]
        pauli_basis = []
        for i in range(n_qubits):
            pauli_basis.append(CliffordGroup1[clifford_labels[i]][1])
        
        shadow_circuit = cirq.Circuit(cirq_clifford_gates_from_labels(qubits, clifford_labels))

        if qvm_simulator:
            full_circuit = merge_circuits_phxz_layers(circuit, shadow_circuit)
        else:
            full_circuit = circuit + shadow_circuit
        full_circuit += measurement_circuit

        results = simulator.run(full_circuit, repetitions=1)
        bit_string = results.data[qubit_ordering].values
        bit_string = [i.item() for i in bit_string[0]]

        outcomes.append([pauli_basis, bit_string])

    return outcomes


def pauli_shadow_sampling_symmetrized(qubits, circuit, simulator=None, repetitions=1, sqrt_iSWAP=False):

    n_qubits = len(qubits)
    qubit_ordering = get_cirq_qubit_order(qubits)

    if simulator is None:
        simulator = qsimcirq.QSimSimulator()
    
    qvm_simulator = isinstance(simulator, cirq_google.ProcessorSampler)
    if qvm_simulator:
        circuit = optimize_circuit_for_SqrtIswapTargetGateset(circuit, align=True)
        sqrt_iSWAP = True
    
    measurement_circuit = cirq.Circuit(cirq.measure(qubit) for qubit in qubits)

    random_local_clifford_labels = np.random.randint(24, size=(repetitions, n_qubits))
    
    outcomes = []
    for j in range(repetitions):
        # Random local Clifford gates
        clifford_labels = random_local_clifford_labels[j]
        pauli_basis = []
        for i in range(n_qubits):
            pauli_basis.append(CliffordGroup1[clifford_labels[i]][1])
        shadow_circuit = cirq.Circuit(cirq_clifford_gates_from_labels(qubits, clifford_labels))
        
        # Random permutation circuit constructed out of adjacent SWAP gates
        # If using QVM, SWAP gates are replaced by two native iSWAP**0.5 gates, which has the same effect for
        # the purposes of the classical shadows protocol
        permutation = np.random.permutation(n_qubits)
        shadow_circuit += cirq.Circuit(swap_circuit_from_permutation(qubits, permutation, sqrt_iSWAP=sqrt_iSWAP))

        if qvm_simulator:
            full_circuit = merge_circuits_phxz_layers(circuit, shadow_circuit)
        else:
            full_circuit = circuit + shadow_circuit
        # print(len(full_circuit))
        # print(full_circuit.to_text_diagram(transpose=True))
        full_circuit += measurement_circuit

        results = simulator.run(full_circuit, repetitions=1)
        bit_string = results.data[qubit_ordering].values
#         bit_string = [i.item() for i in bit_string[0]]
        bit_string = [bit_string[0][i].item() for i in permutation]

#         outcomes.append([[permutation.tolist(), pauli_basis], bit_string])
        outcomes.append([pauli_basis, bit_string])

    return outcomes


def cirq_clifford_gates_from_labels(qubits, clifford_labels):
    
    n_qubits = len(qubits)
    
    for i in range(n_qubits):
        l = clifford_labels[i]
        yield CliffordGroup1_Cirq[l](qubits[i])
    
    return


def swap_circuit_from_permutation(qubits, permutation, sqrt_iSWAP=False):
    
    permutation_decomp = odd_even_sort(permutation)
    for transposition in permutation_decomp:
        i, j = transposition
        if sqrt_iSWAP:
            yield cirq.SQRT_ISWAP(qubits[i], qubits[j])
            yield cirq.SQRT_ISWAP(qubits[i], qubits[j])
        else:
            yield cirq.SWAP(qubits[i], qubits[j])
    
    return


def odd_even_sort(array):
    
    N = len(array)
    array_copy = array.copy()
    sorted_bool = False
    
    decomposition = []
    while not sorted_bool:
        sorted_bool = True
        
        for i in range(1, N - 1, 2):
            if array_copy[i] > array_copy[i + 1]:
                array_copy[i], array_copy[i + 1] = array_copy[i + 1], array_copy[i]
                
                decomposition.append((i, i + 1))
                sorted_bool = False
        
        for i in range(0, N - 1, 2):
            if array_copy[i] > array_copy[i + 1]:
                array_copy[i], array_copy[i + 1] = array_copy[i + 1], array_copy[i]
                
                decomposition.append((i, i + 1))
                sorted_bool = False
                
    return decomposition


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

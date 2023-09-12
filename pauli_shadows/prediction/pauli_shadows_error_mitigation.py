import numpy as np
from itertools import combinations

from qubit_rdm_tools import QubitRDM


def mitigate_pauli_expectations_via_magnetization(qubit_rdm: QubitRDM, magnetization) -> QubitRDM:
    n_qubits = qubit_rdm.n_qubits
    
    one_body_conserved_quantity = magnetization
    two_body_conserved_quantity = 0.5 * (magnetization**2 - n_qubits)
    
    if one_body_conserved_quantity == 0.0:
        one_body_rescaling_factor = 1.0
        print('One-body conserved quantity is zero, leaving one-body expectation values untouched.')
    else:
        noisy_one_body_quantity = 0.0
        for i in range(n_qubits):
            term = ((i, "Z"),)
            noisy_one_body_quantity += qubit_rdm.get_pauli_expectation(term)
        one_body_rescaling_factor = one_body_conserved_quantity / noisy_one_body_quantity

    if two_body_conserved_quantity == 0.0:
        two_body_rescaling_factor = 1.0
        print('Two-body conserved quantity is zero, leaving two-body expectation values untouched.')
    else:
        noisy_two_body_quantity = 0.0
        for i, j in combinations(range(n_qubits), 2):
            term = ((i, "Z"), (j, "Z"))
            noisy_two_body_quantity += qubit_rdm.get_pauli_expectation(term)
        two_body_rescaling_factor = two_body_conserved_quantity / noisy_two_body_quantity

    mitigated_rdm = QubitRDM(n_qubits, order=qubit_rdm.order, subsystems=qubit_rdm.subsystems)
    for term, val in qubit_rdm._pauli_expectations.items():
        if len(term) == 1:
            mitigated_val = val * one_body_rescaling_factor
            mitigated_rdm.add_pauli_expectation(term, mitigated_val, overwrite=True)
        elif len(term) == 2:
            mitigated_val = val * two_body_rescaling_factor
            mitigated_rdm.add_pauli_expectation(term, mitigated_val, overwrite=True)
        else:
            mitigated_rdm.add_pauli_expectation(term, val, overwrite=True)

    return mitigated_rdm

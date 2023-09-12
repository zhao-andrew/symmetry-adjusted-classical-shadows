# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 00:09:36 2021

@author: Andrew Zhao
"""

import numpy as np
from openfermion.ops import QubitOperator
from openfermion.linalg import get_sparse_operator
import cirq
import itertools

Pauli_Set = ['X', 'Y', 'Z']


def interpret_pauli_string(pauli_string: str):
    pauli_string_split = pauli_string.split()

    pauli_string_as_tuple = []
    for pauli_term in pauli_string_split:
        qubit, pauli = interpret_pauli_term(pauli_term)
        pauli_string_as_tuple.append((qubit, pauli))

    return tuple(pauli_string_as_tuple)


def interpret_pauli_term(pauli_term):
    if len(pauli_term) < 2:
        raise ValueError('Pauli term {} must have a qubit and Pauli label).')

    if isinstance(pauli_term, str):
        pauli_label = pauli_term[0]
        qubit_label = int(pauli_term[1:])
    elif hasattr(pauli_term, '__iter__'):
        pauli_label = pauli_term[1]
        qubit_label = int(pauli_term[0])
    else:
        raise TypeError('Pauli term {} must be either a string or an iterable.')

    return qubit_label, pauli_label


def pauli_string_subsystem_info(pauli_string):
    if isinstance(pauli_string, str):
        pauli_string = interpret_pauli_string(pauli_string)

    reduced_subsystem = []
    reduced_pauli_string = []
    for pauli_term in pauli_string:
        qubit, pauli = interpret_pauli_term(pauli_term)
        reduced_subsystem.append(qubit)
        reduced_pauli_string.append(pauli)

    return tuple(reduced_subsystem), tuple(reduced_pauli_string)


class QubitRDM(object):
    def __init__(self, n_qubits: int, order: int = 2, subsystems=None, expectations_dict: dict = None):
        self.n_qubits = n_qubits
        self.order = order

        self._pauli_expectations = {}
        if expectations_dict is not None:
            self.set_pauli_expectations_from_dict(expectations_dict)

        if subsystems is None:
            self.subsystems = []
            for k in range(1, order + 1):
                for subsystem in itertools.combinations(range(n_qubits), k):
                    self.subsystems.append(subsystem)
        else:
            self.subsystems = []
            for subsystem in subsystems:
                if len(subsystem) <= order:
                    self.subsystems.append(subsystem)
    
    def get_rdm(self, subsystem, return_qop: bool = False):
        k = len(subsystem)
        d = 2**k

        qop = QubitOperator((), 1.0)
        for j in range(1, k + 1):
            for subsubsystem_indices in itertools.combinations(range(k), j):
                subsubsystem_qubits = tuple([subsystem[i] for i in subsubsystem_indices])

                for pauli in itertools.product(Pauli_Set, repeat=j):
                    pauli_string = tuple(zip(subsubsystem_qubits, pauli))
                    val = self.get_pauli_expectation(pauli_string)

                    if not return_qop:
                        pauli_string = tuple(zip(subsubsystem_indices, pauli))
                    qop += QubitOperator(pauli_string, val)

        if not return_qop:
            qop = get_sparse_operator(qop, n_qubits=k).todense()

        return qop / d
    
    def get_all_k_rdms(self, k: int, return_qop: bool = False):
        if k > self.order:
            raise ValueError('{}-RDMs requested but order = {}.'.format(k, self.order))
        
        all_k_rdms = {}
        for subsystem in self.subsystems:
            if len(subsystem) == k:
                subsystem_rdm = self.get_rdm(subsystem, return_qop=return_qop)
                all_k_rdms[subsystem] = subsystem_rdm
        
        return all_k_rdms
    
    def get_pauli_expectation(self, pauli):
        try:
            value = self._pauli_expectations[pauli]
        except KeyError:
            value = 0.0
        
        return value
    
    def get_all_pauli_expectations(self):
        expectations = {}
        for subsystem in self.subsystems:
            for pauli in itertools.product(Pauli_Set, repeat=len(subsystem)):
                pauli_string = tuple(zip(subsystem, pauli))
                val = self.get_pauli_expectation(pauli_string)

                expectations[pauli_string] = val
        
        return expectations

    def get_subsystem_pauli_expectations(self, subsystem):
        expectations = {}
        k = len(subsystem)
        for pauli in itertools.product(Pauli_Set, repeat=k):
            pauli_string = tuple([(subsystem[i], pauli[i]) for i in range(k)])
            val = self.get_pauli_expectation(pauli_string)

            expectations[pauli_string] = val

        return expectations

    def compute_observable(self, observable):

        if isinstance(observable, QubitOperator):
            observable = observable.terms

        observable_expectation = 0.0
        for term, coeff in observable.items():
            rdm_element = self.get_pauli_expectation(term)
            observable_expectation += coeff * rdm_element

        return observable_expectation

    def add_pauli_expectation(self, pauli, value, overwrite: bool = False):
        if isinstance(pauli, str):
            pauli = interpret_pauli_string(pauli)

        if overwrite:
            self._pauli_expectations[pauli] = value
        else:
            try:
                self._pauli_expectations[pauli] += value
            except KeyError:
                self._pauli_expectations[pauli] = value

        return

    def set_pauli_expectations_from_dict(self, expectations_dict: dict, overwrite: bool = True):
        for pauli, value in expectations_dict.items():
            self.add_pauli_expectation(pauli, value, overwrite=overwrite)

        return
    
    def set_rdm(self, subsystem, rdm):
        """Does not actually check that the RDM is properly constructed."""
        
        w = len(subsystem)
        
        if w > self.order:
            raise ValueError(
                'Trying to set an RDM over subsystem {} that involves too many qubits for RDM of order {}.'.format(
                    subsystem, self.order)
            )



        return


def pauli_op_to_cirq_gate(pauli, qubits):
    """
    Converts a Pauli operator string (tuple in OpenFermion format) to a Cirq
    gate object.

    Parameters
    ----------
    pauli : tuple of tuples
        Pauli string in OpenFermion format, i.e., of the form
        ((qubit_index, pauli_matrix), ...).
    qubits : Sequence[cirq.Qid]
        Iterable of cirq.Qid objects. Size must be compatible with pauli.

    Returns
    -------
    cirq_pauli : cirq.Operation
        Pauli string as a Cirq operation.

    """
    cirq_pauli = cirq.I(qubits[0])
    
    for i, P in pauli:
        if P == 'X':
            cirq_pauli *= cirq.X(qubits[i])
        elif P == 'Y':
            cirq_pauli *= cirq.Y(qubits[i])
        elif P == 'Z':
            cirq_pauli *= cirq.Z(qubits[i])
    
    return cirq_pauli


def exact_qubit_rdm(qubits, order, circuit=None, state=None, observables=None, return_list=False):
    
    n_qubits = len(qubits)
    
    if observables is None:
        observables = []
        pauli_ops_cirq = []
        for j in range(1, order + 1):
            for ss in itertools.combinations(range(n_qubits), j):
                for P in itertools.product(Pauli_Set, repeat=j):
                    term = tuple(zip(ss, P))
                    
                    observables.append(term)
                    pauli_ops_cirq.append(pauli_op_to_cirq_gate(term, qubits))
    else:
        pauli_ops_cirq = [
            pauli_op_to_cirq_gate(term, qubits) for term in observables]
    
    if circuit is None:
        circuit = cirq.Circuit()
        for qubit in qubits:
            circuit.append(cirq.I(qubit))
    
    sim = cirq.Simulator()
    expectations = sim.simulate_expectation_values(circuit, initial_state=state, observables=pauli_ops_cirq)
    
    if return_list:
        return observables, expectations
    
    else:
        expectations_dict = {observables[i] : expectations[i] for i in range(len(observables))}
        rdm = QubitRDM(n_qubits, order, expectations_dict=expectations_dict)
        
        return rdm


if __name__ == '__main__':

    n_qubits = 8
    d = 2**n_qubits
    k = 2

    state = (np.random.random(d) - 0.5) * 1j
    state += np.random.random(d) - 0.5
    state /= np.linalg.norm(state)

    qubits = cirq.LineQubit.range(n_qubits)
    rdm = exact_qubit_rdm(qubits, k, state=state)
    observables, expectations = exact_qubit_rdm(qubits, k, state=state, return_list=True)
    # expectations = {observables[i] : expectations[i] for i in range(len(observables))}
    expectations = np.array(expectations)

    expectations_from_state_vector = []
    for pauli_term in observables:
        op = get_sparse_operator(QubitOperator(pauli_term), n_qubits=n_qubits)
        val = np.dot(state.conj(), op @ state)
        expectations_from_state_vector.append(val)
    expectations_from_state_vector = np.array(expectations_from_state_vector)

    print(np.linalg.norm(expectations_from_state_vector - expectations))
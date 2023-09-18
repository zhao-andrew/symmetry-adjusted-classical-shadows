"""
Helper tools for manipulating fermionic RDMs.

TODO: Define a fermionic RDM class to store expectations,
      compute arbitrary observables, convert between ladder
      and Majorana representation, etc.
"""
import numpy as np
from scipy.special import comb

import cirq

import itertools

from openfermion.ops import FermionOperator, MajoranaOperator
from openfermion.transforms import get_majorana_operator, jordan_wigner

def rdm_majorana_ops(n, k):
    
    majorana_ops = {}
    for j in range(1, k + 1):
        for mu in itertools.combinations(range(2 * n), 2 * j):
            majorana_ops[mu] = 0.0
    
    return majorana_ops

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

def exact_majorana_rdm(qubits, state_vector=None, state_circuit=None):
    """
    Computes all 2- and 4-degree Majorana operator expectation values with
    respect to the state prepared by state_circuit acting on state_vector. If
    state_vector is not specified, the circuit acts on the all-zeros state. If
    state_circuit is not specified, the circuit is just the identity operator.

    Parameters
    ----------
    qubits : TYPE
        DESCRIPTION.
    state_vector : TYPE, optional
        DESCRIPTION. The default is None.
    state_circuit : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    majorana_rdm : TYPE
        DESCRIPTION.

    """
    n_qubits = len(qubits)
    
    majorana_ops = rdm_majorana_ops(n_qubits, 2)
    
    cirq_majorana_ops = []
    coeffs = []
    for op in majorana_ops:
        pauli_op = jordan_wigner(MajoranaOperator(op, (-1j)**(len(op) // 2)))
        pauli_str = next(iter(pauli_op.terms))
        
        coeffs.append(pauli_op.terms[pauli_str])
        
        cirq_majorana_ops.append(pauli_op_to_cirq_gate(pauli_str, qubits))
    
    if state_circuit is None:
        circuit = cirq.Circuit()
        for qubit in qubits:
            circuit.append(cirq.I(qubit))
    else:
        circuit = state_circuit
    
    sim = cirq.Simulator()
    evs = sim.simulate_expectation_values(circuit, initial_state=state_vector,
                                          observables=cirq_majorana_ops)
    
    majorana_rdm = {}
    i = 0
    for op in majorana_ops:
        majorana_rdm[op] = evs[i] * coeffs[i]
        i += 1
    
    return majorana_rdm

def fermion_indices_to_term(P, R):
    
    fermion_term = []
    for p in P:
        fermion_term.append((p, 1))
    for r in reversed(R):
        fermion_term.append((r, 0))
    
    return tuple(fermion_term)

def majorana_to_fermion_rdm_array(majorana_rdm, n_orbitals, k):
    
    return fermion_rdm_dict_to_array(
        majorana_to_fermion_rdm(majorana_rdm, n_orbitals, k),
        n_orbitals, k)

def majorana_to_fermion_rdm(majorana_rdm, n_orbitals, k):
    
    fermion_rdm_dict = {}
    
    for P, R in itertools.product(itertools.combinations(range(n_orbitals), k),
                                  repeat=2):
        
        ferm_term = fermion_indices_to_term(P, R)
        ferm_op = FermionOperator(ferm_term, 1.0)
        
        maj_op = get_majorana_operator(ferm_op)
        
        rdm_element = 0
        for term, coeff in maj_op.terms.items():
            try:
                rdm_element += majorana_rdm[term] * coeff * 1j**(len(term) // 2)
            except KeyError:
                if term == ():
                    rdm_element += coeff
                else:
                    pass
        
        fermion_rdm_dict[(P, R)] = rdm_element
    
    return fermion_rdm_dict

def fermion_rdm_dict_to_array(fermion_rdm_dict, n_orbitals, k):
    
    dim = int(comb(n_orbitals, k))
    rdm = np.zeros((dim, dim), dtype='complex')
    
    i = 0
    for P in itertools.combinations(range(n_orbitals), k):
        
        j = 0
        for R in itertools.combinations(range(n_orbitals), k):
            
            if i <= j:
                val = fermion_rdm_dict[(P, R)]
                rdm[i, j] = val
                rdm[j, i] = val.conjugate()
            
            j += 1
        
        i += 1
    
    return rdm

def folded_index(p, q, n):
    """Returns the 2-RDM index i <- (p, q), where 0 <= p < q <= n - 1."""
    
    return p * (n - 1) - (p * (p + 1)) // 2 + q - 1

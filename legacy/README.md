# Fermionic partial tomography via classical shadows

> [!IMPORTANT]
> This is legacy code from December 2021 (and earlier), kept for archival purposes. It essentially only gives the operator coverings, with no code for the actual observable prediction step. The circuit compilation scheme for fermionic Gaussian unitaries is also outdated. The current codes should be used instead, which additionally include implementations of error mitigation for shadows, symmetry adaptation, and the improved Gaussian circuit design. The current library also has complementary codes for Pauli shadows, and usage examples with end-to-end numerical simulations.

Open source implementation of https://arxiv.org/abs/2010.16094, a technique to measure the fermionic k-body reduced density matrices (k-RDM) of an arbitrary quantum state. We use the framework of classical shadows, which measures in randomly selected bases defined by a distribution of unitaries. The distributions considered in this work are:

* Fermionic Gaussian Clifford unitaries (FGU)
* Number-conserving (NC) subgroup of FGU, followed by Pauli measurements

Generally speaking, shadow tomography via FGU requires less circuit repetitions than via NC (empirically, roughly half as many); however, NC circuits have half the depth of FGU. The best option is therefore highly dependent on context.

This implementation requires [NumPy](https://numpy.org/), [OpenFermion](https://quantumai.google/openfermion), and [Cirq](https://quantumai.google/cirq).

`construct_random_measurements_FGU`/`construct_random_measurements_NC` will return a dictionary containing the measurement bases one needs to perform in order to estimate all Majorana operators desired. Typically all 2k-degree Majorana operators are desired to reconstruct the k-RDM; an example of how to generate such operators is given at the bottom of `FGU_random_cover.py`/`NC_random_cover.py`.

The measurement bases are encoded as 2n-length permutations (numpy arrays) for FGU, and n-length permutations + Pauli string for NC. Here, n is the number of fermion modes being simulated. To convert the FGU permutations to circuits which one will implement in hardware, compilation routines are provided in `gaussian_circuit_givens_decomposition.py`. The NC circuits can be compiled using `optimal_givens_decomposition` from OpenFermion, and the following Pauli string is the Pauli measurement to implement using standard Hadamard and phase gates.

If you have any questions, please do not hesitate to reach out.

# Classical shadows for fermions (and qubits!)

Open-source codes for performing classical shadows, particularly for quantum simulations of fermions and qubits. Based primarily on the following papers:

* https://arxiv.org/abs/2002.08953 : Introduces the concept of classical shadows and demonstrates the application to local qubit (Pauli) measurements (random single-qubit Clifford gates)
* https://arxiv.org/abs/2010.16094 : Constructs classical shadows protocols for local fermion measurements (random fermionic Gaussian unitary (matchgate) & Clifford circuits)
* [Coming soon] : Develops an error-mitigation strategy using symmetries to introduce robustness to noise in the quantum computer. Also introduces a number of modifications and extensions of the above protocols, including an optimal circuit design for fermionic Gaussian unitaries

The base implementation in Python requires [OpenFermion](https://quantumai.google/openfermion) and [Cirq](https://quantumai.google/cirq). The numerical simulations additionally require [qsimcirq](https://github.com/quantumlib/qsim), [ReCirq](https://github.com/quantumlib/ReCirq), and the Julia package [ITensor](https://github.com/ITensor/ITensors.jl).

The files are organized as follows. For either `fermion_shadows` or `pauli_shadows`, codes in the `prediction` directory provide the routines to postprocess the quantum measurement data (description of random unitary and bit string) into numerical estimates of desired observables (using either a Pauli or Majorana operator decomposition). Included are also the robust estimators, using either calibration data obtained to perform [Robust Shadow Estimation](https://arxiv.org/abs/2011.09636) or symmetry information as described in [Coming soon]. Codes in the `simulation` directory provide examples for running the protocol (i.e., numerically simulating the quantum circuits to obtain samples, which are then postprocessed into observable predictions). These examples include the numerical experiments featured in [Coming soon].

Working examples/test codes are provided for each protocol. If you have any questions, please do not hesitate to reach out.

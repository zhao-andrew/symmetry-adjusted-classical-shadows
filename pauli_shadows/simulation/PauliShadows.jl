using LinearAlgebra
using Random
using Combinatorics
using ITensors


function sample_pauli_shadows(mps::MPS, n_samples)
    
    n_qubits = length(mps)
    sites = siteinds(mps)
    
    random_local_clifford_labels = rand(0:23, (n_samples, n_qubits))
    measurement_record = []
    for j = 1:n_samples
        clifford_labels = random_local_clifford_labels[j, :]
        measurement_gates = local_clifford_gates(clifford_labels, sites)
        
        rotated_mps = apply(measurement_gates, mps)
        outcome = sample!(rotated_mps) .- 1
        
        pauli_basis = pauli_basis_from_clifford_labels(clifford_labels)
        push!(measurement_record, [pauli_basis, outcome])
    end
    
    return measurement_record
end


function local_clifford_gates(clifford_gate_labels, sites) :: Vector{ITensor}
    n_qubits = length(clifford_gate_labels)
    
    local_clifford_gates = Vector{Tuple{String, Int}}()
    for i = 1:n_qubits
        l = clifford_gate_labels[i]
        push!(local_clifford_gates, ("Cl_$l", i))
    end
    
    clifford_itensor = ops(local_clifford_gates, sites)
    
    return clifford_itensor
end


function pauli_basis_from_clifford_labels(clifford_labels)
    pauli_basis = []
    for l in clifford_labels
        push!(pauli_basis, CliffordGroup1[l][2])
    end
    
    return pauli_basis
end


CliffordGroup1 = Dict(
    5  => (ComplexF64[0.5+0.5im 0.5+0.5im; -0.5+0.5im 0.5-0.5im], "+X"),
    16 => (ComplexF64[0.7071067811865476+0.0im 0.0+0.7071067811865476im; 0.0+0.7071067811865476im 0.7071067811865476+0.0im], "-Y"),
    20 => (ComplexF64[0.7071067811865476-0.7071067811865476im 0.0+0.0im; 0.0+0.0im 0.7071067811865476+0.7071067811865476im], "+Z"),
    12 => (ComplexF64[0.7071067811865476+0.0im 0.7071067811865476+0.0im; 0.7071067811865476+0.0im -0.7071067811865476+0.0im], "+X"),
    8  => (ComplexF64[0.5-0.5im 0.5+0.5im; 0.5-0.5im -0.5-0.5im], "-Y"),
    17 => (ComplexF64[0.0+0.7071067811865476im 0.7071067811865476+0.0im; 0.7071067811865476+0.0im 0.0+0.7071067811865476im], "+Y"),
    1  => (ComplexF64[0.0+0.0im 1.0+0.0im; 1.0+0.0im 0.0+0.0im], "-Z"),
    19 => (ComplexF64[0.7071067811865476+0.0im 0.0-0.7071067811865476im; 0.0+0.7071067811865476im -0.7071067811865476+0.0im], "+Y"),
    0  => (ComplexF64[1.0+0.0im 0.0+0.0im; 0.0+0.0im 1.0+0.0im], "+Z"),
    22 => (ComplexF64[0.0+0.0im -0.7071067811865476-0.7071067811865476im; -0.7071067811865476+0.7071067811865476im 0.0+0.0im], "-Z"),
    6  => (ComplexF64[-0.5+0.5im 0.5-0.5im; -0.5-0.5im -0.5-0.5im], "-X"),
    23 => (ComplexF64[0.7071067811865476-0.7071067811865476im 0.0+0.0im; 0.0+0.0im -0.7071067811865476-0.7071067811865476im], "+Z"),
    11 => (ComplexF64[0.5-0.5im -0.5-0.5im; 0.5-0.5im 0.5+0.5im], "+Y"),
    9  => (ComplexF64[0.5+0.5im 0.5-0.5im; -0.5-0.5im 0.5-0.5im], "+Y"),
    14 => (ComplexF64[0.0+0.7071067811865476im 0.0-0.7071067811865476im; 0.0-0.7071067811865476im -0.0-0.7071067811865476im], "-X"),
    3  => (ComplexF64[1.0+0.0im 0.0+0.0im; 0.0+0.0im -1.0+0.0im], "+Z"),
    7  => (ComplexF64[0.5+0.5im -0.5-0.5im; 0.5-0.5im 0.5-0.5im], "-X"),
    4  => (ComplexF64[0.5+0.5im 0.5+0.5im; 0.5-0.5im -0.5+0.5im], "+X"),
    13 => (ComplexF64[0.7071067811865476+0.0im 0.7071067811865476+0.0im; -0.7071067811865476+0.0im 0.7071067811865476+0.0im], "+X"),
    15 => (ComplexF64[0.7071067811865476+0.0im -0.7071067811865476+0.0im; 0.7071067811865476+0.0im 0.7071067811865476+0.0im], "-X"),
    2  => (ComplexF64[0.0+0.0im 0.0-1.0im; 0.0+1.0im 0.0+0.0im], "-Z"),
    10 => (ComplexF64[-0.5+0.5im -0.5-0.5im; 0.5-0.5im -0.5-0.5im], "-Y"),
    18 => (ComplexF64[-0.7071067811865476+0.0im 0.0-0.7071067811865476im; 0.0+0.7071067811865476im 0.7071067811865476+0.0im], "-Y"),
    21 => (ComplexF64[0.0+0.0im 0.7071067811865476-0.7071067811865476im; 0.7071067811865476+0.7071067811865476im 0.0+0.0im], "-Z")
)

ITensors.op(::OpName"Cl_0", ::SiteType"S=1/2") = CliffordGroup1[0][1]
ITensors.op(::OpName"Cl_1", ::SiteType"S=1/2") = CliffordGroup1[1][1]
ITensors.op(::OpName"Cl_2", ::SiteType"S=1/2") = CliffordGroup1[2][1]
ITensors.op(::OpName"Cl_3", ::SiteType"S=1/2") = CliffordGroup1[3][1]
ITensors.op(::OpName"Cl_4", ::SiteType"S=1/2") = CliffordGroup1[4][1]
ITensors.op(::OpName"Cl_5", ::SiteType"S=1/2") = CliffordGroup1[5][1]
ITensors.op(::OpName"Cl_6", ::SiteType"S=1/2") = CliffordGroup1[6][1]
ITensors.op(::OpName"Cl_7", ::SiteType"S=1/2") = CliffordGroup1[7][1]
ITensors.op(::OpName"Cl_8", ::SiteType"S=1/2") = CliffordGroup1[8][1]
ITensors.op(::OpName"Cl_9", ::SiteType"S=1/2") = CliffordGroup1[9][1]
ITensors.op(::OpName"Cl_10", ::SiteType"S=1/2") = CliffordGroup1[10][1]
ITensors.op(::OpName"Cl_11", ::SiteType"S=1/2") = CliffordGroup1[11][1]
ITensors.op(::OpName"Cl_12", ::SiteType"S=1/2") = CliffordGroup1[12][1]
ITensors.op(::OpName"Cl_13", ::SiteType"S=1/2") = CliffordGroup1[13][1]
ITensors.op(::OpName"Cl_14", ::SiteType"S=1/2") = CliffordGroup1[14][1]
ITensors.op(::OpName"Cl_15", ::SiteType"S=1/2") = CliffordGroup1[15][1]
ITensors.op(::OpName"Cl_16", ::SiteType"S=1/2") = CliffordGroup1[16][1]
ITensors.op(::OpName"Cl_17", ::SiteType"S=1/2") = CliffordGroup1[17][1]
ITensors.op(::OpName"Cl_18", ::SiteType"S=1/2") = CliffordGroup1[18][1]
ITensors.op(::OpName"Cl_19", ::SiteType"S=1/2") = CliffordGroup1[19][1]
ITensors.op(::OpName"Cl_20", ::SiteType"S=1/2") = CliffordGroup1[20][1]
ITensors.op(::OpName"Cl_21", ::SiteType"S=1/2") = CliffordGroup1[21][1]
ITensors.op(::OpName"Cl_22", ::SiteType"S=1/2") = CliffordGroup1[22][1]
ITensors.op(::OpName"Cl_23", ::SiteType"S=1/2") = CliffordGroup1[23][1]

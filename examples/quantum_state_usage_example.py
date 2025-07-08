import numpy as np

from aqft.quantum_state import (
    vacuum, 
    create, 
    destroy, 
    qeye,
    tensor,
    basis
)

def main():
    """An example demonstrating the use of the QuantumState class."""
    print("--- Quantum State Usage Example ---")

    # --- Part 1: Single System --- 
    print("\n--- Part 1: Single System Operations ---")
    hilbert_dim = 4
    print(f"Hilbert space dimension: {hilbert_dim}")

    # Create states and operators
    psi_vacuum = vacuum(hilbert_dim)
    a_dag = create(hilbert_dim)
    a = destroy(hilbert_dim)
    number_op = a_dag @ a

    print("\nNumber operator N = a_dag * a:")
    print(number_op)

    # Calculate expectation value using the new .expect() method
    exp_N_vacuum = psi_vacuum.expect(number_op)
    print(f"\nExpectation value <0|N|0>: {np.round(exp_N_vacuum)}")

    # Create a one-photon state |1>
    psi_one = basis(hilbert_dim, 1)
    exp_N_one = psi_one.expect(number_op)
    print(f"Expectation value <1|N|1>: {np.round(exp_N_one)}")

    # Verify orthogonality
    overlap = psi_vacuum.dag() @ psi_one
    print(f"Overlap <0|1>: {np.round(overlap.data[0, 0])}")

    # --- Part 2: Tensor Product and Multi-System Operations ---
    print("\n--- Part 2: Tensor Product and Multi-System Operations ---")
    dim_1, dim_2 = 2, 3
    print(f"System dimensions: dim_1={dim_1}, dim_2={dim_2}")

    # Create operators for each subsystem
    op1 = create(dim_1)
    op2 = destroy(dim_2)
    id1 = qeye(dim_1)
    id2 = qeye(dim_2)

    # Create composite operator O = op1 ⊗ op2
    O_tensor = tensor(op1, op2)
    print("\nComposite operator O = create(2) tensor destroy(3):")
    print(O_tensor)

    # Create a tensor product state |psi> = |1> ⊗ |2>
    psi_tensor = tensor(basis(dim_1, 1), basis(dim_2, 2))
    print("\nTensor product state |psi> = |1> tensor |2>:")
    print(psi_tensor)

    # Create number operators for each subsystem
    N1 = tensor(create(dim_1) @ destroy(dim_1), id2)
    N2 = tensor(id1, create(dim_2) @ destroy(dim_2))

    # Calculate expectation values in the tensor product state
    exp_N1 = psi_tensor.expect(N1)
    exp_N2 = psi_tensor.expect(N2)
    print(f"\nExpectation value <psi|N1|psi>: {np.round(exp_N1)}")
    print(f"Expectation value <psi|N2|psi>: {np.round(exp_N2)}")

    # --- Part 3: Trace Operation ---
    print("\n--- Part 3: Trace Operation ---")
    # The trace of the number operator is the sum of its eigenvalues (0, 1, 2, ...)
    trace_N = number_op.tr()
    expected_trace = sum(range(hilbert_dim))
    print(f"Trace of number operator (dim={hilbert_dim}): {np.round(trace_N)}")
    print(f"Expected trace: {expected_trace}")

    # --- Part 4: Partial Trace Operation ---
    print("\n--- Part 4: Partial Trace Operation ---")
    # Create a Bell state: (|00> + |11>) / sqrt(2)
    psi00 = tensor(basis(2, 0), basis(2, 0))
    psi11 = tensor(basis(2, 1), basis(2, 1))
    bell_state = (psi00 + psi11) / np.sqrt(2)
    print("Bell state |Phi+> = (|00> + |11>) / sqrt(2):")
    print(bell_state)

    # Calculate the partial trace over the first qubit (index 0)
    rho_A = bell_state.ptrace(0)
    print(f"Partial trace over first qubit performed.")
    print(f"Reduced density matrix:\n{rho_A.data.toarray()}")

    # --- 8. Normalization and Fidelity ---
    print("\n--- 8. Normalization and Fidelity ---")

    # Create a non-normalized state
    non_normalized_ket = basis(2, 0) + 2 * basis(2, 1)
    print(f"Non-normalized state created.")
    print(f"Is it normalized? {non_normalized_ket.is_normalized()}")

    # Normalize it
    non_normalized_ket.normalize()
    print("State has been normalized.")
    print(f"Is it normalized now? {non_normalized_ket.is_normalized()}")

    # Create another state to test fidelity
    other_ket = basis(2, 0)
    fidelity = non_normalized_ket.fidelity(other_ket)
    print(f"Fidelity with |0>: {fidelity:.4f}")

    # Test fidelity with itself
    self_fidelity = non_normalized_ket.fidelity(non_normalized_ket)
    print(f"Fidelity with self: {self_fidelity:.4f}")

    # The result should be a maximally mixed state: I/2
    expected_rho = qeye(2) / 2
    print("\nExpected reduced density matrix (I/2):")
    print(expected_rho)

    # Verify the result
    # np.allclose checks if two arrays are element-wise equal within a tolerance
    is_correct = np.allclose(rho_A.data.toarray(), expected_rho.data.toarray())
    print(f"\nIs the reduced density matrix correct? {is_correct}")

if __name__ == "__main__":
    main()

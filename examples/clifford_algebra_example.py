import numpy as np
from aqft_curved.clifford import Layout, Cl

def main():
    """An example demonstrating the use of the clifford library."""
    print("--- Clifford Algebra Example ---")

    # Define the layout for 3D Euclidean space (3 positive signatures)
    layout, blades = Cl(3, names='e')
    globals().update(blades)

    print("Defined a 3D Euclidean space with basis vectors e1, e2, e3.")

    # Create a vector
    v = 1*e1 + 2*e2 + 3*e3
    print(f"\nVector v: {v}")

    # Create a bivector (represents a plane)
    b = e1^e2
    print(f"Bivector b: {b}")

    # Geometric product
    # The geometric product of a vector with itself is a scalar
    v_squared = v * v
    print(f"Geometric product v*v: {v_squared}")

    # The geometric product of a vector and a bivector produces a new multivector
    vb = v * b
    print(f"Geometric product v*b: {vb}")

    # Grade projection
    # A multivector can have parts of different grades (scalar, vector, bivector, etc.)
    print(f"  Grade 1 part of v*b (vector): {vb(1)}")
    print(f"  Grade 3 part of v*b (trivector/pseudoscalar): {vb(3)}")

    # Inverse
    v_inv = ~v
    print(f"\nInverse of v: {v_inv}")
    print(f"v * ~v: {v * v_inv}")

if __name__ == "__main__":
    main()

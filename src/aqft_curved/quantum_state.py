import numpy as np
from scipy.sparse import spmatrix, issparse, diags, csc_matrix, kron

class QuantumState:
    """
    Represents a quantum object, such as a state vector (ket/bra) or an operator.

    This class is a pure Python, NumPy/SciPy-based implementation inspired by
    QuTiP's Qobj. It aims to provide basic, self-contained functionality for
    representing and manipulating quantum states and operators without requiring
    a complex build process.

    Parameters
    ----------
    inpt : array_like or sparse_matrix
        The data for the quantum object.
    dims : list of list of int, optional
        A list containing the dimensions of the underlying Hilbert spaces.
        For a simple ket, bra, or operator, this is `[[dim], [1]]`,
        `[[1], [dim]]`, or `[[dim], [dim]]` respectively. For tensor product
        spaces, the dimensions of each component are listed.
        If not provided, it will be inferred from the shape of `inpt`.
    """
    def __init__(self, inpt, dims=None):
        if not issparse(inpt):
            inpt = csc_matrix(inpt, dtype=np.complex128)

        self._data = inpt
        self.shape = inpt.shape

        if len(self.shape) != 2:
            raise ValueError("Input data must be a 2D array or matrix.")

        # Determine type from shape
        if self.shape[1] == 1:
            self._type = 'ket'
        elif self.shape[0] == 1:
            self._type = 'bra'
        elif self.shape[0] == self.shape[1]:
            self._type = 'oper'
        else:
            self._type = 'other'

        # Set dimensions
        if dims is None:
            if self._type == 'ket':
                self.dims = [[self.shape[0]], [1]]
            elif self._type == 'bra':
                self.dims = [[1], [self.shape[0]]]
            elif self._type == 'oper':
                self.dims = [[self.shape[0]], [self.shape[1]]]
            else:
                self.dims = [[self.shape[0]], [self.shape[1]]] # Fallback
        else:
            # Basic validation for dims
            if not (isinstance(dims, list) and len(dims) == 2):
                raise ValueError("`dims` must be a list of two lists, e.g., [[rows], [cols]].")
            if np.prod(dims[0]) != self.shape[0] or np.prod(dims[1]) != self.shape[1]:
                raise ValueError("Product of `dims` must match the data shape.")
            self.dims = dims

    @property
    def data(self):
        """The underlying data matrix."""
        return self._data

    @property
    def type(self):
        """The type of the quantum object (ket, bra, oper)."""
        return self._type

    @property
    def isket(self):
        """Returns True if the object is a ket."""
        return self.type == 'ket'

    @property
    def isbra(self):
        """Returns True if the object is a bra."""
        return self.type == 'bra'

    @property
    def isoper(self):
        """Returns True if the object is an operator."""
        return self.type == 'oper'

    def dag(self):
        """
        Returns the adjoint (conjugate transpose) of the quantum object.
        """
        return QuantumState(self.data.conj().T, dims=[self.dims[1], self.dims[0]])

    def tr(self):
        """
        Returns the trace of an operator.
        """
        if not self.isoper:
            raise TypeError("Trace is only defined for operators.")
        return self.data.trace()

    def expect(self, op):
        """
        Calculates the expectation value of an operator for this state.

        Parameters
        ----------
        op : QuantumState
            The operator for which to calculate the expectation value.

        Returns
        -------
        complex
            The expectation value.
        """
        if self.isbra:
            return (self @ op @ self.dag()).data[0, 0]
        elif self.isket:
            return (self.dag() @ op @ self).data[0, 0]
        elif self.isoper:
            # Assuming self is a density matrix
            return (self @ op).tr()
        else:
            raise TypeError(f"Expectation value not defined for type '{self.type}'.")

    def ptrace(self, sel):
        """
        Calculates the partial trace of a quantum state.

        If the object is a ket, it is converted to a density matrix first.

        Parameters
        ----------
        sel : int or list of int
            The index/indices of the subsystem(s) to trace out.

        Returns
        -------
        QuantumState
            The reduced density matrix of the remaining subsystem(s).
        """
        if self.isoper:
            op = self
        elif self.isket:
            op = self @ self.dag()
        else:
            raise TypeError("Partial trace is only defined for operators or kets.")

        if isinstance(sel, int):
            sel = [sel]

        if not all(isinstance(s, int) for s in sel):
            raise TypeError("`sel` must be an integer or a list of integers.")

        dims = list(op.dims[0])
        if op.dims[0] != op.dims[1]:
            raise ValueError("Partial trace requires square dimensions.")

        rho_data = op.data.toarray() if issparse(op.data) else op.data.copy()
        
        for i in sorted(sel, reverse=True):
            d_i = dims.pop(i)
            d_before = int(np.prod(dims[:i]))
            d_after = int(np.prod(dims[i:]))

            rho_tensor = rho_data.reshape((d_before, d_i, d_after, d_before, d_i, d_after))
            rho_data = np.trace(rho_tensor, axis1=1, axis2=4)
            
            new_dim = d_before * d_after
            rho_data = rho_data.reshape((new_dim, new_dim))

        new_data = csc_matrix(rho_data)
        new_dims = [dims, dims]
        return QuantumState(new_data, dims=new_dims)

    def is_normalized(self):
        """Checks if the state is normalized."""
        if self.type not in ('ket', 'bra'):
            raise TypeError("Normalization is only defined for kets and bras.")

        if self.type == 'ket':
            # For a ket |psi>, norm^2 is <psi|psi> = psi.H * psi
            norm_sq = (self.data.conj().T @ self.data).toarray()[0, 0]
        else:  # bra
            # For a bra <psi|, norm^2 is <psi|psi> = psi * psi.H
            norm_sq = (self.data @ self.data.conj().T).toarray()[0, 0]

        return np.isclose(np.abs(norm_sq), 1.0)

    def normalize(self):
        """Normalizes the quantum state (ket or bra) in-place and returns self."""
        if self.type not in ('ket', 'bra'):
            raise TypeError("Normalization is only defined for kets and bras.")

        if self.type == 'ket':
            norm_sq = (self.data.conj().T @ self.data).toarray()[0, 0]
        else:  # bra
            norm_sq = (self.data @ self.data.conj().T).toarray()[0, 0]

        norm = np.sqrt(np.abs(norm_sq))

        if norm == 0:
            raise ValueError("Cannot normalize a zero vector.")

        self._data /= norm
        return self

    def fidelity(self, other):
        """
        Calculates the fidelity between two quantum states.

        For two pure states |psi> and |phi>, the fidelity is |<psi|phi>|^2.
        """
        if not isinstance(other, QuantumState):
            raise TypeError("Can only calculate fidelity with another QuantumState.")

        if self.type == 'ket' and other.type == 'ket':
            # <psi|phi>
            if self.dims != other.dims:
                raise ValueError("Both states must have the same dimensions.")
            overlap = (self.data.conj().T @ other.data).toarray()[0, 0]
            return np.abs(overlap)**2
        else:
            # TODO: Add support for density matrices
            raise NotImplementedError("Fidelity is currently only implemented for two ket states.")

    def _check_dims(self, other):
        if self.dims != other.dims:
            raise ValueError(f"Incompatible dimensions for operation: {self.dims} and {other.dims}")

    def __add__(self, other):
        if isinstance(other, QuantumState):
            self._check_dims(other)
            return QuantumState(self.data + other.data, dims=self.dims)
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, QuantumState):
            self._check_dims(other)
            return QuantumState(self.data - other.data, dims=self.dims)
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, QuantumState):
            self._check_dims(other)
            return QuantumState(other.data - self.data, dims=self.dims)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float, complex)):
            return QuantumState(self.data * other, dims=self.dims)
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float, complex)):
            return QuantumState(self.data / other, dims=self.dims)
        return NotImplemented

    def __matmul__(self, other):
        if isinstance(other, QuantumState):
            if self.shape[1] != other.shape[0]:
                raise ValueError(f"Incompatible shapes for matrix multiplication: {self.shape} and {other.shape}")
            
            new_dims = [self.dims[0], other.dims[1]]
            return QuantumState(self.data @ other.data, dims=new_dims)
        return NotImplemented

    def __repr__(self):
        return (f"QuantumState(shape={self.shape}, "
                f"dims={self.dims}, type='{self.type}')")

    def __str__(self):
        s = f"QuantumState: dims = {self.dims}, shape = {self.shape}, type = {self.type}\n"
        s += "Qobj data:\n" + str(self.data)
        return s

# --- Factory functions for creating states and operators ---

def qeye(dim):
    """
    Creates an identity operator.

    Parameters
    ----------
    dim : int
        The dimension of the Hilbert space.

    Returns
    -------
    QuantumState
        The identity operator.
    """
    return QuantumState(np.identity(dim, dtype=np.complex128), dims=[[dim], [dim]])

def destroy(dim):
    """
    Creates the annihilation operator for a harmonic oscillator.

    Parameters
    ----------
    dim : int
        The dimension of the Hilbert space (truncation).

    Returns
    -------
    QuantumState
        The annihilation operator.
    """
    if not isinstance(dim, int) or dim <= 0:
        raise ValueError("Dimension must be a positive integer.")
    data = diags(np.sqrt(np.arange(1, dim)), 1, shape=(dim, dim), dtype=np.complex128)
    return QuantumState(csc_matrix(data), dims=[[dim], [dim]])

def create(dim):
    """
    Creates the creation operator for a harmonic oscillator.

    Parameters
    ----------
    dim : int
        The dimension of the Hilbert space (truncation).

    Returns
    -------
    QuantumState
        The creation operator.
    """
    return destroy(dim).dag()

def fock(dim, n=0):
    """
    Creates a Fock state (number state) |n>.

    Parameters
    ----------
    dim : int
        The dimension of the Hilbert space.
    n : int, optional
        The number state index (0 <= n < dim). Defaults to 0.

    Returns
    -------
    QuantumState
        The Fock state ket.
    """
    if not isinstance(dim, int) or dim <= 0:
        raise ValueError("Dimension must be a positive integer.")
    if not isinstance(n, int) or not (0 <= n < dim):
        raise ValueError("The state `n` must be an integer where 0 <= n < dim.")
    
    data = np.zeros((dim, 1), dtype=np.complex128)
    data[n, 0] = 1.0
    return QuantumState(data, dims=[[dim], [1]])

def basis(dim, n=0):
    """
    Alias for `fock(dim, n)`. Creates a basis ket |n>.
    """
    return fock(dim, n)

def vacuum(dim):
    """
    Creates the vacuum state |0>.
    """
    return fock(dim, 0)

def tensor(*qobjs):
    """
    Computes the tensor product of multiple QuantumState objects.

    Parameters
    ----------
    *qobjs : QuantumState
        A variable number of QuantumState objects.

    Returns
    -------
    QuantumState
        The tensor product of the input objects.
    """
    if not all(isinstance(q, QuantumState) for q in qobjs):
        raise TypeError("All arguments must be QuantumState objects.")

    if len(qobjs) < 2:
        raise ValueError("tensor function requires at least two arguments.")

    # Start with the first Qobj
    current_qobj = qobjs[0]

    for next_qobj in qobjs[1:]:
        # Calculate tensor product of data using scipy.sparse.kron
        new_data = kron(current_qobj.data, next_qobj.data)
        
        # Combine dimensions
        new_dims = [current_qobj.dims[0] + next_qobj.dims[0], 
                    current_qobj.dims[1] + next_qobj.dims[1]]
        
        current_qobj = QuantumState(new_data, dims=new_dims)

    return current_qobj

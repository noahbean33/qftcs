import sympy
from .spacetime import Spacetime
from .field import FieldOperator, ScalarField

class State:
    """
    Represents a quantum state in AQFT, which is a functional that assigns
    a number (the expectation value) to every observable.
    """
    def __init__(self, field):
        """
        Initializes a State.

        Parameters:
            field (FieldOperator): The quantum field associated with the state.
        """
        if not isinstance(field, FieldOperator):
            raise TypeError("field must be an instance of FieldOperator.")
        self.field = field
        self.spacetime = field.spacetime

    def two_point_function(self, point1, point2):
        """
        Computes the two-point correlation function (Wightman function, <phi(x1)phi(x2)>).
        
        This is a highly non-trivial function that defines the state.
        (This is a placeholder for a future numerical or advanced symbolic implementation).
        """
        print(f"Placeholder for two-point function between {point1} and {point2}.")
        return None

    def n_point_function(self, *points):
        """
        Computes the n-point correlation function.
        For many states (like free fields), these can be derived from the two-point function.
        (This is a placeholder).
        """
        print(f"Placeholder for n-point function for {len(points)} points.")
        return None

    def expectation_value(self, operator):
        """
        Computes the expectation value of an operator in this state.
        For a vacuum state, the one-point function is zero.
        """
        # For a single field operator, the vacuum expectation value is 0.
        if isinstance(operator, FieldOperator):
            return 0
        # For products of operators, this requires the n-point functions.
        # This is a placeholder for a more complete implementation.
        else:
            print(f"Expectation value for operator products is not yet implemented.")
            return None


class VacuumState(State):
    """
    Represents a physically relevant vacuum state, characterized by specific
    properties of its correlation functions (e.g., the Hadamard condition).
    """
    def __init__(self, field, state_type='Hadamard'):
        """
        Initializes a VacuumState.

        Parameters:
            field (ScalarField): The quantum field.
            state_type (str): The type of vacuum state (e.g., 'Hadamard', 'Bunch-Davies').
                              This choice determines the form of the correlation functions.
        """
        if not isinstance(field, ScalarField):
            raise TypeError("VacuumState currently only supports ScalarField.")
        super().__init__(field)
        
        supported_states = ['Hadamard', 'Bunch-Davies']
        if state_type not in supported_states:
            raise ValueError(f"Unsupported vacuum state type: {state_type}. Supported types are {supported_states}.")
            
        self.state_type = state_type
        print(f"Initialized {self.state_type} vacuum state for field '{self.field.name}'.")

    def two_point_function(self, point1, point2):
        """
        Returns a symbolic representation of the two-point function G(x1, x2).
        """
        G = sympy.Function('G')
        return G(point1, point2)

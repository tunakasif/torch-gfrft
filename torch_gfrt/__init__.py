from enum import Enum, auto

__version__ = "0.1.0"


class EigvalSortStrategy(Enum):
    """Enum for the different strategies to sort
    eigenvalues and eigenvectors"""

    TOTAL_VARIATION = auto()
    ASCENDING = auto()


class ComplexSortStrategy(Enum):
    """Sorting strategy for complex numbers. For the `REAL` and `ABS` sorting strategies,
    sorting can be done in PyTorch. For the `REAL_IMAG`, `ABS_ANGLE` and `ABS_ANGLE_02pi`
    strategies, NumPy is used, so the input tensor is moved to numpy and back.
    Args:
        Enum (REAL): Sort by real part, ignoring the imaginary part.
        Enum (ABS): Sort by absolute value, ignoring the phase.
        Enum (REAL_IMAG): Sort by real part, then by imaginary part.
        Enum (ABS_ANGLE): Sort by absolute value, then by phase `(-pi, pi]`.
        Enum (ABS_ANGLE_02pi): Sort by absolute value, then by phase in the range `[0, 2pi)`.
    """

    REAL = auto()
    ABS = auto()
    REAL_IMAG = auto()
    ABS_ANGLE = auto()
    ABS_ANGLE_02pi = auto()

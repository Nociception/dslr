import numpy as np
import numpy.typing as npt


def logistic_function_vector(z: npt.NDArray) -> npt.NDArray:
    return 1 / (1 + np.exp(-z))


def logistic_function_scalar(z: float) -> float:
    return 1 / (1 + np.exp(-z))

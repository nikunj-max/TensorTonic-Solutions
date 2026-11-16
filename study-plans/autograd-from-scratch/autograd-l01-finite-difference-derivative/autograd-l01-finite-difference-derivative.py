import numpy as np


def finite_difference_derivative(
    coefficients: list, x: float, h: float
) -> tuple[float, float, float]:
    """
    Returns the value at x, the value at x plus h, and the estimated slope.
    """
    x_val = np.float64(x)
    h_val = np.float64(h)
    x_plus_h = x_val + h_val

    def evaluate_poly(pt: np.float64) -> np.float64:
        acc = np.float64(0.0)
        for coeff in reversed(coefficients):
            acc = acc * pt + np.float64(coeff)
        return acc

    fx = evaluate_poly(x_val)
    fxh = evaluate_poly(x_plus_h)
    slope = (fxh - fx) / h_val

    return (float(fx), float(fxh), float(slope))
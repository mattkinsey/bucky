"""Constrant decorators for the RHS funcs used in the ODE solvers."""
# flake8: noqa: RST301
from functools import wraps

from ..numerical_libs import sync_numerical_libs, xp


@sync_numerical_libs
def constrain_y_range(constraints):
    """
    Decorator wrapping a function to be passed to an ODE solver which constrains the solution space.

    Examples
    --------
    .. code-block:: python

        @constrain([0, 1])
        def f(t, y):
          dy_dt = # your ODE
          return dy_dt

        solver = scipy.integrate.odeint(f, y0)  # use any solver you like!
        solution = solver.solve()

    If solution goes below 0 or above 1, the function f will ignore values of dy_dt which would make it more extreme,
    and treat the previous solution as if it were at 0 or 1.

    Parameters
    ----------
    constraints: set
        Sequence of (low, high) constraints - use None for unconstrained.

    Returns
    -------

    Notes
    -----
    This constrains the dependent variable from going *any further* past the constraints.
    The ODE will still treat it as if it were at the value of the constraint,
    and with a small step size any problems should be minimal,
    but you may still have slightly out-of-range numbers in your solution.
    """
    if all(constraint is not None for constraint in constraints):
        assert constraints[0] < constraints[1]

    def wrap(f):
        """Wrap function (for functools)."""

        @wraps(f)
        def wrapper(t, y, *args, **kwargs):
            """Wrapper function (for functools)."""
            lower, upper = constraints
            if lower is None:
                lower = -xp.inf
            if upper is None:
                upper = xp.inf

            too_low = y <= lower
            too_high = y >= upper

            y = xp.clip(y, a_min=lower, a_max=upper, out=y)

            result = f(t, y, *args, **kwargs)

            result = xp.where(too_low & (result < 0.0), 0.0, result)
            result = xp.where(too_high & (result > 0.0), 0.0, result)

            return result

        return wrapper

    return wrap

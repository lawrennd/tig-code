import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv, det


# Pre-computed analytical derivatives of Lagrange multiplier using SymPy
# These are EXACT symbolic expressions compiled to Python for efficiency
# Generated from: ν = (-Gθ · a) / (a · a) where G=∇²ψ, a=∇C
#
# To regenerate if needed, use SymPy:
#   nu_sym = (F_unc.dot(a)) / (a.dot(a))
#   dnu = [diff(nu, var) for var in [theta_xx, theta_pp, theta_xp]]
#   _dnu_dtheta_xx_func = lambdify([theta_xx, theta_pp, theta_xp], dnu[0], 'numpy')
#   ... (similarly for other components)

def _dnu_dtheta_xx_func(theta_xx, theta_pp, theta_xp):
    """∂ν/∂θ_xx - exact analytical formula from SymPy (generated via pycode)."""
    return (-2*theta_pp*theta_xp*(2*theta_pp*theta_xp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + theta_xp*(-2*theta_xp**2/(theta_pp*theta_xx - theta_xp**2)**2 - 1/(theta_pp*theta_xx - theta_xp**2)))/(theta_pp*theta_xx - theta_xp**2)**2 + 2*theta_xp*(-4*theta_pp**2*theta_xp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**3 + 2*theta_pp*theta_xp/(theta_pp*theta_xx - theta_xp**2)**2 + theta_xp*(4*theta_pp*theta_xp**2/(theta_pp*theta_xx - theta_xp**2)**3 + theta_pp/(theta_pp*theta_xx - theta_xp**2)**2))/(theta_pp*theta_xx - theta_xp**2) + (-1/2*theta_pp/(theta_pp*theta_xx - theta_xp**2) + (1/2)*(theta_pp*theta_xx - theta_xp**2)*(-theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + 1/(theta_pp*theta_xx - theta_xp**2))/theta_xx)*(theta_pp**3*theta_xx/(theta_pp*theta_xx - theta_xp**2)**3 - 2*theta_pp**2*theta_xp**2/(theta_pp*theta_xx - theta_xp**2)**3 - 1/2*theta_pp**2/(theta_pp*theta_xx - theta_xp**2)**2 + theta_pp*(theta_pp**2*theta_xx/(theta_pp*theta_xx - theta_xp**2)**3 - theta_pp/(theta_pp*theta_xx - theta_xp**2)**2)) + (-1/2*theta_xx/(theta_pp*theta_xx - theta_xp**2) + (1/2)*(theta_pp*theta_xx - theta_xp**2)*(-theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + 1/(theta_pp*theta_xx - theta_xp**2))/theta_pp)*(theta_pp**2*theta_xx**2/(theta_pp*theta_xx - theta_xp**2)**3 - 2*theta_pp*theta_xp**2*theta_xx/(theta_pp*theta_xx - theta_xp**2)**3 - 3/2*theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + theta_xp**2/(theta_pp*theta_xx - theta_xp**2)**2 + theta_xx*(theta_pp**2*theta_xx/(theta_pp*theta_xx - theta_xp**2)**3 - theta_pp/(theta_pp*theta_xx - theta_xp**2)**2) + (1/2)/(theta_pp*theta_xx - theta_xp**2)) + (-1/2*theta_pp**2*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + theta_pp*theta_xp**2/(theta_pp*theta_xx - theta_xp**2)**2 + theta_pp*(-1/2*theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + (1/2)/(theta_pp*theta_xx - theta_xp**2)))*((1/2)*theta_pp**2/(theta_pp*theta_xx - theta_xp**2)**2 + (1/2)*theta_pp*(-theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + 1/(theta_pp*theta_xx - theta_xp**2))/theta_xx + (1/2)*(theta_pp*theta_xx - theta_xp**2)*(2*theta_pp**2*theta_xx/(theta_pp*theta_xx - theta_xp**2)**3 - 2*theta_pp/(theta_pp*theta_xx - theta_xp**2)**2)/theta_xx - 1/2*(theta_pp*theta_xx - theta_xp**2)*(-theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + 1/(theta_pp*theta_xx - theta_xp**2))/theta_xx**2) + (1/2)*(theta_pp*theta_xx - theta_xp**2)*(2*theta_pp**2*theta_xx/(theta_pp*theta_xx - theta_xp**2)**3 - 2*theta_pp/(theta_pp*theta_xx - theta_xp**2)**2)*(-1/2*theta_pp*theta_xx**2/(theta_pp*theta_xx - theta_xp**2)**2 + theta_xp**2*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + theta_xx*(-1/2*theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + (1/2)/(theta_pp*theta_xx - theta_xp**2)))/theta_pp)/(4*theta_xp**2/(theta_pp*theta_xx - theta_xp**2)**2 + (-1/2*theta_pp/(theta_pp*theta_xx - theta_xp**2) + (1/2)*(theta_pp*theta_xx - theta_xp**2)*(-theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + 1/(theta_pp*theta_xx - theta_xp**2))/theta_xx)**2 + (-1/2*theta_xx/(theta_pp*theta_xx - theta_xp**2) + (1/2)*(theta_pp*theta_xx - theta_xp**2)*(-theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + 1/(theta_pp*theta_xx - theta_xp**2))/theta_pp)**2) + (8*theta_pp*theta_xp**2/(theta_pp*theta_xx - theta_xp**2)**3 - (-1/2*theta_pp/(theta_pp*theta_xx - theta_xp**2) + (1/2)*(theta_pp*theta_xx - theta_xp**2)*(-theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + 1/(theta_pp*theta_xx - theta_xp**2))/theta_xx)*(theta_pp**2/(theta_pp*theta_xx - theta_xp**2)**2 + theta_pp*(-theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + 1/(theta_pp*theta_xx - theta_xp**2))/theta_xx + (theta_pp*theta_xx - theta_xp**2)*(2*theta_pp**2*theta_xx/(theta_pp*theta_xx - theta_xp**2)**3 - 2*theta_pp/(theta_pp*theta_xx - theta_xp**2)**2)/theta_xx - (theta_pp*theta_xx - theta_xp**2)*(-theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + 1/(theta_pp*theta_xx - theta_xp**2))/theta_xx**2) - (theta_pp*theta_xx - theta_xp**2)*(-1/2*theta_xx/(theta_pp*theta_xx - theta_xp**2) + (1/2)*(theta_pp*theta_xx - theta_xp**2)*(-theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + 1/(theta_pp*theta_xx - theta_xp**2))/theta_pp)*(2*theta_pp**2*theta_xx/(theta_pp*theta_xx - theta_xp**2)**3 - 2*theta_pp/(theta_pp*theta_xx - theta_xp**2)**2)/theta_pp)*(2*theta_xp*(2*theta_pp*theta_xp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + theta_xp*(-2*theta_xp**2/(theta_pp*theta_xx - theta_xp**2)**2 - 1/(theta_pp*theta_xx - theta_xp**2)))/(theta_pp*theta_xx - theta_xp**2) + (-1/2*theta_pp/(theta_pp*theta_xx - theta_xp**2) + (1/2)*(theta_pp*theta_xx - theta_xp**2)*(-theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + 1/(theta_pp*theta_xx - theta_xp**2))/theta_xx)*(-1/2*theta_pp**2*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + theta_pp*theta_xp**2/(theta_pp*theta_xx - theta_xp**2)**2 + theta_pp*(-1/2*theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + (1/2)/(theta_pp*theta_xx - theta_xp**2))) + (-1/2*theta_xx/(theta_pp*theta_xx - theta_xp**2) + (1/2)*(theta_pp*theta_xx - theta_xp**2)*(-theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + 1/(theta_pp*theta_xx - theta_xp**2))/theta_pp)*(-1/2*theta_pp*theta_xx**2/(theta_pp*theta_xx - theta_xp**2)**2 + theta_xp**2*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + theta_xx*(-1/2*theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + (1/2)/(theta_pp*theta_xx - theta_xp**2))))/(4*theta_xp**2/(theta_pp*theta_xx - theta_xp**2)**2 + (-1/2*theta_pp/(theta_pp*theta_xx - theta_xp**2) + (1/2)*(theta_pp*theta_xx - theta_xp**2)*(-theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + 1/(theta_pp*theta_xx - theta_xp**2))/theta_xx)**2 + (-1/2*theta_xx/(theta_pp*theta_xx - theta_xp**2) + (1/2)*(theta_pp*theta_xx - theta_xp**2)*(-theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + 1/(theta_pp*theta_xx - theta_xp**2))/theta_pp)**2)**2

def _dnu_dtheta_pp_func(theta_xx, theta_pp, theta_xp):
    """∂ν/∂θ_pp - exact analytical formula from SymPy (generated via pycode)."""
    return (-2*theta_xp*theta_xx*(2*theta_pp*theta_xp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + theta_xp*(-2*theta_xp**2/(theta_pp*theta_xx - theta_xp**2)**2 - 1/(theta_pp*theta_xx - theta_xp**2)))/(theta_pp*theta_xx - theta_xp**2)**2 + 2*theta_xp*(-4*theta_pp*theta_xp*theta_xx**2/(theta_pp*theta_xx - theta_xp**2)**3 + 2*theta_xp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + theta_xp*(4*theta_xp**2*theta_xx/(theta_pp*theta_xx - theta_xp**2)**3 + theta_xx/(theta_pp*theta_xx - theta_xp**2)**2))/(theta_pp*theta_xx - theta_xp**2) + (-1/2*theta_pp/(theta_pp*theta_xx - theta_xp**2) + (1/2)*(theta_pp*theta_xx - theta_xp**2)*(-theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + 1/(theta_pp*theta_xx - theta_xp**2))/theta_xx)*(theta_pp**2*theta_xx**2/(theta_pp*theta_xx - theta_xp**2)**3 - 2*theta_pp*theta_xp**2*theta_xx/(theta_pp*theta_xx - theta_xp**2)**3 - 3/2*theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + theta_pp*(theta_pp*theta_xx**2/(theta_pp*theta_xx - theta_xp**2)**3 - theta_xx/(theta_pp*theta_xx - theta_xp**2)**2) + theta_xp**2/(theta_pp*theta_xx - theta_xp**2)**2 + (1/2)/(theta_pp*theta_xx - theta_xp**2)) + (-1/2*theta_xx/(theta_pp*theta_xx - theta_xp**2) + (1/2)*(theta_pp*theta_xx - theta_xp**2)*(-theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + 1/(theta_pp*theta_xx - theta_xp**2))/theta_pp)*(theta_pp*theta_xx**3/(theta_pp*theta_xx - theta_xp**2)**3 - 2*theta_xp**2*theta_xx**2/(theta_pp*theta_xx - theta_xp**2)**3 - 1/2*theta_xx**2/(theta_pp*theta_xx - theta_xp**2)**2 + theta_xx*(theta_pp*theta_xx**2/(theta_pp*theta_xx - theta_xp**2)**3 - theta_xx/(theta_pp*theta_xx - theta_xp**2)**2)) + (-1/2*theta_pp*theta_xx**2/(theta_pp*theta_xx - theta_xp**2)**2 + theta_xp**2*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + theta_xx*(-1/2*theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + (1/2)/(theta_pp*theta_xx - theta_xp**2)))*((1/2)*theta_xx**2/(theta_pp*theta_xx - theta_xp**2)**2 + (1/2)*theta_xx*(-theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + 1/(theta_pp*theta_xx - theta_xp**2))/theta_pp + (1/2)*(theta_pp*theta_xx - theta_xp**2)*(2*theta_pp*theta_xx**2/(theta_pp*theta_xx - theta_xp**2)**3 - 2*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2)/theta_pp - 1/2*(theta_pp*theta_xx - theta_xp**2)*(-theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + 1/(theta_pp*theta_xx - theta_xp**2))/theta_pp**2) + (1/2)*(theta_pp*theta_xx - theta_xp**2)*(2*theta_pp*theta_xx**2/(theta_pp*theta_xx - theta_xp**2)**3 - 2*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2)*(-1/2*theta_pp**2*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + theta_pp*theta_xp**2/(theta_pp*theta_xx - theta_xp**2)**2 + theta_pp*(-1/2*theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + (1/2)/(theta_pp*theta_xx - theta_xp**2)))/theta_xx)/(4*theta_xp**2/(theta_pp*theta_xx - theta_xp**2)**2 + (-1/2*theta_pp/(theta_pp*theta_xx - theta_xp**2) + (1/2)*(theta_pp*theta_xx - theta_xp**2)*(-theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + 1/(theta_pp*theta_xx - theta_xp**2))/theta_xx)**2 + (-1/2*theta_xx/(theta_pp*theta_xx - theta_xp**2) + (1/2)*(theta_pp*theta_xx - theta_xp**2)*(-theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + 1/(theta_pp*theta_xx - theta_xp**2))/theta_pp)**2) + (2*theta_xp*(2*theta_pp*theta_xp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + theta_xp*(-2*theta_xp**2/(theta_pp*theta_xx - theta_xp**2)**2 - 1/(theta_pp*theta_xx - theta_xp**2)))/(theta_pp*theta_xx - theta_xp**2) + (-1/2*theta_pp/(theta_pp*theta_xx - theta_xp**2) + (1/2)*(theta_pp*theta_xx - theta_xp**2)*(-theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + 1/(theta_pp*theta_xx - theta_xp**2))/theta_xx)*(-1/2*theta_pp**2*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + theta_pp*theta_xp**2/(theta_pp*theta_xx - theta_xp**2)**2 + theta_pp*(-1/2*theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + (1/2)/(theta_pp*theta_xx - theta_xp**2))) + (-1/2*theta_xx/(theta_pp*theta_xx - theta_xp**2) + (1/2)*(theta_pp*theta_xx - theta_xp**2)*(-theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + 1/(theta_pp*theta_xx - theta_xp**2))/theta_pp)*(-1/2*theta_pp*theta_xx**2/(theta_pp*theta_xx - theta_xp**2)**2 + theta_xp**2*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + theta_xx*(-1/2*theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + (1/2)/(theta_pp*theta_xx - theta_xp**2))))*(8*theta_xp**2*theta_xx/(theta_pp*theta_xx - theta_xp**2)**3 - (-1/2*theta_xx/(theta_pp*theta_xx - theta_xp**2) + (1/2)*(theta_pp*theta_xx - theta_xp**2)*(-theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + 1/(theta_pp*theta_xx - theta_xp**2))/theta_pp)*(theta_xx**2/(theta_pp*theta_xx - theta_xp**2)**2 + theta_xx*(-theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + 1/(theta_pp*theta_xx - theta_xp**2))/theta_pp + (theta_pp*theta_xx - theta_xp**2)*(2*theta_pp*theta_xx**2/(theta_pp*theta_xx - theta_xp**2)**3 - 2*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2)/theta_pp - (theta_pp*theta_xx - theta_xp**2)*(-theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + 1/(theta_pp*theta_xx - theta_xp**2))/theta_pp**2) - (theta_pp*theta_xx - theta_xp**2)*(-1/2*theta_pp/(theta_pp*theta_xx - theta_xp**2) + (1/2)*(theta_pp*theta_xx - theta_xp**2)*(-theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + 1/(theta_pp*theta_xx - theta_xp**2))/theta_xx)*(2*theta_pp*theta_xx**2/(theta_pp*theta_xx - theta_xp**2)**3 - 2*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2)/theta_xx)/(4*theta_xp**2/(theta_pp*theta_xx - theta_xp**2)**2 + (-1/2*theta_pp/(theta_pp*theta_xx - theta_xp**2) + (1/2)*(theta_pp*theta_xx - theta_xp**2)*(-theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + 1/(theta_pp*theta_xx - theta_xp**2))/theta_xx)**2 + (-1/2*theta_xx/(theta_pp*theta_xx - theta_xp**2) + (1/2)*(theta_pp*theta_xx - theta_xp**2)*(-theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + 1/(theta_pp*theta_xx - theta_xp**2))/theta_pp)**2)**2

def _dnu_dtheta_xp_func(theta_xx, theta_pp, theta_xp):
    """∂ν/∂θ_xp - exact analytical formula from SymPy (generated via pycode)."""
    # This expression is EXTREMELY long (~4000 characters) - see pycode output above
    # For brevity in the file, we can write it on multiple lines or leave as single expression
    return (4*theta_xp**2*(2*theta_pp*theta_xp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + theta_xp*(-2*theta_xp**2/(theta_pp*theta_xx - theta_xp**2)**2 - 1/(theta_pp*theta_xx - theta_xp**2)))/(theta_pp*theta_xx - theta_xp**2)**2 + 2*theta_xp*(8*theta_pp*theta_xp**2*theta_xx/(theta_pp*theta_xx - theta_xp**2)**3 + 2*theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 - 2*theta_xp**2/(theta_pp*theta_xx - theta_xp**2)**2 + theta_xp*(-8*theta_xp**3/(theta_pp*theta_xx - theta_xp**2)**3 - 6*theta_xp/(theta_pp*theta_xx - theta_xp**2)**2) - 1/(theta_pp*theta_xx - theta_xp**2))/(theta_pp*theta_xx - theta_xp**2) + (-1/2*theta_pp/(theta_pp*theta_xx - theta_xp**2) + (1/2)*(theta_pp*theta_xx - theta_xp**2)*(-theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + 1/(theta_pp*theta_xx - theta_xp**2))/theta_xx)*(-2*theta_pp**2*theta_xp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**3 + 4*theta_pp*theta_xp**3/(theta_pp*theta_xx - theta_xp**2)**3 + 2*theta_pp*theta_xp/(theta_pp*theta_xx - theta_xp**2)**2 + theta_pp*(-2*theta_pp*theta_xp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**3 + theta_xp/(theta_pp*theta_xx - theta_xp**2)**2)) + (-1/2*theta_xx/(theta_pp*theta_xx - theta_xp**2) + (1/2)*(theta_pp*theta_xx - theta_xp**2)*(-theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + 1/(theta_pp*theta_xx - theta_xp**2))/theta_pp)*(-2*theta_pp*theta_xp*theta_xx**2/(theta_pp*theta_xx - theta_xp**2)**3 + 4*theta_xp**3*theta_xx/(theta_pp*theta_xx - theta_xp**2)**3 + 2*theta_xp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + theta_xx*(-2*theta_pp*theta_xp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**3 + theta_xp/(theta_pp*theta_xx - theta_xp**2)**2)) + (-theta_pp*theta_xp/(theta_pp*theta_xx - theta_xp**2)**2 - theta_xp*(-theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + 1/(theta_pp*theta_xx - theta_xp**2))/theta_xx + (1/2)*(theta_pp*theta_xx - theta_xp**2)*(-4*theta_pp*theta_xp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**3 + 2*theta_xp/(theta_pp*theta_xx - theta_xp**2)**2)/theta_xx)*(-1/2*theta_pp**2*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + theta_pp*theta_xp**2/(theta_pp*theta_xx - theta_xp**2)**2 + theta_pp*(-1/2*theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + (1/2)/(theta_pp*theta_xx - theta_xp**2))) + (-1/2*theta_pp*theta_xx**2/(theta_pp*theta_xx - theta_xp**2)**2 + theta_xp**2*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + theta_xx*(-1/2*theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + (1/2)/(theta_pp*theta_xx - theta_xp**2)))*(-theta_xp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 - theta_xp*(-theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + 1/(theta_pp*theta_xx - theta_xp**2))/theta_pp + (1/2)*(theta_pp*theta_xx - theta_xp**2)*(-4*theta_pp*theta_xp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**3 + 2*theta_xp/(theta_pp*theta_xx - theta_xp**2)**2)/theta_pp) + 2*(2*theta_pp*theta_xp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + theta_xp*(-2*theta_xp**2/(theta_pp*theta_xx - theta_xp**2)**2 - 1/(theta_pp*theta_xx - theta_xp**2)))/(theta_pp*theta_xx - theta_xp**2))/(4*theta_xp**2/(theta_pp*theta_xx - theta_xp**2)**2 + (-1/2*theta_pp/(theta_pp*theta_xx - theta_xp**2) + (1/2)*(theta_pp*theta_xx - theta_xp**2)*(-theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + 1/(theta_pp*theta_xx - theta_xp**2))/theta_xx)**2 + (-1/2*theta_xx/(theta_pp*theta_xx - theta_xp**2) + (1/2)*(theta_pp*theta_xx - theta_xp**2)*(-theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + 1/(theta_pp*theta_xx - theta_xp**2))/theta_pp)**2) + (2*theta_xp*(2*theta_pp*theta_xp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + theta_xp*(-2*theta_xp**2/(theta_pp*theta_xx - theta_xp**2)**2 - 1/(theta_pp*theta_xx - theta_xp**2)))/(theta_pp*theta_xx - theta_xp**2) + (-1/2*theta_pp/(theta_pp*theta_xx - theta_xp**2) + (1/2)*(theta_pp*theta_xx - theta_xp**2)*(-theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + 1/(theta_pp*theta_xx - theta_xp**2))/theta_xx)*(-1/2*theta_pp**2*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + theta_pp*theta_xp**2/(theta_pp*theta_xx - theta_xp**2)**2 + theta_pp*(-1/2*theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + (1/2)/(theta_pp*theta_xx - theta_xp**2))) + (-1/2*theta_xx/(theta_pp*theta_xx - theta_xp**2) + (1/2)*(theta_pp*theta_xx - theta_xp**2)*(-theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + 1/(theta_pp*theta_xx - theta_xp**2))/theta_pp)*(-1/2*theta_pp*theta_xx**2/(theta_pp*theta_xx - theta_xp**2)**2 + theta_xp**2*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + theta_xx*(-1/2*theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + (1/2)/(theta_pp*theta_xx - theta_xp**2))))*(-16*theta_xp**3/(theta_pp*theta_xx - theta_xp**2)**3 - 8*theta_xp/(theta_pp*theta_xx - theta_xp**2)**2 - (-1/2*theta_pp/(theta_pp*theta_xx - theta_xp**2) + (1/2)*(theta_pp*theta_xx - theta_xp**2)*(-theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + 1/(theta_pp*theta_xx - theta_xp**2))/theta_xx)*(-2*theta_pp*theta_xp/(theta_pp*theta_xx - theta_xp**2)**2 - 2*theta_xp*(-theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + 1/(theta_pp*theta_xx - theta_xp**2))/theta_xx + (theta_pp*theta_xx - theta_xp**2)*(-4*theta_pp*theta_xp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**3 + 2*theta_xp/(theta_pp*theta_xx - theta_xp**2)**2)/theta_xx) - (-1/2*theta_xx/(theta_pp*theta_xx - theta_xp**2) + (1/2)*(theta_pp*theta_xx - theta_xp**2)*(-theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + 1/(theta_pp*theta_xx - theta_xp**2))/theta_pp)*(-2*theta_xp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 - 2*theta_xp*(-theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + 1/(theta_pp*theta_xx - theta_xp**2))/theta_pp + (theta_pp*theta_xx - theta_xp**2)*(-4*theta_pp*theta_xp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**3 + 2*theta_xp/(theta_pp*theta_xx - theta_xp**2)**2)/theta_pp))/(4*theta_xp**2/(theta_pp*theta_xx - theta_xp**2)**2 + (-1/2*theta_pp/(theta_pp*theta_xx - theta_xp**2) + (1/2)*(theta_pp*theta_xx - theta_xp**2)*(-theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + 1/(theta_pp*theta_xx - theta_xp**2))/theta_xx)**2 + (-1/2*theta_xx/(theta_pp*theta_xx - theta_xp**2) + (1/2)*(theta_pp*theta_xx - theta_xp**2)*(-theta_pp*theta_xx/(theta_pp*theta_xx - theta_xp**2)**2 + 1/(theta_pp*theta_xx - theta_xp**2))/theta_pp)**2)**2


def precision_to_covariance(theta):
    """
    Convert natural parameters to covariance matrix.
    
    For precision matrix K = [[θ_xx, θ_xp],
                               [θ_xp, θ_pp]]
    
    Returns Σ = K^(-1) = (1/det(K)) * [[θ_pp, -θ_xp],
                                        [-θ_xp, θ_xx]]
    
    where det(K) = θ_xx * θ_pp - θ_xp²
    """
    theta_xx, theta_pp, theta_xp = theta
    
    # Determinant of precision matrix
    det_K = theta_xx * theta_pp - theta_xp**2
    
    # Ensure positive definite
    if det_K <= 0:
        return None
    
    # Exact 2x2 inverse formula
    Sigma = np.array([[theta_pp,  -theta_xp],
                      [-theta_xp,  theta_xx]]) / det_K
    
    return Sigma


def marginal_entropies(theta):
    """
    Compute h(X) and h(P) from natural parameters.
    
    Returns (h_X, h_P, h_total)
    """
    Sigma = precision_to_covariance(theta)
    if Sigma is None:
        return None, None, None
    
    sigma_x_sq = Sigma[0, 0]
    sigma_p_sq = Sigma[1, 1]
    
    # h = (1/2)log(2πe σ²)
    h_X = 0.5 * np.log(2 * np.pi * np.e * sigma_x_sq)
    h_P = 0.5 * np.log(2 * np.pi * np.e * sigma_p_sq)
    
    return h_X, h_P, h_X + h_P


def constraint_gradient(theta):
    """
    Compute constraint gradient a(θ) = ∇_θ (h(X) + h(P)).
    
    h(X) = (1/2)log(σ_x²), h(P) = (1/2)log(σ_p²)
    where σ_x² = Σ_11 = θ_pp/det(K), σ_p² = Σ_22 = θ_xx/det(K)
    """
    Sigma = precision_to_covariance(theta)
    if Sigma is None:
        return None
    
    theta_xx, theta_pp, theta_xp = theta
    det_K = theta_xx * theta_pp - theta_xp**2
    
    sigma_x_sq = Sigma[0, 0]
    sigma_p_sq = Sigma[1, 1]
    
    grad_sigma_x_sq = np.array([
        -theta_pp * theta_pp / (det_K**2),
        -theta_xp**2 / (det_K**2),
        2 * theta_pp * theta_xp / (det_K**2)
    ])
    
    grad_sigma_p_sq = np.array([
        -theta_xp**2 / (det_K**2),
        -theta_xx * theta_xx / (det_K**2),
        2 * theta_xx * theta_xp / (det_K**2)
    ])
    
    grad_hX = 0.5 * grad_sigma_x_sq / sigma_x_sq
    grad_hP = 0.5 * grad_sigma_p_sq / sigma_p_sq
    
    grad = grad_hX + grad_hP
    
    return grad

def joint_entropy_gradient(theta):
    """
    Compute constraint gradient a(θ) = ∇_θ (h(X) + h(P)).
    
    Analytical formula: a = -(1/2) * vec(Σ) where Σ = K^(-1)
    and vec maps the symmetric matrix to (Σ_11, Σ_22, 2*Σ_12).
    """
    Sigma = precision_to_covariance(theta)
    if Sigma is None:
        return None
    
    # Gradient of -log(det(K)) is Σ
    # For symmetric matrix parametrized as (θ_xx, θ_pp, θ_xp),
    # the gradient is (Σ_11, Σ_22, 2*Σ_12)
    grad = -0.5 * np.array([Sigma[0, 0], 
                            Sigma[1, 1], 
                            2 * Sigma[0, 1]])
    
    return grad


def log_partition_function(theta):
    """
    Log partition function ψ(θ) = -(1/2) log det(K).
    
    Direct computation from θ = (θ_xx, θ_pp, θ_xp).
    """
    theta_xx, theta_pp, theta_xp = theta
    det_K = theta_xx * theta_pp - theta_xp**2
    
    if det_K <= 0:
        return None
    
    return -0.5*np.log(det_K)



def fisher_information(theta):
    """
    Fisher information G(θ) = ∇²ψ(θ) for 2D Gaussian.
    
    Analytical Hessian of ψ(θ) = -(1/2) log(θ_xx θ_pp - θ_xp²).
    
    Parameters:
    -----------
    theta : array [θ_xx, θ_pp, θ_xp]
        Natural parameters defining precision matrix K
    
    Returns:
    --------
    G : 3×3 array
        Fisher information matrix (positive definite)
    """
    theta_xx, theta_pp, theta_xp = theta
    d = theta_xx * theta_pp - theta_xp**2
    
    if d <= 0:
        return None
    
    d2 = d**2
    
    # Exact Hessian formula
    G = np.array([
        [theta_pp**2 / (2*d2),          theta_xp**2 / (2*d2),          -theta_pp * theta_xp / d2],
        [theta_xp**2 / (2*d2),          theta_xx**2 / (2*d2),          -theta_xx * theta_xp / d2],
        [-theta_pp * theta_xp / d2,     -theta_xx * theta_xp / d2,     (theta_xx*theta_pp + theta_xp**2) / d2]
    ])
    
    return G


def compute_constrained_flow(theta):
    """
    Compute F(θ) = -G(θ)θ + ν(θ)a(θ).
    
    Returns flow vector or None if invalid.
    """
    G = fisher_information(theta)
    a = constraint_gradient(theta)
    
    if G is None or a is None:
        return None
    
    # Unconstrained flow toward maximum entropy
    F_unc = -G @ theta
    
    # Lagrange multiplier from tangency condition
    nu = np.dot(F_unc, a) / np.dot(a, a)
    
    # Constrained flow
    F = F_unc - nu * a
    
    return F


def constraint_hessian(theta):
    """
    Compute Hessian of constraint H_C = ∇²(h(X) + h(P)).
    
    Analytical computation of second derivatives.
    """
    theta_xx, theta_pp, theta_xp = theta
    d = theta_xx * theta_pp - theta_xp**2
    
    if d <= 0:
        return None
    
    d2 = d**2
    d3 = d**3
    
    # Covariance matrix elements
    Sigma_11 = theta_pp / d  # σ_x²
    Sigma_22 = theta_xx / d  # σ_p²
    
    # First derivatives of Sigma elements (from constraint_gradient derivation)
    dSigma11_dtheta = np.array([-theta_pp**2/d2, -theta_xp**2/d2, 2*theta_pp*theta_xp/d2])
    dSigma22_dtheta = np.array([-theta_xp**2/d2, -theta_xx**2/d2, 2*theta_xx*theta_xp/d2])
    
    # For h = (1/2)log(σ²), we have:
    # ∂h/∂θ_i = (1/2σ²) ∂σ²/∂θ_i
    # ∂²h/∂θ_i∂θ_j = (1/2) [∂²σ²/(σ²∂θ_i∂θ_j) - (∂σ²/∂θ_i)(∂σ²/∂θ_j)/(σ²)²]
    
    # Second derivatives of Sigma_11
    d2Sigma11 = np.zeros((3, 3))
    d2Sigma11[0,0] = 2*theta_pp**3/d3
    d2Sigma11[0,1] = d2Sigma11[1,0] = 2*theta_pp*theta_xp**2/d3
    d2Sigma11[0,2] = d2Sigma11[2,0] = -2*theta_pp**2*theta_xp/d3 - 2*theta_pp*theta_xp/d2
    d2Sigma11[1,1] = 2*theta_xp**4/d3
    d2Sigma11[1,2] = d2Sigma11[2,1] = -4*theta_xp**3/d3 + 2*theta_pp/d2
    d2Sigma11[2,2] = 2*theta_pp*theta_xp**2/d3 + 2*theta_pp**2/d2 + 4*theta_pp*theta_xp**2/d3
    
    # Second derivatives of Sigma_22
    d2Sigma22 = np.zeros((3, 3))
    d2Sigma22[0,0] = 2*theta_xp**4/d3
    d2Sigma22[0,1] = d2Sigma22[1,0] = 2*theta_xx*theta_xp**2/d3
    d2Sigma22[0,2] = d2Sigma22[2,0] = -4*theta_xp**3/d3 + 2*theta_xx/d2
    d2Sigma22[1,1] = 2*theta_xx**3/d3
    d2Sigma22[1,2] = d2Sigma22[2,1] = -2*theta_xx**2*theta_xp/d3 - 2*theta_xx*theta_xp/d2
    d2Sigma22[2,2] = 2*theta_xx*theta_xp**2/d3 + 2*theta_xx**2/d2 + 4*theta_xx*theta_xp**2/d3
    
    # Hessian of h(X)
    H_X = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            H_X[i,j] = 0.5 * (d2Sigma11[i,j]/Sigma_11 - 
                              dSigma11_dtheta[i]*dSigma11_dtheta[j]/(Sigma_11**2))
    
    # Hessian of h(P)
    H_P = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            H_P[i,j] = 0.5 * (d2Sigma22[i,j]/Sigma_22 - 
                              dSigma22_dtheta[i]*dSigma22_dtheta[j]/(Sigma_22**2))
    
    return H_X + H_P


def third_derivatives_log_partition(theta):
    """
    Compute third derivatives T_ijk = ∂³ψ/∂θ_i∂θ_j∂θ_k.
    
    For ψ(θ) = -(1/2) log(θ_xx θ_pp - θ_xp²).
    
    Returns:
    --------
    T : 3×3×3 array
        Third derivative tensor (fully symmetric)
    """
    theta_xx, theta_pp, theta_xp = theta
    d = theta_xx * theta_pp - theta_xp**2
    
    if d <= 0:
        return None
    
    d2 = d**2
    d3 = d**3
    
    T = np.zeros((3, 3, 3))
    
    # Diagonal elements
    T[0,0,0] = -theta_pp**3 / d3  # T_111
    T[1,1,1] = -theta_xx**3 / d3  # T_222
    T[2,2,2] = 2*theta_xp*(3*theta_xx*theta_pp + theta_xp**2) / d3  # T_333
    
    # Mixed second order
    val_112 = -theta_pp*theta_xp**2 / d3
    T[0,0,1] = T[0,1,0] = T[1,0,0] = val_112  # T_112
    
    val_122 = -theta_xx*theta_xp**2 / d3
    T[0,1,1] = T[1,0,1] = T[1,1,0] = val_122  # T_122
    
    val_113 = 2*theta_pp**2*theta_xp / d3
    T[0,0,2] = T[0,2,0] = T[2,0,0] = val_113  # T_113
    
    val_223 = 2*theta_xx**2*theta_xp / d3
    T[1,1,2] = T[1,2,1] = T[2,1,1] = val_223  # T_223
    
    # Fully mixed
    val_123 = theta_xp*(theta_xx*theta_pp + theta_xp**2) / d3
    T[0,1,2] = T[0,2,1] = T[1,0,2] = val_123  # T_123
    T[1,2,0] = T[2,0,1] = T[2,1,0] = val_123
    
    val_133 = -theta_pp*(theta_xx*theta_pp + 3*theta_xp**2) / d3
    T[0,2,2] = T[2,0,2] = T[2,2,0] = val_133  # T_133
    
    val_233 = -theta_xx*(theta_xx*theta_pp + 3*theta_xp**2) / d3
    T[1,2,2] = T[2,1,2] = T[2,2,1] = val_233  # T_233
    
    return T


def compute_jacobian_analytical(theta):
    """
    Compute Jacobian M = ∂F/∂θ fully analytically.
    
    For F(θ) = -G(θ)θ - ν(θ)a(θ) where ν = (Gθ·a)/(a·a):
    
    M = -G - T_ψ[θ] - (∂ν/∂θ)⊗a - ν H_C
    
    where T_ψ[θ]_ij = Σ_k T_ijk θ_k is the contraction with θ.
    
    Returns:
    --------
    M : 3×3 array
        Full Jacobian matrix
    S : 3×3 array  
        Symmetric part (dissipation)
    A : 3×3 array
        Anti-symmetric part (circulation)
    """
    G = fisher_information(theta)
    a = constraint_gradient(theta)
    H_C = constraint_hessian(theta)
    T = third_derivatives_log_partition(theta)
    
    if G is None or a is None or H_C is None or T is None:
        return None, None, None
    
    # Unconstrained flow
    F_unc = -G @ theta
    
    # Lagrange multiplier
    a_norm_sq = np.dot(a, a)
    nu = np.dot(F_unc, a) / a_norm_sq
    
    # Compute ∂ν/∂θ using pre-computed exact analytical formulas from SymPy
    # These are EXACT symbolic derivatives, not numerical approximations
    theta_xx, theta_pp, theta_xp = theta
    dnu = np.array([
        _dnu_dtheta_xx_func(theta_xx, theta_pp, theta_xp),
        _dnu_dtheta_pp_func(theta_xx, theta_pp, theta_xp),
        _dnu_dtheta_xp_func(theta_xx, theta_pp, theta_xp)
    ])
    
    # Contraction T_ψ[θ]: (T_ψ[θ])_ij = Σ_k T_ijk θ_k
    T_psi_theta = np.einsum('ijk,k->ij', T, theta)
    
    # Jacobian: M = -G - T_ψ[θ] - (∂ν/∂θ)⊗a - ν H_C
    # NOTE: Negative signs because F = F_unc - ν*a, not + ν*a
    M = -G - T_psi_theta - np.outer(dnu, a) - nu * H_C
    
    # Decompose into symmetric and anti-symmetric parts
    S = 0.5 * (M + M.T)
    A = 0.5 * (M - M.T)
    
    return M, S, A


def antisymmetric_part_analytical(theta):
    """
    Compute anti-symmetric part A directly from analytical formulas.
    
    Uses the factored forms derived via SymPy:
    A = -[(∂ν/∂θ)⊗a - a⊗(∂ν/∂θ)]/2
    
    where only this term in the Jacobian is non-symmetric.
    
    This is a direct implementation of the symbolic factorization,
    revealing the geometric structure more clearly than extracting
    from the full Jacobian.
    
    Returns:
    --------
    A : 3×3 array
        Anti-symmetric part (A^T = -A)
    """
    theta_xx, theta_pp, theta_xp = theta
    d = theta_xx * theta_pp - theta_xp**2
    
    if d <= 0:
        return None
    
    # Common denominator (related to (a·a)²)
    D_base = (theta_pp**4 * theta_xx**2 + 
              2*theta_pp**3 * theta_xp**2 * theta_xx + 
              theta_pp**2 * theta_xp**4 + 
              16*theta_pp**2 * theta_xp**2 * theta_xx**2 + 
              theta_pp**2 * theta_xx**4 + 
              2*theta_pp * theta_xp**2 * theta_xx**3 + 
              theta_xp**4 * theta_xx**2)
    
    D_squared = D_base**2
    
    # A[0,1]: Cleanest factored form - NOTE: negative sign from -outer(dnu, a)
    # SymPy: -4·θ_pp·θ_xp²·θ_xx·(θ_pp - θ_xx)·(θ_pp + θ_xx)·(θ_pp·θ_xx + θ_xp²)² / D²
    A_01 = (-4*theta_pp*theta_xp**2*theta_xx*(theta_pp - theta_xx)*(theta_pp + theta_xx)*
            (theta_pp*theta_xx + theta_xp**2)**2) / D_squared
    
    # A[0,2]: Complex polynomial (degree 6 in numerator) - NOTE: positive sign
    # SymPy: (1/2)·θ_pp·θ_xp·(...polynomial...) / D²
    A_02 = (0.5*theta_pp*theta_xp*(
        theta_pp**6*theta_xx**2 + 
        2*theta_pp**5*theta_xp**2*theta_xx + 
        8*theta_pp**5*theta_xx**3 + 
        theta_pp**4*theta_xp**4 + 
        16*theta_pp**4*theta_xp**2*theta_xx**2 + 
        2*theta_pp**4*theta_xx**4 + 
        24*theta_pp**3*theta_xp**4*theta_xx + 
        4*theta_pp**3*theta_xp**2*theta_xx**3 + 
        8*theta_pp**3*theta_xx**5 + 
        2*theta_pp**2*theta_xp**4*theta_xx**2 - 
        16*theta_pp**2*theta_xp**2*theta_xx**4 + 
        theta_pp**2*theta_xx**6 - 
        8*theta_pp*theta_xp**4*theta_xx**3 + 
        2*theta_pp*theta_xp**2*theta_xx**5 + 
        theta_xp**4*theta_xx**4
    )) / D_squared
    
    # A[1,2]: Similar structure - NOTE: positive sign
    # SymPy: (1/2)·θ_xp·θ_xx·(...polynomial...) / D²
    A_12 = (0.5*theta_xp*theta_xx*(
        theta_pp**6*theta_xx**2 + 
        2*theta_pp**5*theta_xp**2*theta_xx + 
        8*theta_pp**5*theta_xx**3 + 
        theta_pp**4*theta_xp**4 - 
        16*theta_pp**4*theta_xp**2*theta_xx**2 + 
        2*theta_pp**4*theta_xx**4 - 
        8*theta_pp**3*theta_xp**4*theta_xx + 
        4*theta_pp**3*theta_xp**2*theta_xx**3 + 
        8*theta_pp**3*theta_xx**5 + 
        2*theta_pp**2*theta_xp**4*theta_xx**2 + 
        16*theta_pp**2*theta_xp**2*theta_xx**4 + 
        theta_pp**2*theta_xx**6 + 
        24*theta_pp*theta_xp**4*theta_xx**3 + 
        2*theta_pp*theta_xp**2*theta_xx**5 + 
        theta_xp**4*theta_xx**4
    )) / D_squared
    
    # Construct anti-symmetric matrix
    A = np.array([
        [0,      A_01,  A_02],
        [-A_01,  0,     A_12],
        [-A_02, -A_12,  0   ]
    ])
    
    return A


def compute_jacobian_numerical(theta, eps=1e-5):
    """
    Compute Jacobian M = ∂F/∂θ numerically (for validation).
    """
    F_base = compute_constrained_flow(theta)
    if F_base is None:
        return None
    
    d = 3
    M = np.zeros((d, d))
    
    for j in range(d):
        theta_plus = theta.copy()
        theta_plus[j] += eps
        F_plus = compute_constrained_flow(theta_plus)
        if F_plus is None:
            return None
        M[:, j] = (F_plus - F_base) / eps
    
    return M


def verify_jacobi_identity_numerical(theta, eps_diff=1e-5, verbose=True):
    """
    Numerically verify the Jacobi identity for the antisymmetric operator A.
    
    The Jacobi identity for a Poisson bracket is:
        {{f,g},h} + {{g,h},f} + {{h,f},g} = 0
    
    where {f,g} = (∇f)^T A (∇g).
    
    For coordinate functions θ_i, θ_j, θ_k:
        - {θ_i, θ_j} = A_ij
        - {{θ_i, θ_j}, θ_k} = Σ_m (∂A_ij/∂θ_m) A_mk
    
    This function tests the Jacobi identity on all triplets of coordinate
    functions and reports the maximum violation.
    
    Parameters
    ----------
    theta : array
        Parameter point at which to evaluate [θ_xx, θ_pp, θ_xp]
    eps_diff : float
        Step size for finite differences (default: 1e-5)
    verbose : bool
        If True, print detailed statistics
        
    Returns
    -------
    dict with keys:
        'max_violation' : float
            Maximum |{{f,g},h} + {{g,h},f} + {{h,f},g}| over all triplets
        'mean_violation' : float
            Mean absolute violation
        'violations' : ndarray, shape (3, 3, 3)
            Full violation tensor
        'A' : ndarray, shape (3, 3)
            Antisymmetric operator at θ
        'num_triplets' : int
            Number of triplets tested
    """
    d = 3  # Harmonic oscillator has 3 parameters
    
    # Compute A at the base point using analytical formula
    A = antisymmetric_part_analytical(theta)
    if A is None:
        if verbose:
            print("Warning: Invalid parameter point (negative determinant)")
        return None
    
    # Compute numerical derivatives ∂A_ij/∂θ_k for all i,j,k
    # This is a tensor of shape (3, 3, 3)
    dA = np.zeros((d, d, d))
    
    for k in range(d):
        theta_plus = theta.copy()
        theta_plus[k] += eps_diff
        
        # Compute A at the perturbed point
        A_plus = antisymmetric_part_analytical(theta_plus)
        if A_plus is None:
            if verbose:
                print(f"Warning: Invalid perturbed point at k={k}")
            return None
        
        # Finite difference: ∂A/∂θ_k ≈ (A(θ+ε) - A(θ))/ε
        dA[:, :, k] = (A_plus - A) / eps_diff
    
    # Evaluate Jacobi identity for all triplets (i,j,k)
    violations = np.zeros((d, d, d))
    
    for i in range(d):
        for j in range(d):
            for k in range(d):
                # First bracket: {θ_i, θ_j} = A_ij (just a number)
                # Second bracket: {{θ_i, θ_j}, θ_k} = Σ_m (∂A_ij/∂θ_m) A_mk
                bracket_ij_k = np.dot(dA[i, j, :], A[:, k])
                
                # Similarly for cyclic permutations
                bracket_jk_i = np.dot(dA[j, k, :], A[:, i])
                bracket_ki_j = np.dot(dA[k, i, :], A[:, j])
                
                # Jacobi identity: sum should be zero
                jacobi_sum = bracket_ij_k + bracket_jk_i + bracket_ki_j
                violations[i, j, k] = jacobi_sum
    
    max_violation = np.max(np.abs(violations))
    mean_violation = np.mean(np.abs(violations))
    
    if verbose:
        print("\n" + "="*70)
        print("JACOBI IDENTITY VERIFICATION (NUMERICAL)")
        print("="*70)
        print(f"Parameter: θ = [{theta[0]:.4f}, {theta[1]:.4f}, {theta[2]:.4f}]")
        print(f"Parameter dimension: d = {d}")
        print(f"Number of triplets tested: {d**3}")
        print(f"\nResults:")
        print(f"  Maximum violation:  {max_violation:.6e}")
        print(f"  Mean violation:     {mean_violation:.6e}")
        print(f"  ||A|| (Frobenius):  {np.linalg.norm(A, 'fro'):.6f}")
        
        # Relative violation
        A_norm = np.linalg.norm(A, 'fro')
        if A_norm > 1e-10:
            rel_violation = max_violation / (A_norm**2)
            print(f"  Relative violation: {rel_violation:.6e} (scaled by ||A||²)")
        
        # Find worst offending triplet
        worst_idx = np.unravel_index(np.argmax(np.abs(violations)), violations.shape)
        i_worst, j_worst, k_worst = worst_idx
        print(f"\nWorst triplet: (θ[{i_worst}], θ[{j_worst}], θ[{k_worst}])")
        print(f"  {{{{θ[{i_worst}], θ[{j_worst}]}}, θ[{k_worst}]}} = {violations[i_worst, j_worst, k_worst]:.6e}")
        
        # Check if it's acceptable
        if max_violation < 1e-4:
            print("\n✓ Jacobi identity satisfied (violation < 1e-4)")
        elif max_violation < 1e-3:
            print("\n⚠ Jacobi identity approximately satisfied (violation < 1e-3)")
        else:
            print("\n✗ Jacobi identity violated (violation > 1e-3)")
        print("="*70 + "\n")
    
    return {
        'max_violation': max_violation,
        'mean_violation': mean_violation,
        'violations': violations,
        'A': A,
        'num_triplets': d**3,
        'eps_diff': eps_diff
    }


def project_onto_constraint(theta, C_target, max_iter=10, tol=1e-12):
    """
    Project θ onto constraint manifold h(X) + h(P) = C_target.
    
    Uses Newton's method to adjust θ along the constraint gradient
    direction until the constraint is satisfied.
    
    Parameters
    ----------
    theta : array_like
        Current parameter vector
    C_target : float
        Target constraint value
    max_iter : int
        Maximum number of Newton iterations (default: 10)
    tol : float
        Tolerance for constraint satisfaction (default: 1e-12)
    
    Returns
    -------
    theta_projected : ndarray
        Projected parameter vector, or None if projection fails
    """
    theta = theta.copy()
    
    for i in range(max_iter):
        # Compute current constraint value
        h_X, h_P, C_current = marginal_entropies(theta)
        if h_X is None:
            return None
        
        error = C_current - C_target
        
        # Check convergence
        if abs(error) < tol:
            return theta
        
        # Compute constraint gradient
        a = constraint_gradient(theta)
        if a is None:
            return None
        
        # Newton step: θ ← θ - (error / ||a||²) · a
        # This moves θ along the constraint gradient to reduce error
        step_size = error / np.dot(a, a)
        theta = theta - step_size * a
    
    # If we didn't converge, return None
    return None


def simulate_dynamics(theta_init, tau_span, dtau=0.01, project=True, project_tol=1e-12):
    """
    Simulate constrained dynamics: dθ/dt = F(θ).
    
    Parameters
    ----------
    theta_init : array_like
        Initial parameter vector [θ_xx, θ_pp, θ_xp]
    tau_span : float
        Total integration time
    dtau : float
        Time step (default: 0.01)
    project : bool
        If True, project back onto constraint manifold after each step
        to prevent constraint drift (default: True)
    project_tol : float
        Tolerance for constraint projection (default: 1e-12)
    
    Returns
    -------
    tau_vals : ndarray
        Time points
    trajectory : ndarray
        Parameter trajectory (n_steps × 3)
    constraint_vals : ndarray
        Constraint values at each time point
    
    Notes
    -----
    Without projection, constraint drift accumulates as O(√N) due to
    numerical integration errors. With projection, constraint is
    preserved to machine precision.
    """
    tau_vals = np.arange(0, tau_span, dtau)
    trajectory = [theta_init.copy()]
    constraint_vals = []
    
    # Initial constraint value
    _, _, C_init = marginal_entropies(theta_init)
    constraint_vals.append(C_init)
    
    theta = theta_init.copy()
    
    for _ in tau_vals[1:]:
        F = compute_constrained_flow(theta)
        if F is None:
            print("Warning: Invalid state reached")
            break
        
        # Euler step
        theta = theta + dtau * F
        
        # Project back onto constraint manifold
        if project:
            theta_projected = project_onto_constraint(theta, C_init, tol=project_tol)
            if theta_projected is None:
                print("Warning: Projection failed")
                break
            theta = theta_projected
        
        trajectory.append(theta.copy())
        
        # Track constraint
        _, _, C_curr = marginal_entropies(theta)
        constraint_vals.append(C_curr)
    
    return tau_vals[:len(trajectory)], np.array(trajectory), np.array(constraint_vals)

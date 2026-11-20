"""
@author: Josaphat Dackouo

This code performs the grid convergence test.
With a fixed domain length, the grid discretization size can be changed with the dxs parameter.
NB: Do not forget to adapt the file name in save_table at the end.

Files are directly saved in the file folder, from where plots can be done.
"""

import numpy as np
from tqdm import tqdm   # <-- progress bar

# ----------------------------
# Global Parameters
# ----------------------------
L_domain = 1e-6        # Fixed domain length (1 µm)
dxs = [1e-10]          # Grid discretization
bc_dil = 1e8           # A dilution parameter to handle sharp changes

# Adaptive time-step parameters
dt_init = 1e-14        # initial (and minimum) dt
dt_max = 1e-5          # maximum dt

T_end_ref = 1.0e-6     # Final time

""" Boundary conditions BCs: C = Closed; O = Open """
BCs = 'O'

""" Poisson != 'ON' or 'Off' ==> Electroneutrality
    Poisson == 'ON' ==> PNP """
Poisson = 'ON'

# Diffusion coefficients (m^2/s)
D_Na = 1.33e-9
D_Cl = 2.03e-9

# Ionic charges
z_Na = 1
z_Cl = -1

c_init = 10.0          # Initial concentration

# Physical constants
F = 96485.332123       # Faraday constant (C/mol)
R_const = 8.3145       # Gas constant (J/(mol·K))
T_const = 298.15       # Temperature (K)
epsilon = 6.94004652e-10 # Permittivity (C/(V·m))
w = 0.0                # fixed charge density

F_RT = F / (R_const * T_const)

# ----------------------------
# Index helper
# ----------------------------
def index(i, var):
    if var == 'Na': return 3*i
    elif var == 'Cl': return 3*i + 1
    elif var == 'V': return 3*i + 2
    else: raise ValueError("Unknown variable")

# small positive floor to avoid logs of zero or negative
spf = 1e-20

def log_avg(a, b, tol):
    a = max(a, spf)
    b = max(b, spf)
    ld = np.log(b) - np.log(a)
    if abs(ld) > tol:
        return (b - a) / ld
    else:
        return 0.5 * (a + b)

def d_log_avg_da(a, b, tol):
    a = max(a, spf)
    b = max(b, spf)
    ld = np.log(b) - np.log(a)
    if abs(ld) > tol:
        d_avg = ((b / a) - 1 - ld) / ld**2
        if abs(d_avg) > c_init:
            d_avg = 10.0 * np.sign(d_avg) * a
    else:
        d_avg = 0.5
    return d_avg

def d_log_avg_db(a, b, tol):
    a = max(a, spf)
    b = max(b, spf)
    ld = np.log(b) - np.log(a)
    if abs(ld) > tol:
        d_avg = ((a / b) - 1 + ld) / ld**2
        if abs(d_avg) > c_init:
            d_avg = 10.0 * np.sign(d_avg) * b
    else:
        d_avg = 0.5
    return d_avg

def A(a, b, dx):
    return (b - a) / dx

def dA_da(a, b, dx):
    return -1.0 / dx

def dA_db(a, b, dx):
    return 1.0 / dx

def B(a, b, z, tol):
    return F_RT * z * log_avg(a, b, tol)

def dB_da(a, b, z, tol):
    return F_RT * z * d_log_avg_da(a, b, tol)

def dB_db(a, b, z, tol):
    return F_RT * z * d_log_avg_db(a, b, tol)

# ----------------------------
# Compute Nernst-Planck flux
# ----------------------------
def flux(c_i, c_ip1, V_i, V_ip1, z, D, dx, tol):
    return -D * (A(c_i, c_ip1, dx) + B(c_i, c_ip1, z, tol) * A(V_i, V_ip1, dx))

# ----------------------------
# Flux derivatives for Jacobian
# ----------------------------
def dFlux_dc_i(c_i, c_ip1, V_i, V_ip1, z, D, dx, tol):
    return -D * (dA_da(c_i, c_ip1, dx) + dB_da(c_i, c_ip1, z, tol) * A(V_i, V_ip1, dx))

def dFlux_dc_ip1(c_i, c_ip1, V_i, V_ip1, z, D, dx, tol):
    return -D * (dA_db(c_i, c_ip1, dx) + dB_db(c_i, c_ip1, z, tol) * A(V_i, V_ip1, dx))

def dFlux_dV_i(c_i, c_ip1, V_i, V_ip1, z, D, dx, tol):
    return -D * (B(c_i, c_ip1, z, tol) * dA_da(V_i, V_ip1, dx))

def dFlux_dV_ip1(c_i, c_ip1, V_i, V_ip1, z, D, dx, tol):
    return -D * (B(c_i, c_ip1, z, tol) * dA_db(V_i, V_ip1, dx))

# ----------------------------
# Local Residual for concentrations
# ----------------------------
def residuals_C(i, c_old, c_i, c_ip1, c_im1, V_i, V_ip1, V_im1, z, D, dx, tol, dt, left_close, right_close, N):
    delta_flux = 0.0
    if not right_close:
        delta_flux += flux(c_i, c_ip1, V_i, V_ip1, z, D, dx, tol)
    if not left_close:
        delta_flux += -flux(c_im1, c_i, V_im1, V_i, z, D, dx, tol)
    if BCs == 'O' and i == N-1:
        delta_flux /= bc_dil
    return ((c_i - c_old) / dt) + (delta_flux / dx)

def d_residuals_C_dc_i(i, c_old, c_i, c_ip1, c_im1, V_i, V_ip1, V_im1, z, D, dx, tol, dt, left_close, right_close, N):
    d_delta_flux_dc_i = 0.0
    if not right_close:
        d_delta_flux_dc_i += dFlux_dc_i(c_i, c_ip1, V_i, V_ip1, z, D, dx, tol)
    if not left_close:
        d_delta_flux_dc_i += -dFlux_dc_ip1(c_im1, c_i, V_im1, V_i, z, D, dx, tol)
    if BCs == 'O' and i == N-1:
        d_delta_flux_dc_i /= bc_dil
    return ((1.0 / dt) + (d_delta_flux_dc_i / dx)) * c_i

def d_residuals_C_dc_ip1(i, c_old, c_i, c_ip1, c_im1, V_i, V_ip1, V_im1, z, D, dx, tol, dt, left_close, right_close, N):
    d_delta_flux_dc_ip1 = 0.0
    if not right_close:
        d_delta_flux_dc_ip1 += dFlux_dc_ip1(c_i, c_ip1, V_i, V_ip1, z, D, dx, tol)
    if BCs == 'O' and i == N-1:
        d_delta_flux_dc_ip1 /= bc_dil
    return (d_delta_flux_dc_ip1 / dx) * c_ip1

def d_residuals_C_dc_im1(i, c_old, c_i, c_ip1, c_im1, V_i, V_ip1, V_im1, z, D, dx, tol, dt, left_close, right_close, N):
    d_delta_flux_dc_im1 = 0.0
    if not left_close:
        d_delta_flux_dc_im1 += -dFlux_dc_i(c_im1, c_i, V_im1, V_i, z, D, dx, tol)
    if BCs == 'O' and i == N-1:
        d_delta_flux_dc_im1 /= bc_dil
    return (d_delta_flux_dc_im1 / dx) * c_im1

def d_residuals_C_dV_i(i, c_old, c_i, c_ip1, c_im1, V_i, V_ip1, V_im1, z, D, dx, tol, dt, left_close, right_close, N):
    d_delta_flux_dV_i = 0.0
    if not right_close:
        d_delta_flux_dV_i += dFlux_dV_i(c_i, c_ip1, V_i, V_ip1, z, D, dx, tol)
    if not left_close:
        d_delta_flux_dV_i += -dFlux_dV_ip1(c_im1, c_i, V_im1, V_i, z, D, dx, tol)
    if BCs == 'O' and i == N-1:
        d_delta_flux_dV_i /= bc_dil
    return d_delta_flux_dV_i / dx

def d_residuals_C_dV_ip1(i, c_old, c_i, c_ip1, c_im1, V_i, V_ip1, V_im1, z, D, dx, tol, dt, left_close, right_close, N):
    d_delta_flux_dV_ip1 = 0.0
    if not right_close:
        d_delta_flux_dV_ip1 += dFlux_dV_ip1(c_i, c_ip1, V_i, V_ip1, z, D, dx, tol)
    if BCs == 'O' and i == N-1:
        d_delta_flux_dV_ip1 /= bc_dil
    return d_delta_flux_dV_ip1 / dx

def d_residuals_C_dV_im1(i, c_old, c_i, c_ip1, c_im1, V_i, V_ip1, V_im1, z, D, dx, tol, dt, left_close, right_close, N):
    d_delta_flux_dV_im1 = 0.0
    if not left_close:
        d_delta_flux_dV_im1 += -dFlux_dV_i(c_im1, c_i, V_im1, V_i, z, D, dx, tol)
    if BCs == 'O' and i == N-1:
        d_delta_flux_dV_im1 /= bc_dil
    return d_delta_flux_dV_im1 / dx

# ----------------------------
# Local Residual for potential (Poisson equation)
# ----------------------------
if Poisson == 'ON':
    def residuals_V(c_iNa, c_iCl, V_i, V_ip1, V_im1, i, dx, dt, left_close, right_close):
        if not left_close and not right_close:
            return (((V_ip1 - 2*V_i + V_im1) / dx**2) + F/epsilon*(c_iNa - c_iCl)) * (epsilon/F)/dt
        elif left_close:
            return (((V_ip1 - V_i) / dx**2) + F/epsilon*(c_iNa - c_iCl)) * (epsilon/F)/dt
        elif right_close:
            return (((-V_i + V_im1) / dx**2) + F/epsilon*(c_iNa - c_iCl)) * (epsilon/F)/dt

    def d_residuals_V_dNa_i(c_iNa, c_iCl, V_i, V_ip1, V_im1, i, dt):
        return (1.0 / dt) * c_iNa

    def d_residuals_V_dCl_i(c_iNa, c_iCl, V_i, V_ip1, V_im1, i, dt):
        return (-1.0 / dt) * c_iCl

    def d_residuals_V_dV_i(c_iNa, c_iCl, V_i, V_ip1, V_im1, i, dx, dt, left_close, right_close):
        if not left_close and not right_close:
            return -2.0 / dx**2 * (epsilon/F)/dt
        elif left_close:
            return -1.0 / dx**2 * (epsilon/F)/dt
        elif right_close:
            return -1.0 / dx**2 * (epsilon/F)/dt

    def d_residuals_V_dV_im1(c_iNa, c_iCl, V_i, V_ip1, V_im1, i, dx, dt, left_close, right_close):
        if not left_close and not right_close:
            return 1.0/ dx**2 * (epsilon/F)/dt
        elif left_close:
            return 0.0
        elif right_close:
            return 1.0/ dx**2 * (epsilon/F)/dt

    def d_residuals_V_dV_ip1(c_iNa, c_iCl, V_i, V_ip1, V_im1, i, dx, dt, left_close, right_close):
        if not left_close and not right_close:
            return 1.0/ dx**2 * (epsilon/F)/dt
        elif left_close:
            return 1.0/ dx**2 * (epsilon/F)/dt
        elif right_close:
            return 0.0
else:
    def residuals_V(c_iNa, c_iCl, V_i, V_ip1, V_im1, i, dx, dt, left_close, right_close):
        return F/epsilon*(c_iNa - c_iCl) * (epsilon/F)/dt # simplify F/epsilon * epsilon/F

    def d_residuals_V_dNa_i(c_iNa, c_iCl, V_i, V_ip1, V_im1, i, dt):
        return (1.0 / dt) * c_iNa

    def d_residuals_V_dCl_i(c_iNa, c_iCl, V_i, V_ip1, V_im1, i, dt):
        return (-1.0 / dt) * c_iCl

    def d_residuals_V_dV_i(c_iNa, c_iCl, V_i, V_ip1, V_im1, i, dx, dt, left_close, right_close):
        return 0.0

    def d_residuals_V_dV_im1(c_iNa, c_iCl, V_i, V_ip1, V_im1, i, dx, dt, left_close, right_close):
        return 0.0

    def d_residuals_V_dV_ip1(c_iNa, c_iCl, V_i, V_ip1, V_im1, i, dx, dt, left_close, right_close):
        return 0.0

# ----------------------------
# Assemble Residual
# ----------------------------
def AssembleResidualFV(U_new, U_old, dx_val, dt, BCs, N_local):
    R_local = np.zeros(3 * N_local)
    T_local = np.zeros((3 * N_local, 3 * N_local))

    pin_idx = 0 if BCs != 'C' else 1
    U_new[index(pin_idx, 'V')] = 0.0

    for i in range(1, N_local - 1):
        left_close = (i == 1 and BCs == 'C')
        right_close = (i == N_local - 2 and BCs == 'C')

        V_i = U_new[index(i, 'V')]
        V_ip1 = U_new[index(i + 1, 'V')]
        V_im1 = U_new[index(i - 1, 'V')]
        tol = 1e-8

        # Na
        c_Na_old = U_old[index(i, 'Na')]
        c_Na_i = max(U_new[index(i, 'Na')], 1e-12)
        c_Na_ip1 = max(U_new[index(i + 1, 'Na')], 1e-12)
        c_Na_im1 = max(U_new[index(i - 1, 'Na')], 1e-12)

        R_local[index(i, 'Na')] = residuals_C(i, c_Na_old, c_Na_i, c_Na_ip1, c_Na_im1,
                                       V_i, V_ip1, V_im1, z_Na, D_Na, dx_val, tol, dt, left_close,
                                       right_close, N_local)

        # Cl
        c_Cl_old = U_old[index(i, 'Cl')]
        c_Cl_i = max(U_new[index(i, 'Cl')], 1e-12)
        c_Cl_ip1 = max(U_new[index(i + 1, 'Cl')], 1e-12)
        c_Cl_im1 = max(U_new[index(i - 1, 'Cl')], 1e-12)

        R_local[index(i, 'Cl')] = residuals_C(i, c_Cl_old, c_Cl_i, c_Cl_ip1, c_Cl_im1,
                                       V_i, V_ip1, V_im1, z_Cl, D_Cl, dx_val, tol, dt, left_close,
                                       right_close, N_local)

        # Potential equation
        R_local[index(i, 'V')] = residuals_V(c_Na_i, c_Cl_i, V_i, V_ip1, V_im1, i, dx_val, dt, left_close, right_close)

        # Na eq (Jacobian entries)
        r_na = index(i, 'Na')
        T_local[r_na, index(i, 'Na')] = d_residuals_C_dc_i(i, c_Na_old, c_Na_i, c_Na_ip1, c_Na_im1,
                                                     V_i, V_ip1, V_im1, z_Na, D_Na, dx_val, tol, dt,
                                                     left_close, right_close, N_local)
        T_local[r_na, index(i, 'Cl')] = 0.0
        T_local[r_na, index(i, 'V')] = d_residuals_C_dV_i(i, c_Na_old, c_Na_i, c_Na_ip1, c_Na_im1,
                                                   V_i, V_ip1, V_im1, z_Na, D_Na, dx_val, tol, dt,
                                                   left_close, right_close, N_local)

        T_local[r_na, index(i + 1, 'Na')] = d_residuals_C_dc_ip1(i, c_Na_old, c_Na_i, c_Na_ip1, c_Na_im1,
                                                          V_i, V_ip1, V_im1, z_Na, D_Na, dx_val, tol, dt,
                                                          left_close, right_close, N_local)
        T_local[r_na, index(i + 1, 'Cl')] = 0.0
        T_local[r_na, index(i + 1, 'V')] = d_residuals_C_dV_ip1(i, c_Na_old, c_Na_i, c_Na_ip1, c_Na_im1,
                                                          V_i, V_ip1, V_im1, z_Na, D_Na, dx_val, tol, dt,
                                                          left_close, right_close, N_local)

        T_local[r_na, index(i - 1, 'Na')] = d_residuals_C_dc_im1(i, c_Na_old, c_Na_i, c_Na_ip1, c_Na_im1,
                                                          V_i, V_ip1, V_im1, z_Na, D_Na, dx_val, tol, dt,
                                                          left_close, right_close, N_local)
        T_local[r_na, index(i - 1, 'Cl')] = 0.0
        T_local[r_na, index(i - 1, 'V')] = d_residuals_C_dV_im1(i, c_Na_old, c_Na_i, c_Na_ip1, c_Na_im1,
                                                          V_i, V_ip1, V_im1, z_Na, D_Na, dx_val, tol, dt,
                                                          left_close, right_close, N_local)

        # Cl eq
        r_Cl = index(i, 'Cl')
        T_local[r_Cl, index(i, 'Cl')] = d_residuals_C_dc_i(i, c_Cl_old, c_Cl_i, c_Cl_ip1, c_Cl_im1,
                                                     V_i, V_ip1, V_im1, z_Cl, D_Cl, dx_val, tol, dt,
                                                     left_close, right_close, N_local)
        T_local[r_Cl, index(i, 'Na')] = 0.0
        T_local[r_Cl, index(i, 'V')] = d_residuals_C_dV_i(i, c_Cl_old, c_Cl_i, c_Cl_ip1, c_Cl_im1,
                                                   V_i, V_ip1, V_im1, z_Cl, D_Cl, dx_val, tol, dt,
                                                   left_close, right_close, N_local)

        T_local[r_Cl, index(i + 1, 'Cl')] = d_residuals_C_dc_ip1(i, c_Cl_old, c_Cl_i, c_Cl_ip1, c_Cl_im1,
                                                           V_i, V_ip1, V_im1, z_Cl, D_Cl, dx_val, tol, dt,
                                                           left_close, right_close, N_local)
        T_local[r_Cl, index(i + 1, 'Na')] = 0.0
        T_local[r_Cl, index(i + 1, 'V')] = d_residuals_C_dV_ip1(i, c_Cl_old, c_Cl_i, c_Cl_ip1, c_Cl_im1,
                                                          V_i, V_ip1, V_im1, z_Cl, D_Cl, dx_val, tol, dt,
                                                          left_close, right_close, N_local)

        T_local[r_Cl, index(i - 1, 'Cl')] = d_residuals_C_dc_im1(i, c_Cl_old, c_Cl_i, c_Cl_ip1, c_Cl_im1,
                                                           V_i, V_ip1, V_im1, z_Cl, D_Cl, dx_val, tol, dt,
                                                           left_close, right_close, N_local)
        T_local[r_Cl, index(i - 1, 'Na')] = 0.0
        T_local[r_Cl, index(i - 1, 'V')] = d_residuals_C_dV_im1(i, c_Cl_old, c_Cl_i, c_Cl_ip1, c_Cl_im1,
                                                          V_i, V_ip1, V_im1, z_Cl, D_Cl, dx_val, tol, dt,
                                                          left_close, right_close, N_local)

        # Potential row
        r_v = index(i, 'V')
        T_local[r_v, index(i, 'V')] = d_residuals_V_dV_i(c_Na_i, c_Cl_i, V_i, V_ip1, V_im1, i, dx_val, dt, left_close, right_close)
        T_local[r_v, index(i, 'Na')] = d_residuals_V_dNa_i(c_Na_i, c_Cl_i, V_i, V_ip1, V_im1, i, dt)
        T_local[r_v, index(i, 'Cl')] = d_residuals_V_dCl_i(c_Na_i, c_Cl_i, V_i, V_ip1, V_im1, i, dt)

        T_local[r_v, index(i + 1, 'V')] = d_residuals_V_dV_ip1(c_Na_i, c_Cl_i, V_i, V_ip1, V_im1, i, dx_val, dt, left_close, right_close)
        T_local[r_v, index(i + 1, 'Na')] = 0.0
        T_local[r_v, index(i + 1, 'Cl')] = 0.0

        T_local[r_v, index(i - 1, 'V')] = d_residuals_V_dV_im1(c_Na_i, c_Cl_i, V_i, V_ip1, V_im1, i, dx_val, dt, left_close, right_close)
        T_local[r_v, index(i - 1, 'Na')] = 0.0
        T_local[r_v, index(i - 1, 'Cl')] = 0.0

    # pin the potential value to 0 in the chosen pin cell (Dirichlet)
    r_v = index(pin_idx, 'V')
    R_local[r_v] = U_new[index(pin_idx, 'V')]
    T_local[r_v, :] = 0.0
    T_local[r_v, r_v] = 1.0

    # --- Left boundary: Dirichlet (fixed concentrations)
    r = index(0, 'Na')
    R_local[r] = U_new[r] - c_init
    T_local[r, :] = 0.0
    T_local[r, r] = 1.0

    r = index(0, 'Cl')
    R_local[r] = U_new[r] - c_init
    T_local[r, :] = 0.0
    T_local[r, r] = 1.0

    if BCs == 'C':
        r = index(0, 'V')
        R_local[r] = U_new[r]
        T_local[r, :] = 0.0
        T_local[r, r] = 1.0

    # --- Right boundary
    r = index(N_local - 1, 'Na')
    R_local[r] = U_new[r] - 1e-3
    T_local[r, :] = 0.0
    T_local[r, r] = 1.0

    r = index(N_local - 1, 'Cl')
    R_local[r] = U_new[r] - 1e-3
    T_local[r, :] = 0.0
    T_local[r, r] = 1.0

    r = index(N_local - 1, 'V')
    if BCs != 'O':
        R_local[r] = U_new[r]
        T_local[r, :] = 0.0
        T_local[r, r] = 1.0
    else:
        i = N_local - 1
        r_v = index(i, 'V')
        left_close = False
        right_close = True

        V_i = U_new[index(i, 'V')]
        V_ip1 = 0.0
        V_im1 = U_new[index(i - 1, 'V')]
        tol = 1e-8

        # Na
        c_Na_old = U_old[index(i, 'Na')]
        c_Na_i = max(U_new[index(i, 'Na')], 1e-12)
        c_Na_ip1 = 0.0
        c_Na_im1 = max(U_new[index(i - 1, 'Na')], 1e-12)

        R_local[index(i, 'Na')] = residuals_C(i, c_Na_old, c_Na_i, c_Na_ip1, c_Na_im1,
                                       V_i, V_ip1, V_im1, z_Na, D_Na, dx_val, tol, dt, left_close,
                                       right_close, N_local)

        # Cl
        c_Cl_old = U_old[index(i, 'Cl')]
        c_Cl_i = max(U_new[index(i, 'Cl')], 1e-12)
        c_Cl_ip1 = 0.0
        c_Cl_im1 = max(U_new[index(i - 1, 'Cl')], 1e-12)

        R_local[index(i, 'Cl')] = residuals_C(i, c_Cl_old, c_Cl_i, c_Cl_ip1, c_Cl_im1,
                                       V_i, V_ip1, V_im1, z_Cl, D_Cl, dx_val, tol, dt, left_close,
                                       right_close, N_local)

        R_local[index(i, 'V')] = residuals_V(c_Na_i, c_Cl_i, V_i, V_ip1, V_im1, i, dx_val, dt, left_close, right_close)

        # Na eq (Jacobian entries)
        r_na = index(i, 'Na')
        T_local[r_na, index(i, 'Na')] = d_residuals_C_dc_i(i, c_Na_old, c_Na_i, c_Na_ip1, c_Na_im1,
                                                     V_i, V_ip1, V_im1, z_Na, D_Na, dx_val, tol, dt,
                                                     left_close, right_close, N_local)
        T_local[r_na, index(i, 'Cl')] = 0.0
        T_local[r_na, index(i, 'V')] = d_residuals_C_dV_i(i, c_Na_old, c_Na_i, c_Na_ip1, c_Na_im1,
                                                   V_i, V_ip1, V_im1, z_Na, D_Na, dx_val, tol, dt,
                                                   left_close, right_close, N_local)

        T_local[r_na, index(i - 1, 'Na')] = d_residuals_C_dc_im1(i, c_Na_old, c_Na_i, c_Na_ip1, c_Na_im1,
                                                          V_i, V_ip1, V_im1, z_Na, D_Na, dx_val, tol, dt,
                                                          left_close, right_close, N_local)
        T_local[r_na, index(i - 1, 'Cl')] = 0.0
        T_local[r_na, index(i - 1, 'V')] = d_residuals_C_dV_im1(i, c_Na_old, c_Na_i, c_Na_ip1, c_Na_im1,
                                                          V_i, V_ip1, V_im1, z_Na, D_Na, dx_val, tol, dt,
                                                          left_close, right_close, N_local)

        # Cl eq
        r_Cl = index(i, 'Cl')
        T_local[r_Cl, index(i, 'Cl')] = d_residuals_C_dc_i(i, c_Cl_old, c_Cl_i, c_Cl_ip1, c_Cl_im1,
                                                     V_i, V_ip1, V_im1, z_Cl, D_Cl, dx_val, tol, dt,
                                                     left_close, right_close, N_local)
        T_local[r_Cl, index(i, 'Na')] = 0.0
        T_local[r_Cl, index(i, 'V')] = d_residuals_C_dV_i(i, c_Cl_old, c_Cl_i, c_Cl_ip1, c_Cl_im1,
                                                   V_i, V_ip1, V_im1, z_Cl, D_Cl, dx_val, tol, dt,
                                                   left_close, right_close, N_local)

        T_local[r_Cl, index(i - 1, 'Cl')] = d_residuals_C_dc_im1(i, c_Cl_old, c_Cl_i, c_Cl_ip1, c_Cl_im1,
                                                           V_i, V_ip1, V_im1, z_Cl, D_Cl, dx_val, tol, dt,
                                                           left_close, right_close, N_local)
        T_local[r_Cl, index(i - 1, 'Na')] = 0.0
        T_local[r_Cl, index(i - 1, 'V')] = d_residuals_C_dV_im1(i, c_Cl_old, c_Cl_i, c_Cl_ip1, c_Cl_im1,
                                                          V_i, V_ip1, V_im1, z_Cl, D_Cl, dx_val, tol, dt,
                                                          left_close, right_close, N_local)

        # Potential row
        r_v = index(i, 'V')
        T_local[r_v, index(i, 'V')] = d_residuals_V_dV_i(c_Na_i, c_Cl_i, V_i, V_ip1, V_im1, i, dx_val, dt, left_close, right_close)
        T_local[r_v, index(i, 'Na')] = d_residuals_V_dNa_i(c_Na_i, c_Cl_i, V_i, V_ip1, V_im1, i, dt)
        T_local[r_v, index(i, 'Cl')] = d_residuals_V_dCl_i(c_Na_i, c_Cl_i, V_i, V_ip1, V_im1, i, dt)

        T_local[r_v, index(i - 1, 'V')] = d_residuals_V_dV_im1(c_Na_i, c_Cl_i, V_i, V_ip1, V_im1, i, dx_val, dt, left_close, right_close)
        T_local[r_v, index(i - 1, 'Na')] = 0.0
        T_local[r_v, index(i - 1, 'Cl')] = 0.0

    return R_local, T_local

# ----------------------------
# Newton–Raphson Solver returns 
# ----------------------------
def NewtonRaphsonSolver_FV(U_old, dx_val, dt, N_local):
    U_new = U_old.copy()
    max_iter = 200
    pin_idx = 0 if BCs != 'C' else 1

    for it in range(1, max_iter + 1):
        convergenceNa = True
        convergenceCl = True
        convergenceV = True

        R_local, J_local = AssembleResidualFV(U_new, U_old, dx_val, dt, BCs, N_local)

        if not np.all(np.isfinite(R_local)) or not np.all(np.isfinite(J_local)):
            raise RuntimeError("Non-finite entries detected in residual or Jacobian during Newton.")

        try:
            dU = np.linalg.solve(J_local, -R_local)
        except np.linalg.LinAlgError as e:
            raise RuntimeError(f"Linear solve failed in Newton: {e}")

        for i in range(0, N_local):
            if abs(dU[index(i, 'Na')]) > 3:
                dU[index(i, 'Na')] = 3 * np.sign(dU[index(i, 'Na')])
                convergenceNa = False

            if abs(dU[index(i, 'Cl')]) > 3:
                dU[index(i, 'Cl')] = 3 * np.sign(dU[index(i, 'Cl')])
                convergenceCl = False

            dvmax = 0.1
            if abs(dU[index(i, 'V')]) > dvmax:
                dU[index(i, 'V')] = dvmax * np.sign(dU[index(i, 'V')])
                convergenceV = False

            U_new[index(i, 'Na')] *= np.exp(dU[index(i, 'Na')])
            U_new[index(i, 'Cl')] *= np.exp(dU[index(i, 'Cl')])
            U_new[index(i, 'V')] += dU[index(i, 'V')]

            if i == pin_idx:
                U_new[index(i, 'V')] = 0.0

            if not BCs == 'C':
                U_new[index(0, 'Na')] = c_init
                U_new[index(0, 'Cl')] = c_init

            # Convergence criteria
            if abs(R_local[index(i, 'Na')] * dt) > 1e-10:
                convergenceNa = False

            if abs(R_local[index(i, 'Cl')] * dt) > 1e-10:
                convergenceCl = False

            if abs(R_local[index(i, 'V')] * dt) > 1e-10:
                convergenceV = False

        if convergenceNa and convergenceCl and convergenceV and it > 3:
            if BCs == 'O':
                U_new[index(N_local-1, 'Na')] = 1.0e-3
                U_new[index(N_local-1, 'Cl')] = 1.0e-3
            return U_new, it

    raise RuntimeError(f"Newton failed in {max_iter} iters (‖R‖={np.linalg.norm(R_local):.2e})")

# ----------------------------
# Transient Solver with adaptive dt and tqdm progress bar
# ----------------------------
def TransientSolver_FV(U_initial, T_end, dt_init, dt_max, dx_val, N_local, progress_callback=None):
    t = 0.0
    U = U_initial.copy()
    sol = {}
    step = 0
    dt = dt_init
    dt_min = dt_init

    pbar = tqdm(total=T_end, desc="Propagation Temporelle", unit="s")

    while t < T_end:
        dt = min(dt, T_end - t)
        step += 1

        try:
            U_new, its = NewtonRaphsonSolver_FV(U, dx_val, dt, N_local)
        except RuntimeError:
            dt = max(dt/10.0, dt_min)
            U_new, its = NewtonRaphsonSolver_FV(U, dx_val, dt, N_local)

        if progress_callback is not None:
            progress_callback(step, dt, t + dt)
        sol[t + dt] = U_new.copy()
        U = U_new

        pbar.update(dt)
        t += dt

        if progress_callback is not None:
            progress_callback(step, dt, t)

        if its <= 6:
            dt = min(dt*1.5, dt_max)
        elif its > 15:
            dt = max(dt/10, dt_min)
        else:
            dt += 0.0

    pbar.close()
    return sol

# ----------------------------
# Generate initial condition
# ----------------------------
def generate_initial_condition(N_local, BCs_local):
    U0 = np.zeros(3*N_local)
    if BCs_local == 'C':
        half = (N_local - 2) // 2
        for i in range(0, N_local):
            if i <= half:
                U0[index(i, 'Na')] = c_init
                U0[index(i, 'Cl')] = c_init
            else:
                U0[index(i, 'Na')] = 1.0e-3
                U0[index(i, 'Cl')] = 1.0e-3
    else:
        U0[index(0, 'Na')] = c_init
        U0[index(0, 'Cl')] = c_init
        U0[index(N_local-1, 'Na')] = 1.0e-3
        U0[index(N_local-1, 'Cl')] = 1.0e-3
        for i in range(1, N_local-1):
            U0[index(i,'Na')] = 1.0e-3
            U0[index(i,'Cl')] = 1.0e-3
    for i in range(0, N_local):
        U0[index(i, 'V')] = 0.0
    return U0

# ----------------------------
# Planck–Henderson potential (diffusion-coefficients based)
# ----------------------------
def planck_henderson_potential(c_left, c_right, D_plus=D_Na, D_minus=D_Cl, T=T_const):
    cL = max(c_left, spf)
    cR = max(c_right, spf)
    denom = (D_plus + D_minus)
    if denom <= 0.0:
        factor = 0.0
    else:
        factor = (D_minus - D_plus) / denom
    return factor * (R_const * T / F) * np.log(cR / cL)

# ----------------------------
# Generate_matrix():
# ----------------------------
def generate_matrix(dx_values):
    results = {}

    for j, dx_val in enumerate(dx_values):
        # Dynamic number of cells for 1mm domain: N_local = L_domain / dx_val
        # Add + 2 cells to handle boundary nodes appropriately
        N_local = int(round(L_domain / dx_val)) + 2

        T_end_dx = T_end_ref

        U0 = generate_initial_condition(N_local, BCs)
        sol = TransientSolver_FV(U0, T_end_dx, dt_init, dt_max, dx_val, N_local, progress_callback=None)
        Ufinal = sol[max(sol.keys())]

        # Fill concentration / potential arrays
        Na_arr = np.zeros(N_local)
        Cl_arr = np.zeros(N_local)
        V_arr = np.zeros(N_local)

        for i in range(N_local):
            Na_arr[i] = Ufinal[index(i, 'Na')]
            Cl_arr[i] = Ufinal[index(i, 'Cl')]
            V_arr[i] = Ufinal[index(i, 'V')]

        # Compute Planck-Henderson potentials
        c_salt = 0.5 * (Na_arr + Cl_arr)
        E_interface = np.zeros(N_local - 1)
        for jj in range(0, N_local - 1):
            cL = max(c_salt[jj], spf)
            cR = max(c_salt[jj+1], spf)
            E_interface[jj] = planck_henderson_potential(cL, cR, D_plus=D_Na, D_minus=D_Cl, T=T_const)

        pin_idx = 0 if BCs != 'C' else 1
        phi_ph = np.zeros(N_local)
        phi_ph[pin_idx] = 0.0
        for jj in range(pin_idx, N_local - 1):
            phi_ph[jj+1] = phi_ph[jj] + E_interface[jj]
        for jj in range(pin_idx - 1, -1, -1):
            phi_ph[jj] = phi_ph[jj+1] - E_interface[jj]

        results[dx_val] = {'Na': Na_arr, 'Cl': Cl_arr, 'V': V_arr, 'PH': phi_ph, 'N': N_local}

    return results

# ----------------------------
# Save table:
# ----------------------------
def save_table(filename, results, dx_values, fmt="{:.8e}", sep_between_groups="||", BCs=None):
    if BCs is None:
        BCs = globals().get('BCs', 'unknown')

    M = len(dx_values)
    max_N = max(results[dx]['N'] for dx in dx_values)

    # pre-format all numbers
    formatted = {}
    for j, dx_val in enumerate(dx_values):
        res = results[dx_val]
        formatted[dx_val] = {
            'Na': [fmt.format(x) for x in res['Na']],
            'Cl': [fmt.format(x) for x in res['Cl']],
            'V' : [fmt.format(x) for x in res['V']],
            'PH': [fmt.format(x) for x in res['PH']],
        }

    # compute column widths
    sub_w = []
    group_widths = []
    for j, dx_val in enumerate(dx_values):
        w_na = max(len("Na"), max(len(s) for s in formatted[dx_val]['Na']))
        w_cl = max(len("Cl"), max(len(s) for s in formatted[dx_val]['Cl']))
        w_v  = max(len("V"),  max(len(s) for s in formatted[dx_val]['V']))
        w_ph = max(len("PH"), max(len(s) for s in formatted[dx_val]['PH']))
        sub_w.append((w_na, w_cl, w_v, w_ph))
        group_widths.append(w_na + w_cl + w_v + w_ph + 3)

    cell_w = max(len("cell"), len(str(max_N-1)))
    total_width = cell_w + 3 + sum(group_widths) + (len(sep_between_groups) + 2) * (M - 1)

    with open(filename, "w") as f:
        top_line_parts = [f"BCs={BCs}  " + f"{'cell'.ljust(cell_w)}  "]
        for j, d in enumerate(dx_values):
            gw = group_widths[j]
            label = f"dx={d:.6e}"
            top_line_parts.append(label.center(gw))
            if j < M - 1:
                top_line_parts.append(f" {sep_between_groups} ")
        f.write("".join(top_line_parts) + "\n")

        subh_parts = [f"{''.ljust(cell_w)}  "]
        for j in range(M):
            w_na, w_cl, w_v, w_ph = sub_w[j]
            subh_parts.append(f"{ 'Na'.rjust(w_na)} {'Cl'.rjust(w_cl)} {'V'.rjust(w_v)} {'PH'.rjust(w_ph)}")
            if j < M - 1:
                subh_parts.append(f" {sep_between_groups} ")
        f.write("".join(subh_parts) + "\n")

        f.write("-" * min(total_width, 200) + "\n")

        # Data rows (one per cell index up to max_N)
        for i in range(max_N):
            row_parts = [f"{str(i).rjust(cell_w)}  "]
            for j, dx_val in enumerate(dx_values):
                w_na, w_cl, w_v, w_ph = sub_w[j]
                N_run = results[dx_val]['N']
                if i < N_run:
                    s_na = formatted[dx_val]['Na'][i].rjust(w_na)
                    s_cl = formatted[dx_val]['Cl'][i].rjust(w_cl)
                    s_v  = formatted[dx_val]['V'][i].rjust(w_v)
                    s_ph = formatted[dx_val]['PH'][i].rjust(w_ph)
                else:
                    s_na = "".rjust(w_na)
                    s_cl = "".rjust(w_cl)
                    s_v  = "".rjust(w_v)
                    s_ph = "".rjust(w_ph)
                row_parts.append(f"{s_na} {s_cl} {s_v} {s_ph}")
                if j < M - 1:
                    row_parts.append(f" {sep_between_groups} ")
            f.write("".join(row_parts) + "\n")

    print(f"Saved table: {filename} (max rows={max_N}, dx cols={M}, BCs={BCs})")

# ----------------------------
# generate_and_save_fields
# ----------------------------
def generate_and_save_fields():
    results = generate_matrix(dxs)
    save_table(r"..\file\results_dx10_bcOpnpON.txt", results, dxs, BCs=BCs)

if __name__ == "__main__":
    generate_and_save_fields()

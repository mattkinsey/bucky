from ..numerical_libs import reimport_numerical_libs, xp

#
# RHS for odes - d(sstate)/dt = F(t, state, *mats, *pars)
# NB: requires the state vector be 1d
#


def RHS_func(t, y_flat, mc_inst):
    """RHS function for the ODEs, get's called in ivp.solve_ivp"""

    reimport_numerical_libs("RHS_func")

    # constraint on values
    lower, upper = (0.0, 1.0)  # bounds for state vars

    # grab index of OOB values so we can zero derivatives (stability...)
    too_low = y_flat <= lower
    too_high = y_flat >= upper

    # TODO we're passing in y.state just to overwrite it, we probably need another class
    # reshape to the usual state tensor (compartment, age, node)
    y = mc_inst.state
    y.state = y_flat.reshape(y.state_shape)

    # Clip state to be in bounds
    y.state = xp.clip(y.state, a_min=lower, a_max=upper, out=y.state)

    # init d(state)/dt
    dy = mc_inst.dy

    if mc_inst.npi_active or mc_inst.vacc_active:
        t_index = min(int(t), mc_inst.t_max)  # prevent OOB error when the integrator overshoots
    else:
        t_index = None

    # TODO add a function to mc instance that fills all these in using nonlocal?
    npi = mc_inst.npi_params
    par = mc_inst.epi_params
    BETA_eff = mc_inst.BETA_eff(t_index)
    if hasattr(mc_inst, "scen_beta_scale"):
        BETA_eff = mc_inst.scen_beta_scale[t_index] * BETA_eff

    F_eff = par["F_eff"]
    HOSP = par["H"]
    THETA = y.Rhn * par["THETA"]
    GAMMA = y.Im * par["GAMMA"]
    GAMMA_H = y.Im * par["GAMMA_H"]
    SIGMA = y.En * par["SIGMA"]
    SYM_FRAC = par["SYM_FRAC"]
    CASE_REPORT = par["CASE_REPORT"]

    Cij = mc_inst.Cij(t_index)
    Aij = mc_inst.Aij  # TODO needs to take t and return Aij_eff

    if mc_inst.npi_active:
        Aij_eff = npi["mobility_reduct"][t_index] * Aij
    else:
        Aij_eff = Aij

    S_eff = mc_inst.S_eff(t_index, y)

    Nij = mc_inst.Nij

    # perturb Aij
    # new_R0_fracij = truncnorm(xp, 1.0, .1, size=Aij.shape, a_min=1e-6)
    # new_R0_fracij = xp.clip(new_R0_fracij, 1e-6, None)
    # A = Aij * new_R0_fracij
    # Aij_eff = A / xp.sum(A, axis=0)

    # Infectivity matrix (I made this name up, idk what its really called)
    I_tot = xp.sum(Nij * y.Itot, axis=0) - (1.0 - par["rel_inf_asym"]) * xp.sum(Nij * y.Ia, axis=0)

    I_tmp = I_tot @ Aij  # using identity (A@B).T = B.T @ A.T

    # beta_mat = y.S * xp.squeeze((Cij @ I_tmp.T[..., None]), axis=-1).T
    beta_mat = S_eff * (Cij @ xp.atleast_3d(I_tmp.T)).T[0]
    beta_mat /= Nij

    # dS/dt
    dy.S = -BETA_eff * (beta_mat)
    # dE/dt
    dy.E[0] = BETA_eff * (beta_mat) - SIGMA * y.E[0]
    dy.E[1:] = SIGMA * (y.E[:-1] - y.E[1:])

    # dI/dt
    dy.Ia[0] = (1.0 - SYM_FRAC) * SIGMA * y.E[-1] - GAMMA * y.Ia[0]
    dy.Ia[1:] = GAMMA * (y.Ia[:-1] - y.Ia[1:])

    # dIa/dt
    dy.I[0] = SYM_FRAC * (1.0 - HOSP) * SIGMA * y.E[-1] - GAMMA * y.I[0]
    dy.I[1:] = GAMMA * (y.I[:-1] - y.I[1:])

    # dIc/dt
    dy.Ic[0] = SYM_FRAC * HOSP * SIGMA * y.E[-1] - GAMMA_H * y.Ic[0]
    dy.Ic[1:] = GAMMA_H * (y.Ic[:-1] - y.Ic[1:])

    # dRhi/dt
    dy.Rh[0] = GAMMA_H * y.Ic[-1] - THETA * y.Rh[0]
    dy.Rh[1:] = THETA * (y.Rh[:-1] - y.Rh[1:])

    # dR/dt
    dy.R = GAMMA * (y.I[-1] + y.Ia[-1]) + (1.0 - F_eff) * THETA * y.Rh[-1]

    # dD/dt
    dy.D = F_eff * THETA * y.Rh[-1]

    dy.incH = GAMMA_H * y.Ic[-1]  # SYM_FRAC * HOSP * SIGMA * y.E[-1]
    dy.incC = SYM_FRAC * CASE_REPORT * SIGMA * y.E[-1]

    # bring back to 1d for the ODE api
    dy_flat = dy.state.ravel()

    # zero derivatives for things we had to clip if they are going further out of bounds
    dy_flat = xp.where(too_low & (dy_flat < 0.0), 0.0, dy_flat)
    dy_flat = xp.where(too_high & (dy_flat > 0.0), 0.0, dy_flat)

    return dy_flat

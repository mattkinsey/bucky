"""Submodule that manages some of the calculations for estimating params from the historical data"""

from ..numerical_libs import sync_numerical_libs, xp


@sync_numerical_libs
def estimate_Rt(
    g_data,
    params,
    days_back=7,
    case_reporting=None,
    use_geo_mean=False,
):
    """Estimate R_t from the recent case data"""

    rolling_case_hist = g_data.rolling_inc_cases / params["CASE_REPORT"]

    rolling_case_hist = g_data.rolling_inc_cases[-case_reporting.shape[0] :] / case_reporting

    rolling_case_hist = xp.clip(rolling_case_hist, a_min=0.0, a_max=None)

    tot_case_hist = (g_data.Aij.A.T @ rolling_case_hist.T).T + 1.0  # to avoid weirdness with small numbers

    t_max = rolling_case_hist.shape[0]
    k = params.consts["En"]

    mean = params["Ts"]
    theta = mean / k
    x = xp.arange(0.0, t_max)

    w = 1.0 / (xp.special.gamma(k) * theta ** k) * x ** (k - 1) * xp.exp(-x / theta)
    w = w / (1.0 - w)
    w = w[::-1]
    # adm0
    rolling_case_hist_adm0 = xp.nansum(rolling_case_hist, axis=1)[:, None]
    tot_case_hist_adm0 = xp.nansum(tot_case_hist, axis=1)[:, None]

    n_loc = rolling_case_hist_adm0.shape[1]
    Rt = xp.empty((days_back, n_loc))
    for i in range(days_back):  # TODO we can vectorize by convolving w over case hist
        d = i + 1
        Rt[i] = rolling_case_hist_adm0[-d] / (xp.sum(w[d:, None] * tot_case_hist_adm0[:-d], axis=0))

    # Take harmonic mean
    Rt[~(Rt > 0.0)] = xp.nan
    Rt = 1.0 / xp.nanmean(1.0 / Rt, axis=0)

    Rt_out = xp.full((rolling_case_hist.shape[1],), Rt)

    # adm1
    tot_case_hist_adm1 = g_data.sum_adm1(tot_case_hist.T).T
    rolling_case_hist_adm1 = g_data.sum_adm1(rolling_case_hist.T).T

    n_loc = rolling_case_hist_adm1.shape[1]
    Rt = xp.empty((days_back, n_loc))
    for i in range(days_back):
        d = i + 1
        Rt[i] = rolling_case_hist_adm1[-d] / (xp.sum(w[d:, None] * tot_case_hist_adm1[:-d], axis=0))

    # take harmonic mean
    Rt[~(Rt > 0.0)] = xp.nan
    Rt = 1.0 / xp.nanmean(1.0 / Rt, axis=0)

    # TODO we should mask this before projecting it to adm2...
    Rt = Rt[g_data.adm1_id]
    valid_mask = xp.isfinite(Rt) & (xp.mean(rolling_case_hist_adm1[-7:], axis=0)[g_data.adm1_id] > 25)
    Rt_out[valid_mask] = Rt[valid_mask]

    # adm2
    n_loc = rolling_case_hist.shape[1]
    Rt = xp.empty((days_back, n_loc))
    for i in range(days_back):
        d = i + 1
        Rt[i] = rolling_case_hist[-d] / (xp.sum(w[d:, None] * tot_case_hist[:-d], axis=0))

    Rt[~(Rt > 0.0)] = xp.nan

    # rt_geo = xp.exp(xp.nanmean(xp.log(Rt), axis=0))
    # rt_mean = xp.nanmean(Rt, axis=0)
    # rt_med = xp.nanmedian(Rt, axis=0)
    rt_harm = 1.0 / xp.nanmean(1.0 / Rt, axis=0)

    Rt = rt_harm  # (rt_geo + rt_med) /2.
    # TODO make this max value a param
    valid_mask = xp.isfinite(Rt) & (xp.mean(rolling_case_hist[-7:], axis=0) > 25) & (Rt > 0.1) & (Rt < 2.5)
    Rt_out[valid_mask] = Rt[valid_mask]
    return Rt_out


@sync_numerical_libs
def estimate_doubling_time(
    g_data,
    days_back=7,  # TODO rename, its the number days calc the rolling Td
    doubling_time_window=7,
    mean_time_window=None,
    min_doubling_t=1.0,
    case_reporting=None,
):
    """Calculate the recent doubling time of the historical case data"""

    if mean_time_window is not None:
        days_back = mean_time_window

    cases = g_data.cum_case_hist[-days_back:] / case_reporting[-days_back:]
    cases_old = (
        g_data.cum_case_hist[-days_back - doubling_time_window : -doubling_time_window]
        / case_reporting[-days_back - doubling_time_window : -doubling_time_window]
    )

    # adm0
    adm0_doubling_t = doubling_time_window / xp.log2(xp.nansum(cases, axis=1) / xp.nansum(cases_old, axis=1))

    """
    if self.debug:
        logging.debug("Adm0 doubling time: " + str(adm0_doubling_t))
    if xp.any(~xp.isfinite(adm0_doubling_t)):
        if self.debug:
            logging.debug(xp.nansum(cases, axis=1))
            logging.debug(xp.nansum(cases_old, axis=1))
        raise SimulationException
    """

    doubling_t = xp.repeat(adm0_doubling_t[:, None], cases.shape[-1], axis=1)

    # adm1
    cases_adm1 = g_data.sum_adm1(cases.T)
    cases_old_adm1 = g_data.sum_adm1(cases_old.T)

    adm1_doubling_t = doubling_time_window / xp.log2(cases_adm1 / cases_old_adm1)

    tmp_doubling_t = adm1_doubling_t[g_data.adm1_id].T
    valid_mask = xp.isfinite(tmp_doubling_t) & (tmp_doubling_t > min_doubling_t)

    doubling_t[valid_mask] = tmp_doubling_t[valid_mask]

    # adm2
    adm2_doubling_t = doubling_time_window / xp.log2(cases / cases_old)

    valid_adm2_dt = xp.isfinite(adm2_doubling_t) & (adm2_doubling_t > min_doubling_t)
    doubling_t[valid_adm2_dt] = adm2_doubling_t[valid_adm2_dt]

    # hist_weights = xp.arange(1., days_back + 1.0, 1.0)
    # hist_doubling_t = xp.sum(doubling_t * hist_weights[:, None], axis=0) / xp.sum(
    #    hist_weights
    # )

    # Take mean of most recent values
    if mean_time_window is not None:
        ret = xp.nanmean(doubling_t[-mean_time_window:], axis=0)
    else:
        ret = doubling_t

    return ret

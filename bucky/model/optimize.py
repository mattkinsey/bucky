import datetime
from pprint import pformat

import numpy as np
import pandas as pd
import yaml

from ..numerical_libs import sync_numerical_libs, xp
from ..util.scoring import WIS

best_opt = xp.inf


def opt_func(params, args):
    """Function y = f(params, args) to be minimized"""
    env, hist_daily_cases, hist_daily_deaths, hist_daily_h, fips_mask = args[0]
    columns = ("daily_reported_cases", "daily_deaths", "daily_hospitalizations")
    if hist_daily_cases.shape[-1] < env.base_mc_instance.t_max:
        env.base_mc_instance.set_tmax(hist_daily_cases.shape[-1])

    new_params = {
        "H_fac": {
            "dist": "approx_mPERT",
            "mu": params[0],
            "a": params[0] - params[1] / 2.0,
            "b": params[0] + params[1] / 2.0,
            "gamma": params[2],
        },
        "Rt_fac": {
            "dist": "approx_mPERT",
            "mu": params[6],
            "a": params[6] - params[7] / 2.0,
            "b": params[6] + params[7] / 2.0,
            "gamma": params[8],
        },
        "E_fac": {
            "dist": "approx_mPERT",
            "mu": params[9],
            "a": params[9] - params[10] / 2.0,
            "b": params[9] + params[10] / 2.0,
            "gamma": params[11],
        },
        "R_fac": {
            "dist": "approx_mPERT",
            "mu": params[13],
            "a": params[13] - params[14] / 2.0,
            "b": params[13] + params[14] / 2.0,
            "gamma": params[15],
        },
        "consts": {
            "F_scaling": params[5],
            "reroll_variance": params[12],
            "rh_scaling": params[4],
            "F_RR_var": params[3],
            # "CI_scaling": params[16], "CI_init_scale": params[17], "CI_scaling_acc": params[18],
        },
    }

    print(params)
    print(pformat(new_params))

    env.update_params(new_params)
    data = env.run_multiple(100, 2, columns)
    if data is None:
        return 1e5
    agg_data = {}
    q = xp.arange(0.05, 1, 0.05)

    ci_scale_fac = (q - 0.5) / xp.max(q - 0.5)
    ci_factor_mo = (
        env.consts.CI_scaling * ci_scale_fac
        + env.consts.CI_scaling_acc * xp.sign(ci_scale_fac) * ci_scale_fac * ci_scale_fac
    )
    ci_factor = xp.repeat(ci_factor_mo[..., None, None], hist_daily_cases.shape[-1], axis=-1)
    ci_factor = env.consts.CI_init_scale + xp.cumsum(ci_factor / 30.0, axis=-1)

    for col in columns:
        tmp = xp.array([env.g_data.sum_adm1(run[col][fips_mask, 1:], mask=fips_mask) for run in data])
        # agg_data[col] = ci_factor * xp.percentile(tmp, 100.0 * q, axis=0)
        agg_data[col] = xp.percentile(tmp, 100.0 * q, axis=0)

    # from IPython import embed
    # embed()
    log = False

    ret_c = WIS(hist_daily_cases.ravel(), q, agg_data["daily_reported_cases"].reshape(len(q), -1), norm=True, log=log)
    ret_d = WIS(
        hist_daily_deaths.ravel(), q, agg_data["daily_deaths"].reshape(len(q), -1), norm=True, log=log, embed=False
    )
    med_ind = agg_data["daily_deaths"].reshape(len(q), -1).shape[0] // 2 + 1
    mse_d = (
        xp.abs(agg_data["daily_deaths"].reshape(len(q), -1)[med_ind] - hist_daily_deaths.ravel())
        / (hist_daily_deaths.ravel() + 1)
    ) ** 2
    mse_c = (
        xp.abs(agg_data["daily_reported_cases"].reshape(len(q), -1)[med_ind] - hist_daily_cases.ravel())
        / (hist_daily_cases.ravel() + 1)
    ) ** 2

    # from IPython import embed
    # embed()

    # handle the fact that the hosp data might be more stale than than csse
    if hist_daily_h.shape[-1] > agg_data["daily_hospitalizations"].shape[-1]:
        ret_h = WIS(
            hist_daily_h[..., : agg_data["daily_hospitalizations"].shape[-1]].ravel(),
            q,
            agg_data["daily_hospitalizations"][..., : hist_daily_h.shape[1]].reshape(len(q), -1),
            norm=True,
            log=log,
        )
        mse_h = (
            xp.abs(
                agg_data["daily_hospitalizations"].reshape(len(q), -1)[med_ind, ..., : hist_daily_h.shape[1]]
                - hist_daily_h.ravel()
            )
            / (hist_daily_h.ravel() + 1)
        ) ** 2
    else:
        ret_h = WIS(
            hist_daily_h.ravel(),
            q,
            agg_data["daily_hospitalizations"][..., : hist_daily_h.shape[1]].reshape(len(q), -1),
            norm=True,
            log=log,
        )
        mse_h = (
            xp.abs(
                agg_data["daily_hospitalizations"][..., : hist_daily_h.shape[1]].reshape(len(q), -1)[med_ind]
                - hist_daily_h[..., : agg_data["daily_hospitalizations"].shape[-1]].ravel()
            )
            / (hist_daily_h.ravel() + 1)
        ) ** 2

    # from IPython import embed
    # embed()
    # normalize for diff historical lengths and get the mean
    ret_c /= hist_daily_cases.shape[-1]
    ret_d /= hist_daily_deaths.shape[-1]
    ret_h /= hist_daily_h.shape[-1]
    mse_c /= hist_daily_cases.shape[-1]
    mse_d /= hist_daily_deaths.shape[-1]
    mse_h /= hist_daily_h.shape[-1]

    ret_c = xp.nansum(ret_c)
    ret_d = xp.nansum(ret_d)
    ret_h = xp.nansum(ret_h)
    mse_c = xp.nansum(mse_c)
    mse_d = xp.nansum(mse_d)
    mse_h = xp.nansum(mse_h)

    # from IPython import embed
    # embed()

    ret_wis = ret_c + ret_d + ret_h
    ret_mse = mse_c + mse_d + mse_h

    ret = ret_mse + ret_wis

    print()
    print(
        pformat(
            {
                "wis": ret_wis,
                "wis_c": ret_c,
                "wis_d": ret_d,
                "wis_h": ret_h,
                "mse": ret_mse,
                "mse_c": mse_c,
                "mse_d": mse_d,
                "mse_h": mse_h,
            }
        )
    )
    print(ret)

    # TODO this global var doesnt work when the function is wrapped in a partial()
    global best_opt
    ret = xp.to_cpu(ret).item()
    # from IPython import embed

    # embed()
    if ret < best_opt:
        best_opt = ret
        # with open('best_opt.yml', 'w') as f:
        # TODO need to recursively convert these from np.float to float for this to work
        #    yaml.safe_dump(new_params, f)

    return ret  # xp.to_cpu(ret).item()


@sync_numerical_libs
def test_opt(env, params=None):
    """Wrapper for calling the optimizer"""

    global best_opt
    best_opt = xp.inf
    hist = pd.read_csv("data/cases/csse_hist_timeseries.csv")
    hist.adm2 = hist.adm2.astype(int)
    hist.date = pd.to_datetime(hist.date)

    # get incident data
    hist.set_index(["adm2", "date"], inplace=True)
    hist = hist.groupby(level=0).diff()
    hist.reset_index(inplace=True)

    rolling = True
    if rolling:
        first_day = env.init_date - datetime.timedelta(days=6)
    else:
        first_day = env.init_date

    # filter hist
    hist = hist.loc[hist.date > pd.to_datetime(first_day)]
    hist = hist.loc[hist.adm2.isin(xp.to_cpu(env.g_data.adm2_id))]
    uniq_hist_adm2 = hist.adm2.unique()

    fips_to_mask = []
    for i in env.g_data.adm2_id:
        if not int(i.item()) in uniq_hist_adm2:
            fips_to_mask.append(int(i.item()))

    hist = hist.set_index(["adm2", "date"])
    # align order with the output arrays
    hist = hist.reindex(env.g_data.adm2_id.get(), level=0)

    hist_daily_cases = xp.array(hist.cumulative_reported_cases.unstack().to_numpy())
    hist_daily_deaths = xp.array(hist.cumulative_deaths.unstack().to_numpy())
    fips_mask = xp.array(~np.isin(xp.to_cpu(env.g_data.adm2_id), np.array(fips_to_mask)))

    hist_daily_cases = env.g_data.sum_adm1(hist_daily_cases, mask=fips_mask)
    hist_daily_deaths = env.g_data.sum_adm1(hist_daily_deaths, mask=fips_mask)

    hist_daily_cases = xp.clip(hist_daily_cases, a_min=0.0, a_max=None)
    hist_daily_deaths = xp.clip(hist_daily_deaths, a_min=0.0, a_max=None)

    hist = pd.read_csv("data/cases/hhs_hosps.csv")
    hist.date = pd.to_datetime(hist.date)
    hist = hist.loc[hist.date > pd.to_datetime(first_day)]
    hist = hist.loc[hist.adm1.isin(xp.to_cpu(env.g_data.adm1_id))]
    hist = hist.set_index(["adm1", "date"])
    hist = hist.sort_index()

    from IPython import embed

    # embed()

    hist_daily_h_df = hist.previous_day_admission_adult_covid_confirmed.unstack()
    hist_daily_h = xp.zeros((hist_daily_h_df.index.max() + 1, len(hist_daily_h_df.columns)))
    hist_daily_h[hist_daily_h_df.index.to_numpy()] = hist_daily_h_df.to_numpy()

    hist_daily_h = xp.clip(hist_daily_h, a_min=0.0, a_max=None)

    # from IPython import embed

    # embed()

    # if rolling:
    #    from ..util.rolling_mean import rolling_mean

    #    hist_daily_cases = rolling_mean(hist_daily_cases, axis=1)
    #    hist_daily_deaths = rolling_mean(hist_daily_deaths, axis=1)
    #    hist_daily_h = rolling_mean(hist_daily_h, axis=1)
    spline = True
    dof = 4
    if spline:
        from ..util.spline_smooth import fit

        hist_daily_cases = xp.clip(fit(hist_daily_cases, df=dof), a_min=0.0, a_max=None)
        hist_daily_deaths = xp.clip(fit(hist_daily_deaths, df=dof), a_min=0.0, a_max=None)
        hist_daily_h = xp.clip(fit(hist_daily_h, df=dof), a_min=0.0, a_max=None)

    hist_daily_cases = xp.clip(hist_daily_cases, a_min=0.0, a_max=None)
    hist_daily_deaths = xp.clip(hist_daily_deaths, a_min=0.0, a_max=None)
    hist_daily_h = xp.clip(hist_daily_h, a_min=0.0, a_max=None)

    from functools import partial

    from scipy.optimize import Bounds, differential_evolution, minimize
    from skopt import callbacks, gp_minimize
    from skopt.callbacks import CheckpointSaver
    from skopt.sampler import Lhs
    from skopt.space import Real

    args = ((env, hist_daily_cases, hist_daily_deaths, hist_daily_h, fips_mask),)

    opt_params = np.array(
        [
            env.bucky_params.base_params["H_fac"]["mu"],
            env.bucky_params.base_params["H_fac"]["b"] - env.bucky_params.base_params["H_fac"]["a"],
            env.bucky_params.base_params["H_fac"]["gamma"],
            env.consts.F_RR_var.get(),
            # env.dists.H_RR_var,
            env.consts.rh_scaling.get(),
            env.consts.F_scaling.get(),
            env.bucky_params.base_params["Rt_fac"]["mu"],
            env.bucky_params.base_params["Rt_fac"]["b"] - env.bucky_params.base_params["Rt_fac"]["a"],
            env.bucky_params.base_params["Rt_fac"]["gamma"],
            env.bucky_params.base_params["E_fac"]["mu"],
            env.bucky_params.base_params["E_fac"]["b"] - env.bucky_params.base_params["E_fac"]["a"],
            env.bucky_params.base_params["E_fac"]["gamma"],
            env.consts.reroll_variance.get(),
            env.bucky_params.base_params["R_fac"]["mu"],
            env.bucky_params.base_params["R_fac"]["b"] - env.bucky_params.base_params["R_fac"]["a"],
            env.bucky_params.base_params["R_fac"]["gamma"],
            # env.consts.CI_scaling.get(),
            # env.consts.CI_init_scale.get(),
            # env.consts.CI_scaling_acc.get(),
        ]
    )

    # space = {
    #    "H_fac_mu": hp.uniform("H_fac_mu", 0.1, 2.0),
    #    "H_fac_w": hp.uniform("H_fac_w", 0.0, 0.99),
    #    "H_fac_gamma": hp.uniform("H_fac_gamma", 0.1, 10.0),
    # }

    dims = [
        Real(0.5, 2.0),
        Real(0.1, 10),
        Real(0.1, 10),
        Real(0.1, 100),
        Real(0.00001, 0.3),
        Real(0.00001, 0.3),
        Real(0.5, 2.0),
        Real(0.5, 2.0),
        Real(0.1, 10),
        Real(0.1, 10),
        Real(0.1, 100),
        Real(0.5, 2.0),
        Real(0.1, 10),
        Real(0.1, 10),
        Real(0.1, 100),
        Real(0.02, 0.4),
    ]
    lhs = Lhs(criterion="maximin", iterations=10000)

    dims = [Real(0.5 * opt_params[i], 1.0 / 0.5 * opt_params[i]) for i in range(len(opt_params))]

    checkpoint_saver = CheckpointSaver("./checkpoint.pkl", compress=9)
    if True:
        for j in range(2):
            fac = 0.5 + 0.25 * j
            dims = [Real(fac * opt_params[i], 1.0 / fac * opt_params[i]) for i in range(len(opt_params))]
            res = gp_minimize(
                partial(opt_func, args=args),
                dimensions=dims,
                x0=opt_params.tolist(),
                initial_point_generator=lhs,
                # callback=[checkpoint_saver],
                n_calls=200,
                verbose=True,
            )
            opt_params = np.array(res.x)
        best_params = np.array(res.x)
    else:
        best_params = opt_params
    # print(res)
    result = minimize(
        opt_func,
        best_params,
        (args,),
        options={"disp": True, "adaptive": True, "maxfev": 1000},
        method="Nelder-Mead",
    )
    # from hyperopt import atpe, fmin

    # best = fmin(fn=partial(opt_func, args=args), space=space, algo=atpe.suggest, max_evals=100)

    # TODO need function that will take the array being optimized and cast it to a dict (what opt_func is doing but available more generally)
    # need to be able to dump that dict to a yaml file
    embed()

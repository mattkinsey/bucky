import datetime
from pprint import pformat

import numpy as np
import pandas as pd
from iPython import embed

from ..numerical_libs import sync_numerical_libs, xp
from ..util.scoring import WIS

# TODO better place for these
columns = ("daily_reported_cases", "daily_deaths", "daily_hospitalizations")

n_mc = 100
base_seed = 2
default_ret = 1e5
percentile_params = (0.05, 1, 0.05)  # min, max, delta
log = False

rolling = False
spline = True
dof = 4

local_opt = True


def ravel_3d(a: xp.ndarray):
    """Ravel each element of a, preserving first dimension"""
    return a.reshape(a.shape[0], -1)


def opt_func(params, args):
    """Function y = f(params, args) to be minimized"""
    # Convert param list to dictionary
    print(params)
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
    print(pformat(new_params))
    # Unroll args
    env, hist_daily_cases, hist_daily_deaths, hist_daily_h, fips_mask = args
    hist_data = {
        "daily_reported_cases": hist_daily_cases,
        "daily_deaths": hist_daily_deaths,
        "daily_hospitalizations": hist_daily_h,
    }
    hist_days = {col: vals.shape[-1] for col, vals in hist_data.items()}
    hist_data = {col: vals.ravel() for col, vals in hist_data.items()}

    # Run model
    env.update_params(new_params)
    data = env.run_multiple(n_mc, base_seed, columns)
    if data is None:
        return default_ret
    model_data = {}

    # Convert array of MC runs into array of percentiles
    q = xp.arange(*percentile_params)
    for col in columns:
        # Roll up to admin 1
        tmp = xp.array([env.g_data.sum_adm1(run[col][fips_mask, 1:], mask=fips_mask) for run in data])
        # Cut down to length of available ground truth data
        hist_num_days = hist_days[col]
        model_data[col] = ravel_3d(xp.percentile(tmp, 100.0 * q, axis=0)[..., :hist_num_days])

    # from IPython import embed
    # embed()

    # WIS
    ret = [WIS(hist_data[col], q, model_data[col], norm=True, log=log) for col in columns]
    # Normalize by number of days
    ret = [ret_i / hist_days[col] for ret_i, col in zip(ret, columns)]
    # Sum over admin 2 and days
    ret = [xp.nansum(ret_i) for ret_i in ret]

    # MSE
    med_ind = q.shape[0] // 2 + 1
    mse = [(xp.abs(model_data[col][med_ind] - hist_data[col]) / (hist_data[col] + 1)) ** 2 for col in columns]
    mse = [mse_i / hist_days[col] for mse_i, col in zip(mse, columns)]
    mse = [xp.nansum(mse_i) for mse_i in mse]

    # Sum over cases, deaths, hosp
    ret_wis = xp.sum(ret)  # ret_c + ret_d + ret_h
    ret_mse = xp.sum(mse)  # mse_c + mse_d + mse_h
    print()
    print(
        pformat(
            {
                "wis": ret_wis,
                **dict(zip(["wis_c", "wis_d", "wis_h"], ret)),
                "mse": ret_mse,
                **dict(zip(["mse_c", "mse_d", "mse_h"], mse)),
            }
        )
    )
    # Sum MSE + WIS
    ret = ret_mse + ret_wis
    print(ret)

    ret = xp.to_cpu(ret).item()
    # from IPython import embed

    # embed()

    return ret  # xp.to_cpu(ret).item()


def case_death_df(first_day: datetime.datetime, adm2_filter: xp.ndarray) -> pd.DataFrame:
    """Load historical case and death data and filter to correct dates/counties"""
    # Case and death data
    hist = pd.read_csv("data/cases/csse_hist_timeseries.csv")
    # Types
    hist.adm2 = hist.adm2.astype(int)
    hist.date = pd.to_datetime(hist.date)

    # get incident data
    hist.set_index(["adm2", "date"], inplace=True)
    hist = hist.groupby(level=0).diff()
    hist.reset_index(inplace=True)
    hist = hist.loc[hist.date > pd.to_datetime(first_day)]
    hist = hist.loc[hist.adm2.isin(adm2_filter)]
    hist = hist.set_index(["adm2", "date"])
    hist = hist.reindex(adm2_filter, level=0)
    return hist


def hosp_df(first_day: datetime.datetime, adm1_filter: xp.ndarray) -> pd.DataFrame:
    """Load historical hospitalization data and filter to correct dates/states"""
    hist = pd.read_csv("data/cases/hhs_hosps.csv")
    hist.date = pd.to_datetime(hist.date)
    hist = hist.loc[hist.date > pd.to_datetime(first_day)]
    hist = hist.loc[hist.adm1.isin(adm1_filter)]
    hist = hist.set_index(["adm1", "date"])
    hist = hist.sort_index()
    return hist


@sync_numerical_libs
def test_opt(env):
    """Wrapper for calling the optimizer"""
    # First day of historical data
    first_day = env.init_date
    if rolling:
        first_day -= datetime.timedelta(days=6)

    # Environment admin2 and admin1 values
    env_adm2 = xp.to_cpu(env.g_data.adm2_id)
    env_adm1 = xp.to_cpu(env.g_data.adm1_id)

    # Get historical case and death data
    hist = case_death_df(first_day, env_adm2)

    # Make sure environment end date is same as amount of available historical data
    days_of_hist_data = (
        hist.index.get_level_values(-1).max()
        - datetime.datetime(env.init_date.year, env.init_date.month, env.init_date.day)
    ).days
    if days_of_hist_data != env.base_mc_instance.t_max:
        env.base_mc_instance.set_tmax(days_of_hist_data)

    # Get environment admin2 mask
    good_fips = hist.index.get_level_values("adm2").unique()
    fips_mask = xp.array(np.isin(env_adm2, good_fips))

    # Extract case and death data from data frame
    hist_daily_cases = xp.array(hist.cumulative_reported_cases.unstack().to_numpy())
    hist_daily_deaths = xp.array(hist.cumulative_deaths.unstack().to_numpy())

    # Sum case and death data to state
    hist_daily_cases = env.g_data.sum_adm1(hist_daily_cases, mask=fips_mask)
    hist_daily_deaths = env.g_data.sum_adm1(hist_daily_deaths, mask=fips_mask)

    # Hosp data
    hist = hosp_df(first_day, env_adm1)

    # Move hosp data to xp array where 0-index is admin1 id
    hist_daily_h_df = hist.previous_day_admission_adult_covid_confirmed.unstack()
    hist_daily_h = xp.zeros((hist_daily_h_df.index.max() + 1, len(hist_daily_h_df.columns)))
    hist_daily_h[hist_daily_h_df.index.to_numpy()] = hist_daily_h_df.to_numpy()

    # Collect case, death, hosp data
    hist_vals = [hist_daily_cases, hist_daily_deaths, hist_daily_h]
    # Get rid of negatives
    hist_vals = [xp.clip(vals, a_min=0.0, a_max=None) for vals in hist_vals]

    # Rolling mean
    if rolling:
        from ..util.rolling_mean import rolling_mean

        hist_vals = [rolling_mean(vals, axis=1) for vals in hist_vals]

    # Spline
    if spline:
        from ..util.spline_smooth import fit

        hist_vals = [fit(vals, df=dof) for vals in hist_vals]

    # Get rid of negatives
    hist_vals = [xp.clip(vals, a_min=0.0, a_max=None) for vals in hist_vals]

    from functools import partial

    from scipy.optimize import minimize
    from skopt import gp_minimize
    from skopt.sampler import Lhs
    from skopt.space import Real

    # Opt function args
    args = (env, *hist_vals, fips_mask)

    # Opt function params
    opt_params = np.array(
        [
            env.bucky_params.base_params["H_fac"]["mu"],
            env.bucky_params.base_params["H_fac"]["b"] - env.bucky_params.base_params["H_fac"]["a"],
            env.bucky_params.base_params["H_fac"]["gamma"],
            xp.to_cpu(env.consts.F_RR_var),
            xp.to_cpu(env.consts.rh_scaling),
            xp.to_cpu(env.consts.F_scaling),
            env.bucky_params.base_params["Rt_fac"]["mu"],
            env.bucky_params.base_params["Rt_fac"]["b"] - env.bucky_params.base_params["Rt_fac"]["a"],
            env.bucky_params.base_params["Rt_fac"]["gamma"],
            env.bucky_params.base_params["E_fac"]["mu"],
            env.bucky_params.base_params["E_fac"]["b"] - env.bucky_params.base_params["E_fac"]["a"],
            env.bucky_params.base_params["E_fac"]["gamma"],
            xp.to_cpu(env.consts.reroll_variance),
            env.bucky_params.base_params["R_fac"]["mu"],
            env.bucky_params.base_params["R_fac"]["b"] - env.bucky_params.base_params["R_fac"]["a"],
            env.bucky_params.base_params["R_fac"]["gamma"],
        ]
    )
    # Global search initialization
    lhs = Lhs(criterion="maximin", iterations=10000)

    # Best objective value
    best_opt = xp.inf
    best_params = opt_params

    # 2 Global searches
    if local_opt:
        for j in range(2):
            fac = 0.5 + 0.25 * j
            dims = [Real(fac * best_params[i], 1.0 / fac * best_params[i]) for i in range(len(best_params))]
            res = gp_minimize(
                partial(opt_func, args=args),
                dimensions=dims,
                x0=best_params.tolist(),
                initial_point_generator=lhs,
                # callback=[checkpoint_saver],
                n_calls=200,
                verbose=True,
            )
            if res.fun < best_opt:
                best_opt = res.fun
                best_params = np.array(res.x)

    # Local search
    result = minimize(
        opt_func,
        best_params,
        (args,),
        options={"disp": True, "adaptive": True, "maxfev": 1000},
        method="Nelder-Mead",
    )
    if result.fun < best_opt:
        best_opt = result.fun
        best_params = np.array(result.x)

    print("Best Opt:", best_opt)
    print("Best Params:", best_params)

    # with open('best_opt.yml', 'w') as f:
    # TODO need to recursively convert these from np.float to float for this to work
    #    yaml.safe_dump(new_params, f)
    # TODO need function that will take the array being optimized and cast it to a dict (what opt_func is doing but
    #  available more generally)
    # need to be able to dump that dict to a yaml file

    embed()

import datetime
from pprint import pformat

import numpy as np
import pandas as pd
import yaml

from ..numerical_libs import sync_numerical_libs, xp
from ..util.scoring import WIS

# from iPython import embed


# TODO better place for these
columns = ("daily_reported_cases", "daily_deaths", "daily_hospitalizations")

default_ret = 1e5


def ravel_3d(a: xp.ndarray):
    """Ravel each element of a, preserving first dimension"""
    return a.reshape(a.shape[0], -1)


def extract_values(base_params: dict, to_extract: list):
    """
    Extract numerical values of specified parameters from base params dictionary.

    For example, given the following (in yaml representation for clarity)

        base_params:
           Rt_fac:
              dist: "approx_mPERT"
              mu: 1.
              gamma: 5.
              a: .9
              b: 1.1
           R_fac:
              dist: "approx_mPERT"
              mu: .5
              a: .45
              b: .55
              gamma: 50.
           consts:
              En: 3
              Im: 3
              Rhn: 3

        to_extract:
           - Rt_fac
           - R_fac
           - consts:
              - En
              - Im

    extract_values(base_params, to_extract) would return:

        np.array([1., 5., .2, .5, 50., .1, 3, 3]),
        [("Rt_fac", ["mu", "gamma", "b-a"]), ("R_fac", ["mu", "gamma", "b-a"]), ("consts", ["En", "Im"])]
    """

    base_params = base_params.copy()
    ordered_params = []
    values = []
    for param in to_extract:
        these_values = []
        if type(param) is dict:  # consts
            k0, k1s = list(param.items())[0]
        else:  # type(param) is string
            k0 = param
            vals = base_params[k0]
            if all(k1 in vals for k1 in ["a", "b", "mu"]):
                vals = vals.copy()
                vals["b-a"] = vals.pop("b") - vals.pop("a")
                base_params[k0] = vals
            k1s = list(vals.keys())
        for k1 in k1s:
            these_values.append(base_params[k0][k1])
        numeric_val_indices = [i for i, val in enumerate(these_values) if type(val) is float or type(val) is int]
        ordered_params.append((k0, [k1s[i] for i in numeric_val_indices]))
        values.extend([these_values[i] for i in numeric_val_indices])
    return np.array(values), ordered_params


def rebuild_params(values, keys):
    """
    Build parameter dictionary from flattened values and ordered parameter names.

    For example, given the following:

       values = np.array([1., 5., .2, .5, 50., .1, 3, 3]),
       keys = [("Rt_fac", ["mu", "gamma", "b-a"]), ("R_fac", ["mu", "gamma", "b-a"]), ("consts", ["En", "Im"])]

    rebuild_params(values, keys) would return (in yaml representation for clarity):

       Rt_fac:
          mu: 1.
          gamma: 5.
          a: .9
          b: 1.1
       R_fac:
          mu: .5
          gamma: 50.
          a: .45
          b: .55
       consts:
          En: 3
          Im: 3
    """
    v_i = 0
    d = {}
    for p0, p1s in keys:
        d[p0] = {}
        r = None
        mu = None
        for p1 in p1s:
            if p1 == "b-a":
                r = values[v_i]
            else:
                if p1 == "mu":
                    mu = values[v_i]
                d[p0][p1] = values[v_i]
            v_i += 1
        if r is not None and mu is not None:
            d[p0]["a"] = mu - r / 2
            d[p0]["b"] = mu + r / 2
    return d


def opt_func(params, args):
    """Function y = f(params, args) to be minimized"""
    # Unroll args
    env, hist_daily_cases, hist_daily_deaths, hist_daily_h, fips_mask, keys = args

    # Convert param list to dictionary
    print(params)
    new_params = rebuild_params(params, keys)
    print(pformat(new_params))

    run_params = env.bucky_params.opt_params
    hist_data = {
        "daily_reported_cases": hist_daily_cases,
        "daily_deaths": hist_daily_deaths,
        "daily_hospitalizations": hist_daily_h,
    }
    hist_days = {col: vals.shape[-1] for col, vals in hist_data.items()}
    hist_data = {col: vals.ravel() for col, vals in hist_data.items()}

    # Run model
    env.update_params(new_params)
    data = env.run_multiple(run_params.n_mc, run_params.base_seed, columns)
    if data is None:
        return default_ret
    model_data = {}

    # Convert array of MC runs into array of percentiles
    q = xp.arange(*run_params.percentile_params)
    for col in columns:
        # Roll up to admin 1
        tmp = xp.array([env.g_data.sum_adm1(run[col][fips_mask, 1:], mask=fips_mask) for run in data])
        # Cut down to length of available ground truth data
        hist_num_days = hist_days[col]
        model_data[col] = ravel_3d(xp.percentile(tmp, 100.0 * q, axis=0)[..., :hist_num_days])

    # from IPython import embed
    # embed()

    # WIS
    ret = [WIS(hist_data[col], q, model_data[col], norm=True, log=run_params.log) for col in columns]
    # Normalize by number of days
    ret = [ret_i / hist_days[col] for ret_i, col in zip(ret, columns)]
    # Sum over admin 2 and days
    ret = xp.array([xp.nansum(ret_i) for ret_i in ret])

    # MSE
    med_ind = q.shape[0] // 2 + 1
    mse = [(xp.abs(model_data[col][med_ind] - hist_data[col]) / (hist_data[col] + 1)) ** 2 for col in columns]
    mse = [mse_i / hist_days[col] for mse_i, col in zip(mse, columns)]
    mse = xp.array([xp.nansum(mse_i) for mse_i in mse])

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
    run_params = env.bucky_params.opt_params
    if run_params.rolling:
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
    if run_params.rolling:
        from ..util.rolling_mean import rolling_mean

        hist_vals = [rolling_mean(vals, axis=1) for vals in hist_vals]

    # Spline
    if run_params.spline:
        from ..util.spline_smooth import fit

        hist_vals = [fit(vals, df=run_params.dof) for vals in hist_vals]

    # Get rid of negatives
    hist_vals = [xp.clip(vals, a_min=0.0, a_max=None) for vals in hist_vals]

    from functools import partial

    from scipy.optimize import minimize
    from skopt import gp_minimize
    from skopt.sampler import Lhs
    from skopt.space import Real

    # Opt function params
    opt_params, keys = extract_values(env.bucky_params.base_params, env.bucky_params.opt_params.to_opt)

    # Opt function args
    args = (env, *hist_vals, fips_mask, keys)

    # Global search initialization
    lhs = Lhs(criterion="maximin", iterations=10000)

    # Best objective value
    best_opt = xp.inf
    best_params = opt_params

    # 2 Global searches
    if run_params.global_opt:
        for (lower, upper) in run_params.global_multipliers:
            dims = [Real(lower * p, upper * p) for p in best_params]
            res = gp_minimize(
                partial(opt_func, args=args),
                dimensions=dims,
                x0=best_params.tolist(),
                initial_point_generator=lhs,
                # callback=[checkpoint_saver],
                n_calls=run_params.global_calls,
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
        options={"disp": True, "adaptive": True, "maxfev": run_params.local_calls},  # local_calls
        method="Nelder-Mead",
    )
    if result.fun < best_opt:
        best_opt = result.fun
        best_params = np.array(result.x)

    print("Best Opt:", best_opt)
    print("Best Params:", best_params)

    with open("best_opt.yml", "w") as f:
        best_params = [p.item() for p in best_params]
        new_params = rebuild_params(best_params, keys)
        yaml.safe_dump(new_params, f)

    with open("values.csv", "a") as f:
        f.write("{},{}\n".format(run_params.ID, best_opt))

    # embed()

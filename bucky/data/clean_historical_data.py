import warnings

from IPython import embed
from loguru import logger
from numpy import RankWarning

from ..numerical_libs import sync_numerical_libs, xp
from ..util.array_utils import rolling_window
from ..util.extrapolate import interp_extrap
from ..util.power_transforms import YeoJohnson
from ..util.spline_smooth import fit, lin_reg
from .timeseries import BuckyFittedData


def mask_outliers(ts, window_size=3, frac_err_max=0.1):
    """Find unusual outliers in timeseries compared to other time-local values"""
    windowed_ts = rolling_window(ts, window_size, center=True)
    flat_fitted_windowed_ts = lin_reg(windowed_ts.reshape(-1, window_size), return_fit=True)
    fitted_ts = xp.mean(flat_fitted_windowed_ts, axis=1).reshape(ts.shape)
    frac_err_fit = xp.abs((ts - fitted_ts) / ts)
    # return a mask of nonoutlier points
    return frac_err_fit < frac_err_max


@sync_numerical_libs
def clean_historical_data(csse_data, hhs_data, adm_mapping, force_save_plots=False):
    """Preprocess the historical data to smooth it and remove outliers."""

    n_hist = csse_data.n_days
    csse_adm1 = csse_data.sum_adm_level(1, adm_mapping)

    adm1_case_hist = csse_adm1.cumulative_cases
    adm1_death_hist = csse_adm1.cumulative_deaths

    # mask out days that didnt have some increase > 0 in either cases or deaths
    # to detect unreported days
    adm1_diff_mask_cases = xp.around(xp.diff(adm1_case_hist, axis=0, prepend=adm1_case_hist[0][None, ...]), 2) >= 1.0
    adm1_diff_mask_death = xp.around(xp.diff(adm1_death_hist, axis=0, prepend=adm1_death_hist[0][None, ...]), 2) >= 1.0

    valid_adm1_mask = adm1_diff_mask_cases | adm1_diff_mask_death

    # get mask of time series that avg less than one case/death per day
    adm1_enough_case_data = (adm1_case_hist[-1] - adm1_case_hist[0]) > n_hist
    adm1_enough_death_data = (adm1_death_hist[-1] - adm1_death_hist[0]) > n_hist

    # mask out spikey outliers
    valid_adm1_case_mask = valid_adm1_mask & mask_outliers(adm1_case_hist)
    valid_adm1_death_mask = valid_adm1_mask & mask_outliers(adm1_death_hist)

    # map the adm1 masks back to adm2
    valid_case_mask = valid_adm1_case_mask[:, adm_mapping.adm1_ids]
    valid_death_mask = valid_adm1_death_mask[:, adm_mapping.adm1_ids]
    enough_case_data = adm1_enough_case_data[adm_mapping.adm1_ids]
    enough_death_data = adm1_enough_death_data[adm_mapping.adm1_ids]

    # replace masked values by interpolating/extrapolating nearby values if there is enough data
    new_cum_cases = xp.empty((csse_data.n_loc, csse_data.n_days))
    new_cum_deaths = xp.empty((csse_data.n_loc, csse_data.n_days))

    x = xp.arange(0, csse_data.n_days)
    for i in range(new_cum_cases.shape[0]):
        try:
            if ~enough_case_data[i]:
                new_cum_cases[i] = csse_data.cumulative_cases[:, i]
            else:
                new_cum_cases[i] = interp_extrap(
                    x,
                    x[valid_case_mask[:, i]],
                    csse_data.cumulative_cases[valid_case_mask[:, i], i],
                    n_pts=7,
                    order=2,
                )

            if ~enough_death_data[i]:
                new_cum_deaths[i] = csse_data.cumulative_deaths[:, i]
            else:
                new_cum_deaths[i] = interp_extrap(
                    x,
                    x[valid_death_mask[:, i]],
                    csse_data.cumulative_deaths[valid_death_mask[:, i], i],
                    n_pts=7,
                    order=2,
                )

        except (TypeError, RankWarning, ValueError) as e:
            # TODO not sure that we still need this try/catch
            logger.error(e)

    new_cum_cases = xp.around(new_cum_cases, 6) + 0.0  # plus zero to convert -0 to 0.
    new_cum_deaths = xp.around(new_cum_deaths, 6) + 0.0

    # Apply spline smoothing
    df = max(1 * n_hist // 7 - 1, 4)

    alp = 1.5
    tol = 1.0e-5  # 6
    gam_inc = 2.4  # 8.
    gam_cum = 2.4  # 8.

    # df2 = int(10 * n_hist ** (2.0 / 9.0)) + 1  # from gam book section 4.1.7
    gam_inc = 8.0  # 2.4  # 2.4
    gam_cum = 8.0  # 2.4  # 2.4
    # tol = 1e-3

    spline_cum_cases = xp.clip(
        fit(
            new_cum_cases,
            df=df,
            alp=alp,
            gamma=gam_cum,
            tol=tol,
            label="PIRLS Cumulative Cases",
            standardize=False,
        ),
        a_min=0.0,
        a_max=None,
    )
    spline_cum_deaths = xp.clip(
        fit(
            new_cum_deaths,
            df=df,
            alp=alp,
            gamma=gam_cum,
            tol=tol,
            label="PIRLS Cumulative Deaths",
            standardize=False,
        ),
        a_min=0.0,
        a_max=None,
    )

    inc_cases = xp.clip(xp.gradient(spline_cum_cases, axis=1, edge_order=2), a_min=0.0, a_max=None)
    inc_deaths = xp.clip(xp.gradient(spline_cum_deaths, axis=1, edge_order=2), a_min=0.0, a_max=None)

    # inc_cases = xp.around(inc_cases, 6) + 0.0
    # inc_deaths = xp.around(inc_deaths, 6) + 0.0
    inc_hosp = xp.around(hhs_data.incident_hospitalizations.T, 6) + 0.0

    power_transform1 = YeoJohnson()
    power_transform2 = YeoJohnson()
    power_transform3 = YeoJohnson()
    inc_cases = power_transform1.fit(inc_cases)
    inc_deaths = power_transform2.fit(inc_deaths)
    inc_hosp2 = power_transform3.fit(inc_hosp)

    inc_cases = xp.around(inc_cases, 6) + 0.0
    inc_deaths = xp.around(inc_deaths, 6) + 0.0
    inc_hosp = xp.around(inc_hosp, 6) + 0.0

    inc_fit_args = {
        "alp": alp,
        "df": df,  # df // 2 + 2 - 1,
        "dist": "g",
        "standardize": False,  # True,
        "gamma": gam_inc,
        "tol": tol,
        "clip": (0.0, None),
        "bootstrap": False,  # True,
    }

    all_cached = (
        fit.check_call_in_cache(inc_cases, **inc_fit_args)
        and fit.check_call_in_cache(inc_deaths, **inc_fit_args)
        and fit.check_call_in_cache(inc_hosp2, **inc_fit_args)
    )

    spline_inc_cases = fit(
        inc_cases,
        **inc_fit_args,
        label="PIRLS Incident Cases",
    )
    spline_inc_deaths = fit(
        inc_deaths,
        **inc_fit_args,
        label="PIRLS Incident Deaths",
    )
    spline_inc_hosp = fit(
        inc_hosp2,
        **inc_fit_args,
        label="PIRLS Incident Hospitalizations",
    )
    # TODO move this to the actual fitting method
    for _ in range(5):
        resid = spline_inc_cases - inc_cases
        stddev = xp.quantile(xp.abs(resid), axis=1, q=0.682)
        clean_resid = xp.clip(resid / (6.0 * stddev[:, None] + 1e-8), -1.0, 1.0)
        robust_weights = xp.clip(1.0 - clean_resid ** 2.0, 0.0, 1.0) ** 2.0
        spline_inc_cases = fit(inc_cases, **inc_fit_args, label="PIRLS Incident Cases", w=robust_weights)

        resid = spline_inc_deaths - inc_deaths
        stddev = xp.quantile(xp.abs(resid), axis=1, q=0.682)
        clean_resid = xp.clip(resid / (6.0 * stddev[:, None] + 1e-8), -1.0, 1.0)
        robust_weights = xp.clip(1.0 - clean_resid ** 2.0, 0.0, 1.0) ** 2.0
        spline_inc_deaths = fit(inc_deaths, **inc_fit_args, label="PIRLS Incident Deaths", w=robust_weights)

        resid = spline_inc_hosp - inc_hosp2
        stddev = xp.quantile(xp.abs(resid), axis=1, q=0.682)
        clean_resid = xp.clip(resid / (6.0 * stddev[:, None] + 1e-8), -1.0, 1.0)
        robust_weights = xp.clip(1.0 - clean_resid ** 2.0, 0.0, 1.0) ** 2.0
        spline_inc_hosp = fit(inc_hosp2, **inc_fit_args, label="PIRLS Incident Hosps", w=robust_weights)

    spline_inc_cases = power_transform1.inv(spline_inc_cases)
    spline_inc_deaths = power_transform2.inv(spline_inc_deaths)
    spline_inc_hosp = power_transform3.inv(spline_inc_hosp)

    ret_data = {
        "cumulative_cases": spline_cum_cases,
        "cumulative_deaths": spline_cum_deaths,
        "incident_cases": spline_inc_cases,
        "incident_deaths": spline_inc_deaths,
        "incident_hospitalizations": spline_inc_hosp,
    }

    # Only plot if the fits arent in the cache already
    # TODO this wont update if doing a historical run thats already cached
    save_plots = (not all_cached) or force_save_plots

    save_plots = True
    if save_plots:
        plot_historical_fits(csse_data, hhs_data, adm_mapping, ret_data, valid_adm1_case_mask, valid_adm1_death_mask)

    return ret_data


def plot_historical_fits(csse_data, hhs_data, adm_mapping, fitted_data, valid_adm1_case_mask, valid_adm1_death_mask):
    # pylint: disable=import-outside-toplevel
    import matplotlib

    matplotlib.use("agg")
    import pathlib

    import matplotlib.pyplot as plt
    import numpy as np
    import tqdm
    import us

    # TODO we should drop these in raw_output_dir and have postprocess put them in the run's dir
    # TODO we could also drop the data for viz.plot...
    # if we just drop the data this should be moved to viz.historical_plots or something
    out_dir = pathlib.Path("output") / "_historical_fit_plots"  # TODO TMP
    # out_dir = pathlib.Path(bucky_cfg["output_dir"]) / "_historical_fit_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir.touch(exist_ok=True)  # update mtime

    csse_adm1 = csse_data.sum_adm_level(1, adm_mapping)

    raw_diff_cases = xp.gradient(csse_adm1.cumulative_cases, axis=0, edge_order=2)
    raw_diff_deaths = xp.gradient(csse_adm1.cumulative_deaths, axis=0, edge_order=2)

    fitted_datac = BuckyFittedData(
        2,
        csse_data.adm_ids,
        csse_data.dates,
        fitted_data["cumulative_cases"].T,
        fitted_data["cumulative_deaths"].T,
        fitted_data["incident_cases"].T,
        fitted_data["incident_deaths"].T,
    )

    fitted_adm1 = fitted_datac.sum_adm_level(1, adm_mapping)
    # embed()

    fips_map = us.states.mapping("fips", "abbr")

    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(15, 10))
    x = xp.arange(csse_adm1.n_days)
    # TODO move the sum_adm1 calls out here, its doing that reduction ALOT
    for i in tqdm.tqdm(range(csse_adm1.n_loc), desc="Ploting fits", dynamic_ncols=True):
        adm1_fips = adm_mapping.uniq_adm1_ids[i]
        fips_str = str(adm1_fips).zfill(2)
        if fips_str in fips_map:
            name = fips_map[fips_str] + " (" + fips_str + ")"
        else:
            name = fips_str

        ax = fig.subplots(nrows=2, ncols=4)
        ax[0, 0].plot(xp.to_cpu(csse_adm1.cumulative_cases[:, i]), label="Cumulative Cases")
        ax[0, 0].plot(xp.to_cpu(fitted_adm1.cumulative_cases[:, i]), label="Fit")

        ax[0, 0].fill_between(
            xp.to_cpu(x),
            xp.to_cpu(xp.min(csse_adm1.cumulative_cases[:, i])),
            xp.to_cpu(xp.max(csse_adm1.cumulative_cases[:, i])),
            where=xp.to_cpu(~valid_adm1_case_mask[:, i]),
            color="grey",
            alpha=0.2,
        )
        ax[1, 0].plot(xp.to_cpu(csse_adm1.cumulative_deaths[:, i]), label="Cumulative Deaths")
        ax[1, 0].plot(xp.to_cpu(fitted_adm1.cumulative_deaths[:, i]), label="Fit")
        ax[1, 0].fill_between(
            xp.to_cpu(x),
            xp.to_cpu(xp.min(csse_adm1.cumulative_deaths[:, i])),
            xp.to_cpu(xp.max(csse_adm1.cumulative_deaths[:, i])),
            where=xp.to_cpu(~valid_adm1_death_mask[:, i]),
            color="grey",
            alpha=0.2,
        )

        ax[0, 1].plot(xp.to_cpu(raw_diff_cases[:, i]), label="Incident Cases")
        ax[0, 1].plot(xp.to_cpu(fitted_adm1.incident_cases[:, i]), label="Fit")
        ax[0, 1].fill_between(
            xp.to_cpu(x),
            xp.to_cpu(xp.min(raw_diff_cases[:, i])),
            xp.to_cpu(xp.max(raw_diff_cases[:, i])),
            where=xp.to_cpu(~valid_adm1_case_mask[:, i]),
            color="grey",
            alpha=0.2,
        )

        ax[0, 2].plot(xp.to_cpu(raw_diff_deaths[:, i]), label="Incident Deaths")
        ax[0, 2].plot(xp.to_cpu(fitted_adm1.incident_deaths[:, i]), label="Fit")
        ax[0, 2].fill_between(
            xp.to_cpu(x),
            xp.to_cpu(xp.min(raw_diff_deaths[:, i])),
            xp.to_cpu(xp.max(raw_diff_deaths[:, i])),
            where=xp.to_cpu(~valid_adm1_death_mask[:, i]),
            color="grey",
            alpha=0.2,
        )

        ax[1, 1].plot(xp.to_cpu(xp.log1p(raw_diff_cases[:, i])), label="Log(Incident Cases)")
        ax[1, 1].plot(xp.to_cpu(xp.log1p(fitted_adm1.incident_cases[:, i])), label="Fit")

        ax[1, 2].plot(xp.to_cpu(xp.log1p(raw_diff_deaths[:, i])), label="Log(Incident Deaths)")
        ax[1, 2].plot(xp.to_cpu(xp.log1p(fitted_adm1.incident_deaths[:, i])), label="Fit")

        hind = np.argwhere(xp.to_cpu(hhs_data.adm_ids == adm1_fips.item()))
        if len(hind):
            hind = hind[0][0]
            ax[0, 3].plot(xp.to_cpu(hhs_data.incident_hospitalizations[:, hind]), label="Incident Hosp")
            ax[0, 3].plot(xp.to_cpu(fitted_data["incident_hospitalizations"][hind]), label="Fit")

            ax[1, 3].plot(xp.to_cpu(xp.log1p(hhs_data.incident_hospitalizations[:, hind])), label="Log(Incident Hosp)")
            ax[1, 3].plot(xp.to_cpu(xp.log1p(fitted_data["incident_hospitalizations"][hind])), label="Fit")

        log_cases = xp.to_cpu(xp.log1p(xp.clip(raw_diff_cases[:, i], a_min=0.0, a_max=None)))
        log_deaths = xp.to_cpu(xp.log1p(xp.clip(raw_diff_deaths[:, i], a_min=0.0, a_max=None)))
        if xp.any(xp.array(log_cases > 0)):
            ax[1, 1].set_ylim([0.9 * xp.min(log_cases[log_cases > 0]), 1.1 * xp.max(log_cases)])
        if xp.any(xp.array(log_deaths > 0)):
            ax[1, 2].set_ylim([0.9 * xp.min(log_deaths[log_deaths > 0]), 1.1 * xp.max(log_deaths)])

        ax[0, 0].legend()
        ax[1, 0].legend()
        ax[0, 1].legend()
        ax[1, 1].legend()
        ax[0, 2].legend()
        ax[1, 2].legend()
        ax[0, 3].legend()
        ax[1, 3].legend()

        fig.suptitle(name)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(out_dir / (name + ".png"))
        fig.clf()
    plt.close(fig)
    plt.close("all")

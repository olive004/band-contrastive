

import numpy as np
import pandas as pd


def filter_invalids(df, input_cols, cfg) -> pd.DataFrame:


    if cfg.filt_x_nans:
        df = df[df[input_cols].notna().all(axis=1)]
    if cfg.filt_y_nans:
        for k in cfg.objective:
            df = df[df[k].notna() & (
                np.abs(df[k]) < np.inf)]
    if cfg.filt_sensitivity_nans:
        df = df[(np.abs(df['sensitivity'])
                       < np.inf) & df['sensitivity'].notna()]
    if cfg.filt_precision_nans:
        df = df[(np.abs(df['precision'])
                       < np.inf) & df['precision'].notna()]

    if cfg.filt_response_time_high:

        df['response_time'] = np.where(
            df['response_time'] < np.inf, df['response_time'], np.nan)

        df = df[(df['response_time'] < (cfg.filt_response_time_perc_max *
                                                np.nanmax(df['response_time'])))]

    df = df.reset_index(drop=True)

    return df
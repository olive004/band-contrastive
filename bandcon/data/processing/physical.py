

import numpy as np


def calculate_adaptation(s, p, alpha=3, s_center=7, p_center=7.5):
    """ Adaptation = robustness to noise. 
    s = sensitivity, p = precision 
    High when s > 1 and p > 10
    """
    s_log = np.log10(s)
    p_log = np.log10(p)
    a = -(alpha*(s_log - s_center)**2 + (p_log - p_center)**2) + 1000
    return np.where((a > -np.inf) & (a < np.inf),
                     a,
                     np.nan)
    

def embellish_data(data, zero_log_replacement=-10.0, transform_sensitivity_nans=True):
    data['adaptation'] = calculate_adaptation(
        s=np.array(data['sensitivity']),
        p=np.array(data['precision']), alpha=2)
    if transform_sensitivity_nans:
        data['sensitivity'] = np.where(np.isnan(
            data['sensitivity']), 0, data['sensitivity'])

    def make_log(k, data):
        data[f'Log {k}'] = np.where(
            data[k] != 0, np.log10(data[k]), zero_log_replacement)
        return data

    data = make_log('sensitivity', data)
    data = make_log('precision', data)
    data['Log sensitivity > 0'] = data['Log sensitivity'] > 0
    data['Log precision > 1'] = data['Log precision'] > 1
    data['Adaptable'] = (data['Log sensitivity'] >= 0) & (data['Log precision'] >= 1)
    return data

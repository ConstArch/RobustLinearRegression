import statsmodels.api as sm

from statsmodels.tsa.stattools import adfuller


def DJSR_test(X, significance_level=0.05):

    if adfuller(X, regression='ct')[1] < significance_level:
        return 'TSP + linear trend'

    if sm.OLS(np.diff(X), sm.add_constant(np.arange(len(X) - 1))).fit().pvalues[1] < significance_level:
        return 'DSP + linear trend'
    
    if adfuller(X, regression='c')[1] < significance_level:
        return 'TSP + constant'

    if sm.OLS(np.diff(X), np.ones(len(X) - 1)).fit().pvalues[0] < significance_level:
        return 'DSP + constant'

    if adfuller(X, regression='n')[1] < significance_level:
        return 'TSP'
    else:
        return 'DSP'

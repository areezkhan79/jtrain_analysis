# regression_analysis.py
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
import matplotlib.pyplot as plt

def fit_regression_model(X, y):
    X_const = add_constant(X)
    model = OLS(y, X_const).fit()
    return model

def calculate_vif(X):
    X_const = add_constant(X)
    return pd.Series([variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])], index=X_const.columns)

def breusch_pagan_test(model):
    test = het_breuschpagan(model.resid, model.model.exog)
    return test[2], test[3]  # LM Statistic and p-value

def durbin_watson_test(model):
    return durbin_watson(model.resid)


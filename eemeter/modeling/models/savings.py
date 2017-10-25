import pandas as pd
from scipy.stats.stats import pearsonr
from scipy import stats
import statsmodels.formula.api as smf
import numpy as np

gas_file = "/vagrant/scripts/gas.csv"

def get_data(fname, energy_type):
    data = pd.read_csv(fname)
    if energy_type == 'gas':
        data = data[['heating_coefficient',
                     'intercept_coefficient',
                     'natural_gas_savings_thm']]
    else:
        data = data[['heating_coefficient',
                     'cooling_coefficient',
                     'intercept_coefficient',
                     'electricity_savings_kwh']]
    return data

def prediction(fitted_model_ols, dataframe):
    return fitted_model_ols.predict(dataframe)


def rmse_train(fitted_model_ols):
    return np.sqrt(fitted_model_ols.ssr/fitted_model_ols.nobs)

def fit(dataframe,
        formula):
    ols_model = smf.ols(formula=formula, data=dataframe)
    fitted_model = ols_model.fit()
    return ols_model, fitted_model

def correlation(dataframe, energy_type):
    if energy_type == 'gas':
        dataframe = dataframe[dataframe.heating_coefficient.notnull()]
    if energy_type == 'elec':
        dataframe = dataframe[dataframe.cooling_coefficient.notnull()]
        dataframe = dataframe[dataframe.heating_coefficient.notnull()]

    if energy_type == 'gas':
        return pearsonr(dataframe['natural_gas_savings_th'],
                        dataframe['heating_coefficient'])

    else:
        return pearsonr(dataframe['electricity_savings_kwh'],
                        dataframe['heating_coefficient'])

if __name__ == "__main__":
    energy_type = 'gas'
    df = get_data(gas_file)
    df = df.dropna()

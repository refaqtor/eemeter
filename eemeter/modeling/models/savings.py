import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

GAS_ENERGY = 'NATURAL_GAS_CONSUMPTION_SUPPLIED'
ELECTREICITY_ENERGY = 'ELECTRICITY_CONSUMPTION_SUPPLIED'

def read_savings_data(fname, energy_type):
    df = pd.read_csv(fname)
    if energy_type == 'gas':
        df = df[['heating_coefficient',
                 'intercept_coefficient',
                 'project_climate_zone',
                 'natural_gas_savings_thm']]
    else:
        df = df[['heating_coefficient',
                 'cooling_coefficient',
                 'intercept_coefficient',
                 'project_climate_zone',
                 'electricity_savings_kwh']]
    return df


def train_test_split(df, train_frac=0.8):
    mask = np.random.rand(len(df)) < train_frac
    train = df[mask]
    test = df[~mask]
    return train, test


class SimpeSavingPredictionUsingHistoricalData:
    """
    A simple linear regression based model to predict savings for potential
    clients.
    We build a simple linear regression model from our past savings data which
    has following information for each traces
    1. heating_coefficient, 2. cooling_coefficient, 3. intercept_coef
    4. annualized savings in Kwh
    The goal is to learn weights of coeficients in predicting savings.
    For new clients, we will build our usual Caltrack based models to learn
    coeficients. We can then use the coeficient and weights learned from this
    model to predict annualized savings for client..
    """
    def __init__(self, energy_type,
                 include_climate_zone=False,
                 response_var_name=None,
                 model_weights=None,
                 formula=None):

        self.model_weights = model_weights
        self.energy_type = energy_type
        self.response_var_name = response_var_name
        if formula:
            self.formula = formula
        elif energy_type == GAS_ENERGY:
            self.formula = "natural_gas_savings_thm ~ heating_coefficient + intercept_coefficient"
            self.response_var_name = 'natural_gas_savings_thm'
        elif energy_type == ELECTREICITY_ENERGY:
            self.formula = "electricity_savings_kwh ~ cooling_coefficient +\
            heating_coefficient + intercept_coefficient"
            self.response_var_name = 'electricity_savings_kwh'
        else:
            raise ValueError("")
        if not formula and include_climate_zone:
            self.formula = formula + " + project_climate_zone"

        self.fitted_model = None
        self.model_obj = None

    def __init__(self, model_params):
        self.model_weights = model_params

    def predict(self, df):
        pred = self.fitted_model.predict(df)
        # Series
        return pred

    def predict_with_model_weights(self, feature_values):
        if not self.model_weights:
            raise ValueError("Model Weights Not Set")

        if 'Intercept' not in self.model_weights:
            raise ValueError("Intercept not present in model weights")

        model_weight_keys = self.model_weights.keys() - set("Intercept")
        if model_weight_keys != feature_values.keys():
            raise ValueError("Feature Keys and Model Weights Keys do not match")

        saving = self.model_weights['Intercept']
        for feature_key, value in feature_values.items():
            feature_weight = self.model_weights[feature_key]
            saving += (value * feature_weight)

        return saving

    def out_of_sample_stats(self, df, response_var_name=None):
        prediction = self.predict(df)
        if response_var_name is None:
            reponse_var_name = self.response_var_name
        actual = df[reponse_var_name]
        rmse = sqrt(mean_squared_error(prediction, actual))

        savings = 0.0
        for idx, predicted_saving in enumerate(prediction):
            if predicted_saving > 0.0 and actual.iloc[idx] > 0.0:
                savings += 1.0

        savings_precision = savings / len(df)
        print (savings, ' ', len(df))
        return { 'rmse': rmse,
                 'savings_precision' : savings_precision }

    def fit(self, data_frame):
        ols_model = smf.ols(formula=self.formula, data=data_frame)
        self.fitted_model = ols_model.fit()
        self.model_obj = ols_model
        self.model_weights =  self.fitted_model.params.to_dict()
        return self.model_weights

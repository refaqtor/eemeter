import warnings

import numpy as np
import pandas as pd
import patsy
from scipy.stats import chi2
from sklearn import linear_model


class ElasticNetCVBaseModel(object):
    """
    Used as base for billing and seasonal models, which each provide
    the methods:

        `_model_data_from_input_data`
        `_model_data_from_demand_fixture_data`
        `_patsy_formula`

    The rest is shared in this base model.
    """

    def __init__(self, cooling_base_temp, heating_base_temp):

        self.cooling_base_temp = cooling_base_temp
        self.heating_base_temp = heating_base_temp

        self.base_formula = 'energy ~ 1 + CDD + HDD + CDD:HDD'

        self.l1_ratio = [.01, .1, .3, .5, .7, .8, .9, .95, .99, 1]
        self.params = None
        self.upper = None
        self.lower = None
        self.variance = None
        self.X = None
        self.y = None
        self.estimated = None
        self.r2 = None
        self.rmse = None
        self.cvrmse = None
        self.n = None
        self.input_data = None

    def fit(self, input_data):
        ''' Fits a model to the input data.

        Parameters
        ----------
        input_data : pandas.DataFrame
            Formatted input data as returned by
            :code:`ModelDataFormatter.create_input()`

        Returns
        -------
        out : dict
            Results of this model fit:

            - :code:`"r2"`: R-squared value from this fit.
            - :code:`"model_params"`: Fitted parameters.

              - :code:`X_design_matrix`: patsy design matrix used in
                formatting design matrix.
              - :code:`formula`: patsy formula used in creating design matrix.
              - :code:`coefficients`: ElasticNetCV coefficients.
              - :code:`intercept`: ElasticNetCV intercept.

            - :code:`"rmse"`: Root mean square error
            - :code:`"cvrmse"`: Normalized root mean square error
              (Coefficient of variation of root mean square error).
            - :code:`"upper"`: self.upper,
            - :code:`"lower"`: self.lower,
            - :code:`"n"`: self.n
        '''
        # convert to daily

        self.input_data = input_data
        model_data = self._model_data_from_input_data(input_data)
        formula = self._patsy_formula(model_data)
        y, X = patsy.dmatrices(formula, model_data, return_type='dataframe')

        self.X = X
        self.y = y

        model_obj = linear_model.ElasticNetCV(l1_ratio=self.l1_ratio,
                                              fit_intercept=False)
        model_obj.fit(X, y.values.ravel())

        estimated = pd.Series(model_obj.predict(X),
                              index=model_data.energy.index)

        self.estimated = estimated
        self.model_obj = model_obj

        r2 = model_obj.score(X, y)
        rmse = ((y.values.ravel() - estimated)**2).mean()**.5

        if y.mean != 0:
            cvrmse = rmse / float(y.values.ravel().mean())
        else:
            cvrmse = np.nan

        self.r2 = r2
        self.rmse = rmse
        self.cvrmse = cvrmse

        # For justification of these 95% confidence intervals, based on rmse,
        # see http://stats.stackexchange.com/questions/78079/
        #     confidence-interval-of-rmse
        #
        # > Let xi be your true value for the ith data point and xhat_i the
        # >   estimated value.
        # > If we assume that the differences between the estimated and
        # > true values have
        # >
        # > 1. mean zero (i.e. the xhat_i are distributed around xi)
        # > 2. follow a Normal distribution
        # > 3. and all have the same standard deviation sigma
        # > then you really want a confidence interval for sigma
        # > ...
        #
        # We might decide these assumptions don't hold.

        n = self.estimated.shape[0]

        c1, c2 = chi2.ppf([0.025, 1 - 0.025], n)
        self.lower = np.sqrt(n / c2) * self.rmse
        self.upper = np.sqrt(n / c1) * self.rmse
        self.variance = self.rmse ** 2
        self.n = n

        self.params = {
            "coefficients": list(model_obj.coef_),
            "intercept": model_obj.intercept_,
            "X_design_info": X.design_info,
            "formula": formula,
        }

        output = {
            "r2": self.r2,
            "model_params": self.params,
            "rmse": self.rmse,
            "cvrmse": self.cvrmse,
            "upper": self.upper,
            "lower": self.lower,
            "n": self.n
        }
        return output

    def predict(self, demand_fixture_data, params=None, summed=True):
        ''' Predicts across index using fitted model params

        Parameters
        ----------
        demand_fixture_data : pandas.DataFrame
            Formatted input data as returned by
            :code:`ModelDataFormatter.create_demand_fixture()`
        params : dict, default None
            Parameters found during model fit. If None, `.fit()` must be called
            before this method can be used.

              - :code:`X_design_matrix`: patsy design matrix used in
                formatting design matrix.
              - :code:`formula`: patsy formula used in creating design matrix.
              - :code:`coefficients`: ElasticNetCV coefficients.
              - :code:`intercept`: ElasticNetCV intercept.

        Returns
        -------
        output : pandas.DataFrame
            Dataframe of energy values as given by the fitted model across the
            index given in :code:`demand_fixture_data`.
        '''
        if params is None:
            params = self.params

        design_info = params["X_design_info"]

        model_data = self._model_data_from_demand_fixture_data(
            demand_fixture_data)

        (X,) = patsy.build_design_matrices([design_info],
                                           model_data,
                                           return_type='dataframe')

        model_obj = linear_model.ElasticNetCV(l1_ratio=self.l1_ratio,
                                              fit_intercept=False)

        model_obj.coef_ = np.array(params["coefficients"])
        model_obj.intercept_ = params["intercept"]

        try:
            predicted = pd.Series(model_obj.predict(X), index=X.index)
        except:
            return np.nan, np.nan

        if summed:
            n = len(predicted)
            predicted = np.sum(predicted)
            variance = self.variance * n
        else:
            # add NaNs back in
            predicted = predicted.reindex(model_data.index)
            variance = self.variance

        return predicted, variance

    def calc_gross(self):
        return np.nansum(self.input_data.energy)

    def plot(self):
        ''' Plots fit against input data. Should not be run before the
        :code:`.fit(` method.
        '''

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("Cannot plot - no matplotlib.")
            return None

        plt.title("actual v. estimated w/ 95% confidence")

        self.estimated.plot(color='b', alpha=0.7)

        plt.fill_between(self.estimated.index.to_datetime(),
                         self.estimated + self.upper,
                         self.estimated - self.lower,
                         color='b', alpha=0.3)

        pd.Series(self.y.values.ravel(), index=self.estimated.index).plot(
            color='k', linewidth=1.5)

        plt.show()

## WIP ##
from statsmodels.tsa.arima.model import ARIMA


class ARIMAModel:
    def __init__(self, order):
        self.order = order
        self.model = None

    def fit(self, X, y):
        self.model = ARIMA(y, order=self.order)
        self.model_fit = self.model.fit()

    def predict(self, X):
        return self.model_fit.forecast(steps=len(X))

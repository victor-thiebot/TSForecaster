from fbprophet import Prophet
from .base_model import Model


## WIP --> check how to uses
class ProphetModel(Model):
    def __init__(self, prophet_config=None):
        super().__init__(input_dim=None, output_dim=None)
        self.prophet_config = prophet_config or {}
        self.model = None

    def _create_model(self):
        self.model = Prophet(**self.prophet_config)

    def fit(self, df, **kwargs):
        self._create_model()
        self.model.fit(df, **kwargs)

    def predict(self, df, **kwargs):
        if self.model is None:
            raise ValueError("Model has not been trained. Call 'fit' before 'predict'.")
        forecast = self.model.predict(df, **kwargs)
        return forecast

    def add_seasonality(self, name, period, fourier_order):
        if self.model is None:
            self._create_model()
        self.model.add_seasonality(
            name=name, period=period, fourier_order=fourier_order
        )

    def add_country_holidays(self, country_name):
        if self.model is None:
            self._create_model()
        self.model.add_country_holidays(country_name=country_name)

    def add_regressor(self, name, prior_scale=None, standardize=True, mode=None):
        if self.model is None:
            self._create_model()
        self.model.add_regressor(
            name=name, prior_scale=prior_scale, standardize=standardize, mode=mode
        )


## example use

# prophet_config = {
#     "growth": "logistic",
#     "changepoints": ["2022-01-01", "2023-01-01"],
#     "n_changepoints": 10,
#     "yearly_seasonality": True,
#     "weekly_seasonality": True,
#     "daily_seasonality": False,
# }

# model = ProphetModel(prophet_config=prophet_config)
# model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
# model.add_country_holidays(country_name="US")
# model.add_regressor(name="temperature", prior_scale=10.0)

# model.fit(train_df)
# forecast = model.predict(test_df)

from models.shallow.base import BaseShallowModel
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


class RandomForestModel(BaseShallowModel):
    def __init__(self, n_estimators=100, max_depth=None, random_state=None):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X, y):
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class GradientBoostingModel(BaseShallowModel):
    def __init__(
        self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=None
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X, y):
        self.model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

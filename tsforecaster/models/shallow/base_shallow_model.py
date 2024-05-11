from models.base_model import BaseModel

class BaseShallowModel(BaseModel):
    def __init__(self):
        self.model = None

    def fit(self, X, y):
        raise NotImplementedError("Subclasses must implement the 'fit' method.")

    def predict(self, X):
        raise NotImplementedError("Subclasses must implement the 'predict' method.")

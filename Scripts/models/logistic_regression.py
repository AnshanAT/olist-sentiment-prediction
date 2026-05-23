from sklearn.linear_model import LogisticRegression

class LogisticRegressionModel:

    def __init__(self, C=1):
        self.model = LogisticRegression(
            C=C,
            max_iter=5000, 
            class_weight="balanced"
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
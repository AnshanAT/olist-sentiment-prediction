from sklearn.ensemble import RandomForestClassifier

class RandomForestModel:
    def __init__(self, max_depth=10):
        self.model = RandomForestClassifier(
            max_depth=max_depth, 
            random_state=42, 
            class_weight="balanced"
            )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

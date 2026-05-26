from sklearn.ensemble import RandomForestClassifier

class RandomForestModel:
    def __init__(self, max_depth=5, n_estimators=200):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
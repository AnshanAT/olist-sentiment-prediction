import numpy as np
from collections import defaultdict


class StratifiedKFold:

    def __init__(self, n_splits=10, shuffle=True, random_state=42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y):
        rng = np.random.default_rng(self.random_state)
        indices = np.arange(len(y))

        if self.shuffle:
            rng.shuffle(indices)

        # group indices by class label
        by_class = defaultdict(list)
        for i in indices:
            by_class[int(y[i])].append(i)

        # round-robin assign each class's samples to folds
        buckets = [[] for _ in range(self.n_splits)]
        for class_indices in by_class.values():
            for pos, idx in enumerate(class_indices):
                buckets[pos % self.n_splits].append(idx)

        for k in range(self.n_splits):
            val = np.array(buckets[k])
            train = np.array([i for j, b in enumerate(buckets) for i in b if j != k])
            yield train, val


class HyperparameterGrid:

    def __init__(self, param_grid):
        self.param_grid = param_grid
        self.combos = self._build()

    def _build(self):
        keys = list(self.param_grid.keys())
        vals = list(self.param_grid.values())

        result = [[]]
        for pool in vals:
            result = [x + [y] for x in result for y in pool]

        return [dict(zip(keys, combo)) for combo in result]

    def __iter__(self):
        return iter(self.combos)

    def __len__(self):
        return len(self.combos)


class NestedCV:

    def __init__(self, model_class, param_grid, scoring='f1_macro', n_inner=3, random_state=42):
        self.model_class = model_class
        self.grid = HyperparameterGrid(param_grid)
        self.scoring = scoring
        self.n_inner = n_inner
        self.random_state = random_state

    def fit(self, X, y):
        inner_cv = StratifiedKFold(n_splits=self.n_inner, shuffle=True, random_state=self.random_state)

        best_score = -np.inf
        best_params = {}

        for params in self.grid:
            scores = []

            for train_idx, val_idx in inner_cv.split(X, y):
                model = self.model_class(**params)
                model.fit(X[train_idx], y[train_idx])
                y_pred = model.predict(X[val_idx])
                scores.append(self._score(y[val_idx], y_pred, model, X[val_idx]))

            mean = float(np.mean(scores))
            if mean > best_score:
                best_score = mean
                best_params = params

        self.best_score_ = best_score
        self.best_params_ = best_params
        return best_params

    def _score(self, y_true, y_pred, model, X_val):
        from Scripts.metrics import f1_macro, accuracy, roc_auc

        if self.scoring == 'f1_macro':
            return f1_macro(y_true, y_pred)
        elif self.scoring == 'accuracy':
            return accuracy(y_true, y_pred)
        elif self.scoring == 'auc':
            return roc_auc(y_true, model.predict_proba(X_val))
        else:
            raise ValueError(f"Unknown scoring: {self.scoring}")


def cross_validate(model_class, X, y, param_grid, n_outer=10, n_inner=3,
                   scoring='f1_macro', random_state=42, verbose=True):

    from Scripts.metrics import compute_all_metrics

    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=random_state)

    fold_metrics = []
    best_params = []
    y_true_all, y_pred_all, y_proba_all = [], [], []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        inner = NestedCV(model_class, param_grid, scoring, n_inner, random_state)
        params = inner.fit(X_train, y_train)
        best_params.append(params)

        model = model_class(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

        metrics = compute_all_metrics(y_test, y_pred, y_proba)
        fold_metrics.append(metrics)

        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())
        if y_proba is not None:
            y_proba_all.extend(y_proba.tolist())

        if verbose:
            print(f"fold {fold}/{n_outer}  f1={metrics.get('f1_macro', float('nan')):.4f}  "
                  f"auc={metrics.get('auc', float('nan')):.4f}  params={params}")

    keys = fold_metrics[0].keys()
    mean_metrics = {k: float(np.mean([m[k] for m in fold_metrics])) for k in keys}
    std_metrics  = {k: float(np.std( [m[k] for m in fold_metrics])) for k in keys}

    if verbose:
        print("\nresults:")
        for k in keys:
            print(f"  {k}: {mean_metrics[k]:.4f} +/- {std_metrics[k]:.4f}")

    return {
        'fold_metrics': fold_metrics,
        'best_params':  best_params,
        'y_true_all':   np.array(y_true_all),
        'y_pred_all':   np.array(y_pred_all),
        'y_proba_all':  np.array(y_proba_all) if y_proba_all else None,
        'mean_metrics': mean_metrics,
        'std_metrics':  std_metrics,
    }


if __name__ == '__main__':

    class DummyModel:
        def __init__(self, threshold=0.5):
            self.threshold = threshold
            self.majority = None

        def fit(self, X, y):
            vals, counts = np.unique(y, return_counts=True)
            self.majority = vals[np.argmax(counts)]

        def predict(self, X):
            return np.full(len(X), self.majority)

        def predict_proba(self, X):
            p = 0.7 if self.majority == 1 else 0.3
            return np.column_stack([1 - p * np.ones(len(X)), p * np.ones(len(X))])

    # stub out metrics so smoke test runs without metrics.py
    import sys, types
    stub = types.ModuleType('Scripts.metrics')
    stub.f1_macro = lambda yt, yp: float(np.mean(yt == yp))
    stub.accuracy = lambda yt, yp: float(np.mean(yt == yp))
    stub.roc_auc  = lambda yt, yp: 0.5
    stub.compute_all_metrics = lambda yt, yp, ypr: {
        'f1_macro': float(np.mean(yt == yp)),
        'accuracy': float(np.mean(yt == yp)),
        'auc': 0.5,
    }
    sys.modules.setdefault('Scripts', types.ModuleType('Scripts'))
    sys.modules['Scripts.metrics'] = stub

    np.random.seed(42)
    X_dummy = np.random.randn(500, 10)
    y_dummy = (X_dummy[:, 0] + np.random.randn(500) * 0.5 > 0).astype(int)

    results = cross_validate(
        DummyModel, X_dummy, y_dummy,
        param_grid={'threshold': [0.3, 0.5, 0.7]},
        verbose=True
    )

    print("\nsmoke test passed")

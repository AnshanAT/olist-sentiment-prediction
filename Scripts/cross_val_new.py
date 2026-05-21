import numpy as np
from collections import defaultdict


class StratifiedKFold:

    def __init__(self, n_splits=10, shuffle=True, random_state=42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, df, y):
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
            result = [
                x + [y]
                for x in result
                for y in pool
            ]

        return [
            dict(zip(keys, combo))
            for combo in result
        ]

    def __iter__(self):
        return iter(self.combos)

    def __len__(self):
        return len(self.combos)


class NestedCV:

    def __init__(
        self,
        model_class,
        param_grid,
        scoring="f1_macro",
        n_inner=3,
        random_state=42,
        target_col="review_score"
    ):
        self.model_class = model_class
        self.grid = HyperparameterGrid(param_grid)
        self.scoring = scoring
        self.n_inner = n_inner
        self.random_state = random_state
        self.target_col = target_col

    def fit(self, train_df):

        from Scripts.inter_feature_builder import build_features, prepare_X_y

        y = train_df[self.target_col].astype(int).values

        inner_cv = StratifiedKFold(
            n_splits=self.n_inner,
            shuffle=True,
            random_state=self.random_state
        )

        best_score = -np.inf
        best_params = {}

        for params in self.grid:

            scores = []

            for inner_train_idx, inner_val_idx in inner_cv.split(train_df, y):

                inner_train_df = train_df.iloc[inner_train_idx].copy()
                inner_val_df = train_df.iloc[inner_val_idx].copy()

                inner_train_fe, z_stats = build_features(
                    inner_train_df,
                    inner_train_df,
                    fit=True
                )

                inner_val_fe, _ = build_features(
                    inner_train_df,
                    inner_val_df,
                    fit=False,
                    z_stats=z_stats
                )

                X_inner_train, y_inner_train, _ = prepare_X_y(
                    inner_train_fe,
                    target_col=self.target_col
                )

                X_inner_val, y_inner_val, _ = prepare_X_y(
                    inner_val_fe,
                    target_col=self.target_col
                )

                model = self.model_class(**params)
                model.fit(X_inner_train, y_inner_train)

                y_pred = model.predict(X_inner_val)

                score = self._score(
                    y_inner_val,
                    y_pred
                )

                scores.append(score)

            mean_score = float(np.mean(scores))

            if mean_score > best_score:
                best_score = mean_score
                best_params = params

        self.best_score_ = best_score
        self.best_params_ = best_params

        return best_params

    def _score(self, y_true, y_pred):

        from Scripts.metrics import f1_macro, accuracy

        if self.scoring == "f1_macro":
            return f1_macro(y_true, y_pred)

        elif self.scoring == "accuracy":
            return accuracy(y_true, y_pred)

        else:
            raise ValueError(f"Unknown scoring: {self.scoring}")


def cross_validate(
    model_class,
    df,
    param_grid,
    target_col="review_score",
    n_outer=10,
    n_inner=3,
    scoring="f1_macro",
    random_state=42,
    verbose=True
):

    from Scripts.metrics import compute_all_metrics
    from Scripts.inter_feature_builder import build_features, prepare_X_y

    y = df[target_col].astype(int).values

    outer_cv = StratifiedKFold(
        n_splits=n_outer,
        shuffle=True,
        random_state=random_state
    )

    fold_metrics = []
    best_params = []

    y_true_all = []
    y_pred_all = []

    feature_names = None

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(df, y), 1):

        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

        inner = NestedCV(
            model_class=model_class,
            param_grid=param_grid,
            scoring=scoring,
            n_inner=n_inner,
            random_state=random_state,
            target_col=target_col
        )

        params = inner.fit(train_df)
        best_params.append(params)

        train_fe, z_stats = build_features(
            train_df,
            train_df,
            fit=True
        )

        test_fe, _ = build_features(
            train_df,
            test_df,
            fit=False,
            z_stats=z_stats
        )

        X_train, y_train, feature_names = prepare_X_y(
            train_fe,
            target_col=target_col
        )

        X_test, y_test, _ = prepare_X_y(
            test_fe,
            target_col=target_col
        )

        model = model_class(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        metrics = compute_all_metrics(
            y_test,
            y_pred
        )

        fold_metrics.append(metrics)

        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())

        if verbose:
            print(
                f"fold {fold}/{n_outer} "
                f"{scoring}={metrics.get(scoring, float('nan')):.4f} "
                f"params={params}"
            )

    keys = fold_metrics[0].keys()

    mean_metrics = {
        k: float(np.mean([m[k] for m in fold_metrics]))
        for k in keys
    }

    std_metrics = {
        k: float(np.std([m[k] for m in fold_metrics]))
        for k in keys
    }

    if verbose:
        print("\nresults:")
        for k in keys:
            print(f"  {k}: {mean_metrics[k]:.4f} +/- {std_metrics[k]:.4f}")

    return {
        "fold_metrics": fold_metrics,
        "best_params": best_params,
        "y_true_all": np.array(y_true_all),
        "y_pred_all": np.array(y_pred_all),
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
        "feature_names": feature_names,
    }
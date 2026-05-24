import numpy as np
from sklearn import metrics

from cv_utils import kfold_split, make_param_grid, prepare_X_y, standardize_features
from create_interaction_features import create_interaction_features
from metrics import compute_all_metrics, f1_macro, accuracy


TARGET_COL = "review_score"


def score_result(y_true, y_pred, scoring):
    if scoring == "f1_macro":
        return f1_macro(y_true, y_pred)

    if scoring == "accuracy":
        return accuracy(y_true, y_pred)

    raise ValueError(f"Unknown scoring: {scoring}")


def evaluate_inner_fold(model_class, params, train_df, val_df, scoring, use_interactions=True):
    if use_interactions:
        train_fe, val_fe = create_interaction_features(train_df, val_df)
    else:
        train_fe, val_fe = train_df, val_df

    X_train, y_train, _ = prepare_X_y(train_fe, TARGET_COL)
    X_val, y_val, _ = prepare_X_y(val_fe, TARGET_COL)

    X_train, X_val = standardize_features(X_train, X_val)

    model = model_class(**params)
    print("before model fit")
    model.fit(X_train, y_train)
    print("after model fit")

    y_pred = model.predict(X_val)

    return score_result(y_val, y_pred, scoring)


def select_best_params(model_class, train_df, param_grid, n_inner=3, scoring="f1_macro", use_interactions=True):
    y = train_df[TARGET_COL].astype(int).values

    param_list = make_param_grid(param_grid)

    best_score = -np.inf
    best_params = None

    for params in param_list:
        scores = []

        for inner_train_idx, val_idx in kfold_split(y, n_splits=n_inner):
            inner_train_df = train_df.iloc[inner_train_idx].copy()
            val_df = train_df.iloc[val_idx].copy()

            score = evaluate_inner_fold(
                model_class,
                params,
                inner_train_df,
                val_df,
                scoring,
                use_interactions
            )

            scores.append(score)

        mean_score = float(np.mean(scores))

        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    return best_params, best_score


def evaluate_outer_fold(model_class, params, train_df, test_df, use_interactions=True):
    if use_interactions:
        train_fe, test_fe = create_interaction_features(train_df, test_df)
    else:
        train_fe, test_fe = train_df, test_df

    X_train, y_train, feature_names = prepare_X_y(train_fe, TARGET_COL)
    X_test, y_test, _= prepare_X_y(test_fe, TARGET_COL)

    X_train, X_test = standardize_features(X_train, X_test)

    model = model_class(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    metrics = compute_all_metrics(y_test, y_pred)
    train_metrics = compute_all_metrics(y_train, y_train_pred)
    train_test_gap = (train_metrics["f1_macro"]- metrics["f1_macro"])
    
    metrics["train_f1"] = train_metrics["f1_macro"]
    metrics["train_test_gap"] = train_test_gap

    return metrics, y_test, y_pred, feature_names


def cross_validate(
    model_class,
    df,
    param_grid,
    n_outer=10,
    n_inner=3,
    scoring="f1_macro",
    verbose=True,
    use_interactions=True
):
    y = df[TARGET_COL].astype(int).values

    fold_metrics = []
    best_params_all = []
    y_true_all = []
    y_pred_all = []
    feature_names = None
    print("cv start")
    for fold, (train_idx, test_idx) in enumerate(
        kfold_split(y, n_splits=n_outer),
        start=1
    ):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

        best_params, inner_score = select_best_params(
            model_class=model_class,
            train_df=train_df,
            param_grid=param_grid,
            n_inner=n_inner,
            scoring=scoring,
            use_interactions=use_interactions
        )

        metrics, y_test, y_pred, feature_names = evaluate_outer_fold(
            model_class=model_class,
            params=best_params,
            train_df=train_df,
            test_df=test_df,
            use_interactions=use_interactions
        )

        fold_metrics.append(metrics)
        best_params_all.append(best_params)

        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())

        if verbose:
            print(
                f"fold {fold}/{n_outer} "
                f"{scoring}={metrics.get(scoring, float('nan')):.4f} "
                f"params={best_params}"
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
        "best_params": best_params_all,
        "y_true_all": np.array(y_true_all),
        "y_pred_all": np.array(y_pred_all),
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
        "feature_names": feature_names,
    }
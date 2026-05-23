import numpy as np
from collections import defaultdict

RANDOM_STATE = 42
DROP_COLS = [
    # IDs
    "order_id",
    "seller_id",
    "customer_unique_id",

    # raw datetime
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "shipping_limit_date",
    "review_creation_date",
    "review_answer_timestamp",

    # raw categoricals
    "product_category_name",
    "product_category_name_english",
    "customer_city",
    "seller_city",
    "customer_state",
    "seller_state",
    "order_status",

    # duplicates
    "carrier_pickup_lag",
    "photo_qty",
]

# def prepare_X_y(df, target_col, median_giv=None):
def prepare_X_y(df, target_col):
    data = df.copy()

    y = data[target_col].astype(int).values

    data = data.drop(columns=[c for c in DROP_COLS if c in data.columns], errors="ignore")
    data = data.drop(columns=[target_col], errors="ignore")
        
    # Drop any surviving non-numeric cols (object, category, datetime)
    bad_cols = data.select_dtypes(
        include=[
            "object",
            "category",
            "datetime64[ns]"
        ]
    ).columns
    data = data.drop(columns=bad_cols)

    ### move to the create_interaction_features
    # # Convert infinite values to NaN before
    # data = data.replace([np.inf, -np.inf], np.nan)
    
    # # median not given so gotta calculate the means 
    # to_fit = False
    # if median_giv is None: to_fit = True

    # if to_fit:
    #     median_giv = {}

    #     # Just calculate for all cus key errors later...
    #     for col in data.columns:
    #         median_giv[col] = data[col].median()
    
    # # Median-impute remaining NaNs
    # for col in data.columns:
    #     if data[col].isna().any():
    #         data[col] = data[col].fillna(median_giv[col])

    X = data.values.astype(np.float64)
    feature_names = data.columns.tolist()

    return X, y, feature_names#, median_giv


def standardize_features(X_train, X_test):
    # z-score normalization based on training set stats only (to avoid data leakage)
    # Detect binary features automatically (0/1 only)
    binary_mask = np.all(
        (X_train == 0) | (X_train == 1),
        axis=0
    )

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    # avoid division by zero
    std[std == 0] = 1

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    # standardize non-binary features only
    X_train_scaled[:, ~binary_mask] = (
        X_train[:, ~binary_mask]
        - mean[~binary_mask]
    ) / std[~binary_mask]

    X_test_scaled[:, ~binary_mask] = (
        X_test[:, ~binary_mask]
        - mean[~binary_mask]
    ) / std[~binary_mask]

    return X_train_scaled, X_test_scaled


def kfold_split(y, n_splits=10):
    
    rng = np.random.default_rng(RANDOM_STATE)
    indices = np.arange(len(y))

    rng.shuffle(indices)

    # Group row indices by class label
    by_class = defaultdict(list)

    for i in indices:
        by_class[int(y[i])].append(i)

    folds = [[] for _ in range(n_splits)]

    for class_indices in by_class.values():
        for pos, idx in enumerate(class_indices):
            folds[pos % n_splits].append(idx)

    for k in range(n_splits):
        val_idx = np.array(folds[k])
        train_idx = np.array([
            i
            for j, fold in enumerate(folds)
            for i in fold
            if j != k
        ])

        yield train_idx, val_idx


def make_param_grid(param_grid):
    keys = list(param_grid.keys())
    values = list(param_grid.values())

    combinations = [[]]

    for value_list in values:
        combinations = [
            current + [value]
            for current in combinations
            for value in value_list
        ]

    return [
        dict(zip(keys, combo))
        for combo in combinations
    ]



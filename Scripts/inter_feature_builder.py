import numpy as np
import pandas as pd


def build_foldaware_features(train_df, apply_df):
    apply_df = apply_df.copy()

    # Global fallbacks computed from train only
    global_mean_freight  = train_df['freight_to_price'].mean()
    global_install       = train_df['payment_installments'].mean()

    # Customer zip-code installment norm and deviation
    # Captures whether this customer is borrowing more than is normal for their area
    zip_install = train_df.groupby('customer_zip_code_prefix')['payment_installments'].mean()
    apply_df['fbi_zip_install_mean'] = (
        apply_df['customer_zip_code_prefix'].map(zip_install).fillna(global_install)
    )
    apply_df['fbi_installment_deviation'] = (
        apply_df['payment_installments'] - apply_df['fbi_zip_install_mean']
    )

    # Customer zip freight norm
    # Captures whether this order's shipping cost is high relative to others in the same area
    zip_freight = train_df.groupby('customer_zip_code_prefix')['freight_to_price'].mean()
    apply_df['zip_avg_freight'] = (
        apply_df['customer_zip_code_prefix'].map(zip_freight).fillna(global_mean_freight)
    )

    # Seller statistics (by seller_id)
    # A seller's historical delay, on-time rate, and volume are strong signals
    # None of these use review_score
    if 'seller_id' in train_df.columns:
        seller_stats = train_df.groupby('seller_id').agg(
            seller_avg_delay        =('delivery_delay_days', 'mean'),
            seller_delay_std        =('delivery_delay_days', 'std'),
            seller_on_time_rate     =('delivery_delay_days', lambda x: (x <= 0).mean()),
            seller_order_volume     =('order_id', 'count'),
            seller_avg_processing_hrs=('processing_time_hrs', 'mean'),
        )
        global_seller_delay       = train_df['delivery_delay_days'].mean()
        global_seller_on_time     = (train_df['delivery_delay_days'] <= 0).mean()
        global_seller_volume      = seller_stats['seller_order_volume'].mean()
        global_processing         = train_df['processing_time_hrs'].mean()

        apply_df['seller_avg_delay']         = apply_df['seller_id'].map(seller_stats['seller_avg_delay']).fillna(global_seller_delay)
        apply_df['seller_delay_std']         = apply_df['seller_id'].map(seller_stats['seller_delay_std']).fillna(0)
        apply_df['seller_on_time_rate']      = apply_df['seller_id'].map(seller_stats['seller_on_time_rate']).fillna(global_seller_on_time)
        apply_df['seller_order_volume']      = apply_df['seller_id'].map(seller_stats['seller_order_volume']).fillna(1)
        apply_df['seller_avg_processing_hrs']= apply_df['seller_id'].map(seller_stats['seller_avg_processing_hrs']).fillna(global_processing)

        # How much worse/better is this specific order's freight vs the seller's average
        seller_avg_freight = train_df.groupby('seller_id')['freight_to_price'].mean()
        apply_df['freight_vs_seller_avg'] = (
            apply_df['freight_to_price']
            - apply_df['seller_id'].map(seller_avg_freight).fillna(global_mean_freight)
        )

        # Combined signal: far away from an unreliable seller — highest delivery risk
        apply_df['distance_x_seller_reliability'] = (
            apply_df['geo_distance_km'] * (1 - apply_df['seller_on_time_rate'])
        )

    # Category statistics (by product_category_name)
    # Some categories structurally have higher freight, longer wait, heavier items
    cat_col = 'product_category_name'
    cat_stats = train_df.groupby(cat_col).agg(
        category_avg_price  =(('total_price'   if 'total_price'   in train_df.columns else 'price'),   'mean'),
        category_avg_freight=(('total_freight' if 'total_freight' in train_df.columns else 'freight_value'), 'mean'),
        category_avg_delay  =('delivery_delay_days', 'mean'),
        category_order_count=('order_id', 'count'),
    )
    price_col   = 'total_price'   if 'total_price'   in train_df.columns else 'price'
    freight_col = 'total_freight' if 'total_freight' in train_df.columns else 'freight_value'

    apply_df['category_avg_price']   = apply_df[cat_col].map(cat_stats['category_avg_price']).fillna(train_df[price_col].mean())
    apply_df['category_avg_freight'] = apply_df[cat_col].map(cat_stats['category_avg_freight']).fillna(train_df[freight_col].mean())
    apply_df['category_avg_delay']   = apply_df[cat_col].map(cat_stats['category_avg_delay']).fillna(train_df['delivery_delay_days'].mean())
    apply_df['category_order_count'] = apply_df[cat_col].map(cat_stats['category_order_count']).fillna(1)

    # Customer statistics (by customer_unique_id)
    # Repeat customers have calibrated expectations; their order history provides context
    if 'customer_unique_id' in train_df.columns:
        cust_stats = train_df.groupby('customer_unique_id').agg(
            customer_order_count      =('order_id', 'nunique'),
            customer_avg_payment      =(('payment_value'), 'mean'),
            customer_avg_installments =(('payment_installments'), 'mean'),
            customer_avg_freight_ratio=('freight_to_price', 'mean'),
        )
        apply_df['customer_order_count']       = apply_df['customer_unique_id'].map(cust_stats['customer_order_count']).fillna(1)
        apply_df['customer_avg_payment']       = apply_df['customer_unique_id'].map(cust_stats['customer_avg_payment']).fillna(train_df['payment_value'].mean())
        apply_df['customer_avg_installments']  = apply_df['customer_unique_id'].map(cust_stats['customer_avg_installments']).fillna(global_install)
        apply_df['customer_avg_freight_ratio'] = apply_df['customer_unique_id'].map(cust_stats['customer_avg_freight_ratio']).fillna(global_mean_freight)
        apply_df['customer_is_repeat']         = (apply_df['customer_order_count'] > 1).astype(int)

    foldaware_cols = [
        'fbi_zip_install_mean', 'fbi_installment_deviation', 'zip_avg_freight',
        'seller_avg_delay', 'seller_delay_std', 'seller_on_time_rate',
        'seller_order_volume', 'seller_avg_processing_hrs',
        'freight_vs_seller_avg', 'distance_x_seller_reliability',
        'category_avg_price', 'category_avg_freight', 'category_avg_delay', 'category_order_count',
        'customer_order_count', 'customer_avg_payment', 'customer_avg_installments',
        'customer_avg_freight_ratio', 'customer_is_repeat'
        ]
    
    return apply_df, foldaware_cols


def zscore(series, mu=None, sigma=None):
    if mu is None:    
        mu = series.mean()

    if sigma is None: 
        sigma = series.std()
    
    # case where all is identical (just return all zeros...)
    if sigma == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    
    return (series - mu) / sigma


def build_motif_indices(df, z_stats=None, fit=True):

    d = df.copy()
    if z_stats is None:
        z_stats = {}

    def _z(col):

        if fit:
            mu = d[col].mean()
            sigma = d[col].std()
            z_stats[col] = (mu, sigma)

        else:
            mu, sigma = z_stats.get(col, (0, 1))

        return zscore(d[col], mu, sigma)

    # Frustration Index: late + expensive freight
    d['frustration_index'] = (
        _z('delivery_delay_ratio').fillna(0) +
        _z('freight_to_price').fillna(0)
    ) / 2

    # Financial Burden Index: installment cost + how unusual for this customer's area
    d['financial_burden_index'] = (
        _z('value_per_installment').fillna(0) +
        _z('fbi_installment_deviation').fillna(0)
    ) / 2

    # Consistency Index: order size + price spread + impulse purchase timing
    if 'num_items' in d.columns:
        items_col = 'num_items'

    else:
        'order_item_id'

    d['consistency_index'] = (
        _z(items_col).fillna(0) +
        _z('price_per_item').fillna(0) +
        _z('is_night').fillna(0)
    ) / 3

    return d, z_stats

TARGET_COL = "review_score"
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
    "photo_qty"
]


def prepare_X_y(df):

    d = df.copy()

    y = d[TARGET_COL].astype(int).values

    d = d.drop(
        columns=[c for c in DROP_COLS if c in d.columns],
        errors="ignore"
    )

    d = d.drop(
        columns=[TARGET_COL],
        errors="ignore"
    )

    # remove remaining non-numeric
    bad = d.select_dtypes(
        include=[
            "object",
            "category",
            "datetime64[ns]"
        ]
    ).columns

    d = d.drop(columns=bad)

    # median fill
    for col in d.columns:
        if d[col].isna().any():
            d[col] = d[col].fillna(
                d[col].median()
            )

    X = d.values.astype(np.float64)

    return X, y, d.columns.tolist()


def build_features(
    train_df,
    apply_df,
    fit=True,
    z_stats=None
):
    """
    Build all fold-aware features.

    train_df:
        Training fold only.
        Used to compute statistics.

    apply_df:
        Dataset to transform.
        Can be train, validation, or test.

    fit=True:
        Compute z-score statistics from train fold.

    fit=False:
        Reuse train z-score statistics.
    """

    transformed_df, foldaware_cols = (
        build_foldaware_features(
            train_df,
            apply_df
        )
    )

    transformed_df, z_stats = (
        build_motif_indices(
            transformed_df,
            z_stats=z_stats,
            fit=fit
        )
    )

    return transformed_df, z_stats, foldaware_cols

""" usage example:
from Scripts.inter_feature_builder import build_features

train_fe, z_stats, _ = build_features(
    train_df,
    train_df,
    fit=True
)

test_fe, _, _ = build_features(
    train_df,
    test_df,
    fit=False,
    z_stats=z_stats
)

X_train, y_train, feature_names = prepare_X_y(train_fe)
X_test, y_test, _ = prepare_X_y(test_fe)

"""
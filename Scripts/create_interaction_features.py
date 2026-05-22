import numpy as np
import pandas as pd


def _build_foldaware_features(train_df, apply_df):
    apply_df = apply_df.copy()

    global_mean_freight = train_df["freight_to_price"].mean()
    global_install = train_df["payment_installments"].mean()

    # Customer zip-code installment norm and deviation
    # Captures whether this customer is borrowing more than is normal for their area
    zip_install = train_df.groupby("customer_zip_code_prefix")["payment_installments"].mean()
    apply_df["fbi_zip_install_mean"] = (
        apply_df["customer_zip_code_prefix"].map(zip_install).fillna(global_install)
    )
    apply_df["fbi_installment_deviation"] = (
        apply_df["payment_installments"] - apply_df["fbi_zip_install_mean"]
    )

    # Customer zip freight norm
    # Captures whether this order's shipping cost is high relative to others in the same area
    zip_freight = train_df.groupby("customer_zip_code_prefix")["freight_to_price"].mean()
    apply_df["zip_avg_freight"] = (
        apply_df["customer_zip_code_prefix"].map(zip_freight).fillna(global_mean_freight)
    )

    # Seller statistics (by seller_id)
    # A seller's historical delay, on-time rate, and volume are strong signals
    # None of these use review_score
    seller_stats = train_df.groupby("seller_id").agg(
        seller_avg_delay=("delivery_delay_days", "mean"),
        seller_delay_std=("delivery_delay_days", "std"),
        seller_on_time_rate=("delivery_delay_days", lambda x: (x <= 0).mean()),
        seller_order_volume=("order_id", "count"),
        seller_avg_processing_hrs=("processing_time_hrs", "mean"),
    )

    apply_df["seller_avg_delay"] = (
        apply_df["seller_id"].map(seller_stats["seller_avg_delay"])
        .fillna(train_df["delivery_delay_days"].mean())
    )
    apply_df["seller_delay_std"] = (
        apply_df["seller_id"].map(seller_stats["seller_delay_std"])
        .fillna(0)
    )
    apply_df["seller_on_time_rate"] = (
        apply_df["seller_id"].map(seller_stats["seller_on_time_rate"])
        .fillna((train_df["delivery_delay_days"] <= 0).mean())
    )
    apply_df["seller_order_volume"] = (
        apply_df["seller_id"].map(seller_stats["seller_order_volume"])
        .fillna(1)
    )
    apply_df["seller_avg_processing_hrs"] = (
        apply_df["seller_id"].map(seller_stats["seller_avg_processing_hrs"])
        .fillna(train_df["processing_time_hrs"].mean())
    )

    # How much worse/better is this specific order's freight vs the seller's average
    seller_avg_freight = train_df.groupby("seller_id")["freight_to_price"].mean()
    apply_df["freight_vs_seller_avg"] = (
        apply_df["freight_to_price"]
        - apply_df["seller_id"].map(seller_avg_freight).fillna(global_mean_freight)
    )
    
    # Combined signal: far away from an unreliable seller — highest delivery risk
    apply_df["distance_x_seller_reliability"] = (
        apply_df["geo_distance_km"] * (1 - apply_df["seller_on_time_rate"])
    )

    price_col = "total_price" if "total_price" in train_df.columns else "price"
    freight_col = "total_freight" if "total_freight" in train_df.columns else "freight_value"
    
    # Category statistics (by product_category_name)
    # Some categories structurally have higher freight, longer wait, heavier items
    cat_col = "product_category_name"
    cat_stats = train_df.groupby(cat_col).agg(
        category_avg_price=(price_col, "mean"),
        category_avg_freight=(freight_col, "mean"),
        category_avg_delay=("delivery_delay_days", "mean"),
        category_order_count=("order_id", "count"),
    )

    apply_df["category_avg_price"] = (
        apply_df[cat_col].map(cat_stats["category_avg_price"])
        .fillna(train_df[price_col].mean())
    )
    apply_df["category_avg_freight"] = (
        apply_df[cat_col].map(cat_stats["category_avg_freight"])
        .fillna(train_df[freight_col].mean())
    )
    apply_df["category_avg_delay"] = (
        apply_df[cat_col].map(cat_stats["category_avg_delay"])
        .fillna(train_df["delivery_delay_days"].mean())
    )
    apply_df["category_order_count"] = (
        apply_df[cat_col].map(cat_stats["category_order_count"])
        .fillna(1)
    )

    cust_stats = train_df.groupby("customer_unique_id").agg(
        customer_order_count=("order_id", "nunique"),
        customer_avg_payment=("payment_value", "mean"),
        customer_avg_installments=("payment_installments", "mean"),
        customer_avg_freight_ratio=("freight_to_price", "mean"),
    )

    apply_df["customer_order_count"] = (
        apply_df["customer_unique_id"].map(cust_stats["customer_order_count"])
        .fillna(1)
    )
    apply_df["customer_avg_payment"] = (
        apply_df["customer_unique_id"].map(cust_stats["customer_avg_payment"])
        .fillna(train_df["payment_value"].mean())
    )
    apply_df["customer_avg_installments"] = (
        apply_df["customer_unique_id"].map(cust_stats["customer_avg_installments"])
        .fillna(global_install)
    )
    apply_df["customer_avg_freight_ratio"] = (
        apply_df["customer_unique_id"].map(cust_stats["customer_avg_freight_ratio"])
        .fillna(global_mean_freight)
    )
    apply_df["customer_is_repeat"] = (
        apply_df["customer_order_count"] > 1
    ).astype(int)

    return apply_df


def _zscore(series, mean, std):
    # case where all is identical (just return all zeros...)
    if std == 0 or pd.isna(std):
        return pd.Series(np.zeros(len(series)), index=series.index)

    return (series - mean) / std


def _build_motif_indices(train_df, apply_df):
    apply_df = apply_df.copy()

    z_cols = [
        "delivery_delay_ratio",
        "freight_to_price",
        "value_per_installment",
        "fbi_installment_deviation",
        "num_items" if "num_items" in train_df.columns else "order_item_id",
        "price_per_item",
        "is_night",
    ]

    z_stats = {
        col: (train_df[col].mean(), train_df[col].std())
        for col in z_cols
    }

    def z(col):
        mean, std = z_stats[col]
        return _zscore(apply_df[col], mean, std).fillna(0)

    items_col = "num_items" if "num_items" in apply_df.columns else "order_item_id"

    apply_df["frustration_index"] = (
        z("delivery_delay_ratio") + z("freight_to_price")
    ) / 2

    apply_df["financial_burden_index"] = (
        z("value_per_installment") + z("fbi_installment_deviation")
    ) / 2

    apply_df["consistency_index"] = (
        z(items_col) + z("price_per_item") + z("is_night")
    ) / 3

    return apply_df


def create_interaction_features(train_df, test_df):
    """
    Create post-split interaction/fold-aware features.

    Statistics are computed from train_df only and then applied to both
    train_df and test_df.

    Returns:
        train_fe, test_fe
    """

    train_fe = _build_foldaware_features(train_df, train_df)
    test_fe = _build_foldaware_features(train_df, test_df)

    train_fe = _build_motif_indices(train_fe, train_fe)
    test_fe = _build_motif_indices(train_fe, test_fe)

    return train_fe, test_fe
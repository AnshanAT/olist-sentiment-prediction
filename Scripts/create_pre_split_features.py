import pandas as pd
import numpy as np

# Converted from Notebook to Script form

# 1. joining_notebook.ipynb

# Load all raw datasets
customers = pd.read_csv('./Data/olist_customers_dataset.csv')
orders = pd.read_csv('./Data/olist_orders_dataset.csv')
items = pd.read_csv('./Data/olist_order_items_dataset.csv')
products = pd.read_csv('./Data/olist_products_dataset.csv')
payments = pd.read_csv('./Data/olist_order_payments_dataset.csv')
reviews = pd.read_csv('./Data/olist_order_reviews_dataset.csv')
sellers = pd.read_csv('./Data/olist_sellers_dataset.csv')
translations = pd.read_csv('./Data/product_category_name_translation.csv')
geo_location = pd.read_csv('./Data/olist_geolocation_dataset.csv')

# Merge category name translations
products = products.merge(translations, on='product_category_name', how='left')

# Merge all main tables sequentially into a single dataframe
df = orders.merge(customers, on='customer_id', how='left')
df = df.merge(items, on='order_id', how='left')
df = df.merge(sellers, on='seller_id', how='left')
df = df.merge(products, on='product_id', how='left')
df = df.merge(payments, on='order_id', how='left')
df = df.merge(reviews, on='order_id', how='left')

# Calculate average coordinates for each zip code prefix
geo_centroids = geo_location.groupby('geolocation_zip_code_prefix')[['geolocation_lat', 'geolocation_lng']].mean().reset_index()

# Map centroid coordinates to customers
df = df.merge(geo_centroids.rename(columns={'geolocation_zip_code_prefix': 'customer_zip_code_prefix', 'geolocation_lat': 'customer_lat', 'geolocation_lng': 'customer_lng'}), on='customer_zip_code_prefix', how='left')

# Map centroid coordinates to sellers
df['seller_zip_int'] = pd.to_numeric(df['seller_zip_code_prefix'], errors='coerce').astype('Int64')
df = df.merge(geo_centroids.rename(columns={'geolocation_zip_code_prefix': 'seller_zip_int', 'geolocation_lat': 'seller_lat', 'geolocation_lng': 'seller_lng'}), on='seller_zip_int', how='left')
df.drop(columns='seller_zip_int', inplace=True)

# 2. cleaning_dataset.ipynb

master_df = df

# lenght -> length
master_df['product_name_length'] = master_df['product_name_lenght']        
master_df['product_desc_length'] = master_df['product_description_lenght']
master_df = master_df.drop(columns=['product_description_lenght', 'product_name_lenght'])

# Convert date columns to datetime objects
date_time_cols = ['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date', 'order_estimated_delivery_date', 'shipping_limit_date', 'review_creation_date', 'review_answer_timestamp']
for col in date_time_cols:
    master_df[col] = pd.to_datetime(master_df[col], format='mixed')

# no review creation date, no info...
master_df = master_df.dropna(subset=['review_creation_date'])

# Filter rows based on notebook criteria
master_df = master_df[master_df['review_score'].notna()].copy()
master_df = master_df[master_df['order_status'] == 'delivered'].copy()
master_df = master_df[master_df['seller_id'].notna()].copy()

# Drop rows missing critical IDs
required_cols = ['seller_id', 'product_id', 'order_item_id']
master_df = master_df.dropna(subset=required_cols)

# Fill missing text fields
master_df['review_comment_title'] = master_df['review_comment_title'].fillna('')
master_df['review_comment_message'] = master_df['review_comment_message'].fillna('')
master_df['product_category_name'] = master_df['product_category_name'].fillna('unknown')
master_df['product_category_name_english'] = master_df['product_category_name_english'].fillna('unknown')

# Drop rows missing essential geographic and delivery info
master_df = master_df.dropna(subset=['customer_lat', 'customer_lng', 'seller_lat', 'seller_lng'])
master_df = master_df.dropna(subset=['order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date'])
master_df = master_df.dropna(subset=['payment_sequential', 'payment_value', 'payment_installments', 'payment_type'])

# Impute missing product metrics with category means
dim_cols = ['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']
for col in dim_cols:
    master_df[col] = master_df.groupby('product_category_name_english')[col].transform(lambda x: x.fillna(x.mean()))

meta_cols = ['product_name_length', 'product_desc_length', 'product_photos_qty']

for col in meta_cols:
    master_df[col] = master_df.groupby('product_category_name_english')[col].transform(lambda x: x.fillna(x.mean()))

# Match notebook's exact row duplicate dropping state
master_df = master_df.drop_duplicates()

payment_dummies = pd.get_dummies(
    master_df[['order_id', 'payment_type']],
    columns=['payment_type'],
    prefix='paid'
)
payment_dummies = payment_dummies.groupby('order_id').max().reset_index()

# Aggregate item data to order-level (Moved up to align with notebook aggregation state)
agg_dict = {
    'customer_id':                   'first',
    'customer_unique_id':            'first',
    'order_status':                  'first',
    'order_purchase_timestamp':      'first',
    'order_approved_at':             'first',
    'order_delivered_carrier_date':  'first',
    'order_delivered_customer_date': 'first',
    'order_estimated_delivery_date': 'first',
    'customer_zip_code_prefix':      'first',
    'customer_city':                 'first',
    'customer_state':                'first',
    'customer_lat':                  'first',
    'customer_lng':                  'first',
    'shipping_limit_date':           'min',
    'order_item_id':                 'max',
    'price':                         'sum',
    'freight_value':                 'sum',
    'product_id':                    'first',
    'seller_id':                     'first',
    'seller_zip_code_prefix':        'first',
    'seller_city':                   'first',
    'seller_state':                  'first',
    'seller_lat':                    'first',
    'seller_lng':                    'first',
    'product_category_name':         lambda x: x.mode().iloc[0],
    'product_category_name_english': lambda x: x.mode().iloc[0],
    'product_photos_qty':            'mean',
    'product_weight_g':              'mean',
    'product_length_cm':             'mean',
    'product_height_cm':             'mean',
    'product_width_cm':              'mean',
    'product_name_length':           'mean',
    'product_desc_length':           'mean',
    'payment_sequential':            'max',
    'payment_installments':          'max',
    'payment_value':                 'sum',
    'review_id':                     'first',
    'review_score':                  'first',
    'review_creation_date':          'first',
    'review_answer_timestamp':       'first',
    'review_comment_title':          'first',
    'review_comment_message':        'first',
}
master_df_agg = master_df.groupby('order_id').agg(agg_dict).reset_index()

# Rename aggregated metrics for clarity
master_df_agg = master_df_agg.rename(columns={
    'order_item_id': 'num_items', 
    'price': 'total_price', 
    'freight_value': 'total_freight', 
    'payment_sequential': 'num_payment_methods'
})

# Merge payment indicators back into the aggregated dataframe
master_df = master_df_agg.merge(payment_dummies, on='order_id', how='left')

# Convert numerical floats to integers
int_cols = ['num_items', 'product_photos_qty', 'product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm', 'num_payment_methods', 'payment_installments', 'review_score', 'customer_zip_code_prefix', 'seller_zip_code_prefix', 'product_name_length', 'product_desc_length']
master_df[int_cols] = master_df[int_cols].astype(int)

# Convert categorical features
nominal_cols = ['order_status', 'customer_city', 'customer_state', 'seller_city', 'seller_state', 'paid_boleto', 'paid_credit_card', 'paid_debit_card', 'paid_voucher', 'product_category_name', 'product_category_name_english']
for col in nominal_cols:
    master_df[col] = master_df[col].astype('category')

# Standardize review score structure
master_df['review_score'] = pd.Categorical(master_df['review_score'], categories=[1, 2, 3, 4, 5], ordered=True)

# Drop redundant raw IDs
master_df = master_df.drop(columns=['customer_id', 'product_id', 'review_id'])

# 3. feature_creation_v3.ipynb (Pre-split)
df = master_df

df['review_score'] = df['review_score'].astype(int)

# Cast target variable to int and drop raw review texts
# df['review_score'] = df['review_score'].astype(int)
df = df.drop(columns=['review_comment_title', 'review_comment_message'])

price_col = 'total_price' if 'total_price' in df.columns else 'price'
freight_col = 'total_freight' if 'total_freight' in df.columns else 'freight_value'
desc_col = 'product_desc_length' if 'product_desc_length' in df.columns else 'product_description_lenght'
name_col = 'product_name_length' if 'product_name_length' in df.columns else 'product_name_lenght'
items_col = 'num_items' if 'num_items' in df.columns else 'order_item_id'

# Calculate Haversine distance
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

df['geo_distance_km'] = haversine(df['seller_lat'], df['seller_lng'], df['customer_lat'], df['customer_lng'])

# Calculate operational time features
df['estimated_wait_days'] = ((df['order_estimated_delivery_date'] - df['order_purchase_timestamp']).dt.total_seconds() / 86_400).clip(lower=0.01)
df['delivery_delay_days'] = ((df['order_delivered_customer_date'] - df['order_estimated_delivery_date']).dt.total_seconds() / 86_400)
df['delivery_delay_ratio'] = df['delivery_delay_days'] / df['estimated_wait_days']
df['end_to_end_delivery_days'] = ((df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.total_seconds() / 86_400)
# Actual / estimated — ratio >1 means slower than promised (different from delay_ratio)
df['estimated_vs_actual_ratio'] = df['end_to_end_delivery_days'] / df['estimated_wait_days'].clip(lower=0.01)
df['processing_time_hrs'] = ((df['order_approved_at'] - df['order_purchase_timestamp']).dt.total_seconds() / 3600)
df['carrier_pickup_lag'] = ((df['order_delivered_carrier_date'] - df['order_approved_at']).dt.total_seconds() / 86_400)
# Seller dispatch: approval → carrier handoff (seller's own responsibility window)
df['approval_to_carrier_days'] = ((df['order_delivered_carrier_date'] - df['order_approved_at']).dt.total_seconds() / 86_400)
df['days_to_ship_limit'] = ((df['shipping_limit_date'] - df['order_purchase_timestamp']).dt.total_seconds() / 86_400)
df['days_before_ship_limit'] = ((df['shipping_limit_date'] - df['order_delivered_carrier_date']).dt.total_seconds() / 86_400)
df['days_to_review'] = ((df['review_creation_date'] - df['order_delivered_customer_date']).dt.total_seconds() / 86_400)

# Categorize review speed
# Review speed bucket: 0=fast(<2d), 1=medium(2-7d), 2=slow(>7d)
df['review_speed_bucket'] = pd.cut(df['days_to_review'], bins=[-np.inf, 2, 7, np.inf], labels=[0, 1, 2]).astype(float)

# Calculate freight and price ratios
df['freight_to_price'] = np.where(df[price_col] > 0, df[freight_col] / df[price_col], np.nan)
df['freight_fraction'] = np.where(df[price_col] + df[freight_col] > 0, df[freight_col] / (df[price_col] + df[freight_col]), np.nan)
df['price_per_item'] = df[price_col] / df[items_col].clip(lower=1)
df['freight_per_item'] = df[freight_col] / df[items_col].clip(lower=1)

# Calculate payment metrics
df['value_per_installment'] = np.where(df['payment_installments'] > 0, df['payment_value'] / df['payment_installments'], np.nan)
methods_col = 'num_payment_methods' if 'num_payment_methods' in df.columns else 'payment_sequential'
df['multi_payment'] = (df[methods_col] > 1).astype(int)
df['is_single_installment'] = (df['payment_installments'] == 1).astype(int)
df['high_installments'] = (df['payment_installments'] >= 6).astype(int)
df['freight_installment_ratio'] = np.where(df['payment_installments'] > 0, df[freight_col] / df['payment_installments'], np.nan)

# Product features
df['has_description'] = (df[desc_col].fillna(0) > 0).astype(int)
df['has_photos'] = (df['product_photos_qty'].fillna(0) > 0).astype(int)
df['photo_qty'] = df['product_photos_qty'].fillna(0)
df['total_listing_length'] = df[name_col].fillna(0) + df[desc_col].fillna(0)

# Calculate volume and ratio metrics
df['product_volume_cm3'] = df['product_length_cm'] * df['product_height_cm'] * df['product_width_cm']
# Density: heavy-for-size → expensive to ship, affects carrier handling
df['weight_to_volume_ratio'] = np.where(df['product_volume_cm3'] > 0, df['product_weight_g'] / df['product_volume_cm3'], np.nan)

df['photos_per_desc_char'] = np.where(df[desc_col].fillna(0) > 0, df['product_photos_qty'].fillna(0) / df[desc_col].clip(lower=1), np.nan)
df['is_heavy_product'] = (df['product_weight_g'] > 5_000).astype(int)
df['is_bulky_product'] = (df['product_volume_cm3'] > 40_000).astype(int)

# Extract calendar components
# Calendar / behavioural 
df['purchase_hour'] = df['order_purchase_timestamp'].dt.hour
df['purchase_dow'] = df['order_purchase_timestamp'].dt.dayofweek
df['purchase_month'] = df['order_purchase_timestamp'].dt.month
df['purchase_quarter'] = df['order_purchase_timestamp'].dt.quarter
df['is_weekend'] = (df['purchase_dow'] >= 5).astype(int)
df['is_night'] = (df['purchase_hour'].between(22, 5)).astype(int)

# Geographics
df['same_state'] = (df['customer_state'].astype(str) == df['seller_state'].astype(str)).astype(int)

# Map macro regions
region_map = {
    'SP':'southeast','RJ':'southeast','MG':'southeast','ES':'southeast',
    'RS':'south',    'SC':'south',    'PR':'south',
    'BA':'northeast','PE':'northeast','CE':'northeast','MA':'northeast',
    'PB':'northeast','RN':'northeast','AL':'northeast','SE':'northeast','PI':'northeast',
    'GO':'centerwest','MT':'centerwest','MS':'centerwest','DF':'centerwest',
    'AM':'north','PA':'north','RO':'north','AC':'north',
    'RR':'north','AP':'north','TO':'north',
}
df['customer_region'] = df['customer_state'].map(region_map)
df['seller_region'] = df['seller_state'].map(region_map)

# Encode region indicators
cust_dummies = pd.get_dummies(df['customer_region'], prefix='cust_reg', drop_first=True).astype(int)
sell_dummies = pd.get_dummies(df['seller_region'], prefix='sell_reg', drop_first=True).astype(int)
df = pd.concat([df, cust_dummies, sell_dummies], axis=1)
df = df.drop(columns=['customer_region', 'seller_region'])

# Composite interaction features
# Late AND expensive shipping — double-bad signal
df['delay_x_freight_ratio'] = df['delivery_delay_days'] * df['freight_to_price']
# Complex multi-item orders put more pressure on processing
items_col = 'num_items' if 'num_items' in df.columns else 'order_item_id'
df['processing_x_items'] = df['processing_time_hrs'] * df[items_col]
# High installments on expensive orders = credit burden proxy
df['installment_burden'] = df['payment_installments'] * df['payment_value']
# Delay relative to how long the customer expected to wait
df['delay_per_estimated_day'] = df['delivery_delay_days'] / df['estimated_wait_days'].clip(lower=1)
# Heavy parcel + long distance = highest shipping risk composite
df['weight_x_distance'] = df['product_weight_g'] * df['geo_distance_km']

# Save output
df.to_csv('./Data/pre_split_features.csv', index=False)
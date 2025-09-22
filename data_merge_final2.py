import pandas as pd
import numpy as np
import logging
import gc

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load datasets
print("Loading train and test datasets...")
train = pd.read_csv("train_data.csv")
test = pd.read_csv("test_data.csv")

# Function to process add_event_augmented in chunks
def process_chunks(chunk_size=100000):
    print(f"Processing add_event_augmented.csv in chunks of {chunk_size} rows...")
    
    # Initialize an empty list to store aggregated results
    event_agg_chunks = []

    # Read and process in chunks
    for chunk in pd.read_csv("add_event_augmented.csv", chunksize=chunk_size):
        # Remove constant columns within each chunk
        constant_columns = [col for col in chunk.columns if chunk[col].nunique() == 1]
        chunk = chunk.drop(columns=constant_columns)
        print(f"Removed {len(constant_columns)} constant columns in this chunk: {constant_columns}")
        
        # Drop unnecessary columns
        chunk.drop(columns=['id2', 'id6', 'id7'], inplace=True, errors='ignore')

        # Aggregate by id3 within the chunk
        chunk_agg = chunk.groupby('id3').agg({
            'hour_sin': 'mean',
            'hour_cos': 'mean',
            'day_of_week': 'mean',
            'is_weekend': 'mean',
            'month': 'mean',
            'time_diff': 'mean',
            'click_latency': 'mean',
            'industry_diversity': 'mean',
            'most_frequent_industry': lambda x: x.value_counts().idxmax() if not x.dropna().empty and not x.value_counts().empty else 'unknown',
            'debit_credit_ratio': 'mean',
            'spending_power': 'mean',
            'high_redemption': 'mean',
            'high_discount': 'mean',
            'id8': lambda x: x.value_counts().idxmax() if not x.dropna().empty and not x.value_counts().empty else 'unknown',
            'has_click': ['mean', 'sum'],
            'f375': 'mean',
            'f376': 'mean',
            'avg_trans_amount': 'mean',
            'total_trans_amount': 'sum',
            'std_trans_amount': 'std',
            'max_trans_amount': 'max',
            'trans_count': 'sum',
            'avg_trans_time_seconds': 'mean',
            'most_common_time_bucket': lambda x: x.value_counts().idxmax() if not x.dropna().empty and not x.value_counts().empty else 0,
        }).reset_index()

        # Flatten column names
        chunk_agg.columns = [
            'id3',
            'avg_hour_sin',
            'avg_hour_cos',
            'avg_day_of_week',
            'avg_is_weekend',
            'avg_month',
            'avg_time_diff',
            'avg_click_latency',
            'avg_industry_diversity',
            'most_frequent_industry',
            'avg_debit_credit_ratio',
            'avg_spending_power',
            'prop_high_redemption',
            'prop_high_discount',
            'most_frequent_id8',
            'prop_has_click',
            'total_clicks',
            'avg_redemption_freq',
            'avg_discount_rate',
            'avg_trans_amount',
            'total_trans_amount',
            'std_trans_amount',
            'max_trans_amount',
            'total_trans_count',
            'avg_trans_time_seconds',
            'most_frequent_trans_bucket'
        ]

        # Downcast numeric columns to save memory
        for col in chunk_agg.select_dtypes(include=['float64', 'int64']).columns:
            if col in ['total_clicks', 'total_trans_count', 'most_frequent_trans_bucket']:
                chunk_agg[col] = pd.to_numeric(chunk_agg[col], downcast='integer')
            else:
                chunk_agg[col] = pd.to_numeric(chunk_agg[col], downcast='float')

        # Append to list
        event_agg_chunks.append(chunk_agg)
        print(f"Processed chunk {len(event_agg_chunks)}")
        gc.collect()

    # Combine all chunks
    print("Combining chunked aggregations...")
    event_agg = pd.concat(event_agg_chunks, ignore_index=True)

    # Final aggregation to handle duplicates
    print("Performing final aggregation to handle duplicates...")
    event_agg = event_agg.groupby('id3').agg({
        'avg_hour_sin': 'mean',
        'avg_hour_cos': 'mean',
        'avg_day_of_week': 'mean',
        'avg_is_weekend': 'mean',
        'avg_month': 'mean',
        'avg_time_diff': 'mean',
        'avg_click_latency': 'mean',
        'avg_industry_diversity': 'mean',
        'most_frequent_industry': lambda x: x.value_counts().idxmax() if not x.empty else 'unknown',
        'avg_debit_credit_ratio': 'mean',
        'avg_spending_power': 'mean',
        'prop_high_redemption': 'mean',
        'prop_high_discount': 'mean',
        'most_frequent_id8': lambda x: x.value_counts().idxmax() if not x.empty else 'unknown',
        'prop_has_click': 'mean',
        'total_clicks': 'sum',
        'avg_redemption_freq': 'mean',
        'avg_discount_rate': 'mean',
        'avg_trans_amount': 'mean',
        'total_trans_amount': 'sum',
        'std_trans_amount': 'std',
        'max_trans_amount': 'max',
        'total_trans_count': 'sum',
        'avg_trans_time_seconds': 'mean',
        'most_frequent_trans_bucket': lambda x: x.value_counts().idxmax() if not x.empty else 0
    }).reset_index()

    # Downcast final DataFrame
    for col in event_agg.select_dtypes(include=['float64', 'int64']).columns:
        if col in ['total_clicks', 'total_trans_count', 'most_frequent_trans_bucket']:
            event_agg[col] = pd.to_numeric(event_agg[col], downcast='integer')
        else:
            event_agg[col] = pd.to_numeric(event_agg[col], downcast='float')

    return event_agg

# Process the data in chunks
event_agg = process_chunks(chunk_size=100000)

# Merge with train and test
print("Merging with train dataset...")
train = train.merge(event_agg, on='id3', how='left')

print("Merging with test dataset...")
test = test.merge(event_agg, on='id3', how='left')

# Fill missing values with defaults
print("Filling missing values...")
fill_values = {
    'avg_hour_sin': 0,
    'avg_hour_cos': 0,
    'avg_day_of_week': 0,
    'avg_is_weekend': 0,
    'avg_month': 0,
    'avg_time_diff': 0,
    'avg_click_latency': 0,
    'avg_industry_diversity': 0,
    'most_frequent_industry': 'unknown',
    'avg_debit_credit_ratio': 1.0,
    'avg_spending_power': 0,
    'prop_high_redemption': 0,
    'prop_high_discount': 0,
    'most_frequent_id8': 'unknown',
    'prop_has_click': 0,
    'total_clicks': 0,
    'avg_redemption_freq': 0,
    'avg_discount_rate': 0,
    'avg_trans_amount': 0,
    'total_trans_amount': 0,
    'std_trans_amount': 0,
    'max_trans_amount': 0,
    'total_trans_count': 0,
    'avg_trans_time_seconds': 0,
    'most_frequent_trans_bucket': 0
}
train.fillna(fill_values, inplace=True)
test.fillna(fill_values, inplace=True)

def add_time_features(df):
    df['id4'] = pd.to_datetime(df['id4'], errors='coerce')
    
    # Day of week (0 = Monday, 6 = Sunday)
    df['week_day'] = df['id4'].dt.dayofweek.fillna(-1).astype(int)
    
    # Time bucket
    def time_bucket(hour):
        if pd.isna(hour): return 0
        if hour < 6: return 1       # Night
        elif hour < 12: return 2    # Morning
        elif hour < 18: return 3    # Afternoon
        else: return 4              # Evening

    df['hour'] = df['id4'].dt.hour + df['id4'].dt.minute / 60
    df['time_bucket1'] = df['id4'].dt.hour.apply(time_bucket)

    # Hour sin/cos encoding
    df['hour_rad'] = 2 * np.pi * df['hour'] / 24
    df['hour_sin1'] = np.sin(df['hour_rad'])
    df['hour_cos1'] = np.cos(df['hour_rad'])

    # Clean up
    df.drop(columns=['hour', 'hour_rad'], inplace=True, errors='ignore')
    return df

train = add_time_features(train)
test = add_time_features(test)

# Save augmented datasets
print("Saving augmented datasets...")
train.to_csv("train_augmented.csv", index=False)
test.to_csv("test_augmented.csv", index=False)
print("✅ Augmented datasets saved as train_augmented.csv and test_augmented.csv")
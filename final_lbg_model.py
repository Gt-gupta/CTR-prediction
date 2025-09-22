import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from scipy.stats import boxcox
import gc
import logging
import re
import optuna
from optuna.samplers import TPESampler
import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
CONFIG = {
    'min_category_count': 5,
    'n_splits': 5,
    'random_state': 42,
    'chunk_size': 100000,
    'optuna_trials': 50,
    'optuna_timeout': 7200,
    'lambdarank_position': 7,
    'final_blend_ratio': 0.3  # Blend ratio for ranking adjustment
}

def map_at_7(y_true, y_pred, user_ids):
    """Calculate MAP@7 metric"""
    df = pd.DataFrame({
        'user_id': user_ids,
        'y_true': y_true,
        'y_pred': y_pred
    })
    
    df = df.sort_values(['user_id', 'y_pred'], ascending=[True, False])
    df = df.groupby('user_id').head(7)
    df['rank'] = df.groupby('user_id').cumcount() + 1
    df['cum_gains'] = df.groupby('user_id')['y_true'].cumsum()
    df['precision'] = df['cum_gains'] / df['rank']
    
    user_ap = df[df['y_true'] == 1].groupby('user_id')['precision'].mean()
    return user_ap.mean() if not user_ap.empty else 0.0

class CyclicalFeatureEncoder(BaseEstimator, TransformerMixin):
    """Encode cyclical features into sine/cosine components"""
    def __init__(self, features, max_vals):
        self.features = features
        self.max_vals = max_vals
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = X.copy()
        for feature, max_val in zip(self.features, self.max_vals):
            X[f'{feature}_sin'] = np.sin(2 * np.pi * X[feature] / max_val)
            X[f'{feature}_cos'] = np.cos(2 * np.pi * X[feature] / max_val)
        return X

class FeatureInteractionGenerator(BaseEstimator, TransformerMixin):
    """Generate interaction features between important columns"""
    def __init__(self, interactions):
        self.interactions = interactions
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = X.copy()
        for f1, f2 in self.interactions:
            # Multiplication interaction
            X[f'{f1}x{f2}'] = X[f1] * X[f2]
            
            # Ratio interaction (with protection)
            safe_f2 = np.where(X[f2] == 0, 1e-5, X[f2])
            X[f'{f1}div{f2}'] = X[f1] / safe_f2
            
            # Difference interaction
            X[f'{f1}diff{f2}'] = X[f1] - X[f2]
        return X

class CustomEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, min_category_count=5):
        self.encoders = {}
        self.min_category_count = min_category_count
        self.categorical_cols = []

    def fit(self, X, y=None):
        self.categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in self.categorical_cols:
            if isinstance(X[col].dtype, pd.CategoricalDtype):
                if 'missing' not in X[col].cat.categories:
                    X[col] = X[col].cat.add_categories(['missing', 'other'])
            else:
                X[col] = X[col].astype('object')
            
            X_col = X[col].astype(str).fillna('missing')
            value_counts = X_col.value_counts()
            valid_values = value_counts[value_counts >= self.min_category_count].index
            le = LabelEncoder()
            le.fit(X_col)
            self.encoders[col] = le
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.categorical_cols:
            if col in self.encoders:
                le = self.encoders[col]
                if isinstance(X[col].dtype, pd.CategoricalDtype):
                    if 'missing' not in X[col].cat.categories:
                        X[col] = X[col].cat.add_categories(['missing', 'other'])
                X[col] = X[col].astype(str).fillna('missing')
                valid_classes = set(le.classes_)
                X.loc[~X[col].isin(valid_classes), col] = 'other'
                X[col] = le.transform(X[col])
        return X

def reduce_mem_usage(df, preserve_cols=None):
    """Reduce memory usage of dataframe by converting data types."""
    start_mem = df.memory_usage().sum() / 1024**2
    logging.info(f'Memory usage of dataframe is {start_mem:.2f} MB')
    
    preserve_cols = preserve_cols or ['y']  # Default to preserve 'y'
    for col in df.columns:
        col_type = df[col].dtype
        if col in preserve_cols or col_type.name in ['category', 'object']:
            continue  # Skip preserved columns and non-numeric types
        if pd.api.types.is_numeric_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()
            if pd.api.types.is_integer_dtype(col_type):
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            elif pd.api.types.is_float_dtype(col_type):
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        elif col_type == object:
            if df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype('category')
    
    end_mem = df.memory_usage().sum() / 1024**2
    logging.info(f'Memory usage after optimization is {end_mem:.2f} MB')
    logging.info(f'Reduced by {100 * (start_mem - end_mem) / start_mem:.1f}%')
    return df

def safe_to_seconds(series):
    seconds = []
    for val in series:
        try:
            if isinstance(val, str) and re.fullmatch(r"\d{1,2}:\d{1,2}(:\d{1,2})?", val):
                time_parts = list(map(int, val.split(":")))
                if len(time_parts) == 2:
                    seconds.append(time_parts[0] * 60 + time_parts[1])
                elif len(time_parts) == 3:
                    seconds.append(time_parts[0] * 3600 + time_parts[1] * 60 + time_parts[2])
                else:
                    seconds.append(np.nan)
            else:
                seconds.append(np.nan)
        except:
            seconds.append(np.nan)
    return pd.Series(seconds)

def optimize_hyperparameters(X_train, y_train, X_val, y_val, user_ids_val, cat_features):
    """Optimize hyperparameters using Optuna with MAP@7"""
    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 128, 512),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.95),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.95),
            'bagging_freq': trial.suggest_int('bagging_freq', 3, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 100, 1000),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 100, 1000),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 15, 40),
            'random_state': CONFIG['random_state'],
            'verbosity': -1,
            'n_jobs': -1,
        }
        
        train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
        val_set = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_features, reference=train_set)
        
        model = lgb.train(
            params,
            train_set,
            num_boost_round=1000,
            valid_sets=[val_set],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=100)
            ]
        )
        
        val_pred = model.predict(X_val)
        map7 = map_at_7(y_val, val_pred, user_ids_val)
        return map7

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=CONFIG['random_state']))
    study.optimize(objective, n_trials=CONFIG['optuna_trials'], timeout=CONFIG['optuna_timeout'])
    
    logging.info(f"Best hyperparameters: {study.best_params}")
    logging.info(f"Best MAP@7 from optimization: {study.best_value:.5f}")
    
    base_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'random_state': CONFIG['random_state'],
        'n_jobs': -1,
    }
    return {**base_params, **study.best_params}

def read_data_memory_efficient(file_path):
    """Read data in chunks with memory optimization"""
    logging.info(f"Reading {file_path} in chunks...")
    chunk_list = []
    
    for i, chunk in enumerate(pd.read_csv(file_path, chunksize=CONFIG['chunk_size'], low_memory=False)):
        chunk = reduce_mem_usage(chunk)
        chunk_list.append(chunk)
        logging.info(f"Processed chunk {i+1}")
    
    return pd.concat(chunk_list, ignore_index=True)

def create_advanced_features(df):
    """Generate advanced feature interactions and transformations"""
    # Behavioral features
    df['discount_sensitivity'] = df['avg_discount_rate'] / (df['avg_spending_power'] + 1e-5)
    df['value_ratio'] = df['avg_discount_rate'] / (df['avg_trans_amount'] + 1e-3)
    df['redemption_velocity'] = df['total_clicks'] / (df['avg_redemption_freq'] + 1e-5)
    
    # Time-based features
    if 'avg_time_diff' in df.columns:
        df['recency'] = 1 / (df['avg_time_diff'] + 1)
    
    # Industry interactions
    if 'most_frequent_industry' in df.columns and 'avg_industry_diversity' in df.columns:
        df['industry_stability'] = (df['most_frequent_industry'].astype('category').cat.codes + 1) / (df['avg_industry_diversity'] + 1e-5)
    
    # Spending patterns
    df['spending_consistency'] = df['avg_trans_amount'] / (df['std_trans_amount'] + 1e-5)
    df['premium_offer_affinity'] = (df['max_trans_amount'] - df['avg_trans_amount']) * df['prop_high_redemption']
    
    # Time intelligence features
    df['weekend_affinity'] = df['avg_is_weekend'] * df['prop_has_click']
    df['hour_engagement'] = df['avg_hour_sin'] * df['total_clicks']
    
    # Financial behavior
    df['liquidity_ratio'] = df['total_trans_amount'] / (df['total_trans_count'] + 1e-5)
    df['debit_utilization'] = df['avg_debit_credit_ratio'] * df['avg_trans_amount']
    
    # Complex interactions
    df['offer_value_score'] = (df['avg_discount_rate'] * df['total_trans_count']) / (df['avg_click_latency'] + 1)
    df['user_activity_score'] = (df['total_clicks'] + df['total_trans_count']) * df['prop_has_click']
    
    # Time-based ratios
    if 'avg_day_of_week' in df.columns and 'week_day' in df.columns:
        df['dow_consistency'] = df['avg_day_of_week'] - df['week_day']
    
    # Normalize high-range features
    for col in ['total_trans_amount', 'max_trans_amount', 'total_trans_count']:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(df[col])
    
    return reduce_mem_usage(df)

def diversity_reranking(predictions, user_ids, diversity_features, k=7, alpha=0.3):
    """Re-rank predictions to increase diversity in top positions"""
    df = pd.DataFrame({
        'user_id': user_ids,
        'prediction': predictions
    }).reset_index()
    
    # Add diversity features
    df = pd.concat([df, diversity_features], axis=1)
    
    # Group by user and get top 2k predictions to rerank
    reranked_dfs = []
    for user_id, group in df.groupby('user_id'):
        group = group.sort_values('prediction', ascending=False).head(2000)
        
        # Initialize reranking
        top_k = []
        remaining = group.copy()
        
        # First item is the highest prediction
        first_item = remaining.iloc[[0]]
        top_k.append(first_item)
        remaining = remaining.iloc[1:]
        
        # Build the rest of the list with diversity
        while len(top_k) < k and len(remaining) > 0:
            last_item = top_k[-1]
            
            # Calculate diversity scores
            diversity_scores = []
            for idx, row in remaining.iterrows():
                diversity_score = 0
                for feature in diversity_features.columns:
                    # Categorical diversity
                    if feature in ['most_frequent_industry', 'most_frequent_id8']:
                        diversity_score += 1 if row[feature] != last_item[feature].values[0] else 0
                    # Numerical diversity
                    else:
                        diversity_score += abs(row[feature] - last_item[feature].values[0])
                diversity_scores.append(diversity_score)
            
            # Blend scores
            blended_scores = (
                alpha * remaining['prediction'].values + 
                (1 - alpha) * np.array(diversity_scores) / max(diversity_scores))
            
            # Select next item
            next_idx = np.argmax(blended_scores)
            next_item = remaining.iloc[[next_idx]]
            top_k.append(next_item)
            remaining = remaining.drop(next_item.index)
        
        reranked_dfs.append(pd.concat(top_k))
    
    return pd.concat(reranked_dfs)['prediction'].values

def apply_pipeline_in_chunks(pipeline, df, feature_cols, chunk_size=CONFIG['chunk_size']):
    """Apply feature pipeline in chunks to save memory."""
    chunks = []
    for start in range(0, len(df), chunk_size):
        end = min(start + chunk_size, len(df))
        chunk = df.iloc[start:end][feature_cols]
        chunk_features = pd.DataFrame(pipeline.transform(chunk))
        chunks.append(chunk_features)
        logging.info(f"Processed feature chunk {start//chunk_size + 1}")
    return pd.concat(chunks).reset_index(drop=True)

def main():
    # ========== Load Base Datasets ==========
    logging.info("Loading base datasets...")
    train = read_data_memory_efficient("train_augmented.csv")
    test = read_data_memory_efficient("test_augmented.csv")
    
    # Reduce memory immediately after loading
    train = reduce_mem_usage(train, preserve_cols=['y'])
    test = reduce_mem_usage(test)
    
    # Preserve user IDs
    train_user_ids = train['id2'].copy()
    test_user_ids = test['id2'].copy()
    train_groups = train['id3'].copy()
    
    # ========== Enhanced Feature Engineering ==========
    logging.info("Creating advanced features...")
    train = create_advanced_features(train)
    test = create_advanced_features(test)
    
    # Reduce memory after feature creation
    train = reduce_mem_usage(train, preserve_cols=['y'])
    test = reduce_mem_usage(test)
    
    # ========== Feature Engineering Pipeline ==========
    logging.info("Running feature engineering pipeline...")
    cyclical_features = ['avg_hour_sin', 'avg_day_of_week', 'avg_month', 'week_day']
    max_vals = [1, 7, 12, 7]
    
    # Define important interactions
    interactions = [
        ('avg_discount_rate', 'avg_spending_power'),
        ('total_clicks', 'prop_has_click'),
        ('avg_trans_amount', 'std_trans_amount'),
        ('avg_redemption_freq', 'prop_high_redemption'),
        ('avg_time_diff', 'total_trans_count'),
        ('avg_hour_sin', 'total_clicks'),
        ('avg_debit_credit_ratio', 'avg_trans_amount')
    ]
    
    # Build pipeline
    feature_pipeline = FeatureUnion([
        ('cyclical', CyclicalFeatureEncoder(cyclical_features, max_vals)),
        ('interactions', FeatureInteractionGenerator(interactions))
    ])
    
    # Fit pipeline on a sample to initialize encoders
    feature_cols = [col for col in train.columns if col not in ['id1', 'id2', 'id3', 'id5', 'offer_id', 'y']]
    sample_size = min(10000, len(train))
    feature_pipeline.fit(train[feature_cols].iloc[:sample_size])
    
    # Apply pipeline in chunks
    train_features = apply_pipeline_in_chunks(feature_pipeline, train, feature_cols)
    test_features = apply_pipeline_in_chunks(feature_pipeline, test, feature_cols)
    
    # Log memory usage before and after concat
    logging.info(f"Train memory before concat: {train.memory_usage().sum() / 1024**2:.2f} MB")
    logging.info(f"Train features memory: {train_features.memory_usage().sum() / 1024**2:.2f} MB")
    
    # Combine with original data
    train = pd.concat([train, train_features.add_prefix('fe_')], axis=1)
    test = pd.concat([test, test_features.add_prefix('fe_')], axis=1)
    
    logging.info(f"Train memory after concat: {train.memory_usage().sum() / 1024**2:.2f} MB")
    
    # Reduce memory
    train = reduce_mem_usage(train, preserve_cols=['y'])
    test = reduce_mem_usage(test)
    
    # ========== Prepare Model Features ==========
    logging.info("Preparing model columns...")
    common_cols = [col for col in train.columns if col in test.columns and col not in ['id1', 'id2', 'id3', 'id5', 'offer_id', 'y']]
    X = train[common_cols].copy()
    y = train['y'].copy().astype(np.int8)
    X_test = test[common_cols].copy()
    
    # ========== Encode Categorical Features ==========
    logging.info("Encoding categorical features...")
    cat_cols = [col for col in X.select_dtypes(include=['object', 'category']).columns]
    encoder = CustomEncoder(CONFIG['min_category_count'])
    X = encoder.fit_transform(X, y)
    X_test = encoder.transform(X_test)
    
    # Reduce memory
    X = reduce_mem_usage(X)
    X_test = reduce_mem_usage(X_test)
    
    # ========== Hyperparameter Optimization ==========
    logging.info("Optimizing hyperparameters...")
    X_hyper, X_val, y_hyper, y_val, groups_hyper, groups_val = train_test_split(
        X, y, train_groups, test_size=0.2, 
        random_state=CONFIG['random_state'], stratify=y
    )
    
    cat_features = list(set(cat_cols).intersection(set(X_hyper.columns)))
    best_params = optimize_hyperparameters(
        X_hyper, y_hyper, X_val, y_val, train_user_ids.loc[X_val.index], cat_features
    )
    
    # Clean up memory
    del X_hyper, X_val, y_hyper, y_val
    gc.collect()
    
    # ========== Two-Stage Modeling ==========
    logging.info("Starting two-stage modeling...")
    test_preds_stage1 = np.zeros(len(X_test))
    test_preds_stage2 = np.zeros(len(X_test))
    cv_map_scores = []
    
    # Use GroupKFold with id3 groups
    gkf = GroupKFold(n_splits=CONFIG['n_splits'])
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=train_groups)):
        logging.info(f"\n====== Fold {fold+1}/{CONFIG['n_splits']} ======")
        
        # Stage 1: Binary Classification
        logging.info("Training Stage 1: Binary Classifier")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        user_val = train_user_ids.iloc[val_idx]
        
        train_set = lgb.Dataset(X_train, label=y_train)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
        
        model_stage1 = lgb.train(
            best_params,
            train_set,
            num_boost_round=1000,
            valid_sets=[val_set],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=100)
            ]
        )
        
        # Stage 1 predictions
        val_pred_stage1 = model_stage1.predict(X_val)
        test_pred_stage1 = model_stage1.predict(X_test)
        test_preds_stage1 += test_pred_stage1 / CONFIG['n_splits']
        
        # Stage 2: Ranking Model
        logging.info("Training Stage 2: Ranking Model")
        # Add stage1 predictions as features
        X_train['stage1_pred'] = model_stage1.predict(X_train)
        X_val['stage1_pred'] = val_pred_stage1
        X_test_fold = X_test.copy()
        X_test_fold['stage1_pred'] = test_pred_stage1
        
        # Ranking parameters
        rank_params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [CONFIG['lambdarank_position']],
            'lambdarank_truncation_level': CONFIG['lambdarank_position'],
            'learning_rate': 0.03,
            'num_leaves': 192,
            'min_data_in_leaf': 150,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbosity': -1,
            'random_state': CONFIG['random_state'],
            'force_row_wise': True
        }
        
        # Group by user for ranking
        train_groups_size = X_val.groupby(train_user_ids.iloc[val_idx]).size().values
        
        train_set_rank = lgb.Dataset(X_train, label=y_train, group=[len(train_idx)])
        val_set_rank = lgb.Dataset(X_val, label=y_val, group=train_groups_size, reference=train_set_rank)
        
        model_stage2 = lgb.train(
            rank_params,
            train_set_rank,
            num_boost_round=500,
            valid_sets=[val_set_rank],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30, verbose=False),
                lgb.log_evaluation(period=50)
            ]
        )
        
        # Stage 2 predictions
        val_pred_stage2 = model_stage2.predict(X_val)
        test_pred_stage2 = model_stage2.predict(X_test_fold)
        test_preds_stage2 += test_pred_stage2 / CONFIG['n_splits']
        
        # Calculate MAP@7 for both stages
        map_stage1 = map_at_7(y_val, val_pred_stage1, user_val)
        map_stage2 = map_at_7(y_val, val_pred_stage2, user_val)
        cv_map_scores.append(map_stage2)
        
        logging.info(f"Stage 1 MAP@7: {map_stage1:.5f}")
        logging.info(f"Stage 2 MAP@7: {map_stage2:.5f}")
        
        # Clean up
        del model_stage1, model_stage2, train_set, val_set, train_set_rank, val_set_rank
        gc.collect()
    
    logging.info(f"\nMean CV MAP@7: {np.mean(cv_map_scores):.5f}")
    
    # ========== Blending and Re-ranking ==========
    logging.info("Blending predictions and re-ranking...")
    # Blend stage1 and stage2 predictions
    blended_preds = (
        CONFIG['final_blend_ratio'] * test_preds_stage1 +
        (1 - CONFIG['final_blend_ratio']) * test_preds_stage2
    )
    
    # Select diversity features for re-ranking
    diversity_cols = [
        'most_frequent_industry', 'avg_industry_diversity',
        'most_frequent_id8', 'avg_spending_power'
    ]
    diversity_features = test[diversity_cols].copy()
    
    # Apply diversity re-ranking
    reranked_preds = diversity_reranking(
        blended_preds, 
        test_user_ids, 
        diversity_features,
        k=7,
        alpha=0.3
    )
    
    # ========== Generate Submission ==========
    logging.info("Generating submission...")
    submission = test[['id1', 'id2', 'id3', 'id5']].copy()
    submission['pred'] = reranked_preds
    
    # Group-wise calibration
    if 'dominant_industry_code' in submission.columns:
        segment_col = 'dominant_industry_code'
        segment_means = train.groupby(segment_col)['y'].mean()
        global_mean = train['y'].mean()
        submission['segment_mean'] = submission[segment_col].map(segment_means).fillna(global_mean)
        submission['pred'] = submission['pred'] * (submission['segment_mean'] / global_mean)
    
    submission['pred'] = submission.groupby('id2')['pred'].rank(pct=True)
    submission[['id1', 'id2', 'id3', 'id5', 'pred']].to_csv("enhanced_ctr_submission.csv", index=False)
    
    logging.info("Submission saved successfully")

if __name__ == "__main__":
    main()
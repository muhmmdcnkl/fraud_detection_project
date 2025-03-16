import pandas as pd
import numpy as np
import os
import pickle
import time
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, make_scorer
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Set display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

# Define file path
file_path = "DS Classification Case Data.csv"

# Define batch size and temp directory
BATCH_SIZE = 100000  # Adjust based on your system's memory
TEMP_DIR = "temp_fraud_data"

# Define location standardization mappings
COUNTRY_MAPPING = {
    'TURKEY': 'Turkey',
    'TÜRKİYE': 'Turkey',
    'TÜRKIYE': 'Turkey',
    'Türkiye': 'Turkey',
    'türkiye': 'Turkey',
    'turkey': 'Turkey',
    'TR': 'Turkey'
}

CITY_MAPPING = {
    'ISTANBUL': 'Istanbul',
    'İSTANBUL': 'Istanbul',
    'istanbul': 'Istanbul',
    'ANKARA': 'Ankara',
    'ankara': 'Ankara',
    'IZMIR': 'Izmir',
    'İZMIR': 'Izmir',
    'İZMİR': 'Izmir',
    'izmir': 'Izmir',
    'ANTALYA': 'Antalya',
    'antalya': 'Antalya'
}

def ensure_temp_dir():
    """Create temporary directory if it doesn't exist"""
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
        print(f"Created temporary directory: {TEMP_DIR}")

def count_rows(file_path):
    """Count the number of rows in the CSV file without loading it entirely"""
    print("Counting rows in the dataset...")
    try:
        # Try with utf-8 encoding first
        with open(file_path, 'r', encoding='utf-8') as f:
            # Count header
            row_count = -1
            for row_count, _ in enumerate(f):
                if row_count % 1000000 == 0 and row_count > 0:
                    print(f"Counted {row_count} rows so far...")
        return row_count + 1
    except UnicodeDecodeError:
        # If utf-8 fails, try with latin-1 encoding (which can handle all byte values)
        print("UTF-8 encoding failed, trying with latin-1 encoding...")
        with open(file_path, 'r', encoding='latin-1') as f:
            # Count header
            row_count = -1
            for row_count, _ in enumerate(f):
                if row_count % 1000000 == 0 and row_count > 0:
                    print(f"Counted {row_count} rows so far...")
        return row_count + 1

def clean_location_data(df):
    """
    Clean and standardize location data (buyer_city, buyer_country)
    
    Args:
        df: DataFrame containing location data
        
    Returns:
        DataFrame with standardized location data
    """
    print("Cleaning location data...")
    
    # Standardize country names
    if 'buyer_country' in df.columns:
        # Convert to string to handle any non-string values
        df['buyer_country'] = df['buyer_country'].astype(str)
        
        # Apply mapping for known values
        df['buyer_country'] = df['buyer_country'].apply(
            lambda x: COUNTRY_MAPPING.get(x, x) if x in COUNTRY_MAPPING else x
        )
        
        # For values not in mapping, capitalize properly
        df['buyer_country'] = df['buyer_country'].apply(
            lambda x: x.title() if x not in COUNTRY_MAPPING.values() else x
        )
    
    # Standardize city names
    if 'buyer_city' in df.columns:
        # Convert to string to handle any non-string values
        df['buyer_city'] = df['buyer_city'].astype(str)
        
        # Apply mapping for known values
        df['buyer_city'] = df['buyer_city'].apply(
            lambda x: CITY_MAPPING.get(x, x) if x in CITY_MAPPING else x
        )
        
        # For values not in mapping, capitalize properly
        df['buyer_city'] = df['buyer_city'].apply(
            lambda x: x.title() if x not in CITY_MAPPING.values() else x
        )
    
    return df

def process_in_batches(file_path, batch_size=BATCH_SIZE, max_rows=None):
    """
    Process the dataset in batches to calculate aggregated statistics
    
    Args:
        file_path: Path to the CSV file
        batch_size: Number of rows to process in each batch
        max_rows: Maximum number of rows to process (None for all)
        
    Returns:
        Dictionary with aggregated statistics
    """
    print(f"Processing data in batches of {batch_size} rows...")
    
    # Initialize statistics
    stats = {
        'merchant_stats': {},
        'payment_type_stats': {},
        'payment_channel_stats': {},
        'card_stats': {},
        'total_rows': 0,
        'fraud_count': 0
    }
    
    # Calculate total rows to process
    total_rows = count_rows(file_path) - 1  # Subtract header row
    if max_rows is not None:
        total_rows = min(total_rows, max_rows)
    
    print(f"Processing {total_rows} rows in total")
    
    # Process in batches
    for i in range(0, total_rows, batch_size):
        end_row = min(i + batch_size, total_rows)
        print(f"Processing batch {i//batch_size + 1}: rows {i} to {end_row}")
        
        # Read batch with appropriate encoding
        try:
            # Try with utf-8 encoding first
            df_batch = pd.read_csv(file_path, skiprows=range(1, i+1) if i > 0 else None, 
                                  nrows=min(batch_size, end_row-i), header=0 if i == 0 else 0,
                                  encoding='utf-8')
        except UnicodeDecodeError:
            # If utf-8 fails, try with latin-1 encoding
            print("UTF-8 encoding failed, trying with latin-1 encoding...")
            df_batch = pd.read_csv(file_path, skiprows=range(1, i+1) if i > 0 else None, 
                                  nrows=min(batch_size, end_row-i), header=0 if i == 0 else 0,
                                  encoding='latin-1')
        
        # Clean location data
        df_batch = clean_location_data(df_batch)
        
        # Update total statistics
        stats['total_rows'] += len(df_batch)
        stats['fraud_count'] += df_batch['is_fraud_transaction'].sum()
        
        # Update merchant statistics
        merchant_batch = df_batch.groupby('merchant_id').agg({
            'payment_id': 'count',
            'price': ['sum', 'mean', 'std', 'min', 'max'],
            'is_fraud_transaction': 'sum'
        })
        
        merchant_batch.columns = ['_'.join(col).strip() for col in merchant_batch.columns.values]
        
        for merchant_id, row in merchant_batch.iterrows():
            if merchant_id not in stats['merchant_stats']:
                stats['merchant_stats'][merchant_id] = {
                    'transaction_count': 0,
                    'price_sum': 0,
                    'price_values': [],
                    'fraud_count': 0
                }
            
            stats['merchant_stats'][merchant_id]['transaction_count'] += row['payment_id_count']
            stats['merchant_stats'][merchant_id]['price_sum'] += row['price_sum']
            stats['merchant_stats'][merchant_id]['price_values'].extend(df_batch[df_batch['merchant_id'] == merchant_id]['price'].tolist())
            stats['merchant_stats'][merchant_id]['fraud_count'] += row['is_fraud_transaction_sum']
        
        # Update payment type statistics
        if 'payment_type' in df_batch.columns:
            payment_type_batch = df_batch.groupby('payment_type').agg({
                'payment_id': 'count',
                'is_fraud_transaction': 'sum'
            })
            
            for payment_type, row in payment_type_batch.iterrows():
                if payment_type not in stats['payment_type_stats']:
                    stats['payment_type_stats'][payment_type] = {
                        'transaction_count': 0,
                        'fraud_count': 0
                    }
                
                stats['payment_type_stats'][payment_type]['transaction_count'] += row['payment_id']
                stats['payment_type_stats'][payment_type]['fraud_count'] += row['is_fraud_transaction']
        
        # Update payment channel statistics
        if 'payment_channel' in df_batch.columns:
            payment_channel_batch = df_batch.groupby('payment_channel').agg({
                'payment_id': 'count',
                'is_fraud_transaction': 'sum'
            })
            
            for payment_channel, row in payment_channel_batch.iterrows():
                if payment_channel not in stats['payment_channel_stats']:
                    stats['payment_channel_stats'][payment_channel] = {
                        'transaction_count': 0,
                        'fraud_count': 0
                    }
                
                stats['payment_channel_stats'][payment_channel]['transaction_count'] += row['payment_id']
                stats['payment_channel_stats'][payment_channel]['fraud_count'] += row['is_fraud_transaction']
        
        # Update card statistics
        if 'bin_number' in df_batch.columns and 'last_four_digits' in df_batch.columns:
            card_batch = df_batch.groupby(['bin_number', 'last_four_digits']).agg({
                'payment_id': 'count',
                'price': 'mean',
                'is_fraud_transaction': 'sum'
            })
            
            for (bin_number, last_four_digits), row in card_batch.iterrows():
                card_key = f"{bin_number}_{last_four_digits}"
                if card_key not in stats['card_stats']:
                    stats['card_stats'][card_key] = {
                        'transaction_count': 0,
                        'price_sum': 0,
                        'price_count': 0,
                        'fraud_count': 0
                    }
                
                stats['card_stats'][card_key]['transaction_count'] += row['payment_id']
                stats['card_stats'][card_key]['price_sum'] += row['price'] * row['payment_id']
                stats['card_stats'][card_key]['price_count'] += row['payment_id']
                stats['card_stats'][card_key]['fraud_count'] += row['is_fraud_transaction']
    
    # Calculate derived statistics
    print("Calculating derived statistics...")
    
    # Merchant fraud rates
    for merchant_id, merchant_data in stats['merchant_stats'].items():
        merchant_data['fraud_rate'] = merchant_data['fraud_count'] / merchant_data['transaction_count'] if merchant_data['transaction_count'] > 0 else 0
        merchant_data['avg_price'] = merchant_data['price_sum'] / merchant_data['transaction_count'] if merchant_data['transaction_count'] > 0 else 0
        
        # Calculate price statistics
        if merchant_data['price_values']:
            merchant_data['price_std'] = np.std(merchant_data['price_values']) if len(merchant_data['price_values']) > 1 else 0
            merchant_data['price_min'] = min(merchant_data['price_values'])
            merchant_data['price_max'] = max(merchant_data['price_values'])
        else:
            merchant_data['price_std'] = 0
            merchant_data['price_min'] = 0
            merchant_data['price_max'] = 0
        
        # Remove raw price values to save memory
        del merchant_data['price_values']
    
    # Payment type fraud rates
    for payment_type, data in stats['payment_type_stats'].items():
        data['fraud_rate'] = data['fraud_count'] / data['transaction_count'] if data['transaction_count'] > 0 else 0
    
    # Payment channel fraud rates
    for payment_channel, data in stats['payment_channel_stats'].items():
        data['fraud_rate'] = data['fraud_count'] / data['transaction_count'] if data['transaction_count'] > 0 else 0
    
    # Card statistics
    for card_key, data in stats['card_stats'].items():
        data['avg_price'] = data['price_sum'] / data['price_count'] if data['price_count'] > 0 else 0
        data['fraud_rate'] = data['fraud_count'] / data['transaction_count'] if data['transaction_count'] > 0 else 0
        # Remove intermediate calculations
        del data['price_sum']
        del data['price_count']
    
    # Overall fraud rate
    stats['overall_fraud_rate'] = stats['fraud_count'] / stats['total_rows'] if stats['total_rows'] > 0 else 0
    
    print(f"Processed {stats['total_rows']} rows in total")
    print(f"Overall fraud rate: {stats['overall_fraud_rate'] * 100:.4f}%")
    
    # Save statistics to file
    with open(os.path.join(TEMP_DIR, 'fraud_stats.pkl'), 'wb') as f:
        pickle.dump(stats, f)
    
    return stats

def enrich_batch_with_features(df_batch, stats):
    """
    Enrich a batch with engineered features using pre-computed statistics
    
    Args:
        df_batch: DataFrame batch to enrich
        stats: Pre-computed statistics
        
    Returns:
        DataFrame with additional features
    """
    print("Enriching batch with features...")
    
    # Clean location data
    df_batch = clean_location_data(df_batch)
    
    # Convert payment_date to datetime
    df_batch['payment_date'] = pd.to_datetime(df_batch['payment_date'])
    
    # Extract date features
    df_batch['payment_year'] = df_batch['payment_date'].dt.year
    df_batch['payment_month'] = df_batch['payment_date'].dt.month
    df_batch['payment_day'] = df_batch['payment_date'].dt.day
    df_batch['payment_hour'] = df_batch['payment_date'].dt.hour
    df_batch['payment_dayofweek'] = df_batch['payment_date'].dt.dayofweek
    df_batch['is_weekend'] = df_batch['payment_dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Add time-based features
    df_batch['is_night'] = df_batch['payment_hour'].apply(lambda x: 1 if (x >= 22 or x <= 6) else 0)
    df_batch['is_morning'] = df_batch['payment_hour'].apply(lambda x: 1 if (x > 6 and x <= 12) else 0)
    df_batch['is_afternoon'] = df_batch['payment_hour'].apply(lambda x: 1 if (x > 12 and x <= 18) else 0)
    df_batch['is_evening'] = df_batch['payment_hour'].apply(lambda x: 1 if (x > 18 and x < 22) else 0)
    
    # Add merchant features
    df_batch['merchant_transaction_count'] = df_batch['merchant_id'].apply(
        lambda x: stats['merchant_stats'].get(x, {}).get('transaction_count', 0)
    )
    
    df_batch['merchant_avg_price'] = df_batch['merchant_id'].apply(
        lambda x: stats['merchant_stats'].get(x, {}).get('avg_price', 0)
    )
    
    df_batch['merchant_price_std'] = df_batch['merchant_id'].apply(
        lambda x: stats['merchant_stats'].get(x, {}).get('price_std', 0)
    )
    
    df_batch['merchant_fraud_rate'] = df_batch['merchant_id'].apply(
        lambda x: stats['merchant_stats'].get(x, {}).get('fraud_rate', 0)
    )
    
    # Create price-related features
    df_batch['price_to_avg_ratio'] = df_batch['price'] / df_batch['merchant_avg_price'].replace(0, 1)
    
    # Create z-score of price for each merchant
    df_batch['price_zscore'] = (df_batch['price'] - df_batch['merchant_avg_price']) / df_batch['merchant_price_std'].replace(0, 1)
    
    # Add payment type features
    if 'payment_type' in df_batch.columns:
        df_batch['payment_type_fraud_rate'] = df_batch['payment_type'].apply(
            lambda x: stats['payment_type_stats'].get(x, {}).get('fraud_rate', 0)
        )
    
    # Add payment channel features
    if 'payment_channel' in df_batch.columns:
        df_batch['payment_channel_fraud_rate'] = df_batch['payment_channel'].apply(
            lambda x: stats['payment_channel_stats'].get(x, {}).get('fraud_rate', 0)
        )
    
    # Add card features
    if 'bin_number' in df_batch.columns and 'last_four_digits' in df_batch.columns:
        df_batch['card_key'] = df_batch['bin_number'].astype(str) + '_' + df_batch['last_four_digits'].astype(str)
        
        df_batch['card_transaction_count'] = df_batch['card_key'].apply(
            lambda x: stats['card_stats'].get(x, {}).get('transaction_count', 0)
        )
        
        df_batch['card_avg_price'] = df_batch['card_key'].apply(
            lambda x: stats['card_stats'].get(x, {}).get('avg_price', 0)
        )
        
        df_batch['card_fraud_rate'] = df_batch['card_key'].apply(
            lambda x: stats['card_stats'].get(x, {}).get('fraud_rate', 0)
        )
        
        # Price deviation from card average
        df_batch['price_deviation_from_card_avg'] = df_batch['price'] - df_batch['card_avg_price']
        df_batch['price_deviation_ratio'] = df_batch['price'] / df_batch['card_avg_price'].replace(0, 1)
        
        # Drop temporary key
        df_batch.drop('card_key', axis=1, inplace=True)
    
    # Create interaction features
    if 'payment_type' in df_batch.columns and 'payment_channel' in df_batch.columns:
        df_batch['type_channel_interaction'] = df_batch['payment_type'] + '_' + df_batch['payment_channel']
    
    # Drop original datetime columns
    df_batch.drop(['payment_date'], axis=1, inplace=True)
    
    # Drop unnecessary columns
    drop_cols = ['payment_id']
    encrypted_cols = ['buyer_name', 'buyer_surname', 'buyer_email', 'buyer_gsm']
    drop_cols.extend([col for col in encrypted_cols if col in df_batch.columns])
    
    df_batch.drop([col for col in drop_cols if col in df_batch.columns], axis=1, inplace=True)
    
    return df_batch

def reduce_cardinality(df, categorical_cols, threshold=0.01):
    """Reduce the cardinality of categorical features"""
    for col in categorical_cols:
        if col in df.columns:
            # Calculate value counts and frequencies
            value_counts = df[col].value_counts(normalize=True)
            
            # Identify rare categories
            rare_categories = value_counts[value_counts < threshold].index.tolist()
            
            # Replace rare categories with 'Other'
            if rare_categories:
                df[col] = df[col].apply(lambda x: 'Other' if x in rare_categories else x)
    
    return df

def train_model_on_batches(file_path, batch_size=BATCH_SIZE, max_rows=None, sample_for_training=500000, tune_hyperparameters=False):
    """
    Train a model using batch processing
    
    Args:
        file_path: Path to the CSV file
        batch_size: Number of rows to process in each batch
        max_rows: Maximum number of rows to process (None for all)
        sample_for_training: Number of rows to use for training (None for all processed rows)
        tune_hyperparameters: Whether to perform hyperparameter tuning
        
    Returns:
        Trained model
    """
    ensure_temp_dir()
    
    # Step 1: Process the entire dataset in batches to calculate statistics
    if os.path.exists(os.path.join(TEMP_DIR, 'fraud_stats.pkl')):
        print("Loading pre-computed statistics...")
        with open(os.path.join(TEMP_DIR, 'fraud_stats.pkl'), 'rb') as f:
            stats = pickle.load(f)
    else:
        stats = process_in_batches(file_path, batch_size, max_rows)
    
    # Step 2: Prepare data for training
    print(f"Preparing data for training using up to {sample_for_training} rows...")
    
    # Calculate how many batches we need for the training sample
    rows_to_process = min(sample_for_training, stats['total_rows']) if sample_for_training else stats['total_rows']
    batches_needed = (rows_to_process + batch_size - 1) // batch_size
    
    # Initialize lists to store training data
    X_train_parts = []
    X_test_parts = []
    y_train_parts = []
    y_test_parts = []
    
    # Process batches for training
    rows_processed = 0
    for i in range(batches_needed):
        current_batch_size = min(batch_size, rows_to_process - rows_processed)
        if current_batch_size <= 0:
            break
            
        print(f"Processing batch {i+1}/{batches_needed} for training...")
        
        # Read batch with appropriate encoding
        try:
            # Try with utf-8 encoding first
            skip_rows = range(1, i * batch_size + 1) if i > 0 else None
            df_batch = pd.read_csv(file_path, skiprows=skip_rows, nrows=current_batch_size, 
                                  header=0 if i == 0 else 0, encoding='utf-8')
        except UnicodeDecodeError:
            # If utf-8 fails, try with latin-1 encoding
            print("UTF-8 encoding failed, trying with latin-1 encoding...")
            skip_rows = range(1, i * batch_size + 1) if i > 0 else None
            df_batch = pd.read_csv(file_path, skiprows=skip_rows, nrows=current_batch_size, 
                                  header=0 if i == 0 else 0, encoding='latin-1')
        
        # Enrich batch with features
        df_batch = enrich_batch_with_features(df_batch, stats)
        
        # Identify categorical columns
        categorical_cols = df_batch.select_dtypes(include=['object']).columns.tolist()
        
        # Reduce cardinality
        df_batch = reduce_cardinality(df_batch, categorical_cols)
        
        # One-hot encode categorical variables
        df_batch = pd.get_dummies(df_batch, columns=categorical_cols, drop_first=True)
        
        # Split features and target
        X_batch = df_batch.drop('is_fraud_transaction', axis=1)
        y_batch = df_batch['is_fraud_transaction']
        
        # Split into train and test
        X_train_batch, X_test_batch, y_train_batch, y_test_batch = train_test_split(
            X_batch, y_batch, test_size=0.2, random_state=42, stratify=y_batch
        )
        
        # Store batch data
        X_train_parts.append(X_train_batch)
        X_test_parts.append(X_test_batch)
        y_train_parts.append(y_train_batch)
        y_test_parts.append(y_test_batch)
        
        rows_processed += current_batch_size
    
    # Combine all batches
    print("Combining batches for training...")
    
    # Get common columns across all batches
    common_train_cols = set(X_train_parts[0].columns)
    for df in X_train_parts[1:]:
        common_train_cols = common_train_cols.intersection(df.columns)
    
    common_test_cols = set(X_test_parts[0].columns)
    for df in X_test_parts[1:]:
        common_test_cols = common_test_cols.intersection(df.columns)
    
    # Use only common columns
    common_cols = list(common_train_cols.intersection(common_test_cols))
    print(f"Using {len(common_cols)} common features across all batches")
    
    # Combine using only common columns
    X_train = pd.concat([df[common_cols] for df in X_train_parts], ignore_index=True)
    X_test = pd.concat([df[common_cols] for df in X_test_parts], ignore_index=True)
    y_train = pd.concat(y_train_parts, ignore_index=True)
    y_test = pd.concat(y_test_parts, ignore_index=True)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Check class imbalance
    fraud_ratio = y_train.mean()
    print(f"Class imbalance in training data: {fraud_ratio * 100:.4f}% fraud transactions")
    
    # Check for missing values
    missing_values = X_train.isnull().sum().sum()
    if missing_values > 0:
        print(f"Found {missing_values} missing values in training data. Filling with appropriate values...")
        
        # Fill missing values
        for col in X_train.columns:
            if X_train[col].isnull().any():
                # For numeric columns, fill with median
                if X_train[col].dtype in ['int64', 'float64']:
                    median_value = X_train[col].median()
                    X_train[col].fillna(median_value, inplace=True)
                    X_test[col].fillna(median_value, inplace=True)
                    print(f"Filled missing values in {col} with median: {median_value}")
                else:
                    # For non-numeric columns, fill with most frequent value
                    most_frequent = X_train[col].mode()[0]
                    X_train[col].fillna(most_frequent, inplace=True)
                    X_test[col].fillna(most_frequent, inplace=True)
                    print(f"Filled missing values in {col} with most frequent value: {most_frequent}")
    
    # Apply SMOTE to handle class imbalance if fraud ratio is low
    if fraud_ratio < 0.1:  # Less than 10% fraud transactions
        print("Applying SMOTE to handle class imbalance...")
        try:
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            
            # Print resampling results
            print(f"Original class distribution: {pd.Series(y_train).value_counts(normalize=True) * 100}")
            print(f"Resampled class distribution: {pd.Series(y_train_resampled).value_counts(normalize=True) * 100}")
            
            # Use resampled data for training
            X_train = X_train_resampled
            y_train = y_train_resampled
        except Exception as e:
            print(f"Error applying SMOTE: {e}")
            print("Continuing with original imbalanced data...")
    
    # Step 3: Train the model
    if tune_hyperparameters:
        print("Performing hyperparameter tuning...")
        
        # Create a smaller subset for hyperparameter tuning to avoid memory issues
        print("Creating a smaller subset for hyperparameter tuning...")
        subset_size = min(100000, len(X_train))  # Use at most 100,000 samples for tuning
        
        # Ensure balanced classes in the subset
        fraud_indices = np.where(y_train == 1)[0]
        non_fraud_indices = np.where(y_train == 0)[0]
        
        # Calculate how many samples to take from each class
        n_fraud = min(subset_size // 2, len(fraud_indices))
        n_non_fraud = min(subset_size - n_fraud, len(non_fraud_indices))
        
        # Randomly sample from each class
        np.random.seed(42)
        selected_fraud_indices = np.random.choice(fraud_indices, n_fraud, replace=False)
        selected_non_fraud_indices = np.random.choice(non_fraud_indices, n_non_fraud, replace=False)
        
        # Combine indices and create the subset
        selected_indices = np.concatenate([selected_fraud_indices, selected_non_fraud_indices])
        X_train_subset = X_train.iloc[selected_indices]
        y_train_subset = y_train.iloc[selected_indices]
        
        print(f"Created subset for tuning with {len(X_train_subset)} samples")
        print(f"Subset class distribution: {pd.Series(y_train_subset).value_counts(normalize=True) * 100}")
        
        # Define a smaller parameter grid
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, None],
            'min_samples_split': [5, 10],
            'max_features': ['sqrt', 'log2']
        }
        
        # Define the scoring metric (weighted F1 score to account for class imbalance)
        scoring = make_scorer(lambda y_true, y_pred: 
                             classification_report(y_true, y_pred, output_dict=True)['1']['f1-score'])
        
        # Create a base model
        rf = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
        
        # Use StratifiedKFold to maintain class distribution in each fold
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        # Create the grid search
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            verbose=2
        )
        
        # Fit the grid search on the subset
        start_time = time.time()
        grid_search.fit(X_train_subset, y_train_subset)
        tuning_time = time.time() - start_time
        
        # Get the best parameters
        best_params = grid_search.best_params_
        print(f"Best parameters found: {best_params}")
        print(f"Hyperparameter tuning completed in {tuning_time:.2f} seconds")
        
        # Train the model with the best parameters on the full dataset
        print("Training model with best parameters on full dataset...")
        model = RandomForestClassifier(
            **best_params,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        # Save best parameters
        with open(os.path.join(TEMP_DIR, 'best_params.pkl'), 'wb') as f:
            pickle.dump(best_params, f)
    else:
        print("Training Random Forest model with default parameters...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
    
    model.fit(X_train, y_train)
    
    # Step 4: Evaluate the model
    print("Evaluating model...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC: {auc_score:.4f}")
    
    # Find optimal threshold
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
    optimal_threshold = thresholds[np.argmax(f1_scores)]
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    
    # Save optimal threshold
    with open(os.path.join(TEMP_DIR, 'optimal_threshold.txt'), 'w') as f:
        f.write(str(optimal_threshold))
    
    # Apply optimal threshold
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 20 Most Important Features:")
    print(feature_importance.head(20))
    
    # Save model
    with open(os.path.join(TEMP_DIR, 'fraud_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    
    # Save feature list
    with open(os.path.join(TEMP_DIR, 'feature_list.pkl'), 'wb') as f:
        pickle.dump(list(X_train.columns), f)
    
    return model, feature_importance

def main():
    print("Fraud Detection with Batch Processing")
    print("=====================================")
    
    # Ask user for parameters
    max_rows = input("Enter maximum number of rows to process (leave blank for all): ")
    max_rows = int(max_rows) if max_rows.strip() else None
    
    sample_for_training = input("Enter number of rows to use for training (recommended: 500000, leave blank for all processed rows): ")
    sample_for_training = int(sample_for_training) if sample_for_training.strip() else None
    
    batch_size = input("Enter batch size (recommended: 100000, leave blank for default): ")
    batch_size = int(batch_size) if batch_size.strip() else BATCH_SIZE
    
    tune_hyperparameters = input("Perform hyperparameter tuning? (y/n, default: n): ").lower() == 'y'
    
    # Train model using batch processing
    model, feature_importance = train_model_on_batches(
        file_path, 
        batch_size=batch_size,
        max_rows=max_rows,
        sample_for_training=sample_for_training,
        tune_hyperparameters=tune_hyperparameters
    )
    
    print("\nFraud detection model completed!")
    print(f"Model and features saved in {TEMP_DIR} directory")

if __name__ == "__main__":
    main() 
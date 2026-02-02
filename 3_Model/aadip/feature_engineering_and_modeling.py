import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. DATA LOADING & PREPROCESSING
# ============================================================================

def load_and_clean_data(filepath):
    """Load data and handle basic cleaning"""
    df = pd.read_csv(filepath, parse_dates=[0], index_col=0)
    
    # Convert all columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill missing values with forward fill then backward fill
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    return df

# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================

def create_time_features(df):
    """Create time-based features"""
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['weekday'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['day_of_year'] = df.index.dayofyear
    
    # Cyclical encoding for hour, month, weekday
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
    
    # Weekend indicator
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    
    return df

def create_lag_features(df, target_col, lags=[1, 2, 3, 6, 12, 24, 48, 96]):
    """Create lag features for target variable"""
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    return df

def create_rolling_features(df, target_col, windows=[24, 48, 96, 168]):
    """Create rolling window statistics"""
    for window in windows:
        df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
        df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
        df[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(window=window).min()
        df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window=window).max()
    return df

def create_energy_features(df):
    """Create domain-specific energy features"""
    # Total renewable generation
    renewable_cols = ['Hydro Run-of-River', 'Biomass', 'Wind offshore', 
                      'Wind onshore', 'Solar', 'Hydro water reservoir']
    if all(col in df.columns for col in renewable_cols):
        df['total_renewable'] = df[renewable_cols].sum(axis=1)
    
    # Total fossil generation
    fossil_cols = ['Fossil brown coal / lignite', 'Fossil hard coal', 
                   'Fossil oil', 'Fossil gas', 'Fossil coal-derived gas']
    fossil_cols = [col for col in fossil_cols if col in df.columns]
    if fossil_cols:
        df['total_fossil'] = df[fossil_cols].sum(axis=1)
    
    # Renewable ratio
    if 'total_renewable' in df.columns and 'Load' in df.columns:
        df['renewable_ratio'] = df['total_renewable'] / (df['Load'] + 1e-6)
    
    # Net generation
    if 'Load' in df.columns and 'Cross border electricity trading' in df.columns:
        df['net_generation'] = df['Load'] - df['Cross border electricity trading']
    
    return df

def engineer_features(df, target_col='Day Ahead Auction (DE-LU)'):
    """Main feature engineering pipeline"""
    print("Starting feature engineering...")
    
    # Create time features
    df = create_time_features(df)
    print("✓ Time features created")
    
    # Create lag features
    df = create_lag_features(df, target_col)
    print("✓ Lag features created")
    
    # Create rolling features
    df = create_rolling_features(df, target_col)
    print("✓ Rolling features created")
    
    # Create energy-specific features
    df = create_energy_features(df)
    print("✓ Energy features created")
    
    # Drop rows with NaN values (from lag/rolling features)
    df = df.dropna()
    print(f"✓ Feature engineering complete. Shape: {df.shape}")
    
    return df

# ============================================================================
# 3. MODEL TRAINING
# ============================================================================

def prepare_train_test(df, target_col='Day Ahead Auction (DE-LU)', test_size=0.2):
    """Prepare train/test split"""
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Time-based split (no shuffle for time series)
    split_idx = int(len(df) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_train.columns

def train_models(X_train, X_test, y_train, y_test):
    """Train multiple models and compare"""
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    models = {
        'Ridge': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, max_depth=7, learning_rate=0.1, random_state=42)
    }
    
    results = {}
    trained_models = {}
    
    print("\nTraining models...")
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
        trained_models[name] = model
        
        print(f"✓ {name}: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.4f}")
    
    return trained_models, results

# ============================================================================
# 4. LSTM/GRU MODEL
# ============================================================================

def prepare_sequences(X, y, seq_length=96):
    """Prepare sequences for LSTM/GRU"""
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

def train_lstm_model(X_train, X_test, y_train, y_test, seq_length=96):
    """Train LSTM model"""
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
        
        print("\nPreparing sequences for LSTM...")
        X_train_seq, y_train_seq = prepare_sequences(X_train, y_train.values, seq_length)
        X_test_seq, y_test_seq = prepare_sequences(X_test, y_test.values, seq_length)
        
        print(f"Training data shape: {X_train_seq.shape}")
        
        # Build LSTM model
        model = Sequential([
            LSTM(128, activation='relu', return_sequences=True, input_shape=(seq_length, X_train_seq.shape[2])),
            Dropout(0.2),
            LSTM(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        print("Training LSTM...")
        history = model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_test_seq, y_test_seq),
            epochs=50,
            batch_size=64,
            callbacks=[early_stop],
            verbose=1
        )
        
        # Evaluate
        y_pred = model.predict(X_test_seq).flatten()
        mae = mean_absolute_error(y_test_seq, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_seq, y_pred))
        r2 = r2_score(y_test_seq, y_pred)
        
        print(f"\n✓ LSTM: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.4f}")
        
        return model, history, {'MAE': mae, 'RMSE': rmse, 'R2': r2}
    
    except ImportError:
        print("TensorFlow not installed. Skipping LSTM model.")
        return None, None, None

# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Load data (replace with your actual file path)
    print("Loading data...")
    # df = load_and_clean_data('your_data.csv')
    
    # For demonstration, creating sample data structure
    print("Note: Replace the data loading section with your actual CSV file path")
    print("\nTo use this script:")
    print("1. Update the file path in load_and_clean_data()")
    print("2. Run: python feature_engineering_and_modeling.py")
    
    # Example workflow (uncomment when you have data):
    """
    # Load and engineer features
    df = load_and_clean_data('your_data.csv')
    df = engineer_features(df)
    
    # Prepare train/test
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_train_test(df)
    
    # Train traditional models
    trained_models, results = train_models(X_train, X_test, y_train, y_test)
    
    # Train LSTM model
    lstm_model, history, lstm_results = train_lstm_model(X_train, X_test, y_train, y_test)
    
    # Save best model
    import joblib
    best_model_name = max(results, key=lambda x: results[x]['R2'])
    joblib.dump(trained_models[best_model_name], f'{best_model_name}_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print(f"\nBest model ({best_model_name}) saved!")
    """

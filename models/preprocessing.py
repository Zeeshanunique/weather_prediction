import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import glob

class DataPreprocessor:
    def __init__(self, data_path='data'):
        """Initialize the DataPreprocessor.
        
        Args:
            data_path (str): Path to directory containing CSV files
        """
        self.data_path = data_path
        self.scaler = MinMaxScaler()
        self.data = None
        self.station_data = {}
        self.pollutants = ['CO', 'NO', 'NO2', 'NOx', 'O3', 'PM10', 'PM2.5', 'SO2']
        
    def load_data(self):
        """Load all CSV files from the data directory."""
        csv_files = glob.glob(os.path.join(self.data_path, '*.csv'))
        
        all_data = []
        for file in csv_files:
            station_name = os.path.basename(file).split(' ')[0]
            df = pd.read_csv(file)
            
            # Check column names and standardize case
            df.columns = [col.strip() for col in df.columns]  # Remove any whitespace
            
            # Find date column (case-insensitive)
            date_col = None
            for col in df.columns:
                if col.lower() == 'date':
                    date_col = col
                    break
            
            if date_col is None:
                print(f"Warning: No date column found in {file}. Available columns: {df.columns.tolist()}")
                # If no date column, create one with default values
                df['Date'] = pd.date_range(start='2023-01-01', periods=len(df), freq='D')
            else:
                # Rename to standardized 'Date'
                df.rename(columns={date_col: 'Date'}, inplace=True)
                # Convert date to datetime
                try:
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                except Exception as e:
                    print(f"Error converting dates in {file}: {e}")
                    # Try different date formats
                    try:
                        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
                    except:
                        try:
                            df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
                        except:
                            print(f"Could not parse dates in {file}, using default dates")
                            df['Date'] = pd.date_range(start='2023-01-01', periods=len(df), freq='D')
            
            # Store data by station for specific access
            self.station_data[station_name] = df
            
            all_data.append(df)
            
        # Combine all data
        if all_data:
            self.data = pd.concat(all_data, ignore_index=True)
            
            # Sort by date
            self.data.sort_values('Date', inplace=True)
        else:
            print("Warning: No data was loaded. Check if CSV files exist in the data directory.")
            self.data = pd.DataFrame()
        
        return self.data
    
    def clean_data(self, df=None):
        """Clean and preprocess data.
        
        Args:
            df (pandas.DataFrame, optional): DataFrame to clean. If None, uses self.data.
            
        Returns:
            pandas.DataFrame: Cleaned DataFrame
        """
        if df is None:
            if self.data is None:
                print("No data loaded. Loading data first...")
                self.load_data()
            df = self.data.copy()
        else:
            df = df.copy()
        
        if df.empty:
            print("Warning: Empty DataFrame. Cannot clean data.")
            return df
        
        # Ensure Date column exists and is properly formatted
        if 'Date' not in df.columns:
            print("Warning: Date column not found after standardization. Returning data as is.")
            return df
        
        # Convert Date to datetime if not already
        if not pd.api.types.is_datetime64_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Standardize pollutant column names
        rename_map = {}
        for col in df.columns:
            # Handle variations like 'PM2.5', 'PM 2.5', 'PM_2.5'
            clean_col = col.replace(' ', '').replace('_', '')
            if clean_col == 'PM2.5':
                rename_map[col] = 'PM2.5'
            elif clean_col == 'PM10':
                rename_map[col] = 'PM10'
        
        if rename_map:
            df.rename(columns=rename_map, inplace=True)
        
        # Handle missing values
        for col in self.pollutants + ['AMB_TEMP', 'RH', 'WIND_SPEED']:
            if col in df.columns:
                # Fill missing values with the mean of the column
                if df[col].isna().all():
                    print(f"Warning: Column {col} has all missing values. Filling with zeros.")
                    df[col].fillna(0, inplace=True)
                else:
                    df[col].fillna(df[col].mean(), inplace=True)
        
        # Remove any remaining rows with NaN values in pollutant columns
        pollutants_in_df = [p for p in self.pollutants if p in df.columns]
        if pollutants_in_df:  # Only drop if we have pollutant columns
            df.dropna(subset=pollutants_in_df, inplace=True)
        
        return df
    
    def prepare_time_series_data(self, df, target_col, lookback=24, forecast_horizon=24):
        """Prepare time series data for LSTM and other models.
        
        Args:
            df (pandas.DataFrame): DataFrame with time series data
            target_col (str): Column to predict
            lookback (int): Number of previous time steps to use for prediction
            forecast_horizon (int): Number of future time steps to predict
            
        Returns:
            tuple: (X, y) where X is input data and y is target data
        """
        data = df[target_col].values
        X, y = [], []
        
        for i in range(len(data) - lookback - forecast_horizon + 1):
            X.append(data[i:i+lookback])
            y.append(data[i+lookback:i+lookback+forecast_horizon])
        
        return np.array(X), np.array(y)
    
    def prepare_multivariate_data(self, df, target_col, feature_cols=None, lookback=24, forecast_horizon=24):
        """Prepare multivariate time series data.
        
        Args:
            df (pandas.DataFrame): DataFrame with time series data
            target_col (str): Column to predict
            feature_cols (list): List of columns to use as features
            lookback (int): Number of previous time steps to use for prediction
            forecast_horizon (int): Number of future time steps to predict
            
        Returns:
            tuple: (X, y) where X is input data and y is target data
        """
        if feature_cols is None:
            # Use all pollutants by default
            feature_cols = [col for col in self.pollutants if col in df.columns]
            # Add weather features
            weather_cols = ['AMB_TEMP', 'RH', 'WIND_SPEED']
            feature_cols.extend([col for col in weather_cols if col in df.columns])
        
        # Scale features
        df_scaled = df.copy()
        df_scaled[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        
        # Create sequences
        X, y = [], []
        values = df_scaled[feature_cols].values
        target = df[target_col].values
        
        for i in range(len(df) - lookback - forecast_horizon + 1):
            X.append(values[i:i+lookback])
            y.append(target[i+lookback:i+lookback+forecast_horizon])
        
        return np.array(X), np.array(y)
    
    def prepare_forecasting_data(self, station_name, target_pollutant, prediction_length):
        """Prepare data for forecasting with different horizons.
        
        Args:
            station_name (str): Name of the station
            target_pollutant (str): Pollutant to predict
            prediction_length (str): '1day', '1week', or '1month'
            
        Returns:
            tuple: (X_train, y_train, X_test, y_test)
        """
        if station_name not in self.station_data:
            raise ValueError(f"Station {station_name} not found in data")
            
        df = self.station_data[station_name].copy()
        df = self.clean_data(df)
        
        # Set lookback and forecast horizon based on prediction length
        lookback = 24  # Default lookback of 24 hours
        if prediction_length == '1day':
            forecast_horizon = 24  # 24 hours in a day
        elif prediction_length == '1week':
            forecast_horizon = 24 * 7  # 7 days
        elif prediction_length == '1month':
            forecast_horizon = 24 * 30  # Approximate month
        else:
            raise ValueError("prediction_length must be '1day', '1week', or '1month'")
        
        # Prepare features
        feature_cols = [col for col in ['AMB_TEMP', 'RH', 'WIND_SPEED', 'RAINFALL'] if col in df.columns]
        feature_cols.extend([p for p in self.pollutants if p != target_pollutant and p in df.columns])
        
        X, y = self.prepare_multivariate_data(
            df, 
            target_pollutant, 
            feature_cols=feature_cols,
            lookback=lookback,
            forecast_horizon=forecast_horizon
        )
        
        # Split data into train and test sets (80% train, 20% test)
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        return X_train, y_train, X_test, y_test
        
    def prepare_data(self, df, target_column, sequence_length=24, include_all_features=True, test_only=False):
        """Prepare data for universal models that work with all stations and pollutants.
        
        Args:
            df (pandas.DataFrame): DataFrame with time series data
            target_column (str): Column to predict
            sequence_length (int): Length of input sequence (default: 24 for 1day)
            include_all_features (bool): Whether to include all features (True) or just the target (False)
            test_only (bool): If True, only return test data (for prediction)
            
        Returns:
            tuple: (X, y) for training or test data
        """
        # Clean the data
        df = self.clean_data(df)
        
        if df.empty:
            raise ValueError("Empty DataFrame after cleaning. Cannot prepare data.")
            
        # Check if target column exists
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame. Available columns: {df.columns.tolist()}")
            
        # Define feature columns
        if include_all_features:
            # Use all available features except the target
            feature_cols = [col for col in df.columns if col != target_column and col != 'Date']
        else:
            # Use just the target column
            feature_cols = [target_column]
        
        # Scale features
        self.scaler = MinMaxScaler()
        df_scaled = df.copy()
        
        # Make sure all columns are numeric
        for col in feature_cols + [target_column]:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df.fillna(df[col].mean(), inplace=True)
                except:
                    print(f"Warning: Could not convert column {col} to numeric. Dropping column.")
                    feature_cols.remove(col)
        
        # Scale the features
        for col in feature_cols + [target_column]:
            if col in df.columns:
                df_scaled[col] = self.scaler.fit_transform(df[[col]])
        
        # Create sequences for input and output
        X, y = [], []
        target_data = df[target_column].values
        
        # For multivariate input
        if include_all_features:
            feature_data = df_scaled[feature_cols].values
            
            for i in range(len(df) - 2*sequence_length + 1):
                # Input sequence includes all features
                X.append(feature_data[i:i+sequence_length])
                # Output sequence is just the target
                y.append(target_data[i+sequence_length:i+2*sequence_length])
        else:
            # For univariate input (target only)
            for i in range(len(df) - 2*sequence_length + 1):
                X.append(target_data[i:i+sequence_length])
                y.append(target_data[i+sequence_length:i+2*sequence_length])
        
        X = np.array(X)
        y = np.array(y)
        
        if test_only:
            # For prediction, use only the most recent data
            if len(X) > 0:
                return X[-1:], y[-1:] if len(y) > 0 else None
            else:
                return None, None
        else:
            # For training, split data (80% train, 20% test)
            if len(X) > 0:
                train_size = int(len(X) * 0.8)
                X_train, X_test = X[:train_size], X[train_size:]
                y_train, y_test = y[:train_size], y[train_size:]
                return X_train, y_train, X_test, y_test
            else:
                # If no data, return empty arrays
                return np.array([]), np.array([]), np.array([]), np.array([]) 
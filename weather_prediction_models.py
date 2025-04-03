#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Weather Prediction Model Training

This script contains code blocks for training different models for air pollution prediction.
Each section trains a specific type of model and saves it to a folder.

To use as a Jupyter notebook:
1. Create a new notebook
2. Copy sections of this code into different cells
3. Run cells individually to train models one at a time
"""

# %% [markdown]
# # Weather Prediction Model Training
# 
# This notebook allows you to train different models for air pollution prediction. Each code block trains a specific type of model.

# %% [markdown]
# ## Setup and Data Loading

# %%
import os
import numpy as np
import pandas as pd
import glob
import torch
import matplotlib.pyplot as plt
from models import LSTMModel, ANNModel, RandomForestModel, LinearRegressionModel, MultipleLinearRegressionModel
from models.preprocessing import DataPreprocessor
import torch.nn as nn

# Create directory for saved models
models_dir = 'saved_models'
os.makedirs(models_dir, exist_ok=True)

# Set prediction length
prediction_length = '1day'  # '1day', '1week', or '1month'

# Define sequence length based on prediction length
if prediction_length == '1day':
    sequence_length = 24  # 24 hours
elif prediction_length == '1week':
    sequence_length = 24 * 7  # 7 days
elif prediction_length == '1month':
    sequence_length = 24 * 30  # 30 days

# Load and prepare data
def load_all_data(data_path='data'):
    # Get all station data files
    data_files = glob.glob(os.path.join(data_path, '*.csv'))
    if not data_files:
        print(f"No data files found in {data_path}")
        return None
    
    print(f"Found {len(data_files)} data files")
    
    # Load and combine all data
    combined_data = []
    station_names = []
    
    for file_path in data_files:
        station_name = os.path.basename(file_path).split('.')[0]
        station_names.append(station_name)
        
        try:
            # Load data
            df = pd.read_csv(file_path)
            
            # Add station column
            df['station'] = station_name
            
            combined_data.append(df)
            print(f"Loaded data for station: {station_name}")
        except Exception as e:
            print(f"Error loading data for {station_name}: {e}")
    
    if not combined_data:
        print("No data could be loaded. Exiting.")
        return None
    
    # Combine all data
    all_data = pd.concat(combined_data, ignore_index=True)
    print(f"Combined data shape: {all_data.shape}")
    
    # Get all pollutants
    pollutant_columns = [col for col in all_data.columns if col not in 
                         ['Date', 'station', 'AMB_TEMP', 'RH', 'WIND_SPEED', 'RAINFALL']]
    
    print(f"Detected pollutants: {pollutant_columns}")
    
    # Create one-hot encoding for stations
    all_data = pd.get_dummies(all_data, columns=['station'], prefix='station')
    
    return all_data, pollutant_columns

# Data cleaning function to handle NaN and Inf values
def clean_data_for_training(X_data, y_data):
    """Clean data by removing or replacing NaN and Inf values."""
    # Check for NaN values
    if np.isnan(X_data).any() or np.isnan(y_data).any():
        print("Warning: NaN values detected in the data. Replacing with zeros.")
        X_data = np.nan_to_num(X_data, nan=0.0)
        y_data = np.nan_to_num(y_data, nan=0.0)
    
    # Check for infinite values
    if np.isinf(X_data).any() or np.isinf(y_data).any():
        print("Warning: Infinite values detected in the data. Replacing with large values.")
        X_data = np.clip(X_data, -1e15, 1e15)
        y_data = np.clip(y_data, -1e15, 1e15)
    
    return X_data, y_data

# Prepare training data for all pollutants
def prepare_training_data(all_data, pollutant_columns, sequence_length):
    preprocessor = DataPreprocessor()
    combined_X = {}
    combined_y = {}
    
    for pollutant in pollutant_columns:
        print(f"Preparing data for pollutant: {pollutant}")
        
        # Prepare data
        try:
            X_train, y_train, X_test, y_test = preprocessor.prepare_data(
                all_data, 
                target_column=pollutant,
                sequence_length=sequence_length,
                include_all_features=True
            )
            
            if pollutant not in combined_X:
                combined_X[pollutant] = []
                combined_y[pollutant] = []
            
            # Only add data if we have valid training samples
            if len(X_train) > 0:
                # Clean data before adding
                X_train, y_train = clean_data_for_training(X_train, y_train)
                combined_X[pollutant].append(X_train)
                combined_y[pollutant].append(y_train)
        except Exception as e:
            print(f"Error preparing data for {pollutant}: {e}")
    
    return combined_X, combined_y

# Load data
all_data, pollutant_columns = load_all_data()

# Prepare training data
combined_X, combined_y = prepare_training_data(all_data, pollutant_columns, sequence_length)

# %% [markdown]
# ## 1. Train LSTM Model
# 
# Long Short-Term Memory networks are a type of recurrent neural network well-suited for sequence prediction problems.

# %%
def train_lstm_model(combined_X, combined_y, sequence_length, all_data):
    """Train a universal LSTM model."""
    # Combine all data for LSTM
    all_X = []
    all_y = []
    
    for pollutant in combined_X:
        for X_data, y_data in zip(combined_X[pollutant], combined_y[pollutant]):
            # LSTM expects 3D input
            all_X.append(X_data)
            all_y.append(y_data)
    
    # Concatenate all data
    if all_X and all_y:
        X_concat = np.concatenate(all_X)
        y_concat = np.concatenate(all_y)
        
        # Clean data again to ensure no NaNs or Infs
        X_concat, y_concat = clean_data_for_training(X_concat, y_concat)
        
        # Normalize data to range [0, 1] for better numerical stability
        X_mean = np.mean(X_concat, axis=(0, 1), keepdims=True)
        X_std = np.std(X_concat, axis=(0, 1), keepdims=True) + 1e-8  # Add small epsilon to avoid division by zero
        X_concat = (X_concat - X_mean) / X_std
        
        # Get the actual input dimension from the data
        actual_input_dim = X_concat.shape[2]
        
        # Initialize LSTM model with gradient clipping
        try:
            # Create model - note: using parameters that match the actual implementation
            print("Creating LSTM model with normalized data...")
            lstm_model = LSTMModel(
                input_shape=(sequence_length, actual_input_dim),
                output_length=sequence_length
            )
            
            # Lower batch size for more stability
            batch_size = 32
            
            # Train model with error handling
            try:
                print("Training LSTM model with normalized data...")
                # The train_model method already has early stopping built in
                lstm_model.train_model(
                    X_concat, 
                    y_concat, 
                    epochs=50, 
                    batch_size=batch_size,
                    validation_split=0.2  # This parameter is supported
                )
                
                # Save model
                lstm_path, lstm_meta_path = lstm_model.save(os.path.join(models_dir, f"lstm_universal_{prediction_length}"))
                print(f"LSTM model saved to {lstm_path}")
                
                return lstm_model
            except Exception as e:
                print(f"Error during training: {e}")
                print("Trying with a simpler model configuration...")
                
                # Try with a simpler configuration - use standard parameters only
                # Create a new model with the same architecture but different hyperparameters
                print("Creating simpler LSTM model...")
                
                # Custom LSTM implementation with simpler params
                lstm_model = LSTMModel(
                    input_shape=(sequence_length, actual_input_dim),
                    output_length=sequence_length
                )
                
                # Customize the LSTM model with simpler architecture
                # Directly modify relevant attributes instead of using constructor params
                # Reduce complexity by using smaller network, lower learning rate
                lstm_model.dropout1 = nn.Dropout(0.1)  # Lower dropout rate
                lstm_model.dropout2 = nn.Dropout(0.1)
                
                # Reduce batch size further for stability
                batch_size = 16
                
                lstm_model.train_model(
                    X_concat, 
                    y_concat, 
                    epochs=30,  # Fewer epochs
                    batch_size=batch_size,
                    validation_split=0.2  # Already supported
                )
                
                lstm_path, lstm_meta_path = lstm_model.save(os.path.join(models_dir, f"lstm_simple_{prediction_length}"))
                print(f"Simplified LSTM model saved to {lstm_path}")
                
                return lstm_model
                
        except Exception as e:
            print(f"Failed to create LSTM model: {e}")
            raise
    
    raise ValueError("No data available for LSTM training")

# Train LSTM model
try:
    print("\nTraining LSTM model...")
    lstm_model = train_lstm_model(combined_X, combined_y, sequence_length, all_data)
    print("LSTM model training completed!")
except Exception as e:
    print(f"Error training LSTM model: {e}")

# %% [markdown]
# ## 2. Train ANN Model
# 
# Artificial Neural Networks are simple feed-forward networks that can learn complex patterns.

# %%
def train_ann_model(combined_X, combined_y, all_data, sequence_length):
    """Train a universal ANN model."""
    # Combine all data for ANN
    all_X = []
    all_y = []
    
    for pollutant in combined_X:
        for X_data, y_data in zip(combined_X[pollutant], combined_y[pollutant]):
            # ANN expects 2D input, use the last time step
            all_X.append(X_data[:, -1, :])
            all_y.append(y_data)
    
    # Concatenate all data
    if all_X and all_y:
        X_concat = np.concatenate(all_X)
        y_concat = np.concatenate(all_y)
        
        # Clean data
        X_concat, y_concat = clean_data_for_training(X_concat, y_concat)
        
        # Normalize data
        X_mean = np.mean(X_concat, axis=0, keepdims=True)
        X_std = np.std(X_concat, axis=0, keepdims=True) + 1e-8
        X_concat = (X_concat - X_mean) / X_std
        
        # Get the actual input dimension from the data
        actual_input_dim = X_concat.shape[1]
        
        try:
            # Initialize ANN model
            print("Creating ANN model...")
            ann_model = ANNModel(
                input_shape=actual_input_dim,
                output_length=sequence_length
            )
            
            # Train model with error handling
            try:
                print("Training ANN model with normalized data...")
                # ANNModel already has early stopping built in
                ann_model.train_model(
                    X_concat, 
                    y_concat, 
                    epochs=50, 
                    batch_size=32,
                    validation_split=0.2
                )
                
                # Save model
                ann_path, ann_meta_path = ann_model.save(os.path.join(models_dir, f"ann_universal_{prediction_length}"))
                print(f"ANN model saved to {ann_path}")
                
                return ann_model
            except Exception as e:
                print(f"Error during ANN training: {e}")
                print("Trying with a simpler ANN configuration...")
                
                # Try with a simpler model by directly modifying the model
                print("Creating simpler ANN model...")
                
                # Create a new ANN model with the same architecture but simpler configuration
                ann_model = ANNModel(
                    input_shape=actual_input_dim,
                    output_length=sequence_length
                )
                
                # Modify the model to be simpler by replacing the sequential model
                ann_model.model = nn.Sequential(
                    nn.Linear(actual_input_dim, 64),  # Smaller first layer
                    nn.ReLU(),
                    nn.Dropout(0.1),  # Lower dropout
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, sequence_length)
                )
                
                # Train with reduced batch size
                ann_model.train_model(
                    X_concat, 
                    y_concat, 
                    epochs=30, 
                    batch_size=16,
                    validation_split=0.2
                )
                
                ann_path, ann_meta_path = ann_model.save(os.path.join(models_dir, f"ann_simple_{prediction_length}"))
                print(f"Simplified ANN model saved to {ann_path}")
                
                return ann_model
        except Exception as e:
            print(f"Failed to create ANN model: {e}")
            raise
    
    raise ValueError("No data available for ANN training")

# Train ANN model
try:
    print("\nTraining ANN model...")
    ann_model = train_ann_model(combined_X, combined_y, all_data, sequence_length)
    print("ANN model training completed!")
except Exception as e:
    print(f"Error training ANN model: {e}")

# %% [markdown]
# ## 3. Train Random Forest Model
# 
# Random Forest is an ensemble learning method that operates by constructing multiple decision trees.

# %%
def train_random_forest_model(combined_X, combined_y, sequence_length):
    """Train a universal Random Forest model."""
    # Combine all data for RF
    all_X = []
    all_y = []
    
    for pollutant in combined_X:
        for X_data, y_data in zip(combined_X[pollutant], combined_y[pollutant]):
            # RF expects 2D input, flatten the sequence
            all_X.append(X_data.reshape(X_data.shape[0], -1))
            all_y.append(y_data)
    
    # Concatenate all data
    if all_X and all_y:
        X_concat = np.concatenate(all_X)
        y_concat = np.concatenate(all_y)
        
        # Clean data
        X_concat, y_concat = clean_data_for_training(X_concat, y_concat)
        
        try:
            # Initialize RF model
            print("Creating Random Forest model...")
            rf_model = RandomForestModel(n_estimators=100, max_depth=None)
            
            # Train model - note: RandomForestModel uses train() method, not train_model()
            print("Training Random Forest model...")
            rf_model.train(X_concat, y_concat)
            
            # Save model
            rf_path, rf_meta_path = rf_model.save(os.path.join(models_dir, f"random_forest_universal_{prediction_length}"))
            print(f"Random Forest model saved to {rf_path}")
            
            return rf_model
        except Exception as e:
            print(f"Error during Random Forest training: {e}")
            print("Trying with reduced complexity...")
            
            # Try with reduced complexity
            print("Creating simpler Random Forest model...")
            rf_model = RandomForestModel(n_estimators=50, max_depth=10)
            rf_model.train(X_concat, y_concat)
            
            rf_path, rf_meta_path = rf_model.save(os.path.join(models_dir, f"random_forest_simple_{prediction_length}"))
            print(f"Simplified Random Forest model saved to {rf_path}")
            
            return rf_model
    
    raise ValueError("No data available for Random Forest training")

# Train Random Forest model
try:
    print("\nTraining Random Forest model...")
    rf_model = train_random_forest_model(combined_X, combined_y, sequence_length)
    print("Random Forest model training completed!")
except Exception as e:
    print(f"Error training Random Forest model: {e}")

# %% [markdown]
# ## 4. Train Linear Regression Model
# 
# Linear Regression is a simple approach that models the relationship between variables using a linear predictor function.

# %%
def train_linear_regression_model(combined_X, combined_y, sequence_length):
    """Train a universal Linear Regression model."""
    # Combine all data for Linear Regression
    all_X = []
    all_y = []
    
    for pollutant in combined_X:
        for X_data, y_data in zip(combined_X[pollutant], combined_y[pollutant]):
            # LR expects 2D input, use the last time step
            all_X.append(X_data[:, -1, :])
            all_y.append(y_data)
    
    # Concatenate all data
    if all_X and all_y:
        X_concat = np.concatenate(all_X)
        y_concat = np.concatenate(all_y)
        
        # Clean data
        X_concat, y_concat = clean_data_for_training(X_concat, y_concat)
        
        try:
            # Initialize LR model
            print("Creating Linear Regression model...")
            lr_model = LinearRegressionModel()
            
            # Train model - using train() method instead of train_model()
            print("Training Linear Regression model...")
            lr_model.train(X_concat, y_concat)
            
            # Save model
            lr_path, lr_meta_path = lr_model.save(os.path.join(models_dir, f"linear_regression_universal_{prediction_length}"))
            print(f"Linear Regression model saved to {lr_path}")
            
            return lr_model
        except Exception as e:
            print(f"Error during Linear Regression training: {e}")
            # Linear regression is already simple, but we can try to handle failures
            # by reducing data size if needed
            if len(X_concat) > 10000:
                print("Reducing data size for Linear Regression...")
                indices = np.random.choice(len(X_concat), 10000, replace=False)
                X_reduced = X_concat[indices]
                y_reduced = y_concat[indices]
                
                lr_model = LinearRegressionModel()
                lr_model.train(X_reduced, y_reduced)
                
                lr_path, lr_meta_path = lr_model.save(os.path.join(models_dir, f"linear_regression_reduced_{prediction_length}"))
                print(f"Reduced Linear Regression model saved to {lr_path}")
                
                return lr_model
            else:
                raise
    
    raise ValueError("No data available for Linear Regression training")

# Train Linear Regression model
try:
    print("\nTraining Linear Regression model...")
    lr_model = train_linear_regression_model(combined_X, combined_y, sequence_length)
    print("Linear Regression model training completed!")
except Exception as e:
    print(f"Error training Linear Regression model: {e}")

# %% [markdown]
# ## 5. Train Multiple Linear Regression (MLR) Model
# 
# Multiple Linear Regression extends simple linear regression to include multiple input variables.

# %%
def train_mlr_model(combined_X, combined_y, sequence_length):
    """Train a universal MLR model."""
    # Combine all data for MLR
    all_X = []
    all_y = []
    
    for pollutant in combined_X:
        for X_data, y_data in zip(combined_X[pollutant], combined_y[pollutant]):
            # MLR expects 2D input, flatten the sequence
            all_X.append(X_data.reshape(X_data.shape[0], -1))
            all_y.append(y_data)
    
    # Concatenate all data
    if all_X and all_y:
        X_concat = np.concatenate(all_X)
        y_concat = np.concatenate(all_y)
        
        # Clean data
        X_concat, y_concat = clean_data_for_training(X_concat, y_concat)
        
        try:
            # Initialize MLR model
            print("Creating Multiple Linear Regression model...")
            mlr_model = MultipleLinearRegressionModel()
            
            # Train model - using train() method instead of train_model()
            print("Training Multiple Linear Regression model...")
            mlr_model.train(X_concat, y_concat)
            
            # Save model
            mlr_path, mlr_meta_path = mlr_model.save(os.path.join(models_dir, f"mlr_universal_{prediction_length}"))
            print(f"MLR model saved to {mlr_path}")
            
            return mlr_model
        except Exception as e:
            print(f"Error during MLR training: {e}")
            # Like with LR, reduce data size if needed
            if len(X_concat) > 10000:
                print("Reducing data size for MLR...")
                indices = np.random.choice(len(X_concat), 10000, replace=False)
                X_reduced = X_concat[indices]
                y_reduced = y_concat[indices]
                
                mlr_model = MultipleLinearRegressionModel()
                mlr_model.train(X_reduced, y_reduced)
                
                mlr_path, mlr_meta_path = mlr_model.save(os.path.join(models_dir, f"mlr_reduced_{prediction_length}"))
                print(f"Reduced MLR model saved to {mlr_path}")
                
                return mlr_model
            else:
                raise
    
    raise ValueError("No data available for MLR training")

# Train MLR model
try:
    print("\nTraining MLR model...")
    mlr_model = train_mlr_model(combined_X, combined_y, sequence_length)
    print("MLR model training completed!")
except Exception as e:
    print(f"Error training MLR model: {e}")

# %% [markdown]
# ## Running the Web Interface
# 
# After training the models, you can run the web interface to interact with them.

# %%
# Run the Flask application
print("To start the web interface, run the following command in your terminal:")
print("python app.py") 
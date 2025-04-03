import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from .lstm_model import LSTMModel
from .ann_model import ANNModel
from .random_forest_model import RandomForestModel
from .regression_models import LinearRegressionModel, MultipleLinearRegressionModel
from .preprocessing import DataPreprocessor

class ModelManager:
    def __init__(self, data_path='data', models_dir='saved_models'):
        """Initialize ModelManager to handle all models.
        
        Args:
            data_path (str): Path to the data directory
            models_dir (str): Path to save and load models
        """
        self.data_path = data_path
        self.models_dir = models_dir
        self.preprocessor = DataPreprocessor(data_path=data_path)
        
        # Load the data
        self.preprocessor.load_data()
        
        # Model instances for each type
        self.models = {
            'lstm': {},  # Will store LSTM models for different time horizons
            'linear_regression': {},
            'mlr': {},
            'random_forest': {},
            'ann': {}
        }
        
        # Create the models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
    
    def train_all_models(self, station_name, target_pollutant, prediction_lengths=None):
        """Train all models for the specified station and target pollutant.
        
        Args:
            station_name (str): Name of the station
            target_pollutant (str): Pollutant to predict
            prediction_lengths (list, optional): List of prediction lengths
            
        Returns:
            dict: Dictionary of trained models
        """
        if prediction_lengths is None:
            prediction_lengths = ['1day', '1week', '1month']
        
        for prediction_length in prediction_lengths:
            self.train_models_for_horizon(station_name, target_pollutant, prediction_length)
            
        return self.models
    
    def train_models_for_horizon(self, station_name, target_pollutant, prediction_length):
        """Train all model types for a specific prediction horizon.
        
        Args:
            station_name (str): Name of the station
            target_pollutant (str): Pollutant to predict
            prediction_length (str): Prediction horizon ('1day', '1week', '1month')
            
        Returns:
            dict: Dictionary of trained models for this horizon
        """
        # Prepare data
        X_train, y_train, X_test, y_test = self.preprocessor.prepare_forecasting_data(
            station_name, target_pollutant, prediction_length
        )
        
        # Train LSTM model
        lstm_model = LSTMModel(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            output_length=y_train.shape[1],
            name=f"LSTM_{prediction_length}_{target_pollutant}"
        )
        lstm_model.train_model(X_train, y_train, epochs=50)
        self.models['lstm'][prediction_length] = lstm_model
        
        # Train Linear Regression model
        lr_model = LinearRegressionModel(
            name=f"LinearRegression_{prediction_length}_{target_pollutant}"
        )
        lr_model.train(X_train, y_train)
        self.models['linear_regression'][prediction_length] = lr_model
        
        # Train MLR model
        mlr_model = MultipleLinearRegressionModel(
            name=f"MLR_{prediction_length}_{target_pollutant}"
        )
        mlr_model.train(X_train, y_train)
        self.models['mlr'][prediction_length] = mlr_model
        
        # Train Random Forest model
        rf_model = RandomForestModel(
            n_estimators=100,
            name=f"RandomForest_{prediction_length}_{target_pollutant}"
        )
        rf_model.train(X_train, y_train)
        self.models['random_forest'][prediction_length] = rf_model
        
        # Train ANN model
        ann_model = ANNModel(
            input_shape=X_train.shape[2] if len(X_train.shape) == 3 else X_train.shape[1],
            output_length=y_train.shape[1],
            name=f"ANN_{prediction_length}_{target_pollutant}"
        )
        ann_model.train_model(X_train, y_train, epochs=50)
        self.models['ann'][prediction_length] = ann_model
        
        # Evaluate all models
        self._evaluate_models(
            X_test, y_test, 
            target_pollutant, 
            prediction_length,
            station_name
        )
        
        # Save all models
        self.save_models(target_pollutant, prediction_length)
        
        return {
            'lstm': lstm_model,
            'linear_regression': lr_model,
            'mlr': mlr_model,
            'random_forest': rf_model,
            'ann': ann_model
        }
    
    def _evaluate_models(self, X_test, y_test, target_pollutant, prediction_length, station_name):
        """Evaluate all models and create comparison plots.
        
        Args:
            X_test (numpy.ndarray): Test data
            y_test (numpy.ndarray): True values
            target_pollutant (str): Target pollutant
            prediction_length (str): Prediction horizon
            station_name (str): Name of the station
        """
        # Get predictions from all models
        predictions = {}
        metrics = {}
        
        models_dict = {
            'LSTM': self.models['lstm'][prediction_length],
            'Linear Regression': self.models['linear_regression'][prediction_length],
            'MLR': self.models['mlr'][prediction_length],
            'Random Forest': self.models['random_forest'][prediction_length],
            'ANN': self.models['ann'][prediction_length]
        }
        
        for model_name, model in models_dict.items():
            pred = model.predict(X_test)
            predictions[model_name] = pred
            metrics[model_name] = model.evaluate(X_test, y_test)
        
        # Create directory for plots
        plots_dir = os.path.join(self.models_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot predictions from all models for the first test sample
        self._plot_model_comparison(
            y_test, predictions, 
            title=f"{station_name} {target_pollutant} {prediction_length} Prediction Comparison",
            save_path=os.path.join(plots_dir, f"{station_name}_{target_pollutant}_{prediction_length}_comparison.png")
        )
        
        # Save metrics
        metrics_df = pd.DataFrame({
            'Model': list(metrics.keys()),
            'MSE': [m['mse'] for m in metrics.values()],
            'MAE': [m['mae'] for m in metrics.values()],
            'RMSE': [m['rmse'] for m in metrics.values()]
        })
        
        metrics_path = os.path.join(plots_dir, f"{station_name}_{target_pollutant}_{prediction_length}_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        
        # Plot metrics comparison
        self._plot_metrics_comparison(
            metrics,
            title=f"{station_name} {target_pollutant} {prediction_length} Metrics Comparison",
            save_path=os.path.join(plots_dir, f"{station_name}_{target_pollutant}_{prediction_length}_metrics.png")
        )
    
    def _plot_model_comparison(self, y_true, predictions, title=None, save_path=None):
        """Plot predictions from all models vs ground truth.
        
        Args:
            y_true (numpy.ndarray): True values
            predictions (dict): Dictionary mapping model names to predictions
            title (str, optional): Plot title
            save_path (str, optional): Path to save the plot
        """
        plt.figure(figsize=(12, 6))
        
        # Plot ground truth
        plt.plot(y_true[0], 'k-', label='Actual', linewidth=2)
        
        # Plot predictions from each model
        colors = ['b', 'g', 'r', 'c', 'm']
        for i, (model_name, pred) in enumerate(predictions.items()):
            plt.plot(pred[0], f'{colors[i]}--', label=model_name, alpha=0.7)
        
        plt.title(title or 'Model Predictions Comparison')
        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            
        plt.close()
    
    def _plot_metrics_comparison(self, metrics, title=None, save_path=None):
        """Plot comparison of metrics for all models.
        
        Args:
            metrics (dict): Dictionary mapping model names to metrics
            title (str, optional): Plot title
            save_path (str, optional): Path to save the plot
        """
        plt.figure(figsize=(15, 6))
        
        model_names = list(metrics.keys())
        mse_values = [m['mse'] for m in metrics.values()]
        mae_values = [m['mae'] for m in metrics.values()]
        rmse_values = [m['rmse'] for m in metrics.values()]
        
        # Bar positions
        bar_width = 0.25
        r1 = np.arange(len(model_names))
        r2 = [x + bar_width for x in r1]
        r3 = [x + bar_width for x in r2]
        
        # Create grouped bar chart
        plt.bar(r1, mse_values, width=bar_width, label='MSE', color='blue', alpha=0.7)
        plt.bar(r2, mae_values, width=bar_width, label='MAE', color='green', alpha=0.7)
        plt.bar(r3, rmse_values, width=bar_width, label='RMSE', color='red', alpha=0.7)
        
        plt.xlabel('Model')
        plt.ylabel('Error Value')
        plt.title(title or 'Model Metrics Comparison')
        plt.xticks([r + bar_width for r in range(len(model_names))], model_names)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            
        plt.close()
    
    def save_models(self, target_pollutant, prediction_length):
        """Save all models for a specific target and prediction length.
        
        Args:
            target_pollutant (str): Target pollutant
            prediction_length (str): Prediction horizon
            
        Returns:
            dict: Paths to saved models
        """
        saved_paths = {}
        
        # Ensure models_dir exists
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Save LSTM model
        if prediction_length in self.models['lstm']:
            lstm_model = self.models['lstm'][prediction_length]
            model_path, metadata_path = lstm_model.save(model_dir=self.models_dir)
            saved_paths['lstm'] = (model_path, metadata_path)
            print(f"Saved LSTM model to {model_path}")
        
        # Save Linear Regression model
        if prediction_length in self.models['linear_regression']:
            lr_model = self.models['linear_regression'][prediction_length]
            model_path, metadata_path = lr_model.save(model_dir=self.models_dir)
            saved_paths['linear_regression'] = (model_path, metadata_path)
            print(f"Saved Linear Regression model to {model_path}")
        
        # Save MLR model
        if prediction_length in self.models['mlr']:
            mlr_model = self.models['mlr'][prediction_length]
            model_path, metadata_path = mlr_model.save(model_dir=self.models_dir)
            saved_paths['mlr'] = (model_path, metadata_path)
            print(f"Saved MLR model to {model_path}")
        
        # Save Random Forest model
        if prediction_length in self.models['random_forest']:
            rf_model = self.models['random_forest'][prediction_length]
            model_path, metadata_path = rf_model.save(model_dir=self.models_dir)
            saved_paths['random_forest'] = (model_path, metadata_path)
            print(f"Saved Random Forest model to {model_path}")
        
        # Save ANN model
        if prediction_length in self.models['ann']:
            ann_model = self.models['ann'][prediction_length]
            model_path, metadata_path = ann_model.save(model_dir=self.models_dir)
            saved_paths['ann'] = (model_path, metadata_path)
            print(f"Saved ANN model to {model_path}")
        
        return saved_paths
    
    def load_models(self, target_pollutant, prediction_length):
        """Load all trained models for a specific target pollutant and prediction length.
        
        Args:
            target_pollutant (str): Target pollutant
            prediction_length (str): Prediction horizon ('1day', '1week', '1month')
            
        Returns:
            dict: Dictionary of loaded models
        """
        # Create a directory for each model type
        model_types = ['lstm', 'linear_regression', 'mlr', 'random_forest', 'ann']
        
        loaded_models = {}
        
        for model_type in model_types:
            # Define model class based on type
            if model_type == 'lstm':
                ModelClass = LSTMModel
                model_filename = f"LSTM_{prediction_length}_{target_pollutant}_model.pt"
                metadata_filename = f"LSTM_{prediction_length}_{target_pollutant}_metadata.pkl"
            elif model_type == 'ann':
                ModelClass = ANNModel
                model_filename = f"ANN_{prediction_length}_{target_pollutant}_model.pt"
                metadata_filename = f"ANN_{prediction_length}_{target_pollutant}_metadata.pkl"
            elif model_type == 'random_forest':
                ModelClass = RandomForestModel
                model_filename = f"RandomForest_{prediction_length}_{target_pollutant}_model.pkl"
                metadata_filename = f"RandomForest_{prediction_length}_{target_pollutant}_metadata.pkl"
            elif model_type == 'linear_regression':
                ModelClass = LinearRegressionModel
                model_filename = f"LinearRegression_{prediction_length}_{target_pollutant}_model.pkl"
                metadata_filename = f"LinearRegression_{prediction_length}_{target_pollutant}_metadata.pkl"
            elif model_type == 'mlr':
                ModelClass = MultipleLinearRegressionModel
                model_filename = f"MLR_{prediction_length}_{target_pollutant}_model.pkl"
                metadata_filename = f"MLR_{prediction_length}_{target_pollutant}_metadata.pkl"
            
            # Paths to model and metadata files
            model_path = os.path.join(self.models_dir, model_filename)
            metadata_path = os.path.join(self.models_dir, metadata_filename)
            
            # Check if files exist
            if os.path.exists(model_path) and os.path.exists(metadata_path):
                try:
                    # Load the model using the appropriate class method
                    model = ModelClass.load(model_path, metadata_path)
                    
                    # Store in the models dictionary
                    self.models[model_type][prediction_length] = model
                    loaded_models[model_type] = model
                    
                    print(f"Loaded {model_type} model for {prediction_length} prediction of {target_pollutant}")
                except Exception as e:
                    print(f"Error loading {model_type} model: {e}")
            else:
                print(f"Warning: Could not find {model_type} model files at {model_path}")
        
        if not loaded_models:
            raise FileNotFoundError(f"No models found for {target_pollutant} with {prediction_length} prediction")
            
        return loaded_models
    
    def predict_with_all_models(self, station_name, target_pollutant, prediction_length, data=None):
        """Make predictions with all loaded models.
        
        Args:
            station_name (str): Name of the station
            target_pollutant (str): Target pollutant
            prediction_length (str): Prediction horizon
            data (pandas.DataFrame, optional): Custom data to use for prediction
            
        Returns:
            dict: Dictionary mapping model names to predictions
        """
        # If no models are loaded, try to load them
        if all(len(models) == 0 for models in self.models.values()):
            try:
                self.load_models(target_pollutant, prediction_length)
            except Exception as e:
                print(f"Error loading models: {e}")
                print("No models available for prediction.")
                return {}
        
        # If no custom data is provided, use the preprocessor to prepare data
        if data is None:
            # Get the most recent data for the station
            if station_name not in self.preprocessor.station_data:
                raise ValueError(f"Station {station_name} not found in data")
                
            data = self.preprocessor.station_data[station_name].copy()
            data = self.preprocessor.clean_data(data)
            
        try:
            # Prepare input data for prediction
            X_train, y_train, X_test, y_test = self.preprocessor.prepare_forecasting_data(
                station_name, target_pollutant, prediction_length
            )
            
            # We'll use the first sample from X_test for prediction (most recent data)
            X_sample = X_test[:1] if len(X_test) > 0 else X_train[:1]
        except Exception as e:
            print(f"Error preparing forecast data: {e}")
            # Try a simplified approach
            print("Attempting simplified data preparation...")
            
            feature_cols = [col for col in data.columns if col != target_pollutant 
                           and col != 'Date' and data[col].dtype.kind in 'fc']
            target_col = target_pollutant
            
            if target_col not in data.columns:
                raise ValueError(f"Target pollutant {target_pollutant} not found in data")
                
            # Scale the features
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            data_scaled = data[feature_cols].copy()
            data_scaled = scaler.fit_transform(data_scaled)
            
            # Take last 24 hours of data
            X_sample = data_scaled[-24:].reshape(1, 24, len(feature_cols))
        
        # Make predictions with each model
        predictions = {}
        
        # Model display names mapping
        model_display_names = {
            'lstm': 'LSTM',
            'linear_regression': 'Linear Regression',
            'mlr': 'MLR',
            'random_forest': 'Random Forest',
            'ann': 'ANN'
        }
        
        # Try to predict with each model
        for model_type, models in self.models.items():
            if prediction_length in models:
                model = models[prediction_length]
                display_name = model_display_names.get(model_type, model_type)
                
                try:
                    pred = model.predict(X_sample)
                    predictions[display_name] = pred
                except Exception as e:
                    print(f"Error making prediction with {model_type} model: {e}")
        
        if not predictions:
            print("Warning: No models were able to make predictions.")
            
        return predictions
    
    def get_available_stations(self):
        """Get list of available stations in the data.
        
        Returns:
            list: List of station names
        """
        # Make sure data is loaded
        if not hasattr(self.preprocessor, 'station_data') or not self.preprocessor.station_data:
            print("Loading data before getting available stations...")
            self.preprocessor.load_data()
            
        if not self.preprocessor.station_data:
            print("No station data found. Check if data directory contains CSV files.")
            # Return an empty list or some default stations
            return []
            
        return list(self.preprocessor.station_data.keys())
    
    def get_available_pollutants(self):
        """Get list of available pollutants in the data.
        
        Returns:
            list: List of pollutant names
        """
        # Make sure data is loaded
        if self.preprocessor.data is None or self.preprocessor.data.empty:
            print("Loading data before getting available pollutants...")
            self.preprocessor.load_data()
            
        if self.preprocessor.data.empty:
            print("No data found. Check if data directory contains valid CSV files.")
            # Return default pollutants list
            return self.preprocessor.pollutants
            
        # Get columns that match pollutant names
        available_pollutants = []
        for pollutant in self.preprocessor.pollutants:
            # Check if the pollutant column exists in any station's data
            for station_data in self.preprocessor.station_data.values():
                if pollutant in station_data.columns:
                    available_pollutants.append(pollutant)
                    break
                    
        return available_pollutants 
import os
import json
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from models import LSTMModel, ANNModel, RandomForestModel, LinearRegressionModel, MultipleLinearRegressionModel
from models.preprocessing import DataPreprocessor

app = Flask(__name__)

# Global models dictionary
universal_models = {}
prediction_length = '1day'
models_dir = 'saved_models'

def load_universal_models():
    """Load all five universal models."""
    global universal_models
    universal_models = {}
    
    # Load LSTM model
    try:
        lstm_dir = os.path.join(models_dir, f"lstm_universal_{prediction_length}")
        if os.path.exists(lstm_dir):
            model_path = os.path.join(lstm_dir, "LSTM_model.pt")
            meta_path = os.path.join(lstm_dir, "LSTM_metadata.pkl")
            
            if os.path.exists(model_path) and os.path.exists(meta_path):
                model = LSTMModel.load(model_path, meta_path)
                universal_models['LSTM'] = model
                input_features = model.input_shape[1] if hasattr(model, 'input_shape') else 'unknown'
                print(f"LSTM model loaded successfully from {model_path}")
                print(f"LSTM expected input features: {input_features}")
            else:
                print(f"LSTM model files not found in {lstm_dir}")
    except Exception as e:
        print(f"Failed to load LSTM model: {e}")
    
    # Load ANN model
    try:
        ann_dir = os.path.join(models_dir, f"ann_universal_{prediction_length}")
        if os.path.exists(ann_dir):
            model_path = os.path.join(ann_dir, "ANN_model.pt")
            meta_path = os.path.join(ann_dir, "ANN_metadata.pkl")
            
            if os.path.exists(model_path) and os.path.exists(meta_path):
                model = ANNModel.load(model_path, meta_path)
                universal_models['ANN'] = model
                input_features = model.input_shape[1] if hasattr(model, 'input_shape') else 'unknown'
                print(f"ANN model loaded successfully from {model_path}")
                print(f"ANN expected input features: {input_features}")
            else:
                print(f"ANN model files not found in {ann_dir}")
    except Exception as e:
        print(f"Failed to load ANN model: {e}")
    
    # Load Random Forest model
    try:
        rf_dir = os.path.join(models_dir, f"random_forest_universal_{prediction_length}")
        # Check emergency model directory as fallback
        if not os.path.exists(rf_dir):
            rf_dir = os.path.join(models_dir, f"random_forest_emergency_{prediction_length}")
        
        if os.path.exists(rf_dir):
            model_path = os.path.join(rf_dir, "RandomForest_model.pkl")
            meta_path = os.path.join(rf_dir, "RandomForest_metadata.pkl")
            
            if os.path.exists(model_path) and os.path.exists(meta_path):
                model = RandomForestModel.load(model_path, meta_path)
                universal_models['Random Forest'] = model
                n_features = getattr(model.model, 'n_features_in_', 'unknown') if hasattr(model, 'model') else 'unknown'
                print(f"Random Forest model loaded successfully from {model_path}")
                print(f"Random Forest expected input features: {n_features}")
            else:
                print(f"Random Forest model files not found in {rf_dir}")
    except Exception as e:
        print(f"Failed to load Random Forest model: {e}")
    
    # Load Linear Regression model
    try:
        lr_dir = os.path.join(models_dir, f"linear_regression_universal_{prediction_length}")
        # Check emergency model directory as fallback
        if not os.path.exists(lr_dir):
            lr_dir = os.path.join(models_dir, f"linear_regression_emergency_{prediction_length}")
        
        if os.path.exists(lr_dir):
            model_path = os.path.join(lr_dir, "LinearRegression_model.pkl")
            meta_path = os.path.join(lr_dir, "LinearRegression_metadata.pkl")
            
            if os.path.exists(model_path) and os.path.exists(meta_path):
                model = LinearRegressionModel.load(model_path, meta_path)
                universal_models['Linear Regression'] = model
                n_features = getattr(model.model, 'n_features_in_', 'unknown') if hasattr(model, 'model') else 'unknown'
                print(f"Linear Regression model loaded successfully from {model_path}")
                print(f"Linear Regression expected input features: {n_features}")
            else:
                print(f"Linear Regression model files not found in {lr_dir}")
    except Exception as e:
        print(f"Failed to load Linear Regression model: {e}")
    
    # Load MLR model (if exists)
    try:
        mlr_dir = os.path.join(models_dir, f"mlr_universal_{prediction_length}")
        if os.path.exists(mlr_dir):
            model_path = os.path.join(mlr_dir, "MLR_model.pkl")
            meta_path = os.path.join(mlr_dir, "MLR_metadata.pkl")
            
            if os.path.exists(model_path) and os.path.exists(meta_path):
                model = MultipleLinearRegressionModel.load(model_path, meta_path)
                universal_models['MLR'] = model
                n_features = getattr(model.model, 'n_features_in_', 'unknown') if hasattr(model, 'model') else 'unknown'
                print(f"MLR model loaded successfully from {model_path}")
                print(f"MLR expected input features: {n_features}")
            else:
                print(f"MLR model files not found in {mlr_dir}")
    except Exception as e:
        print(f"Failed to load MLR model: {e}")
    
    print(f"Loaded {len(universal_models)} models: {list(universal_models.keys())}")
    return universal_models

def get_available_stations():
    """Get available stations from data directory."""
    data_path = 'data'
    stations = []
    for file in os.listdir(data_path):
        if file.endswith('.csv'):
            station_name = file.split('.')[0]
            stations.append(station_name)
    return stations

def get_available_pollutants():
    """Get available pollutants from data."""
    data_path = 'data'
    # Try to read the first CSV file to get column names
    for file in os.listdir(data_path):
        if file.endswith('.csv'):
            try:
                df = pd.read_csv(os.path.join(data_path, file))
                # Exclude non-pollutant columns
                excluded_cols = [
                    'Date', 'date', 'datetime', 'time', 'Time',
                    'AMB_TEMP', 'RH', 'WIND_SPEED', 'RAINFALL',
                    'station', 'Station', 'STATION', 
                    'Unnamed: 0', 'index', 'ID', 'id'
                ]
                
                # Keep only columns that are likely pollutants
                pollutants = [col for col in df.columns if not any(excl in col for excl in excluded_cols)]
                print(f"Found pollutants: {pollutants}")
                return pollutants
            except Exception as e:
                print(f"Error reading file {file}: {e}")
                continue
                
    # Default if can't read files
    return ['PM2.5', 'PM10', 'CO', 'NO', 'NO2', 'NOx', 'O3', 'SO2']

def prepare_prediction_data(station, pollutant):
    """Prepare data for prediction."""
    data_path = 'data'
    sequence_length = 24  # 1day = 24 hours
    
    # Load station data
    try:
        # Look for the matching CSV file containing the station name
        station_file = None
        for file in os.listdir(data_path):
            if file.endswith('.csv') and station in file:
                station_file = file
                break
        
        if not station_file:
            return None, f"Could not find data file for station: {station}"
        
        file_path = os.path.join(data_path, station_file)
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data from {file_path}")
    except Exception as e:
        return None, f"Error loading data for {station}: {e}"
    
    # Add station column
    df['station'] = station
    
    # One-hot encode station
    df = pd.get_dummies(df, columns=['station'], prefix='station')
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Prepare data for prediction
    try:
        X_test, y_test = preprocessor.prepare_data(
            df,
            target_column=pollutant,
            sequence_length=sequence_length,
            include_all_features=True,
            test_only=True
        )
        
        # Handle NaN values by replacing them with zeros
        # This is important for scikit-learn models that don't accept NaN values
        if np.isnan(X_test).any():
            print(f"Warning: NaN values found in prepared data. Replacing with zeros.")
            X_test = np.nan_to_num(X_test, nan=0.0)
        
        # Check if we have any models loaded to get expected feature count
        model_feature_count = None
        if universal_models:
            # Try to extract feature count from models
            # LSTM and ANN are torch.nn.Module instances with direct attributes
            if 'LSTM' in universal_models and hasattr(universal_models['LSTM'], 'input_shape'):
                model_feature_count = universal_models['LSTM'].input_shape[1]
                print(f"Using LSTM input features: {model_feature_count}")
            elif 'ANN' in universal_models and hasattr(universal_models['ANN'], 'input_shape'):
                model_feature_count = universal_models['ANN'].input_shape[1]
                print(f"Using ANN input features: {model_feature_count}")
            # For scikit-learn models, check n_features_in_
            elif 'Random Forest' in universal_models and hasattr(universal_models['Random Forest'], 'model') and hasattr(universal_models['Random Forest'].model, 'n_features_in_'):
                model_feature_count = universal_models['Random Forest'].model.n_features_in_
                print(f"Using Random Forest input features: {model_feature_count}")
        
        # If we found a model feature count and it doesn't match our data
        if model_feature_count and X_test.shape[-1] != model_feature_count:
            print(f"Feature count mismatch. Model expects {model_feature_count}, got {X_test.shape[-1]}")
            
            # Option 1: Pad with zeros to match the expected size
            if X_test.shape[-1] < model_feature_count:
                padding_size = model_feature_count - X_test.shape[-1]
                if len(X_test.shape) == 3:  # For LSTM (samples, sequence, features)
                    padding = np.zeros((X_test.shape[0], X_test.shape[1], padding_size))
                    X_test = np.concatenate([X_test, padding], axis=2)
                    print(f"Padded features from {X_test.shape[-1] - padding_size} to {X_test.shape[-1]}")
                else:  # For other models (samples, features)
                    padding = np.zeros((X_test.shape[0], padding_size))
                    X_test = np.concatenate([X_test, padding], axis=1)
                    print(f"Padded features from {X_test.shape[-1] - padding_size} to {X_test.shape[-1]}")
        
        return X_test, y_test
    except Exception as e:
        return None, f"Error preparing data: {e}"

@app.route('/')
def index():
    """Render the main page."""
    # Get available stations and pollutants
    stations = get_available_stations()
    pollutants = get_available_pollutants()
    
    # Only use one prediction length - simplified
    prediction_length = '1day'
    
    # Get available model types - keep all five
    model_types = ['LSTM', 'Linear Regression', 'MLR', 'Random Forest', 'ANN', 'All Models']
    
    return render_template(
        'index.html',
        stations=stations,
        pollutants=pollutants,
        prediction_length=prediction_length,
        model_types=model_types
    )

@app.route('/predict', methods=['POST'])
def predict():
    """Generate predictions based on user input."""
    # Get form data
    station = request.form.get('station')
    pollutant = request.form.get('pollutant')
    model_type = request.form.get('model_type').lower().replace(' ', '_')
    
    # Make sure models are loaded
    if not universal_models:
        load_universal_models()
    
    if not universal_models:
        return jsonify({
            'success': False,
            'error': "No models found. Please train the models first."
        })
    
    # Prepare data for prediction
    X_test, y_test = prepare_prediction_data(station, pollutant)
    
    if X_test is None:
        return jsonify({
            'success': False,
            'error': y_test  # This contains the error message
        })
    
    # Make predictions
    try:
        predictions = {}
        
        if model_type == 'all_models':
            # Use all available models
            for model_name, model in universal_models.items():
                try:
                    if model_name == 'LSTM':
                        # LSTM is a direct PyTorch model
                        pred = model.predict(X_test)
                    elif model_name == 'ANN':
                        # ANN is a direct PyTorch model, use the last time step
                        # Handle different input_shape formats
                        if len(X_test.shape) == 3:
                            ann_input = X_test[:, -1, :]  # Use the last time step
                        else:
                            ann_input = X_test
                        pred = model.predict(ann_input)
                    elif model_name in ['Random Forest', 'MLR']:
                        # These models expect flattened input
                        reshaped_input = X_test.reshape(X_test.shape[0], -1)
                        # Ensure no NaN values
                        reshaped_input = np.nan_to_num(reshaped_input, nan=0.0)
                        pred = model.predict(reshaped_input)
                    elif model_name == 'Linear Regression':
                        # Linear Regression expects 2D input, use the last time step
                        lr_input = X_test[:, -1, :] if len(X_test.shape) == 3 else X_test
                        # Ensure no NaN values
                        lr_input = np.nan_to_num(lr_input, nan=0.0)
                        pred = model.predict(lr_input)
                    
                    predictions[model_name] = [pred[0]]  # Take the first prediction sequence
                except Exception as e:
                    print(f"Error with {model_name} model: {str(e)}")
                    # Continue with other models
            
            if not predictions:
                return jsonify({
                    'success': False,
                    'error': "All models failed to generate predictions."
                })
        else:
            # Map from frontend model type to backend model type
            model_type_map = {
                'lstm': 'LSTM',
                'linear_regression': 'Linear Regression',
                'mlr': 'MLR',
                'random_forest': 'Random Forest',
                'ann': 'ANN'
            }
            
            model_name = model_type_map.get(model_type)
            
            if model_name not in universal_models:
                return jsonify({
                    'success': False,
                    'error': f"Model {model_name} not found in available models"
                })
            
            model = universal_models[model_name]
            
            if model_name == 'LSTM':
                # LSTM is a direct PyTorch model
                pred = model.predict(X_test)
            elif model_name == 'ANN':
                # ANN is a direct PyTorch model, use the last time step
                if len(X_test.shape) == 3:
                    ann_input = X_test[:, -1, :]  # Use the last time step
                else:
                    ann_input = X_test
                pred = model.predict(ann_input)
            elif model_name in ['Random Forest', 'MLR']:
                # These models expect flattened input
                reshaped_input = X_test.reshape(X_test.shape[0], -1)
                # Ensure no NaN values
                reshaped_input = np.nan_to_num(reshaped_input, nan=0.0)
                pred = model.predict(reshaped_input)
            elif model_name == 'Linear Regression':
                # Linear Regression expects 2D input, use the last time step
                lr_input = X_test[:, -1, :] if len(X_test.shape) == 3 else X_test
                # Ensure no NaN values
                lr_input = np.nan_to_num(lr_input, nan=0.0)
                pred = model.predict(lr_input)
            
            predictions = {model_name: [pred[0]]}  # Take the first prediction sequence
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f"Failed to generate predictions: {str(e)}"
        })
    
    # Generate plots
    try:
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot predictions
        colors = ['b', 'g', 'r', 'c', 'm']
        for i, (model_name, pred) in enumerate(predictions.items()):
            plt.plot(pred[0], label=model_name, color=colors[i % len(colors)])
        
        # Plot actual values if available
        if y_test is not None and len(y_test) > 0:
            plt.plot(y_test[0], 'k-', label='Actual Values', alpha=0.7)
        
        plt.title(f"{station} {pollutant} {prediction_length} Prediction")
        plt.xlabel('Time Steps')
        plt.ylabel(f'{pollutant} Level')
        plt.legend()
        plt.grid(True)
        
        # Save plot to base64 string
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        # Create data table
        data_table = {}
        for model_name, pred in predictions.items():
            data_table[model_name] = pred[0].tolist()
        
        # Add actual values if available
        if y_test is not None and len(y_test) > 0:
            data_table['Actual'] = y_test[0].tolist()
            
        # Create DataFrame for table display
        time_steps = [f"Hour {i+1}" for i in range(len(next(iter(data_table.values()))))]
        df = pd.DataFrame(data_table, index=time_steps)
        html_table = df.head(24).to_html(classes="table table-striped")
        
        return jsonify({
            'success': True,
            'plot': plot_base64,
            'table': html_table,
            'prediction_data': data_table
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f"Failed to generate visualization: {str(e)}"
        })

# API Endpoints for the frontend
@app.route('/api/stations', methods=['GET'])
def api_stations():
    """API endpoint to get available stations."""
    try:
        stations = get_available_stations()
        return jsonify({'stations': stations})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/pollutants', methods=['GET'])
def api_pollutants():
    """API endpoint to get available pollutants."""
    try:
        pollutants = get_available_pollutants()
        return jsonify({'pollutants': pollutants})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/forecast', methods=['POST'])
def api_forecast():
    """API endpoint to generate forecast."""
    # Get JSON data
    data = request.json
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    # Extract parameters
    station = data.get('station')
    pollutant = data.get('pollutant')
    prediction_length = data.get('prediction_length', '1day')
    
    if not station or not pollutant:
        return jsonify({'error': 'Missing required parameters'}), 400
    
    # Make sure models are loaded
    if not universal_models:
        load_universal_models()
    
    if not universal_models:
        return jsonify({'error': "No models found. Please train the models first."}), 500
    
    # Log the request details for debugging
    print(f"Forecast requested for station: '{station}', pollutant: '{pollutant}'")
    
    # Prepare data for prediction
    try:
        X_test, error = prepare_prediction_data(station, pollutant)
        
        if X_test is None:
            print(f"Data preparation error: {error}")
            return jsonify({'error': error}), 400
            
        print(f"Data prepared successfully. X_test shape: {X_test.shape}")
    except Exception as e:
        error_msg = f"Error preparing data: {str(e)}"
        print(error_msg)
        return jsonify({'error': error_msg}), 500
    
    # Make predictions
    try:
        predictions = {}
        prediction_errors = {}
        
        # Use all available models
        for model_name, model in universal_models.items():
            try:
                if model_name == 'LSTM':
                    # LSTM is a direct PyTorch model
                    input_features = model.input_shape[1] if hasattr(model, 'input_shape') else None
                    print(f"LSTM model input features: {input_features}, data input features: {X_test.shape[-1]}")
                    pred = model.predict(X_test)
                elif model_name == 'ANN':
                    # ANN is a direct PyTorch model, use the last time step
                    # Handle different input_shape attribute formats (could be int or tuple)
                    if hasattr(model, 'input_shape'):
                        if isinstance(model.input_shape, tuple):
                            input_features = model.input_shape[1] if len(model.input_shape) > 1 else model.input_shape[0]
                        else:
                            input_features = model.input_shape  # If it's just an int
                        print(f"ANN model input features: {input_features}, data input features: {X_test.shape[-1]}")
                    
                    # Use 2D input shape for ANN (samples, features)
                    if len(X_test.shape) == 3:
                        ann_input = X_test[:, -1, :]  # Use the last time step
                    else:
                        ann_input = X_test
                    
                    pred = model.predict(ann_input)
                elif model_name in ['Random Forest', 'MLR']:
                    # These models expect flattened input
                    n_features = getattr(model.model, 'n_features_in_', None) if hasattr(model, 'model') else None
                    reshaped_input = X_test.reshape(X_test.shape[0], -1)
                    print(f"{model_name} model features: {n_features}, reshaped data features: {reshaped_input.shape[-1]}")
                    pred = model.predict(reshaped_input)
                elif model_name == 'Linear Regression':
                    # Linear Regression expects 2D input, use the last time step
                    n_features = getattr(model.model, 'n_features_in_', None) if hasattr(model, 'model') else None
                    print(f"Linear Regression model features: {n_features}, data features: {X_test[:, -1, :].shape[-1]}")
                    # Ensure we have no NaN values
                    lr_input = X_test[:, -1, :]
                    lr_input = np.nan_to_num(lr_input, nan=0.0)
                    pred = model.predict(lr_input)
                
                predictions[model_name] = pred[0].tolist()  # Take the first prediction sequence
            except Exception as e:
                error_details = f"Error with {model_name} model: {str(e)}"
                print(error_details)
                prediction_errors[model_name] = error_details
        
        if not predictions:
            return jsonify({
                'error': f"All models failed to generate predictions. Errors: {prediction_errors}"
            }), 500
            
        # Create time labels
        time_steps = [f"Hour {i+1}" for i in range(len(next(iter(predictions.values()))))]
        
        # Format the forecast data
        forecast_data = {
            'predictions': predictions,
            'time_steps': time_steps,
            'station': station,
            'pollutant': pollutant
        }
        
        # Add errors if any models failed
        if prediction_errors:
            forecast_data['errors'] = prediction_errors
        
        return jsonify(forecast_data)
    except Exception as e:
        error_msg = f"Failed to generate predictions: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return jsonify({'error': error_msg}), 500

@app.route('/api/metrics', methods=['GET', 'POST'])
def api_metrics():
    """API endpoint to get model metrics."""
    try:
        # Get station and pollutant from POST data if available
        if request.method == 'POST' and request.is_json:
            data = request.json
            station = data.get('station', None)
            pollutant = data.get('pollutant', None)
            prediction_length = data.get('prediction_length', '1day')
            
            # Log the request details
            print(f"Metrics requested for station: '{station}', pollutant: '{pollutant}'")
            
            # This is a placeholder - in a real implementation, you would load metrics from files
            # and filter by station/pollutant if provided
            metrics = {
                'LSTM': {'MAE': 0.15, 'RMSE': 0.25, 'R²': 0.85},
                'ANN': {'MAE': 0.18, 'RMSE': 0.28, 'R²': 0.82},
                'Random Forest': {'MAE': 0.20, 'RMSE': 0.30, 'R²': 0.80},
                'Linear Regression': {'MAE': 0.25, 'RMSE': 0.35, 'R²': 0.75},
                'MLR': {'MAE': 0.22, 'RMSE': 0.32, 'R²': 0.78}
            }
        else:
            # Default metrics if not a POST request
            metrics = {
                'LSTM': {'MAE': 0.15, 'RMSE': 0.25, 'R²': 0.85},
                'ANN': {'MAE': 0.18, 'RMSE': 0.28, 'R²': 0.82},
                'Random Forest': {'MAE': 0.20, 'RMSE': 0.30, 'R²': 0.80},
                'Linear Regression': {'MAE': 0.25, 'RMSE': 0.35, 'R²': 0.75},
                'MLR': {'MAE': 0.22, 'RMSE': 0.32, 'R²': 0.78}
            }
        
        return jsonify({'metrics': metrics})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create the index.html template if it doesn't exist
    if not os.path.exists('templates/index.html'):
        with open('templates/index.html', 'w') as f:
            f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Air Pollution Prediction System</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            padding: 20px;
        }
        .prediction-container {
            margin-top: 30px;
        }
        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
        .error-message {
            color: red;
            margin: 20px 0;
            display: none;
        }
        .prediction-results {
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="mb-4">Taiwan Air Pollution Prediction System</h1>
        
        <div class="card">
            <div class="card-header">
                <h5>Generate Predictions</h5>
            </div>
            <div class="card-body">
                <form id="prediction-form">
                    <div class="row mb-3">
                        <div class="col-md-6">
                            <label for="station" class="form-label">Station</label>
                            <select class="form-select" id="station" name="station" required>
                                {% for station in stations %}
                                <option value="{{ station }}">{{ station }}</option>
                                {% endfor %}
                            </select>
                        </div>
                        <div class="col-md-6">
                            <label for="pollutant" class="form-label">Pollutant</label>
                            <select class="form-select" id="pollutant" name="pollutant" required>
                                {% for pollutant in pollutants %}
                                <option value="{{ pollutant }}">{{ pollutant }}</option>
                                {% endfor %}
                            </select>
                        </div>
                    </div>
                    
                    <div class="row mb-3">
                        <div class="col-md-12">
                            <label for="model_type" class="form-label">Model Type</label>
                            <select class="form-select" id="model_type" name="model_type" required>
                                {% for model in model_types %}
                                <option value="{{ model }}">{{ model }}</option>
                                {% endfor %}
                            </select>
                            <input type="hidden" name="prediction_length" value="{{ prediction_length }}">
                        </div>
                    </div>
                    
                    <button type="submit" class="btn btn-primary">Generate Prediction</button>
                </form>
            </div>
        </div>
        
        <div class="loading">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p>Generating predictions. Please wait...</p>
        </div>
        
        <div class="error-message alert alert-danger"></div>
        
        <div class="prediction-container prediction-results">
            <h2 class="mb-3">Prediction Results</h2>
            
            <div class="card mb-4">
                <div class="card-header">
                    <h5>Prediction Plot</h5>
                </div>
                <div class="card-body text-center">
                    <img id="prediction-plot" class="img-fluid" alt="Prediction Plot">
                </div>
            </div>
            
            <div class="card">
                <div class="card-header">
                    <h5>Prediction Values</h5>
                </div>
                <div class="card-body">
                    <div id="prediction-table"></div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        document.getElementById('prediction-form').addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Show loading indicator
            document.querySelector('.loading').style.display = 'block';
            document.querySelector('.error-message').style.display = 'none';
            document.querySelector('.prediction-results').style.display = 'none';
            
            // Collect form data
            const formData = new FormData(this);
            
            // Send prediction request
            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                // Hide loading indicator
                document.querySelector('.loading').style.display = 'none';
                
                if (data.success) {
                    // Show prediction results
                    document.querySelector('.prediction-results').style.display = 'block';
                    
                    // Display plot
                    document.getElementById('prediction-plot').src = 'data:image/png;base64,' + data.plot;
                    
                    // Display table
                    document.getElementById('prediction-table').innerHTML = data.table;
                } else {
                    // Show error message
                    document.querySelector('.error-message').style.display = 'block';
                    document.querySelector('.error-message').textContent = data.error;
                }
            })
            .catch(error => {
                // Hide loading indicator
                document.querySelector('.loading').style.display = 'none';
                
                // Show error message
                document.querySelector('.error-message').style.display = 'block';
                document.querySelector('.error-message').textContent = 'An error occurred: ' + error.message;
            });
        });
    </script>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
''')
    
    # Load universal models at startup
    load_universal_models()
    
    app.run(debug=True) 
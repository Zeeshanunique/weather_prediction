# Air Pollution Prediction System

This application provides air pollution forecasting for various stations in Taiwan. It uses both traditional machine learning and Google's Gemini API to generate predictions for different air pollutants.

## Features

- Predict air pollutant levels (PM2.5, PM10, CO, NO, etc.) for multiple weather stations
- Compare predictions from multiple models (LSTM, ANN, Random Forest, Linear Regression, MLR)
- Interactive visualization of prediction results
- Model performance metrics comparison

## Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/weather_prediction.git
cd weather_prediction
```

2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## API Key Setup

This application uses Google's Gemini API for generating predictions. You need to set up your API key:

1. Get a Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a `.env` file in the project root directory
3. Add your API key to the `.env` file:
```
GOOGLE_API_KEY=your_api_key_here
```

## Data

The application uses weather station data stored in CSV files in the `data/` directory. Each file contains historical air quality measurements for different stations.

## Usage

1. Start the Flask application:
```bash
python main.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Select a station, pollutant, and model type
4. Click "Generate Prediction" to view forecasts

## Models

The system uses multiple prediction models:

- LSTM (Long Short-Term Memory)
- ANN (Artificial Neural Network)
- Random Forest
- Linear Regression
- MLR (Multiple Linear Regression)

## License

[MIT License](LICENSE) 
// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // DOM elements
    const predictionForm = document.getElementById('prediction-form');
    const stationSelect = document.getElementById('station-select');
    const pollutantSelect = document.getElementById('pollutant-select');
    const predictionResult = document.getElementById('prediction-result');
    const predictionTitle = document.getElementById('prediction-title');
    const predictionDetails = document.getElementById('prediction-details');
    const metricsCard = document.getElementById('metrics-card');
    const metricsTable = document.getElementById('metrics-table').querySelector('tbody');
    const errorToast = document.getElementById('error-toast');
    const errorMessage = document.getElementById('error-message');
    const loadingModal = new bootstrap.Modal(document.getElementById('loading-modal'));
    
    // Model table mappings
    const modelTables = {
        'LSTM': document.getElementById('lstm-table').querySelector('tbody'),
        'Linear Regression': document.getElementById('linear-table').querySelector('tbody'),
        'MLR': document.getElementById('mlr-table').querySelector('tbody'),
        'Random Forest': document.getElementById('rf-table').querySelector('tbody'),
        'ANN': document.getElementById('ann-table').querySelector('tbody')
    };
    
    // Bootstrap toast object for error messages
    const toast = new bootstrap.Toast(errorToast);
    
    // Initialize the application
    init();
    
    // Event listeners
    predictionForm.addEventListener('submit', handlePredictionSubmit);
    
    // Function to initialize the application
    async function init() {
        try {
            // Load stations
            const stations = await fetchStations();
            populateSelect(stationSelect, stations);
            
            // Load pollutants
            const pollutants = await fetchPollutants();
            populateSelect(pollutantSelect, pollutants);
        } catch (error) {
            showError(`Failed to initialize: ${error.message}`);
        }
    }
    
    // Function to handle prediction form submission
    async function handlePredictionSubmit(event) {
        event.preventDefault();
        
        // Get form values
        const station = stationSelect.value;
        const pollutant = pollutantSelect.value;
        const predictionLength = document.querySelector('input[name="prediction-length"]:checked').value;
        
        // Validate form
        if (!station || !pollutant || !predictionLength) {
            showError('Please fill in all fields');
            return;
        }
        
        try {
            // Show loading modal
            loadingModal.show();
            
            // Get forecast
            const forecast = await fetchForecast(station, pollutant, predictionLength);
            
            // Get metrics
            const metrics = await fetchMetrics(station, pollutant, predictionLength);
            
            // Hide loading modal
            loadingModal.hide();
            
            // Display results
            displayForecast(forecast);
            displayMetrics(metrics);
            
        } catch (error) {
            loadingModal.hide();
            showError(`Failed to generate prediction: ${error.message}`);
        }
    }
    
    // Function to fetch available stations
    async function fetchStations() {
        const response = await fetch('/api/stations');
        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`);
        }
        const data = await response.json();
        return data.stations;
    }
    
    // Function to fetch available pollutants
    async function fetchPollutants() {
        const response = await fetch('/api/pollutants');
        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`);
        }
        const data = await response.json();
        return data.pollutants;
    }
    
    // Function to fetch forecast
    async function fetchForecast(station, pollutant, predictionLength) {
        const response = await fetch('/api/forecast', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                station,
                pollutant,
                prediction_length: predictionLength
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || `HTTP error ${response.status}`);
        }
        
        return await response.json();
    }
    
    // Function to fetch metrics
    async function fetchMetrics(station, pollutant, predictionLength) {
        const response = await fetch('/api/metrics', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                station,
                pollutant,
                prediction_length: predictionLength
            })
        });
        
        if (!response.ok) {
            // If metrics aren't available, just return empty array
            if (response.status === 404) {
                return [];
            }
            
            const error = await response.json();
            throw new Error(error.error || `HTTP error ${response.status}`);
        }
        
        return await response.json();
    }
    
    // Function to populate select elements
    function populateSelect(selectElement, options) {
        // Clear existing options except the first one
        const defaultOption = selectElement.options[0];
        selectElement.innerHTML = '';
        selectElement.appendChild(defaultOption);
        
        // Add new options
        options.forEach(option => {
            const optionElement = document.createElement('option');
            optionElement.value = option;
            optionElement.textContent = option;
            selectElement.appendChild(optionElement);
        });
    }
    
    // Function to display forecast
    function displayForecast(forecast) {
        // Update title
        predictionTitle.textContent = `${forecast.station} ${forecast.pollutant} Prediction`;
        
        // Check if we have a plot image
        if (forecast.plot) {
            console.log("Plot received, length:", forecast.plot.length);
            const imgElem = document.createElement('img');
            imgElem.src = `data:image/png;base64,${forecast.plot}`;
            imgElem.alt = "Prediction Plot";
            imgElem.className = "prediction-image img-fluid mb-3";
            
            // Add error handling to debug image loading issues
            imgElem.onerror = function() {
                console.error("Error loading image");
                predictionResult.innerHTML = `
                    <div class="alert alert-danger">
                        <p>Failed to load prediction image</p>
                        <p>Data length: ${forecast.plot ? forecast.plot.length : 'No data'}</p>
                    </div>
                `;
            };
            
            predictionResult.innerHTML = '';
            predictionResult.appendChild(imgElem);
        } else {
            // If no plot image, create placeholder
            predictionResult.innerHTML = `
                <div class="alert alert-info">Prediction successful. Detailed view not available.</div>
            `;
        }
        
        // Clear existing table data
        Object.values(modelTables).forEach(table => {
            table.innerHTML = '';
        });
        
        // Populate model tables
        for (const [modelName, predictions] of Object.entries(forecast.predictions)) {
            const tableBody = modelTables[modelName];
            if (tableBody) {
                // Get time steps array if available, or generate one
                const timeSteps = forecast.time_steps || Array.from({length: predictions.length}, (_, i) => `Hour ${i+1}`);
                
                predictions.forEach((value, index) => {
                    const row = document.createElement('tr');
                    
                    const timeCell = document.createElement('td');
                    timeCell.textContent = timeSteps[index] || `Time ${index + 1}`;
                    
                    const valueCell = document.createElement('td');
                    valueCell.textContent = parseFloat(value).toFixed(2);
                    
                    row.appendChild(timeCell);
                    row.appendChild(valueCell);
                    tableBody.appendChild(row);
                });
            }
        }
        
        // Show any errors from specific models
        if (forecast.errors) {
            const errorsList = document.createElement('div');
            errorsList.className = 'alert alert-warning';
            errorsList.innerHTML = '<h5>Some models had errors:</h5><ul>';
            
            for (const [modelName, error] of Object.entries(forecast.errors)) {
                errorsList.innerHTML += `<li>${modelName}: ${error}</li>`;
            }
            
            errorsList.innerHTML += '</ul>';
            predictionResult.appendChild(errorsList);
        }
        
        // Show prediction details
        predictionDetails.style.display = 'block';
    }
    
    // Function to display metrics
    function displayMetrics(metricsResponse) {
        // Check if we have metrics data
        if (!metricsResponse || !metricsResponse.metrics) {
            metricsCard.style.display = 'none';
            return;
        }
        
        // Clear existing table data
        metricsTable.innerHTML = '';
        
        // Process metrics data - convert from object to array format
        const metricsData = [];
        for (const [modelName, modelMetrics] of Object.entries(metricsResponse.metrics)) {
            metricsData.push({
                Model: modelName,
                MAE: modelMetrics.MAE || 0,
                RMSE: modelMetrics.RMSE || 0,
                MSE: modelMetrics['R²'] || 0, // Using R² as MSE for display
            });
        }
        
        // If we don't have any metrics data, hide the card
        if (metricsData.length === 0) {
            metricsCard.style.display = 'none';
            return;
        }
        
        // Find the best model (lowest RMSE)
        const bestModel = metricsData.reduce((prev, current) => 
            (prev.RMSE < current.RMSE) ? prev : current
        );
        
        // Populate metrics table
        metricsData.forEach(metric => {
            const row = document.createElement('tr');
            
            // Highlight the best model
            if (metric.Model === bestModel.Model) {
                row.classList.add('best-model');
            }
            
            const modelCell = document.createElement('td');
            modelCell.textContent = metric.Model;
            
            const mseCell = document.createElement('td');
            mseCell.textContent = metric.MSE.toFixed(4);
            
            const maeCell = document.createElement('td');
            maeCell.textContent = metric.MAE.toFixed(4);
            
            const rmseCell = document.createElement('td');
            rmseCell.textContent = metric.RMSE.toFixed(4);
            
            row.appendChild(modelCell);
            row.appendChild(mseCell);
            row.appendChild(maeCell);
            row.appendChild(rmseCell);
            
            metricsTable.appendChild(row);
        });
        
        // Show metrics card
        metricsCard.style.display = 'block';
    }
    
    // Function to format prediction length for display
    function formatPredictionLength(predictionLength) {
        switch (predictionLength) {
            case '1day':
                return '1 Day';
            case '1week':
                return '1 Week';
            case '1month':
                return '1 Month';
            default:
                return predictionLength;
        }
    }
    
    // Function to show error message
    function showError(message) {
        errorMessage.textContent = message;
        toast.show();
    }
}); 
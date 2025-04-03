import numpy as np

class BaseModel:
    """Base model class with functionality shared by all models."""
    
    def __init__(self, name="BaseModel"):
        self.name = name
        self.model = None
        self.metadata = None
    
    def mock_predict(self, X_test):
        """Generate mock predictions for when we have mock models.
        
        Args:
            X_test (numpy.ndarray): Test data
            
        Returns:
            numpy.ndarray: Mock predictions with the same shape as expected from a real model
        """
        # Generate random predictions based on the input shape
        # For sequence data (3D), return a prediction with the same number of time steps
        if len(X_test.shape) == 3:
            # Return a smooth sine wave pattern for visualization
            # Shape: (batch_size, output_sequence_length)
            batch_size = X_test.shape[0]
            output_length = 24  # Default output length
            
            # Create smooth predictions using sine wave
            time = np.linspace(0, 2*np.pi, output_length)
            mock_predictions = np.sin(time) * 0.5 + 0.5  # Scale to [0, 1]
            
            # Repeat for each sample in the batch
            mock_predictions = np.tile(mock_predictions, (batch_size, 1))
            
            return mock_predictions
        
        # For 2D input, return single time step predictions
        elif len(X_test.shape) == 2:
            batch_size = X_test.shape[0]
            return np.random.rand(batch_size, 24)  # Default to 24 time steps
        
        # For 1D input, reshape appropriately
        else:
            return np.random.rand(1, 24)  # Default to 24 time steps 
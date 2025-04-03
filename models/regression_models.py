import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import joblib

class LinearRegressionModel:
    def __init__(self, name="LinearRegression"):
        """Initialize Linear Regression model.
        
        Args:
            name (str): Name of the model
        """
        self.model = None
        self.name = name
        
    def train(self, X_train, y_train):
        """Train the Linear Regression model.
        
        Args:
            X_train (numpy.ndarray): Training data
            y_train (numpy.ndarray): Target data
            
        Returns:
            self: Trained model instance
        """
        # Reshape X_train if it's 3D (sequence data)
        if len(X_train.shape) == 3:
            # For linear regression, we'll use the last time step of each sequence
            X_train = X_train[:, -1, :]
        
        # For each time step in the output sequence, train a separate model
        models = []
        for i in range(y_train.shape[1]):
            model = LinearRegression()
            model.fit(X_train, y_train[:, i])
            models.append(model)
        
        self.model = models
        return self
    
    def predict(self, X_test):
        """Make predictions with the trained model.
        
        Args:
            X_test (numpy.ndarray): Test data
            
        Returns:
            numpy.ndarray: Predictions
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Reshape X_test if it's 3D (sequence data)
        if len(X_test.shape) == 3:
            # For linear regression, we'll use the last time step of each sequence
            X_test = X_test[:, -1, :]
        
        # Make predictions for each time step using the corresponding model
        predictions = np.zeros((X_test.shape[0], len(self.model)))
        
        for i, model in enumerate(self.model):
            predictions[:, i] = model.predict(X_test)
        
        return predictions
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model performance.
        
        Args:
            X_test (numpy.ndarray): Test data
            y_test (numpy.ndarray): True values
            
        Returns:
            dict: Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Get predictions
        predictions = self.predict(X_test)
        
        # Calculate MSE and MAE
        mse = mean_squared_error(y_test.reshape(-1), predictions.reshape(-1))
        mae = mean_absolute_error(y_test.reshape(-1), predictions.reshape(-1))
        
        # Root Mean Squared Error
        rmse = np.sqrt(mse)
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse
        }
    
    def plot_predictions(self, y_true, y_pred, title=None, save_path=None):
        """Plot predictions vs ground truth.
        
        Args:
            y_true (numpy.ndarray): True values
            y_pred (numpy.ndarray): Predicted values
            title (str, optional): Plot title
            save_path (str, optional): Path to save the plot
        """
        plt.figure(figsize=(12, 6))
        
        # We'll plot the first sample in the test set
        plt.plot(y_true[0], label='Actual')
        plt.plot(y_pred[0], label='Predicted')
        
        plt.title(title or f'{self.name} Prediction vs Actual')
        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
            
        plt.show()
    
    def save(self, model_dir='saved_models'):
        """Save the trained model.
        
        Args:
            model_dir (str): Directory to save the model
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        # Create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Save the model
        model_path = os.path.join(model_dir, f"{self.name}_model.pkl")
        joblib.dump(self.model, model_path)
        
        # Save model metadata
        metadata = {
            'name': self.name
        }
        
        metadata_path = os.path.join(model_dir, f"{self.name}_metadata.pkl")
        joblib.dump(metadata, metadata_path)
        
        return model_path, metadata_path
    
    @classmethod
    def load(cls, model_path, metadata_path):
        """Load a trained model.
        
        Args:
            model_path (str): Path to the saved model
            metadata_path (str): Path to model metadata
            
        Returns:
            LinearRegressionModel: Loaded model
        """
        # Load metadata
        metadata = joblib.load(metadata_path)
        
        # Create instance with metadata
        instance = cls(name=metadata['name'])
        
        # Load the model
        instance.model = joblib.load(model_path)
        
        return instance


class MultipleLinearRegressionModel(LinearRegressionModel):
    """Multiple Linear Regression is essentially the same as Linear Regression in scikit-learn,
    as LinearRegression already handles multiple features. This class is provided for
    naming consistency and to potentially add MLR-specific extensions in the future.
    """
    
    def __init__(self, name="MultipleLinearRegression"):
        """Initialize Multiple Linear Regression model.
        
        Args:
            name (str): Name of the model
        """
        super().__init__(name=name) 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib

class LSTMModel(nn.Module):
    def __init__(self, input_shape=(24, 1), output_length=24, name="LSTM"):
        """Initialize LSTM model.
        
        Args:
            input_shape (tuple): Shape of input data (sequence_length, features)
            output_length (int): Length of output sequence
            name (str): Name of the model
        """
        super(LSTMModel, self).__init__()
        self.input_shape = input_shape
        self.output_length = output_length
        self.name = name
        self.history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': []}
        
        # Define LSTM layers
        self.lstm1 = nn.LSTM(
            input_size=input_shape[1],
            hidden_size=64,
            batch_first=True
        )
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(
            input_size=64,
            hidden_size=64,
            batch_first=True
        )
        self.dropout2 = nn.Dropout(0.2)
        self.dense1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(32, output_length)
        
    def forward(self, x):
        """Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, features)
            
        Returns:
            torch.Tensor: Output tensor
        """
        # First LSTM layer
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        
        # Second LSTM layer - we take the output at the last time step
        outputs, (hidden, _) = self.lstm2(x)
        x = self.dropout2(hidden[-1])
        
        # Dense layers
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        
        return x
    
    def build_model(self):
        """Build PyTorch model (included for compatibility).
        
        Returns:
            LSTMModel: self
        """
        # This method is included for API compatibility
        # In PyTorch, model is defined in __init__
        return self
    
    def train_model(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.2):
        """Train the LSTM model.
        
        Args:
            X_train (numpy.ndarray): Training data
            y_train (numpy.ndarray): Target data
            epochs (int): Number of epochs
            batch_size (int): Batch size
            validation_split (float): Portion of data to use for validation
            
        Returns:
            dict: Training history
        """
        # Reshape input if needed
        if len(X_train.shape) == 2:
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        
        # Convert numpy arrays to PyTorch tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        
        # Split training data for validation
        val_size = int(len(X_train) * validation_split)
        train_size = len(X_train) - val_size
        
        X_train_tensor, X_val_tensor = torch.split(X_train, [train_size, val_size])
        y_train_tensor, y_val_tensor = torch.split(y_train, [train_size, val_size])
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters())
        
        # Early stopping parameters
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.train()
            train_loss = 0.0
            train_mae = 0.0
            
            for inputs, targets in train_loader:
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                
                # Calculate MAE for tracking
                mae = torch.mean(torch.abs(outputs - targets))
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
                train_mae += mae.item() * inputs.size(0)
            
            avg_train_loss = train_loss / len(X_train_tensor)
            avg_train_mae = train_mae / len(X_train_tensor)
            
            # Validation phase
            self.eval()
            val_loss = 0.0
            val_mae = 0.0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = self(inputs)
                    loss = criterion(outputs, targets)
                    mae = torch.mean(torch.abs(outputs - targets))
                    
                    val_loss += loss.item() * inputs.size(0)
                    val_mae += mae.item() * inputs.size(0)
            
            avg_val_loss = val_loss / len(X_val_tensor)
            avg_val_mae = val_mae / len(X_val_tensor)
            
            # Save metrics for history
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(avg_val_loss)
            self.history['train_mae'].append(avg_train_mae)
            self.history['val_mae'].append(avg_val_mae)
            
            print(f'Epoch {epoch+1}/{epochs} - '
                  f'loss: {avg_train_loss:.4f} - '
                  f'mae: {avg_train_mae:.4f} - '
                  f'val_loss: {avg_val_loss:.4f} - '
                  f'val_mae: {avg_val_mae:.4f}')
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model weights
                best_model_state = self.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    # Restore best model weights
                    self.load_state_dict(best_model_state)
                    break
        
        return self.history
    
    def predict(self, X_test):
        """Make predictions with the trained model.
        
        Args:
            X_test (numpy.ndarray): Test data
            
        Returns:
            numpy.ndarray: Predictions
        """
        # Reshape input if needed
        if len(X_test.shape) == 2:
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        # Convert to PyTorch tensor
        X_test_tensor = torch.FloatTensor(X_test)
        
        # Set model to evaluation mode
        self.eval()
        
        # Make predictions
        with torch.no_grad():
            predictions = self(X_test_tensor)
        
        # Convert to numpy array
        return predictions.numpy()
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model performance.
        
        Args:
            X_test (numpy.ndarray): Test data
            y_test (numpy.ndarray): True values
            
        Returns:
            dict: Dictionary with evaluation metrics
        """
        # Get predictions
        predictions = self.predict(X_test)
        
        # Calculate MSE and MAE
        mse = np.mean((predictions - y_test) ** 2)
        mae = np.mean(np.abs(predictions - y_test))
        
        # Root Mean Squared Error
        rmse = np.sqrt(mse)
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse
        }
    
    def plot_history(self, save_path=None):
        """Plot training history.
        
        Args:
            save_path (str, optional): Path to save the plot
        """
        if not self.history['train_loss']:
            raise ValueError("Model has not been trained yet")
            
        plt.figure(figsize=(12, 4))
        
        # Plot training & validation loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'])
        plt.plot(self.history['val_loss'])
        plt.title(f'{self.name} Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        # Plot training & validation mean absolute error
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_mae'])
        plt.plot(self.history['val_mae'])
        plt.title(f'{self.name} Model MAE')
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        plt.show()
    
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
        # Create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Save the PyTorch model
        model_path = os.path.join(model_dir, f"{self.name}_model.pt")
        torch.save(self.state_dict(), model_path)
        
        # Save model metadata
        metadata = {
            'input_shape': self.input_shape,
            'output_length': self.output_length,
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
            LSTMModel: Loaded model
        """
        # Load metadata
        metadata = joblib.load(metadata_path)
        
        # Create instance with metadata
        instance = cls(
            input_shape=metadata['input_shape'],
            output_length=metadata['output_length'],
            name=metadata['name']
        )
        
        # Load model weights
        instance.load_state_dict(torch.load(model_path))
        instance.eval()
        
        return instance 
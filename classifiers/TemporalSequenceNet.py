import os
import logging
import json
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False


if TORCH_AVAILABLE:
    class SequenceDataset(Dataset):
        """Custom Dataset for sequence windows."""
        def __init__(self, X: np.ndarray, y: np.ndarray):
            # X: (N, T, K)
            # y: (N,)
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.long)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]


    class GRUNet(nn.Module):
        """Gated Recurrent Unit (GRU) network for presence sequence classification."""
        def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, num_layers: int = 1):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, num_classes)

        def forward(self, x):
            # x: (Batch, SeqLen, InputDim)
            # Make sure input has correct dtype
            x = x.to(torch.float32)
            out, _ = self.gru(x)
            # Pull output of the last time step
            out = self.fc(out[:, -1, :])
            return out


    class CNN1DNet(nn.Module):
        """1D Convolutional Neural Network for presence sequence classification."""
        def __init__(self, input_dim: int, seq_len: int, num_classes: int):
            super().__init__()
            self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool1d(kernel_size=2)
            
            # Calculate size after pooling
            pooled_len = seq_len // 2
            self.fc = nn.Linear(32 * pooled_len, num_classes)

        def forward(self, x):
            # Input shape: (Batch, SeqLen, InputDim)
            # Conv1d expects shape: (Batch, Channels/InputDim, SeqLen)
            x = x.to(torch.float32)
            x = x.transpose(1, 2)
            x = self.conv1(x)
            x = self.relu(x)
            x = self.pool(x)
            x = x.flatten(start_dim=1)
            x = self.fc(x)
            return x
else:
    class SequenceDataset:
        pass
    class GRUNet:
        pass
    class CNN1DNet:
        pass



class TemporalSequenceNet:
    """Wrapper that builds, trains, and exports PyTorch sequence models to ONNX."""

    def __init__(self, model_type: str = "GRU", hidden_dim: int = 64, epochs: int = 60, batch_size: int = 32):
        self.logger = logging.getLogger(__name__)
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.classes: List[str] = []
        self._modelTrained = False
        self._sensorKeys: List[str] = []
        self._windowSteps: int = 15
        
        # ONNX Runtime session for inference
        self.ort_session: Optional[ort.InferenceSession] = None
        self.onnx_path: Optional[str] = None

        if not TORCH_AVAILABLE:
            self.logger.warning("PyTorch (torch) is not installed. Neural network training will be unavailable.")
        if not ORT_AVAILABLE:
            self.logger.warning("onnxruntime is not installed. Sequence model inference will be unavailable.")

    def train_model(
        self,
        X_trainval: np.ndarray,
        y_trainval: np.ndarray,
        sensor_keys: List[str],
        classes: List[str],
        save_path: str
    ) -> bool:
        """Trains the sequence network on CPU and exports it to ONNX format.
        
        Args:
            X_trainval: Numpy array of shape (N, WindowSteps, NumSensors)
            y_trainval: Numpy array of string labels of shape (N,)
            sensor_keys: Sensor key list matching dimensions.
            classes: List of unique room labels.
            save_path: Path to save the compiled ONNX model.
        """
        if not TORCH_AVAILABLE:
            self.logger.error("Cannot train model: PyTorch is not installed.")
            return False

        if len(X_trainval) < 10:
            self.logger.warning("Insufficient data to train sequence model.")
            return False

        self._sensorKeys = sensor_keys
        self._windowSteps = X_trainval.shape[1]
        self.classes = sorted(list(set(classes)))
        num_classes = len(self.classes)

        # Label encoding
        label_to_idx = {lbl: idx for idx, lbl in enumerate(self.classes)}
        y_indexed = np.array([label_to_idx[lbl] for lbl in y_trainval], dtype=np.int64)

        # Handle NaNs inside sequence array (fill with 9999.0)
        X_trainval = np.nan_to_num(X_trainval, nan=9999.0)

        # Dataset & Dataloader
        dataset = SequenceDataset(X_trainval, y_indexed)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        input_dim = len(sensor_keys)
        seq_len = self._windowSteps

        # Model instantiation
        if self.model_type == "CNN1D":
            model = CNN1DNet(input_dim, seq_len, num_classes)
        else:
            model = GRUNet(input_dim, self.hidden_dim, num_classes)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.005)

        model.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * batch_x.size(0)

        self.logger.info(f"Finished training PyTorch {self.model_type} model.")

        # Export to ONNX
        try:
            model.eval()
            dummy_input = torch.randn(1, seq_len, input_dim)
            
            # Export the model
            torch.onnx.export(
                model,
                dummy_input,
                save_path,
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            self.logger.info(f"Exported ONNX model successfully to {save_path}")
            self.onnx_path = save_path
            
            # Initialize ONNX runtime session for inference
            if ORT_AVAILABLE:
                self.ort_session = ort.InferenceSession(save_path)
                self._modelTrained = True
                return True
        except Exception as e:
            self.logger.exception(f"ONNX export or initialization failed: {e}")
            
        return False

    def load_onnx_model(self, onnx_path: str, classes: List[str], sensor_keys: List[str], window_steps: int) -> bool:
        """Loads an existing pre-compiled ONNX model for inference."""
        if not ORT_AVAILABLE:
            self.logger.error("Cannot load ONNX model: onnxruntime is not installed.")
            return False
            
        if not os.path.exists(onnx_path):
            self.logger.warning(f"ONNX model file not found at {onnx_path}")
            return False

        try:
            self.ort_session = ort.InferenceSession(onnx_path)
            self.onnx_path = onnx_path
            self.classes = sorted(list(classes))
            self._sensorKeys = sensor_keys
            self._windowSteps = window_steps
            self._modelTrained = True
            self.logger.info(f"Loaded ONNX model successfully from {onnx_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load ONNX model: {e}")
            return False

    def predict_sequence(self, sensor_df: pd.DataFrame) -> Tuple[Optional[str], float]:
        """Runs inference via ONNX Runtime on a single time-series sequence window.
        
        Args:
            sensor_df: DataFrame of shape (WindowSteps, NumSensors)
        """
        if not self._modelTrained or not self.ort_session:
            return None, 0.0

        if len(sensor_df) < self._windowSteps:
            return None, 0.0

        # Reindex to match sensor key order
        aligned_df = sensor_df.reindex(columns=self._sensorKeys).fillna(9999.0)
        
        # Convert to float32 numpy array of shape (1, T, K)
        seq_array = np.expand_dims(aligned_df.to_numpy(), axis=0).astype(np.float32)

        try:
            # Run inference
            ort_inputs = {'input': seq_array}
            ort_outs = self.ort_session.run(None, ort_inputs)
            logits = ort_outs[0][0]
            
            # Apply softmax to calculate probabilities
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            
            pred_idx = int(np.argmax(probs))
            label = self.classes[pred_idx]
            confidence = float(probs[pred_idx])
            
            return label, confidence
        except Exception as e:
            self.logger.error(f"ONNX sequence prediction failed: {e}")
            return None, 0.0

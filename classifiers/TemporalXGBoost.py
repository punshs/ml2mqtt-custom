import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from typing import TypedDict, Optional, List, Dict, Any, Union
import logging

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class TemporalXGBoostParams(TypedDict):
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    min_child_weight: int
    reg_alpha: float
    reg_lambda: float


DEFAULT_PARAMS: TemporalXGBoostParams = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
}


class TemporalXGBoost:
    """XGBoost model trained on flattened temporal sequences.
    
    Flattens a window of shape (WindowSteps, NumSensors) into a single row of shape
    (WindowSteps * NumSensors) to capture time-series context while leveraging 
    XGBoost's native support for missing values (NaNs).
    """

    def __init__(self, params: Optional[TemporalXGBoostParams] = None):
        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "xgboost is not installed. Install it with: pip install xgboost"
            )

        self.params: TemporalXGBoostParams = {
            **DEFAULT_PARAMS,
            **(params or {}),
        }
        self.logger: logging.Logger = logging.getLogger(__name__)
        self.logger.info(f"TemporalXGBoost initialized with params: {self.params}")

        self.labelEncoder: LabelEncoder = LabelEncoder()
        self._pipeline: Optional[Pipeline] = None
        self._X_test: Optional[pd.DataFrame] = None
        self._y_test: Optional[np.ndarray] = None
        self._modelTrained: bool = False
        self._featureNames: List[str] = []
        self._sensorKeys: List[str] = []
        self._windowSteps: int = 15

    def _flatten_sequence(self, X_seq: np.ndarray, sensor_keys: List[str]) -> pd.DataFrame:
        """Flattens 3D sequence array of shape (N, T, K) to 2D DataFrame of shape (N, T * K)."""
        N, T, K = X_seq.shape
        flat_data = X_seq.reshape(N, T * K)
        
        # Build column names: f"{sensor}_t-{T-1-t}"
        columns = []
        for t in range(T):
            lag = T - 1 - t
            for key in sensor_keys:
                columns.append(f"{key}_t-{lag}")
                
        self._featureNames = columns
        return pd.DataFrame(flat_data, columns=columns)

    def _robust_train_test_split(
        self, X: pd.DataFrame, y: np.ndarray, test_size: float = 0.3, random_state: Optional[int] = None
    ) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        unique_classes, class_counts = np.unique(y, return_counts=True)
        if np.min(class_counts) >= 2:
            return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
        
        rng = np.random.RandomState(random_state)
        train_idx = []
        test_idx = []
        for c in unique_classes:
            c_indices = np.where(y == c)[0]
            train_idx.append(c_indices[0])
            if len(c_indices) > 1:
                c_test_size = int(np.ceil(len(c_indices) * test_size))
                remaining = list(c_indices[1:])
                rng.shuffle(remaining)
                test_idx.extend(remaining[:c_test_size])
                train_idx.extend(remaining[c_test_size:])
        
        rng.shuffle(train_idx)
        rng.shuffle(test_idx)
        
        if not test_idx and len(y) > len(unique_classes):
            representatives = {np.where(y == c)[0][0] for c in unique_classes}
            candidates = [idx for idx in train_idx if idx not in representatives]
            if candidates:
                test_idx = [candidates[0]]
                train_idx.remove(candidates[0])
        
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        return X_train, X_test, y_train, y_test

    def populateDataframe(self, X_seq: np.ndarray, y_labels: np.ndarray, sensor_keys: List[str]) -> None:
        """Trains the model on sequence windows.
        
        Args:
            X_seq: Numpy array of shape (N, WindowSteps, NumSensors)
            y_labels: List/array of labels of shape (N,)
            sensor_keys: List of sensor names matching the K dimension of X_seq.
        """
        if len(X_seq) == 0 or len(y_labels) == 0:
            self.logger.warning("No time-series data available for training.")
            self._modelTrained = False
            return

        self._sensorKeys = sensor_keys
        self._windowSteps = X_seq.shape[1]

        # Flatten sequences
        X = self._flatten_sequence(X_seq, sensor_keys)
        y = self.labelEncoder.fit_transform(y_labels)

        num_classes = len(np.unique(y))
        if num_classes < 2:
            self.logger.warning("Need at least 2 distinct classes to train TemporalXGBoost model.")
            self._modelTrained = False
            return

        xgb_params = {**self.params}
        if num_classes == 2:
            xgb_params["objective"] = "binary:logistic"
            xgb_params["eval_metric"] = "logloss"
        else:
            xgb_params["objective"] = "multi:softprob"
            xgb_params["eval_metric"] = "mlogloss"
            xgb_params["num_class"] = num_classes

        xgb_params["tree_method"] = "hist"
        xgb_params["n_jobs"] = -1
        xgb_params["verbosity"] = 0

        self._pipeline = Pipeline(
            steps=[
                ("classifier", XGBClassifier(**xgb_params)),
            ]
        )

        try:
            X_train, X_test, y_train, y_test = self._robust_train_test_split(X, y, test_size=0.3)
            
            # Fill float NaNs with np.nan for native XGBoost handling
            X_train = X_train.fillna(np.nan)
            X_test = X_test.fillna(np.nan)
            
            self._pipeline.fit(X_train, y_train)
            self._X_test = X_test
            self._y_test = y_test
            self._modelTrained = True
            self.logger.info("TemporalXGBoost training completed successfully.")
        except Exception as e:
            self.logger.exception(f"TemporalXGBoost training failed: {e}")
            self._modelTrained = False

    def predictLabel(self, sensor_df: pd.DataFrame) -> tuple[Optional[str], float]:
        """Predicts the active label for a single sliding window.
        
        Args:
            sensor_df: DataFrame of shape (WindowSteps, NumSensors)
        """
        if not self._pipeline or not self._modelTrained:
            return None, 0.0

        # Assert shape correctness
        if len(sensor_df) < self._windowSteps:
            return None, 0.0

        # Align columns to match order of sensor keys
        aligned_df = sensor_df.reindex(columns=self._sensorKeys)
        
        # Flatten the window
        seq_array = np.expand_dims(aligned_df.to_numpy(), axis=0) # shape (1, T, K)
        X = self._flatten_sequence(seq_array, self._sensorKeys)
        
        # Align columns to test set
        X = X.reindex(columns=self._X_test.columns, fill_value=np.nan)

        try:
            y_pred = self._pipeline.predict(X)
            y_prob = self._pipeline.predict_proba(X)
            label = self.labelEncoder.inverse_transform(y_pred.astype(int))[0]
            confidence = max(y_prob[0])
            return label, float(confidence)
        except Exception as e:
            self.logger.error(f"TemporalXGBoost prediction failed: {e}")
            return None, 0.0

    def getFeatureImportance(self) -> Optional[Dict[str, float]]:
        if not self._modelTrained or self._pipeline is None:
            return None
        try:
            clf = self._pipeline.named_steps["classifier"]
            featureNames = self._X_test.columns
            importances = clf.feature_importances_
            return dict(zip(featureNames, importances))
        except Exception as e:
            self.logger.error(f"Feature importances retrieval failed: {e}")
            return None

    def getAccuracy(self) -> Optional[float]:
        if not self._modelTrained or self._pipeline is None:
            return None
        try:
            y_pred = self._pipeline.predict(self._X_test)
            return float(accuracy_score(self._y_test, y_pred.astype(int)))
        except Exception as e:
            self.logger.error(f"Accuracy calculation failed: {e}")
            return None

    def getLabelStats(self) -> Optional[Dict[str, Any]]:
        if not self._modelTrained or self._pipeline is None:
            return None
        try:
            y_pred = self._pipeline.predict(self._X_test)
            report = classification_report(
                self._y_test,
                y_pred.astype(int),
                labels=np.arange(len(self.labelEncoder.classes_)),
                target_names=self.labelEncoder.classes_,
                output_dict=True,
                zero_division=0,
            )
            return {
                label: {
                    "support": int(stats["support"]),
                    "precision": round(stats["precision"], 3),
                    "recall": round(stats["recall"], 3),
                    "f1": round(stats["f1-score"], 3),
                }
                for label, stats in report.items()
                if label in self.labelEncoder.classes_
            }
        except Exception as e:
            self.logger.error(f"Label stats generation failed: {e}")
            return None

    def getModelParameters(self) -> TemporalXGBoostParams:
        return self.params

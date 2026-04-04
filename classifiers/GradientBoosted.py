import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from typing import TypedDict, Optional, List, Dict, Any, Union
import logging
from ModelStore import ModelObservation

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class GradientBoostedParams(TypedDict):
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    min_child_weight: int
    reg_alpha: float
    reg_lambda: float


DEFAULT_GRADIENT_BOOSTED_PARAMS: GradientBoostedParams = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
}


class GradientBoosted:
    """XGBoost gradient boosted tree classifier.
    
    Key advantages over Random Forest for BLE room tracking:
    - Native missing value handling (no need for NullHandler's 9999 substitution)
    - Typically 5-15% better accuracy on structured/tabular data
    - Better handling of sensor dropout patterns
    - Faster training with n_jobs=-1
    """

    def __init__(self, params: Optional[GradientBoostedParams] = None):
        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "xgboost is not installed. Install it with: pip install xgboost"
            )

        self.params: GradientBoostedParams = {
            **DEFAULT_GRADIENT_BOOSTED_PARAMS,
            **(params or {}),
        }
        self.logger: logging.Logger = logging.getLogger(__name__)
        self.logger.info(f"GradientBoosted initialized with params: {self.params}")

        self.labelEncoder: LabelEncoder = LabelEncoder()
        self._pipeline: Optional[Pipeline] = None
        self._X_test: Optional[pd.DataFrame] = None
        self._y_test: Optional[np.ndarray] = None
        self._modelTrained: bool = False
        self._categoricalCols: List[str] = []
        self._ordinalEncoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )

    def populateDataframe(self, observations: List[ModelObservation]) -> None:
        data: List[Dict[str, Any]] = []
        labels: List[str] = []

        for obs in observations:
            data.append(obs.sensorValues)
            labels.append(obs.label)

        if not data or not labels:
            self.logger.warning("No data available for training.")
            self._modelTrained = False
            return

        X = pd.DataFrame(data)
        y = self.labelEncoder.fit_transform(labels)

        self._categoricalCols = X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        numericalCols = X.select_dtypes(include=[np.number]).columns.tolist()

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", self._ordinalEncoder, self._categoricalCols),
                ("num", "passthrough", numericalCols),
            ]
        )

        num_classes = len(np.unique(y))
        xgb_params = {**self.params}

        # XGBoost needs objective set based on class count
        if num_classes == 2:
            xgb_params["objective"] = "binary:logistic"
            xgb_params["eval_metric"] = "logloss"
        else:
            xgb_params["objective"] = "multi:softprob"
            xgb_params["eval_metric"] = "mlogloss"
            xgb_params["num_class"] = num_classes

        xgb_params["tree_method"] = "hist"  # Fast, supports missing values natively
        xgb_params["n_jobs"] = -1
        xgb_params["verbosity"] = 0

        self._pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", XGBClassifier(**xgb_params)),
            ]
        )

        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
            self._pipeline.fit(X_train, y_train)
            self._X_test = X_test
            self._y_test = y_test
            self._modelTrained = True
        except ValueError as e:
            self.logger.info(f"Not enough data to train the model: {e}")
            nan_columns = X.columns[X.isna().any()].tolist()
            self.logger.error(f"Columns with NaNs: {nan_columns}")
            self._modelTrained = False

    def predictLabel(
        self, sensorValues: Dict[str, Any]
    ) -> tuple[Optional[str], int]:
        if not self._pipeline or not self._modelTrained:
            return None, 0

        X = pd.DataFrame([sensorValues])
        X = X.reindex(columns=self._X_test.columns, fill_value=None)

        try:
            y_pred = self._pipeline.predict(X)
            y_prob = self._pipeline.predict_proba(X)
            label = self.labelEncoder.inverse_transform(y_pred.astype(int))[0]
            confidence = max(y_prob[0])
            return label, confidence
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return None, 0

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
            return accuracy_score(self._y_test, y_pred.astype(int))
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

    def optimizeParameters(
        self, observations: List[ModelObservation]
    ) -> Dict[str, Any]:
        data: List[Dict[str, Any]] = []
        labels: List[str] = []

        for obs in observations:
            data.append(obs.sensorValues)
            labels.append(obs.label)

        if not data or not labels:
            self.logger.warning("No data available for optimization.")
            return {}

        X = pd.DataFrame(data)
        y = self.labelEncoder.fit_transform(labels)

        self._categoricalCols = X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        numericalCols = X.select_dtypes(include=[np.number]).columns.tolist()

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", self._ordinalEncoder, self._categoricalCols),
                ("num", "passthrough", numericalCols),
            ]
        )

        X_trainval, X_test_final, y_trainval, y_test_final = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        num_classes = len(np.unique(y))

        paramGrid = {
            "classifier__n_estimators": [100, 200, 300, 400, 500],
            "classifier__max_depth": [3, 4, 5, 6, 8, 10],
            "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "classifier__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "classifier__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "classifier__min_child_weight": [1, 3, 5, 7],
            "classifier__reg_alpha": [0.0, 0.01, 0.1, 1.0],
            "classifier__reg_lambda": [0.5, 1.0, 2.0, 5.0],
        }

        xgb_base_params = {
            "tree_method": "hist",
            "n_jobs": -1,
            "verbosity": 0,
        }

        if num_classes == 2:
            xgb_base_params["objective"] = "binary:logistic"
            xgb_base_params["eval_metric"] = "logloss"
        else:
            xgb_base_params["objective"] = "multi:softprob"
            xgb_base_params["eval_metric"] = "mlogloss"
            xgb_base_params["num_class"] = num_classes

        try:
            pipeline = Pipeline(
                [
                    ("preprocessor", preprocessor),
                    ("classifier", XGBClassifier(**xgb_base_params)),
                ]
            )

            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=paramGrid,
                n_iter=30,
                scoring="accuracy",
                cv=3,
                n_jobs=-1,
                random_state=42,
                verbose=1,
                refit=True,
            )

            search.fit(X_trainval, y_trainval)
            bestParams = {
                k.replace("classifier__", ""): v
                for k, v in search.best_params_.items()
            }

            self.logger.info(f"Best GradientBoosted parameters: {bestParams}")
            self.params = bestParams

            self._pipeline = search.best_estimator_
            self._X_test = X_test_final
            self._y_test = y_test_final
            self._modelTrained = True

            finalAccuracy = accuracy_score(
                y_test_final, search.best_estimator_.predict(X_test_final).astype(int)
            )
            self.logger.info(
                f"Final accuracy on held-out test set: {round(finalAccuracy, 4)}"
            )

            return bestParams

        except Exception as e:
            self.logger.error(f"Hyperparameter optimization failed: {e}")
            return {}

    def getModelParameters(self) -> GradientBoostedParams:
        return self.params

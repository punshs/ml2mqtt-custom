import logging
import json
from typing import Any, Dict, List, Optional, Union

from ModelStore import ModelStore, ModelObservation, EntityKey
from classifiers.RandomForest import RandomForest, RandomForestParams
from classifiers.KNNClassifier import KNNClassifier, KNNParams
from classifiers.GradientBoosted import GradientBoosted, GradientBoostedParams
from MqttClient import MqttClient
from postprocessors.PostprocessorFactory import PostprocessorFactory
from postprocessors.base import BasePostprocessor
from preprocessors.base import BasePreprocessor
from preprocessors.PreprocessorFactory import PreprocessorFactory
from nodered.nodered_generator import NodeRedGenerator
DISABLED_LABEL = "Disabled"

import re

def shorten_sensor_name(entity_id: str) -> str:
    """Extract a human-readable room name from a Bermuda entity ID.
    e.g. 'sensor.bermuda_ec_0e_36_57_a4_69_unfiltered_distance_to_office_sensor' -> 'Office'
    """
    # Try to match 'distance_to_<room>_sensor' pattern
    match = re.search(r'distance_to_(.+?)(?:_sensor(?:_\d+)?)?$', entity_id)
    if match:
        raw = match.group(1)
        # Convert underscores to spaces, title case
        parts = raw.replace('_', ' ').strip().title()
        # Append sensor number suffix if present
        suffix_match = re.search(r'sensor_(\d+)$', entity_id)
        if suffix_match:
            parts += f' {suffix_match.group(1)}'
        return parts
    # Fallback: last meaningful segment
    parts = entity_id.split('.')[-1].replace('_', ' ').title()
    return parts


class ModelService:
    def __init__(self, mqttClient: MqttClient, modelstore: ModelStore):
        self._mqttClient = mqttClient
        self._modelstore: ModelStore = modelstore
        self._model = None
        self._logger = logging.getLogger(__name__)
        self._postProcessorFactory = PostprocessorFactory()
        self._postprocessors: List[BasePostprocessor] = []

        self._preprocessorFactory = PreprocessorFactory()
        self._preprocessors: List[BasePreprocessor] = []
        
        self._modelType: str
        self._allParams: Dict[str, Dict[str, Any]] = {}
        self._recentMqtt = []
        self._collectingLabel: Optional[str] = None
        self._lastPrediction: Optional[str] = None
        self._lastConfidence: float = 0.0
        self._lastSensorReadings: Dict[str, Any] = {}
        self._sensorDisplayNames: Dict[str, str] = {}  # custom overrides
        self._predictionBuffer: List[str] = []  # for temporal smoothing
        self._smoothedPrediction: Optional[str] = None
        self._smoothingWindow: int = 5
        self._newObsCount: int = 0  # auto-retrain counter
        self._autoRetrainThreshold: int = 20
        self._populateModel()
        self._loadPostprocessors()
        self._loadPreprocessors()
        self._loadSensorDisplayNames()

    def dispose(self) -> None:
        topic = self.getMqttTopic()
        self._mqttClient.unsubscribe(f"{topic}/set", self.predictLabel)
        self._modelstore.close()

    def subscribeToMqttTopics(self) -> None:
        topic = self.getMqttTopic()
        self._logger.info("Subscribing to MQTT topic: %s/set", topic)
        self._mqttClient.subscribe(f"{topic}/set", self.predictLabel)

    def _populateModel(self) -> None:
        settings = self._modelstore.getDict('model_settings') or {}
        self._modelType = settings.get("model_type", "RandomForest")
        self._allParams = settings.get("model_parameters", {})

        paramsForThisModel = self._allParams.get(self._modelType, {})

        self._logger.info(f"Loading with settings {settings}")

        if self._modelType == "KNN":
            self._model = KNNClassifier(params=paramsForThisModel)
        elif self._modelType == "GradientBoosted":
            self._model = GradientBoosted(params=paramsForThisModel)
        else:
            self._model = RandomForest(params=paramsForThisModel)

        observations = self._modelstore.getObservations()
        self._model.populateDataframe(observations)

    def _loadPostprocessors(self) -> None:
        """Load postprocessors from model settings."""
        postProcessors = self._modelstore.getPostprocessors()
        self._postprocessors = []
        
        for postprocessor_data in postProcessors:
            try:
                postprocessor = self._postProcessorFactory.create(postprocessor_data.type, postprocessor_data.id, postprocessor_data.params)
                self._postprocessors.append(postprocessor)
            except ValueError as e:
                self._logger.warning(f"Failed to load postprocessor: {e}")


    def _loadPreprocessors(self) -> None:
        """Load postprocessors from model settings."""
        preprocessors = self._modelstore.getPreprocessors()
        self._preprocessors = []
        
        for preprocessor_data in preprocessors:
            try:
                postprocessor = self._preprocessorFactory.create(preprocessor_data.type, preprocessor_data.id, preprocessor_data.params)
                self._preprocessors.append(postprocessor)
            except ValueError as e:
                self._logger.warning(f"Failed to load preprocessor: {e}")


    def getEntityKeys(self) -> List[EntityKey]:
        features = self._model.getFeatureImportance() or {}
        entities = self._modelstore.getEntityKeys()
        for entity in entities:
            entity.significance = features.get(entity.name, 0.0)
        return entities

    def getAccuracy(self) -> Optional[float]:
        return self._model.getAccuracy()

    def predictLabel(self, msg: Any) -> None:
        self._recentMqtt.append(msg)
        if len(self._recentMqtt) > 10:
            self._recentMqtt.pop(0)

        messageStr: str
        if hasattr(msg, "payload"):
            try:
                messageStr = msg.payload.decode()
            except Exception as e:
                self._logger.warning("Could not decode MQTT payload: %s", e)
                return
        else:
            messageStr = str(msg)

        try:
            parsed = json.loads(messageStr)
        except json.JSONDecodeError:
            self._logger.warning("Invalid JSON: %s", messageStr)
            return

        label: str = DISABLED_LABEL
        entityMap: Dict[str, Any] = {}

        if isinstance(parsed, list):
            # Original ml2mqtt format: [{"entity_id": "...", "state": "..."}, ...]
            for entity in parsed:
                if "label" in entity:
                    label = entity["label"]
                elif "entity_id" in entity and "state" in entity:
                    entityMap[entity["entity_id"]] = entity["state"]
        elif isinstance(parsed, dict):
            # Flat dict format from HA automation: {"sensor.xxx": "3.54", ...}
            for key, value in parsed.items():
                if key == "label":
                    label = value
                else:
                    entityMap[key] = value
        else:
            self._logger.warning("Unexpected JSON structure: %s", type(parsed))
            return

        previousEntityMap = self._modelstore.getDict("mqtt_observations")
        if "history" in previousEntityMap:
            previousEntityMap['history'].append(entityMap)
            if len(previousEntityMap['history']) > 10:
                previousEntityMap['history'].pop(0)
        else:
            previousEntityMap['history'] = [entityMap]    
        self._modelstore.saveDict("mqtt_observations", previousEntityMap)

        # Apply Preprocessors
        processor_storage = self._modelstore.getDict("processor_storage")
        for preprocessor in self._preprocessors:
            if not preprocessor.dbId in processor_storage:
                processor_storage[preprocessor.dbId] = {}
            entityMap = preprocessor.process(entityMap, processor_storage[preprocessor.dbId])
            if not entityMap:
                self._logger.debug("No entity values to process.")
                return
        self._modelstore.saveDict("processor_storage", processor_storage)

        if not entityMap:
            self._logger.debug("No entity values to process.")
            return        

        entityValues = {k: v for k, v in entityMap.items() if v is not None}

        # Store latest sensor readings for the live API
        self._lastSensorReadings = dict(entityMap)

        # Determine the active label: from MQTT message OR from collection mode
        activeLabel = label if label != DISABLED_LABEL else None
        if not activeLabel and self._collectingLabel:
            activeLabel = self._collectingLabel

        if activeLabel:
            learningType = self.getLearningType()
            shouldSave = False
            if learningType == "LAZY":
                prediction, confidence = self._model.predictLabel(entityValues)
                if prediction != activeLabel or confidence < 0.8:
                    shouldSave = True
            elif learningType == "EAGER":
                shouldSave = True

            if shouldSave:
                sortedVals = self._modelstore.sortEntityValues(entityMap, True)
                self._logger.info("Adding training observation for label: %s", activeLabel)
                self._modelstore.addObservation(activeLabel, sortedVals)
                self._newObsCount += 1
                # Auto-retrain every N observations
                if self._newObsCount >= self._autoRetrainThreshold:
                    self._logger.info("Auto-retraining after %d new observations", self._newObsCount)
                    self._newObsCount = 0
                    self._populateModel()

        prediction, confidence = self._model.predictLabel(entityValues)
        confidence = round(float(confidence), 4)

        # Store raw prediction for the live API
        self._lastPrediction = prediction
        self._lastConfidence = confidence

        # Temporal majority vote smoothing for MQTT output
        smoothedPrediction = prediction
        if prediction:
            self._predictionBuffer.append(prediction)
            if len(self._predictionBuffer) > self._smoothingWindow:
                self._predictionBuffer.pop(0)
            # Pick the most frequent prediction in the buffer
            from collections import Counter
            counts = Counter(self._predictionBuffer)
            smoothedPrediction = counts.most_common(1)[0][0]
            self._smoothedPrediction = smoothedPrediction

        # Apply postprocessors
        observation = entityValues
        for postprocessor in self._postprocessors:
            observation, smoothedPrediction = postprocessor.process(observation, smoothedPrediction, confidence)
            if smoothedPrediction is None:
                return

        topic = self.getMqttTopic()
        self._mqttClient.publish(f"{topic}/state", json.dumps({"state": smoothedPrediction, "confidence": confidence}))
        self._logger.info(f"Predicted: {prediction} → Smoothed: {smoothedPrediction} (conf: {confidence})")

    def getMqttTopic(self) -> str:
        return self._modelstore.getMqttTopic() or ""

    def setMqttTopic(self, mqttTopic: str) -> None:
        self._modelstore.setMqttTopic(mqttTopic)

    def getName(self) -> str:
        return self._modelstore.getName() or ""

    def setName(self, modelName: str) -> None:
        self._modelstore.setName(modelName)

    def getObservations(self) -> List[ModelObservation]:
        return self._modelstore.getObservations()

    def getModelSize(self) -> int:
        return self._modelstore.getModelSize()

    def getLabels(self) -> List[str]:
        db_labels = self._modelstore.getLabels()
        config_labels = self.getModelConfig("labels", [])
        # Deduplicate case-insensitively, preserving first-seen casing
        seen: Dict[str, str] = {}
        for label in db_labels + config_labels:
            key = label.lower()
            if key not in seen:
                seen[key] = label
        return sorted(seen.values(), key=str.lower)

    def deleteEntity(self, entityName: str) -> None:
        self._modelstore.deleteEntity(entityName)
        # Rebuild the model after entity deletion
        self._populateModel()

    def getLabelStats(self) -> Optional[Dict[str, Any]]:
        labelStats = self._model.getLabelStats() or {}
        for extraLabel in self.getLabels():
            if not extraLabel in labelStats.keys():   
                labelStats[extraLabel] = {
                    "support":0,
                    "precision": 0,
                    "recall": 0,
                    "f1": 0,
                }        
        return labelStats

    def addLabel(self, label: str) -> None:
        """Add a new label to the model without requiring training data."""
        presavedLabels = self.getModelConfig("labels", [])
        if label not in presavedLabels:
            presavedLabels.append(label)
            self.setModelConfig("labels", sorted(presavedLabels))
            self._logger.info("Added label: %s", label)

    def generateSyntheticObservations(self, label: str, count: int = 50) -> int:
        """Generate synthetic observations where all sensors report the null replacement value.
        
        This is useful for creating training data for labels like 'Away' where all
        sensors would report out-of-range values (the Whoop is not visible to any proxy).
        """
        import random
        import time as time_module

        entities = self._modelstore.getEntityKeys()
        entity_names = [e.name for e in entities]

        # If no recorded observations yet, bootstrap entity names from recent MQTT payloads
        if not entity_names:
            recent = self.getMostRecentMqttObservations()
            if recent:
                entity_names = sorted(recent[-1].keys())
            else:
                raise ValueError(
                    "No sensor entities registered and no MQTT data received yet. "
                    "Ensure the MQTT automation is running and data is flowing."
                )

        # Find the null replacement value from NullHandler preprocessor config
        null_value = 9999.0  # default
        for preprocessor in self._preprocessors:
            if hasattr(preprocessor, 'config') and 'nullReplacement' in preprocessor.config:
                try:
                    null_value = float(preprocessor.config['nullReplacement'])
                except (ValueError, TypeError):
                    pass
                break

        # Ensure label exists in config
        self.addLabel(label)

        # Generate observations with jittered sensor values
        base_time = time_module.time()
        for i in range(count):
            sensor_values = {
                name: null_value + random.uniform(-0.5, 0.5)
                for name in entity_names
            }
            sorted_vals = self._modelstore.sortEntityValues(sensor_values, True)
            self._modelstore.addObservation(label, sorted_vals, base_time - (i * 10))

        self._logger.info("Generated %d synthetic observations for label: %s", count, label)
        self._populateModel()
        return count

    def deleteObservationsByLabel(self, label: str) -> None:
        """Delete all observations with the given label."""
        self._modelstore.deleteObservationsByLabel(label)

        presavedLabels = self.getModelConfig("labels", [])
        if label in presavedLabels:
            presavedLabels.remove(label)
        self.setModelConfig("labels", presavedLabels)

        # Rebuild the model after deletion
        self._populateModel()

    def deleteObservation(self, time: int) -> None:
        """Delete an observation by its timestamp."""
        self._modelstore.deleteObservation(time)
        # Rebuild the model after deletion
        self._populateModel()

    def deleteObservationsSince(self, timestamp: int) -> None:
            """Delete an observation by its timestamp."""
            self._modelstore.deleteObservationsSince(timestamp)
            # Rebuild the model after deletion
            self._populateModel()

    def optimizeParameters(self) -> None:
        best_params = self._model.optimizeParameters(self._modelstore.getObservations())

        modelSettings = self.getModelSettings()
        modelSettings["model_parameters"] = modelSettings.get("model_parameters", {})
        modelSettings["model_parameters"][self._modelType] = best_params
        self._modelstore.saveDict("model_settings", modelSettings)

    def getModelSettings(self) -> Dict[str, Any]:
        settings = self._modelstore.getDict('model_settings') or {}
        if not settings:
            settings = {
                "model_type": "RandomForest",
                "model_parameters": {
                    "RandomForest": {
                        "n_estimators": 100,
                        "max_depth": None,
                        "min_samples_split": 2,
                        "min_samples_leaf": 1,
                        "max_features": "sqrt",
                        "class_weight": None,
                        "bootstrap": True,
                        "oob_score": False
                    }, "KNN": {
                        "n_neighbors": 5,
                        "weights": "uniform",
                        "algorithm": "auto",
                        "leaf_size": 30,
                        "metric": "minkowski",
                        "p": 2
                    }
                }
            }
        return settings

    def setModelSettings(self, settings: Dict[str, Any]) -> None:
        self._logger.info(f"Setting model settings: {settings}");
        self._modelType = settings.get("model_type", "RandomForest")
        self._allParams = settings.get("model_parameters", {})
        self._modelstore.saveDict("model_settings", settings)
        self._populateModel()

    def getPostprocessors(self) -> List[BasePostprocessor]:
        """Get list of postprocessors."""
        return self._postprocessors

    def getPreprocessors(self) -> List[BasePreprocessor]:
        """Get list of preprocessors."""
        return self._preprocessors

    def addPostprocessor(self, type: str, params: Dict[str, Any]) -> None:
        """Add a new postprocessor."""
        try:
            # First add to database to get the ID
            dbId = self._modelstore.addPostprocessor(type, params)
            # Then create the postprocessor instance
            postprocessor = self._postProcessorFactory.create(type, dbId, params)
            self._postprocessors.append(postprocessor)
        except Exception as e:
            # If postprocessor creation fails, delete from database
            if 'dbId' in locals():
                self._modelstore.deletePostprocessor(dbId)
            raise e

    def addPreprocessor(self, type: str, params: Dict[str, Any]) -> None:
        """Add a new preprocessor."""
        try:
            # First add to database to get the ID
            dbId = self._modelstore.addPreprocessor(type, params)
            # Then create the postprocessor instance
            preprocessor = self._preprocessorFactory.create(type, dbId, params)
            self._preprocessors.append(preprocessor)
            self.deleteObservationsSince(0)
        except Exception as e:
            # If postprocessor creation fails, delete from database.
            if 'dbId' in locals():
                self._modelstore.deletePreprocessor(dbId)
            raise e

    def removePostprocessor(self, index: int) -> None:
        """Remove a postprocessor by index."""
        if 0 <= index < len(self._postprocessors):
            deletedProcessor = self._postprocessors.pop(index)
            
            self._modelstore.deletePostprocessor(deletedProcessor.dbId)

    def removePreprocessor(self, index: int) -> None:
        """Remove a preprocessor by index."""
        if 0 <= index < len(self._preprocessors):
            deletedProcessor = self._preprocessors.pop(index)
            
            self._modelstore.deletePreprocessor(deletedProcessor.dbId)
            self.deleteObservationsSince(0)

    def reorderPreprocessors(self, from_index: int, to_index: int) -> None:
        """Reorder preprocessors."""
        if 0 <= from_index < len(self._preprocessors) and 0 <= to_index < len(self._preprocessors):
            self._logger.info("Previous preprocessors: %s", list(map(lambda p: p, self._preprocessors)))
            preprocessor = self._preprocessors.pop(from_index)
            self._preprocessors.insert(to_index, preprocessor)
            self._logger.info("Reordering preprocessors: %s", list(map(lambda p: p, self._preprocessors)))
            self._modelstore.reorderPreprocessors(map(lambda p: p.dbId, self._preprocessors))
            self.deleteObservationsSince(0)

    def reorderPostprocessors(self, from_index: int, to_index: int) -> None:
        """Reorder postprocessors."""
        if 0 <= from_index < len(self._postprocessors) and 0 <= to_index < len(self._postprocessors):
            self._logger.info("Previous postprocessors: %s", list(map(lambda p: p, self._postprocessors)))
            postprocessor = self._postprocessors.pop(from_index)
            self._postprocessors.insert(to_index, postprocessor)
            self._logger.info("Reordering postprocessors: %s", list(map(lambda p: p, self._postprocessors)))
            self._modelstore.reorderPostprocessors(map(lambda p: p.dbId, self._postprocessors))

    def getLearningType(self):
        settings = self.getModelSettings() or {}
        learningType = settings.get("learning_type", "DISABLED")
        self._logger.info(f"Getting learning type: {learningType}")
        return learningType
    
    def setLearningType(self, learningType: str) -> None:
        settings = self.getModelSettings() or {}
        settings["learning_type"] = learningType
        self._logger.info(f"Setting learning type: {learningType}")
        self._modelstore.saveDict("model_settings", settings)

    def getMostRecentMqttObservations(self):
        previousObservations = self._modelstore.getDict("mqtt_observations")
        if 'history' in previousObservations:
            return previousObservations['history']
        else:
            return []
    
    def setModelConfig(self, key, value):
        current = self._modelstore.getDict("config")
        current[key] = value
        self._modelstore.saveDict("config", current)
    
    def getModelConfig(self, key, default):
        config = self._modelstore.getDict("config")
        if key in config:
            return config[key]
        else:
            return default

    def generateNodeRed(self) -> str:
        nodeRedGenerator = NodeRedGenerator(self)
        return nodeRedGenerator.generate()
    
    def getRecentMqtt(self) -> str:
        return self._recentMqtt

    # ── Collection Mode ──────────────────────────────────────────────

    def startCollecting(self, label: str) -> None:
        """Start collecting observations with the given label."""
        self._collectingLabel = label
        # Ensure learning type is at least EAGER when starting collection
        currentType = self.getLearningType()
        if currentType == "DISABLED":
            self.setLearningType("EAGER")
        self._logger.info(f"Started collecting for label: {label} (mode: {self.getLearningType()})")

    def stopCollecting(self) -> None:
        """Stop collecting observations."""
        self._collectingLabel = None
        self._logger.info("Stopped collecting")

    def isCollecting(self) -> bool:
        return self._collectingLabel is not None

    def getCollectingLabel(self) -> Optional[str]:
        return self._collectingLabel

    # ── Live Data API ────────────────────────────────────────────────

    def getLiveData(self) -> Dict[str, Any]:
        """Return current prediction, confidence, sensor readings, and stats for the training UI."""
        # Build sensor readings with shortened display names
        sensors = []
        for entity_id, value in self._lastSensorReadings.items():
            display_name = self._sensorDisplayNames.get(
                entity_id,
                shorten_sensor_name(entity_id)
            )
            # Determine status for color coding
            status = "normal"
            if value == "unknown" or value is None:
                status = "unknown"
            elif value == "unavailable":
                status = "unavailable"
            else:
                try:
                    fval = float(value)
                    if fval >= 9998:
                        status = "null"
                except (ValueError, TypeError):
                    status = "unknown"

            sensors.append({
                "entity_id": entity_id,
                "display_name": display_name,
                "value": value,
                "status": status,
            })

        # Get observation stats per label
        label_stats = self.getLabelStats() or {}
        obs_count = len(self._modelstore.getObservations())

        return {
            "prediction": self._lastPrediction,
            "smoothed_prediction": self._smoothedPrediction,
            "confidence": self._lastConfidence,
            "sensors": sensors,
            "collecting": self._collectingLabel is not None,
            "collecting_label": self._collectingLabel,
            "learning_type": self.getLearningType(),
            "labels": self.getLabels(),
            "accuracy": float(self.getAccuracy()) if self.getAccuracy() is not None else None,
            "observation_count": obs_count,
            "label_stats": {k: v.get("support", 0) for k, v in label_stats.items()},
            "feature_importance": {k: float(v) for k, v in (self._model.getFeatureImportance() or {}).items()} if hasattr(self._model, 'getFeatureImportance') else None,
        }

    def setSensorDisplayName(self, entity_id: str, display_name: str) -> None:
        """Set a custom display name for a sensor entity."""
        self._sensorDisplayNames[entity_id] = display_name
        # Persist to model config
        names = self.getModelConfig("sensor_display_names", {})
        names[entity_id] = display_name
        self.setModelConfig("sensor_display_names", names)

    def _loadSensorDisplayNames(self) -> None:
        """Load custom sensor display names from config."""
        self._sensorDisplayNames = self.getModelConfig("sensor_display_names", {})

    # ── Data Management ──────────────────────────────────────────────

    def clearLabelData(self, label: str) -> Dict[str, Any]:
        """Delete all observations for a specific label and retrain."""
        self._modelstore.deleteObservationsByLabel(label)
        self._populateModel()
        obs_count = len(self._modelstore.getObservations())
        return {"deleted_label": label, "remaining_observations": obs_count}

    def retrain(self) -> Dict[str, Any]:
        """Force re-populate the model from current observations."""
        self._populateModel()
        self._newObsCount = 0
        accuracy = self.getAccuracy()
        obs_count = len(self._modelstore.getObservations())
        return {"accuracy": accuracy, "observation_count": obs_count}

    def getDataHealth(self) -> Dict[str, Any]:
        """Return data quality metrics for the training UI."""
        observations = self._modelstore.getObservations()
        label_counts: Dict[str, int] = {}
        for obs in observations:
            label_counts[obs.label] = label_counts.get(obs.label, 0) + 1

        total = len(observations)
        max_count = max(label_counts.values()) if label_counts else 0
        min_count = min(label_counts.values()) if label_counts else 0

        # Generate warnings
        warnings = []
        if total < 30:
            warnings.append({"type": "low_data", "msg": f"Only {total} total observations. Aim for 50+ per room."})
        for label, count in label_counts.items():
            if count < 20:
                warnings.append({"type": "low_label", "msg": f"{label}: only {count} observations (need 20+)"})
            if max_count > 0 and count < max_count * 0.4:
                warnings.append({"type": "imbalance", "msg": f"{label} is underrepresented ({count} vs {max_count})"})

        return {
            "total_observations": total,
            "label_counts": label_counts,
            "balance_ratio": round(min_count / max_count, 2) if max_count > 0 else 0,
            "warnings": warnings,
            "accuracy": self.getAccuracy(),
        }

    def getConfusionMatrix(self) -> Optional[Dict[str, Any]]:
        """Return confusion matrix data from the test set."""
        if not hasattr(self._model, '_X_test') or self._model._X_test is None:
            return None
        if not hasattr(self._model, '_modelTrained') or not self._model._modelTrained:
            return None

        try:
            import numpy as np
            y_pred = self._model._pipeline.predict(self._model._X_test)
            if hasattr(y_pred[0], 'item'):
                y_pred = y_pred.astype(int)
            y_true = self._model._y_test
            labels = list(self._model.labelEncoder.classes_)
            n = len(labels)

            # Build confusion matrix manually
            matrix = [[0] * n for _ in range(n)]
            for true_idx, pred_idx in zip(y_true, y_pred):
                matrix[int(true_idx)][int(pred_idx)] += 1

            return {
                "labels": labels,
                "matrix": matrix,
            }
        except Exception as e:
            self._logger.error(f"Confusion matrix failed: {e}")
            return None

    def deleteSensor(self, entity_name: str) -> Dict[str, Any]:
        """Remove a sensor from the model and all observations, then retrain."""
        try:
            self._modelstore.deleteEntity(entity_name)
            self._populateModel()
            return {"deleted": entity_name, "success": True}
        except ValueError as e:
            return {"error": str(e), "success": False}

    def getSensorEntities(self) -> List[Dict[str, Any]]:
        """Return list of sensor entities with importance."""
        entities = self._modelstore.getEntityKeys()
        importance = {}
        if hasattr(self._model, 'getFeatureImportance'):
            importance = self._model.getFeatureImportance() or {}

        return [
            {
                "entity_id": ek.name,
                "display_name": self._sensorDisplayNames.get(ek.name, shorten_sensor_name(ek.name)),
                "type": ek.display_type,
                "importance": round(float(importance.get(ek.name, 0)), 4),
            }
            for ek in entities
        ]
from flask import Blueprint, render_template, request, url_for, redirect, abort, Response, jsonify
from jinja2 import TemplateNotFound
from typing import Dict, Any, List, Optional
import json
import math
import logging
from ModelStore import ModelObservation, EntityKey
from classifiers.RandomForest import RandomForestParams
from classifiers.KNNClassifier import KNNParams
from classifiers.GradientBoosted import GradientBoostedParams
from utils.helpers import slugify
from postprocessors.PostprocessorFactory import PostprocessorFactory
from preprocessors.PreprocessorFactory import PreprocessorFactory
from ModelManager import ModelManager
from PreprocessorEvaluator import PreprocessorEvaluator
from datetime import timedelta, datetime

logger = logging.getLogger("ml2mqtt.routes.model")
model_bp = Blueprint('model', __name__)

def init_model_routes(model_manager: ModelManager):
    @model_bp.route("/")
    def home() -> str:
        models = []
        modelMap = model_manager.getModels()
        for model in modelMap:
            modelName = modelMap[model].getName()
            models.append({
                "name": modelName,
                "mqtt_topic": modelMap[model].getMqttTopic()
            })
        return render_template("home.html", title="Home", active_page="home", models=models, mqtt_connected=model_manager._mqttClient._connected)

    @model_bp.route("/check-model-name")
    def checkModel() -> str:
        model_name = request.args.get("name", "").strip().lower()
        slug = slugify(model_name)
        is_taken = model_manager.modelExists(slug)
        return json.dumps({"exists": is_taken})

    @model_bp.route("/api/models", methods=["GET"])
    def apiListModels() -> Response:
        """JSON API endpoint listing all models."""
        modelMap = model_manager.getModels()
        models = []
        for key in modelMap:
            m = modelMap[key]
            models.append({
                "name": m.getName(),
                "mqtt_topic": m.getMqttTopic(),
            })
        return jsonify({"success": True, "models": models})

    @model_bp.route("/create-model", methods=["GET", "POST"])
    def createModel() -> Response:
        if request.method == "POST":
            modelName = request.form.get("model_name")
            defaultValue = request.form.get("default_value")
            mqttTopic = request.form.get("mqtt_topic")
            labels = []
            try:
                labels = json.loads(request.form.get("labels", "[]"))
                if not isinstance(labels, list):
                    raise ValueError
                labels = sorted(set(labels))
            except (json.JSONDecodeError, ValueError):
                pass

            logger.error(f"Request {request.form}")

            if modelName is None:
                abort(400, "Missing model name")
            if mqttTopic is None:
                abort(400, "Missing MQTT topic")
            newModel = model_manager.addModel(modelName)
            newModel.setMqttTopic(mqttTopic)
            newModel.setName(modelName)
            newModel.setModelConfig("labels", sorted(list(set(labels))))
            newModel.setModelConfig("input_count", int(request.form.get("input_count")))
            newModel.addPreprocessor("type_caster", { 'sensor': [{"SELECT_ALL": True }]})
            newModel.addPreprocessor("null_handler", { 'sensor': [{"SELECT_ALL": True }], 'replacementType': 'float', 'nullReplacement': defaultValue})
            newModel.addPostprocessor("only_diff", {})
            newModel.setLearningType("EAGER")
            newModel.subscribeToMqttTopics()

            return redirect(url_for("model.home"))

        return render_template("create-model.html", title="Add Model", active_page="create_model")

    @model_bp.route("/api/create-model", methods=["POST"])
    def apiCreateModel() -> Response:
        """JSON API endpoint for creating a model (used by Lovelace card)."""
        data = request.get_json(force=True) if request.is_json else {}
        modelName = data.get("model_name")
        defaultValue = data.get("default_value", "9999")
        mqttTopic = data.get("mqtt_topic")
        labels = data.get("labels", [])
        input_count = data.get("input_count", 17)

        if not modelName:
            return jsonify({"success": False, "error": "Missing model_name"}), 400
        if not mqttTopic:
            return jsonify({"success": False, "error": "Missing mqtt_topic"}), 400

        if model_manager.modelExists(slugify(modelName)):
            return jsonify({"success": False, "error": "Model already exists"}), 409

        if isinstance(labels, str):
            try:
                labels = json.loads(labels)
            except json.JSONDecodeError:
                labels = []
        labels = sorted(set(labels))

        newModel = model_manager.addModel(modelName)
        newModel.setMqttTopic(mqttTopic)
        newModel.setName(modelName)
        newModel.setModelConfig("labels", labels)
        newModel.setModelConfig("input_count", int(input_count))
        newModel.addPreprocessor("type_caster", {'sensor': [{"SELECT_ALL": True}]})
        newModel.addPreprocessor("null_handler", {'sensor': [{"SELECT_ALL": True}], 'replacementType': 'float', 'nullReplacement': defaultValue})
        newModel.addPostprocessor("only_diff", {})
        newModel.setLearningType("EAGER")
        newModel.subscribeToMqttTopics()

        return jsonify({"success": True, "model_name": modelName})

    @model_bp.route("/delete-model/<string:modelName>/", methods=["POST"])
    def deleteModel(modelName: str) -> Response:
        model_manager.removeModel(modelName)
        return redirect(url_for("model.home"))

    @model_bp.route("/edit-model/<string:modelName>/<section>")
    def editModel(modelName: str, section: str = "settings") -> str:
        validSections = ["settings", "entities", "observations", "preprocessors", "postprocessors", "mqtt", "nodered"]

        if section not in validSections:
            abort(404)

        class ViewModel:
            def __init__(self):
                self.name: str = modelName
                self.params: Dict[str, Any] = {}
                self.observations: List[ModelObservation] = []
                self.entities: List[EntityKey] = []
                self.labels: List[str] = []
                self.currentPage: int = 0
                self.totalPages: int = 0

        model = ViewModel()

        if section == "observations":
            page = int(request.args.get("page", 1))
            pageSize = 50
            allObservations = model_manager.getModel(modelName).getObservations()
            total = len(allObservations)

            start = (page - 1) * pageSize
            end = start + pageSize
            paginated = allObservations[start:end]

            model.observations = paginated
            model.currentPage = page
            model.labels = model_manager.getModel(modelName).getLabels()
            model.totalPages = math.ceil(total / pageSize)

        elif section == "settings":
            logger.info(f"Model settings: {model_manager.getModel(modelName).getModelSettings()}")
            model.params = { 
                "accuracy": model_manager.getModel(modelName).getAccuracy(),
                "observationCount": len(model_manager.getModel(modelName).getObservations()),
                "modelSize": model_manager.getModel(modelName).getModelSize(),
                "modelParameters": model_manager.getModel(modelName).getModelSettings(),
                "labelStats": model_manager.getModel(modelName).getLabelStats(),
                "learningType": model_manager.getModel(modelName).getLearningType(),
            }
        elif section == "postprocessors":
            logger.info(f"{list(map(lambda processor: processor.to_dict(),model_manager.getModel(modelName).getPostprocessors()))}")
            model.postprocessors = map(lambda processor: processor.to_dict(),model_manager.getModel(modelName).getPostprocessors())
        elif section == "preprocessors":
            logger.info(f"{list(map(lambda processor: processor.to_dict(),model_manager.getModel(modelName).getPreprocessors()))}")
            recentObservations = model_manager.getModel(modelName).getMostRecentMqttObservations()
            model.recentMqtt = None if len(recentObservations) == 0 else recentObservations[-1]
            evaluator = PreprocessorEvaluator(model_manager.getModel(modelName).getPreprocessors())
            model.preprocessors = evaluator.evaluate(recentObservations)
            if len(model.preprocessors) == 0:
                model.lastSensors = model.recentMqtt
            else:
                model.lastSensors = model.preprocessors[-1]['produces']

        elif section == "entities":
            model.entities = model_manager.getModel(modelName).getEntityKeys()
        elif section == "mqtt":
            model.params = {
                "mqttTopic": model_manager.getModel(modelName).getMqttTopic(),
            }
        elif section == "nodered":
            model.params = {
                "noderedConfig": model_manager.getModel(modelName).generateNodeRed()
            }

        sectionTemplate = f"edit_model/{section}.html"
        return render_template(
            "edit_model.html",
            title=f"Edit Model: {model.name}",
            activePage=None,
            activeSection=section,
            model=model,
            sectionTemplate=sectionTemplate,
            availablePostprocessors=PostprocessorFactory().get_available_postprocessors(),
            availablePreprocessors=PreprocessorFactory().get_available_preprocessors(),
        )

    @model_bp.route("/edit-model/<string:modelName>/change-model", methods=["POST"])
    def changeModel(modelName: str) -> str:
        try:
            modelType = request.form.get("modelType", "RandomForest")
            currentSettings = model_manager.getModel(modelName).getModelSettings()
            currentSettings["model_type"] = modelType
            model_manager.getModel(modelName).setModelSettings(currentSettings)
            return jsonify(success=True)
        except Exception as e:
            return jsonify(success=False, error=str(e)), 400
    
    @model_bp.route("/model/<modelName>/changeLearning", methods=["POST"])
    def changeLearning(modelName):
        learningType = request.form.get("learningType") # 'DISABLED', 'LAZY', 'EAGER'
        logger.info(f"Changing learning type for model '{modelName}' to {learningType}")
        model_manager.getModel(modelName).setLearningType(learningType)

        return jsonify(success=True)

    @model_bp.route("/edit-model/<string:modelName>/settings/update", methods=["POST"])
    def updateModelSettings(modelName: str) -> str:
        def get_int(name: str, default: Optional[int] = None) -> Optional[int]:
            val = request.form.get(name)
            return int(val) if val and val.isdigit() else default

        def get_optional_int(name: str) -> Optional[int]:
            val = request.form.get(name)
            return int(val) if val and val.isdigit() else None

        def get_bool(name: str) -> bool:
            return request.form.get(name) in ["true", "on", "1"]

        def get_str_or_none(name: str) -> Optional[str]:
            val = request.form.get(name)
            return val if val not in ["None", "", None] else None

        try:
            modelType = request.form.get("modelType", "RandomForest")

            settings: Dict[str, Any] = model_manager.getModel(modelName).getModelSettings()

            if modelType == "RandomForest":
                rfParams: RandomForestParams = {
                    "n_estimators": get_int("nEstimators", 100),
                    "max_depth": get_optional_int("maxDepth"),
                    "min_samples_split": get_int("minSamplesSplit", 2),
                    "min_samples_leaf": get_int("minSamplesLeaf", 1),
                    "max_features": get_str_or_none("maxFeatures"),
                    "class_weight": get_str_or_none("classWeight"),
                    "bootstrap": get_bool("bootstrap"),
                    "oob_score": get_bool("oobScore"),
                }
                settings["model_parameters"]["RandomForest"] = rfParams

            elif modelType == "KNN":
                knnParams: KNNParams = {
                    "n_neighbors": get_int("nNeighbors", 5),
                    "weights": request.form.get("weights", "uniform"),
                    "metric": request.form.get("metric", "minkowski"),
                }
                settings["model_parameters"]["KNN"] = knnParams

            elif modelType == "GradientBoosted":
                gbParams: GradientBoostedParams = {
                    "n_estimators": get_int("nEstimators", 200),
                    "max_depth": get_int("maxDepth", 6),
                    "learning_rate": float(request.form.get("learningRate", 0.1)),
                    "subsample": float(request.form.get("subsample", 0.8)),
                    "colsample_bytree": float(request.form.get("colsampleBytree", 0.8)),
                    "min_child_weight": get_int("minChildWeight", 1),
                    "reg_alpha": float(request.form.get("regAlpha", 0.0)),
                    "reg_lambda": float(request.form.get("regLambda", 1.0)),
                }
                settings["model_parameters"]["GradientBoosted"] = gbParams

            else:
                return jsonify(success=False, error=f"Unknown model type '{modelType}'"), 400

            model_manager.getModel(modelName).setModelSettings(settings)
            return jsonify(success=True)

        except Exception as e:
            return jsonify(success=False, error=str(e)), 400

    @model_bp.route("/edit-model/<string:modelName>/settings/autotune", methods=["POST"])
    def autoTuneModel(modelName: str) -> str:
        model_manager.getModel(modelName).optimizeParameters()
        return json.dumps({"success": True})

    @model_bp.route("/api/model/<string:modelName>/observation/<float:observationTime>/delete", methods=["POST"])
    def apiDeleteObservation(modelName: str, observationTime: float) -> str:
        try:
            model_manager.getModel(modelName).deleteObservation(observationTime)
            return jsonify({"success": True})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        
    @model_bp.route("/api/model/<string:modelName>/observations/delete", methods=["POST"])
    def apiDeleteObservations(modelName: str) -> str:
        try:
            data = request.get_json()
            scope = data.get("scope")

            if not scope:
                return jsonify({"error": "Scope parameter is required"}), 400

            now = datetime.utcnow()

            if scope == "all":
                timestamp = 0  # Effectively deletes all observations
            elif scope == "hour":
                timestamp = (now - timedelta(hours=1)).timestamp()
            elif scope == "day":
                timestamp = (now - timedelta(days=1)).timestamp()
            elif scope == "week":
                timestamp = (now - timedelta(weeks=1)).timestamp()
            else:
                return jsonify({"error": "Invalid scope parameter"}), 400

            model_manager.getModel(modelName).deleteObservationsSince(timestamp)
            return jsonify({"success": True})
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    @model_bp.route("/edit-model/<string:modelName>/model-settings/<string:modelType>")
    def getModelSettingsTemplate(modelName: str, modelType: str) -> str:
        if modelType not in ["RandomForest", "KNN", "GradientBoosted"]:
            abort(404)
        model = ViewModel(modelName)
        settings = model_manager.getModel(modelName).getModelSettings()
        
        # Initialize default parameters if they don't exist
        if "model_parameters" not in settings:
            settings["model_parameters"] = {}
        
        if modelType == "KNN":
            if "KNN" not in settings["model_parameters"]:
                settings["model_parameters"]["KNN"] = {
                    "n_neighbors": 5,
                    "weights": "uniform",
                    "metric": "minkowski"
                }
        
        model.params = {
            "modelParameters": {
                "model_type": modelType,
                "model_parameters": settings["model_parameters"]
            }
        }
        return render_template(f"edit_model/partials/model_settings/{modelType.lower()}.html", model=model)

    @model_bp.route("/api/model/<string:modelName>/entity/<string:entityName>/delete", methods=["POST"])
    def deleteEntity(modelName: str, entityName: str) -> Response:
        try:
            model_manager.getModel(modelName).deleteEntity(entityName)
            return jsonify({"success": True})
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            return jsonify({"error": "Internal server error"}), 500

    @model_bp.route("/delete-label/<string:modelName>/<string:label>", methods=["POST"])
    def deleteLabel(modelName: str, label: str) -> Response:
        try:
            model_manager.getModel(modelName).deleteObservationsByLabel(label)
            return redirect(url_for("model.editModel", modelName=modelName, section="settings"))
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @model_bp.route("/api/model/<string:modelName>/label/add", methods=["POST"])
    def addLabel(modelName: str) -> Response:
        """Add a new label to the model without requiring training data."""
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "Missing JSON payload"}), 400
            label = data.get("label", "").strip()
            if not label:
                return jsonify({"error": "Label name is required"}), 400
            model_manager.getModel(modelName).addLabel(label)
            return jsonify({"success": True, "label": label})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @model_bp.route("/api/model/<string:modelName>/generate-synthetic", methods=["POST"])
    def generateSynthetic(modelName: str) -> Response:
        """Generate synthetic training observations for a label (e.g., 'Away')."""
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "Missing JSON payload"}), 400
            label = data.get("label", "Away").strip()
            count = int(data.get("count", 50))
            if count < 1 or count > 500:
                return jsonify({"error": "Count must be between 1 and 500"}), 400
            generated = model_manager.getModel(modelName).generateSyntheticObservations(label, count)
            return jsonify({"success": True, "generated": generated, "label": label})
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @model_bp.route("/edit-model/<string:modelName>/postprocessor/add", methods=["POST"])
    def addPostprocessor(modelName: str) -> Response:
        try:
            data = request.get_json()
            if data is None:
                return jsonify({"error": "Missing or invalid JSON payload"}), 400
            logger.info(f"Adding postprocessor: {data}")
            model_manager.getModel(modelName).addPostprocessor(data['type'], data['params'])
            return jsonify({"success": True})
        except ValueError as e:
            logger.error(f"Error adding postprocessor: {e}")
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.error(f"Error adding postprocessor: {e}")
            return jsonify({"error": "Internal server error"}), 500

    @model_bp.route("/edit-model/<string:modelName>/preprocessor/add", methods=["POST"])
    def addPreprocessor(modelName: str) -> Response:
        try:
            data = request.get_json()
            if data is None:
                return jsonify({"error": "Missing or invalid JSON payload"}), 400
            logger.info(f"Adding preprocessor: {data}")
            model_manager.getModel(modelName).addPreprocessor(data['type'], data['params'])
            return jsonify({"success": True})
        except ValueError as e:
            logger.error(f"Error adding preprocessor: {e}")
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.error(f"Error adding preprocessor: {e}")
            return jsonify({"error": "Internal server error"}), 500

    @model_bp.route("/edit-model/<string:modelName>/postprocessor/delete", methods=["POST"])
    def deletePostprocessor(modelName: str) -> Response:
        try:
            data = request.get_json()
            if data is None or "index" not in data:
                return jsonify({"error": "Missing index in payload"}), 400
                
            model_manager.getModel(modelName).removePostprocessor(data["index"])
            return jsonify({"success": True})
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            return jsonify({"error": "Internal server error"}), 500

    @model_bp.route("/edit-model/<string:modelName>/preprocessor/delete", methods=["POST"])
    def deletePreprocessor(modelName: str) -> Response:
        try:
            data = request.get_json()
            if not data or "index" not in data:
                return jsonify({"success": False, "error": "Missing 'index' in payload"}), 400
            
            index = data["index"]

            if not isinstance(index, int):
                return jsonify({"success": False, "error": "'index' must be an integer"}), 400

            model = model_manager.getModel(modelName)
            if not model:
                return jsonify({"success": False, "error": f"Model '{modelName}' not found"}), 404

            try:
                model.removePreprocessor(index)
            except IndexError:
                return jsonify({"success": False, "error": f"Index '{index}' out of range"}), 400
            except ValueError as e:
                return jsonify({"success": False, "error": str(e)}), 400

            return jsonify({"success": True})

        except Exception as e:
            app.logger.exception(f"Error deleting preprocessor for model '{modelName}': {e}")
            return jsonify({"success": False, "error": "Internal server error"}), 500

    @model_bp.route("/edit-model/<string:modelName>/postprocessor/reorder", methods=["POST"])
    def reorderPostprocessors(modelName: str) -> Response:
        try:
            data = request.get_json()
            if data is None or "fromIndex" not in data or "toIndex" not in data:
                return jsonify({"error": "Missing fromIndex or toIndex in payload"}), 400
                
            model_manager.getModel(modelName).reorderPostprocessors(data["fromIndex"], data["toIndex"])
            return jsonify({"success": True})
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            return jsonify({"error": "Internal server error"}), 500

    @model_bp.route("/edit-model/<string:modelName>/preprocessor/reorder", methods=["POST"])
    def reorderPreprocessors(modelName: str) -> Response:
        try:
            data = request.get_json()
            if data is None or "fromIndex" not in data or "toIndex" not in data:
                return jsonify({"error": "Missing fromIndex or toIndex in payload"}), 400
                    
            model_manager.getModel(modelName).reorderPreprocessors(data["fromIndex"], data["toIndex"])
            return jsonify({"success": True})
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            return jsonify({"error": "Internal server error"}), 500

    @model_bp.route("/render_preprocessor/<string:preprocessor_type>", methods=["POST"])
    def render_preprocessor(preprocessor_type):
        data = request.get_json()
        sensors = data.get("sensors", [])

        try:
            return render_template(f"preprocessors/{preprocessor_type}.html", sensors=sensors)
        except TemplateNotFound:
            return jsonify({"error": "Template not found"}), 404

    @model_bp.route("/render_postprocessor/<string:postprocessor_type>", methods=["POST"])
    def render_postprocessor(postprocessor_type):
        # In the future, we might want to pass data to the template,
        # similar to how 'sensors' are passed to preprocessor templates.
        # For now, an empty context is sufficient.
        # data = request.get_json()
        # context = data.get("context", {}) 
        try:
            return render_template(f"postprocessors/{postprocessor_type}.html") #, **context)
        except TemplateNotFound:
            return jsonify({"error": "Template not found"}), 404

    @model_bp.route("/mqtt_history/<string:modelName>", methods=["GET"])
    def render_mqtt(modelName: str) -> Response:
        return jsonify(model_manager.getModel(modelName).getRecentMqtt())
    
    @model_bp.route("/mqtt_topic/<string:modelName>", methods=["PUT"])
    def set_mqtt_base_topic(modelName: str) -> Response:
        try:
            data = request.get_json()
            if not data or "mqttTopic" not in data:
                return jsonify({"error": "mqttTopic parameter is required"}), 400

            mqttTopic = data["mqttTopic"]

            # Update the model's MQTT topic
            model = model_manager.getModel(modelName)
            if not model:
                return jsonify({"error": f"Model '{modelName}' not found"}), 404

            model.setMqttTopic(mqttTopic)

            return jsonify({"success": True, "mqttTopic": mqttTopic})

        except Exception as e:
            logger.exception(f"Error setting MQTT topic for model '{modelName}': {e}")
            return jsonify({"error": str(e)}), 500

    # ── Mobile Training Page ─────────────────────────────────────────

    @model_bp.route("/train/<string:modelName>")
    def trainModel(modelName: str) -> str:
        model = model_manager.getModel(modelName)
        if not model:
            abort(404)
        return render_template(
            "train.html",
            title=f"Train: {modelName}",
            model_name=modelName,
            labels=model.getLabels(),
            active_page="train",
        )

    @model_bp.route("/api/model/<string:modelName>/live", methods=["GET"])
    def liveData(modelName: str) -> Response:
        """Return current prediction, sensors, and stats for the training UI."""
        try:
            model = model_manager.getModel(modelName)
            if not model:
                return jsonify({"error": f"Model '{modelName}' not found"}), 404
            return jsonify(model.getLiveData())
        except Exception as e:
            logger.exception(f"Error getting live data for model '{modelName}': {e}")
            return jsonify({"error": str(e)}), 500

    @model_bp.route("/api/model/<string:modelName>/collect", methods=["POST"])
    def toggleCollection(modelName: str) -> Response:
        """Start or stop collecting observations."""
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "Missing JSON payload"}), 400

            model = model_manager.getModel(modelName)
            if not model:
                return jsonify({"error": f"Model '{modelName}' not found"}), 404

            action = data.get("action", "toggle")
            label = data.get("label", "")

            if action == "start":
                if not label:
                    return jsonify({"error": "Label is required to start collecting"}), 400
                model.startCollecting(label)
                return jsonify({"success": True, "collecting": True, "label": label})
            elif action == "stop":
                model.stopCollecting()
                return jsonify({"success": True, "collecting": False})
            else:
                # Toggle
                if model.isCollecting():
                    model.stopCollecting()
                    return jsonify({"success": True, "collecting": False})
                else:
                    if not label:
                        return jsonify({"error": "Label is required to start collecting"}), 400
                    model.startCollecting(label)
                    return jsonify({"success": True, "collecting": True, "label": label})
        except Exception as e:
            logger.exception(f"Error toggling collection for model '{modelName}': {e}")
            return jsonify({"error": str(e)}), 500

    @model_bp.route("/api/model/<string:modelName>/learning-type", methods=["POST"])
    def setLearningTypeApi(modelName: str) -> Response:
        """Change learning type from the training UI."""
        try:
            data = request.get_json()
            if not data or "learning_type" not in data:
                return jsonify({"error": "learning_type is required"}), 400
            model = model_manager.getModel(modelName)
            if not model:
                return jsonify({"error": f"Model '{modelName}' not found"}), 404
            model.setLearningType(data["learning_type"])
            return jsonify({"success": True, "learning_type": data["learning_type"]})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ── Data Management APIs ─────────────────────────────────────────

    @model_bp.route("/api/model/<string:modelName>/label/<string:label>/data", methods=["DELETE"])
    def clearLabelData(modelName: str, label: str) -> Response:
        """Delete all observations for a specific label."""
        try:
            model = model_manager.getModel(modelName)
            if not model:
                return jsonify({"error": f"Model '{modelName}' not found"}), 404
            result = model.clearLabelData(label)
            return jsonify({"success": True, **result})
        except Exception as e:
            logger.exception(f"Error clearing label data: {e}")
            return jsonify({"error": str(e)}), 500

    @model_bp.route("/api/model/<string:modelName>/retrain", methods=["POST"])
    def retrainModel(modelName: str) -> Response:
        """Force model retrain from current observations."""
        try:
            model = model_manager.getModel(modelName)
            if not model:
                return jsonify({"error": f"Model '{modelName}' not found"}), 404
            result = model.retrain()
            return jsonify({"success": True, **result})
        except Exception as e:
            logger.exception(f"Error retraining model: {e}")
            return jsonify({"error": str(e)}), 500

    @model_bp.route("/api/model/<string:modelName>/data-health", methods=["GET"])
    def dataHealth(modelName: str) -> Response:
        """Return data quality metrics."""
        try:
            model = model_manager.getModel(modelName)
            if not model:
                return jsonify({"error": f"Model '{modelName}' not found"}), 404
            return jsonify(model.getDataHealth())
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @model_bp.route("/api/model/<string:modelName>/confusion-matrix", methods=["GET"])
    def confusionMatrix(modelName: str) -> Response:
        """Return confusion matrix data."""
        try:
            model = model_manager.getModel(modelName)
            if not model:
                return jsonify({"error": f"Model '{modelName}' not found"}), 404
            matrix = model.getConfusionMatrix()
            return jsonify({"success": True, "data": matrix})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @model_bp.route("/api/model/<string:modelName>/sensors", methods=["GET"])
    def sensorEntities(modelName: str) -> Response:
        """Return sensor entities with importance scores."""
        try:
            model = model_manager.getModel(modelName)
            if not model:
                return jsonify({"error": f"Model '{modelName}' not found"}), 404
            return jsonify({"success": True, "sensors": model.getSensorEntities()})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @model_bp.route("/api/model/<string:modelName>/sensor/<path:entityName>", methods=["DELETE"])
    def deleteSensor(modelName: str, entityName: str) -> Response:
        """Remove a sensor from the model and all observations."""
        try:
            model = model_manager.getModel(modelName)
            if not model:
                return jsonify({"error": f"Model '{modelName}' not found"}), 404
            result = model.deleteSensor(entityName)
            if result.get("success"):
                return jsonify(result)
            return jsonify(result), 400
        except Exception as e:
            logger.exception(f"Error deleting sensor: {e}")
            return jsonify({"error": str(e)}), 500

    return model_bp 
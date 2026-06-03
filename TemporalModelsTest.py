import unittest
import numpy as np
import pandas as pd
from classifiers.TemporalXGBoost import TemporalXGBoost, XGBOOST_AVAILABLE
from classifiers.TemporalSequenceNet import TemporalSequenceNet, TORCH_AVAILABLE, ORT_AVAILABLE
from postprocessors.hmm_smoother import HMMSmootherPostprocessor

class TestTemporalModels(unittest.TestCase):
    def test_hmm_smoother_viterbi(self):
        """Test HMM Viterbi trajectory smoothing postprocessor."""
        # Initialize smoother with window size 5, transition probability 0.1
        smoother = HMMSmootherPostprocessor(window_size=5, transition_probability=0.1, dbId=1)
        
        # Scenario: steady state of "Office"
        obs = {}
        _, label = smoother.process(obs, "Office", 0.9)
        self.assertEqual(label, "Office")
        
        _, label = smoother.process(obs, "Office", 0.9)
        self.assertEqual(label, "Office")
        
        # A single quick outlier predicted "Bedroom" with low confidence should be smoothed out
        _, label = smoother.process(obs, "Bedroom", 0.3)
        self.assertEqual(label, "Office")
        
        # Continued "Office" predictions
        _, label = smoother.process(obs, "Office", 0.9)
        self.assertEqual(label, "Office")
        
        # If we persistently predict "Bedroom" with high confidence, it should eventually transition
        _, label = smoother.process(obs, "Bedroom", 0.9)
        _, label = smoother.process(obs, "Bedroom", 0.95)
        self.assertEqual(label, "Bedroom")

    def test_temporal_xgboost(self):
        """Test TemporalXGBoost classifier training and prediction."""
        if not XGBOOST_AVAILABLE:
            self.skipTest("XGBoost not installed")
            
        # Create synthetic sequence data
        # shape: (N, T, K) -> (20 observations, 15 time steps, 3 sensors)
        np.random.seed(42)
        X_seq = np.random.uniform(0.5, 5.0, (20, 15, 3))
        
        # Let's make label "Office" have low sensor 0 values, "Bedroom" have high sensor 0 values
        X_seq[:10, :, 0] = np.random.uniform(0.5, 1.5, (10, 15))
        X_seq[10:, :, 0] = np.random.uniform(3.5, 5.0, (10, 15))
        
        y_labels = np.array(["Office"] * 10 + ["Bedroom"] * 10)
        sensor_keys = ["sensor.basement", "sensor.kitchen", "sensor.living_room"]
        
        clf = TemporalXGBoost()
        clf.populateDataframe(X_seq, y_labels, sensor_keys)
        
        self.assertTrue(clf._modelTrained)
        self.assertIsNotNone(clf.getAccuracy())
        
        # Test feature importance
        importances = clf.getFeatureImportance()
        self.assertIsNotNone(importances)
        
        # Test prediction on a single window (Office-like)
        test_df_office = pd.DataFrame(
            np.random.uniform(0.5, 1.5, (15, 3)),
            columns=sensor_keys
        )
        pred_label, conf = clf.predictLabel(test_df_office)
        self.assertEqual(pred_label, "Office")
        self.assertGreater(conf, 0.5)
        
        # Test prediction on a single window (Bedroom-like)
        test_df_bedroom = pd.DataFrame(
            np.random.uniform(0.5, 5.0, (15, 3)),
            columns=sensor_keys
        )
        test_df_bedroom["sensor.basement"] = np.random.uniform(3.5, 5.0, 15)
        pred_label, conf = clf.predictLabel(test_df_bedroom)
        self.assertEqual(pred_label, "Bedroom")
        self.assertGreater(conf, 0.5)

    def test_temporal_sequence_net_availability(self):
        """Test that TemporalSequenceNet handles unavailability of PyTorch or ONNX Runtime cleanly."""
        clf = TemporalSequenceNet()
        
        # If Torch/ORT are not available, it shouldn't raise exceptions on instantiation
        self.assertEqual(clf.model_type, "GRU")
        
        if not TORCH_AVAILABLE:
            # Check training returns False
            success = clf.train_model(
                X_trainval=np.empty((0, 15, 0)),
                y_trainval=np.empty((0,)),
                sensor_keys=[],
                classes=[],
                save_path="test_model.onnx"
            )
            self.assertFalse(success)
            
        if not ORT_AVAILABLE:
            # Check loading returns False
            success = clf.load_onnx_model("dummy_path.onnx", [], [], 15)
            self.assertFalse(success)
            
            # Check prediction returns None, 0.0
            label, conf = clf.predict_sequence(pd.DataFrame())
            self.assertIsNone(label)
            self.assertEqual(conf, 0.0)

if __name__ == "__main__":
    unittest.main()

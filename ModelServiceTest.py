import unittest
from unittest.mock import MagicMock
import os
import json
import time
from ModelStore import ModelStore
from ModelService import ModelService

class TestModelService(unittest.TestCase):
    def setUp(self):
        self.db_path = "test_modelservice.db"
        try:
            os.remove(self.db_path)
            os.remove(self.db_path + "-shm")
            os.remove(self.db_path + "-wal")
        except FileNotFoundError:
            pass
        self.db = ModelStore(self.db_path)
        self.mqtt_client = MagicMock()
        
        # Initialize ModelService
        self.service = ModelService(self.mqtt_client, self.db)

    def tearDown(self):
        self.service.dispose()
        try:
            os.remove(self.db_path)
            os.remove(self.db_path + "-shm")
            os.remove(self.db_path + "-wal")
        except FileNotFoundError:
            pass

    def test_auto_learning_cap_when_untrained(self):
        """Test that AUTO learning type stops saving observations at 200 if the model is not trained."""
        # 1. Set mode to AUTO
        self.service.setLearningType("AUTO")
        
        # 2. Start collecting for "Office"
        self.service.startCollecting("Office")
        
        # 3. Simulate getting MQTT updates. 
        # Add 199 observations directly to database to speed up, then test boundary
        for i in range(199):
            self.db.addObservation("Office", {"Basement": 1.0})
            
        # The service retrains on a background state check.
        self.service.retrain()
        self.assertFalse(self.service._model._modelTrained) # single class, should be untrained
        
        # 4. Trigger predictLabel (since count in db is 199, which is < 200, it should save)
        # Ensure we bypass rate limit by initializing last save time to long ago
        self.service._lastSaveTimes["Office"] = 0.0
        payload = json.dumps({"sensor.distance_to_basement_sensor": 1.5})
        self.service.predictLabel(payload)
        
        self.assertEqual(self.db.getLabelObservationCount("Office"), 200)
        
        # 5. Reset last save time to bypass rate limit
        self.service._lastSaveTimes["Office"] = 0.0
        
        # 6. Now count is 200 (which is >= 200). Since model is not trained, it should NOT save the next one.
        self.service.predictLabel(payload)
        self.assertEqual(self.db.getLabelObservationCount("Office"), 200) # should remain at 200

    def test_save_rate_limiting(self):
        """Test that saving observations for a label is rate-limited to once every 10 seconds."""
        self.service.setLearningType("EAGER") # eager mode saves everything
        self.service.startCollecting("Office")
        
        payload = json.dumps({"sensor.distance_to_basement_sensor": 1.5})
        
        # First save should succeed
        self.service._lastSaveTimes["Office"] = 0.0
        self.service.predictLabel(payload)
        self.assertEqual(self.db.getLabelObservationCount("Office"), 1)
        
        # Second save immediately after should be blocked by rate limit
        self.service.predictLabel(payload)
        self.assertEqual(self.db.getLabelObservationCount("Office"), 1) # count shouldn't increase
        
        # Third save with simulated elapsed time should succeed
        self.service._lastSaveTimes["Office"] = time.time() - 11.0
        self.service.predictLabel(payload)
        self.assertEqual(self.db.getLabelObservationCount("Office"), 2)

    def test_lazy_learning_behavior(self):
        """Test that LAZY learning only saves observations when predictions are wrong or confidence is low."""
        self.service.setLearningType("LAZY")
        self.service.startCollecting("Office")
        
        # Mock the model to be trained
        self.service._model = MagicMock()
        self.service._model._modelTrained = True
        
        # Scenario 1: Correct prediction with high confidence (should NOT save)
        self.service._model.predictLabel.return_value = ("Office", 0.9)
        self.service._lastSaveTimes["Office"] = 0.0
        payload = json.dumps({"sensor.distance_to_basement_sensor": 1.5})
        self.service.predictLabel(payload)
        self.assertEqual(self.db.getLabelObservationCount("Office"), 0)
        
        # Scenario 2: Incorrect prediction (should save)
        self.service._model.predictLabel.return_value = ("Bedroom", 0.9)
        self.service._lastSaveTimes["Office"] = 0.0
        self.service.predictLabel(payload)
        self.assertEqual(self.db.getLabelObservationCount("Office"), 1)
        
        # Scenario 3: Correct prediction but low confidence (should save)
        self.service._model.predictLabel.return_value = ("Office", 0.5)
        self.service._lastSaveTimes["Office"] = 0.0
        self.service.predictLabel(payload)
        self.assertEqual(self.db.getLabelObservationCount("Office"), 2)

    def test_start_collecting_mode_transitions(self):
        """Test startCollecting switches DISABLED to AUTO, but preserves other active modes."""
        # 1. Start from DISABLED -> should change to AUTO
        self.service.setLearningType("DISABLED")
        self.service.startCollecting("Office")
        self.assertEqual(self.service.getLearningType(), "AUTO")
        
        # 2. Start from LAZY -> should remain LAZY
        self.service.setLearningType("LAZY")
        self.service.startCollecting("Office")
        self.assertEqual(self.service.getLearningType(), "LAZY")
        
        # 3. Start from EAGER -> should remain EAGER
        self.service.setLearningType("EAGER")
        self.service.startCollecting("Office")
        self.assertEqual(self.service.getLearningType(), "EAGER")

    def test_undo_last_session(self):
        """Test that undoLastSession deletes observations from the session window and clears session config."""
        # 1. Start collecting
        self.service.startCollecting("Office")
        start_time = self.service.getModelConfig("last_session_start", 0)
        self.assertIsNotNone(start_time)
        self.assertEqual(self.service.getModelConfig("last_session_label", ""), "Office")

        # 2. Add some observations inside and outside the session window
        self.db.addObservation("Office", {"Basement": 1.0}, assignedTime=start_time + 0.005)
        self.db.addObservation("Office", {"Basement": 1.0}, assignedTime=start_time + 0.010)
        # observation before session start
        self.db.addObservation("Office", {"Basement": 1.0}, assignedTime=start_time - 1.000)

        self.service.stopCollecting()
        end_time = self.service.getModelConfig("last_session_end", 0)
        self.assertIsNotNone(end_time)

        # 3. Confirm we have 3 observations in total
        obs_before = self.db.getObservations()
        self.assertEqual(len(obs_before), 3)

        # 4. Perform undo
        res = self.service.undoLastSession()
        self.assertTrue(res["success"])
        self.assertEqual(res["deleted_count"], 2)

        # 5. Confirm only 1 observation remains (the one before session start)
        obs = self.db.getObservations()
        self.assertEqual(len(obs), 1)
        self.assertAlmostEqual(obs[0].time, start_time - 1.000, places=4)

        # 6. Verify session markers are cleared
        self.assertIsNone(self.service.getModelConfig("last_session_start", None))
        self.assertIsNone(self.service.getModelConfig("last_session_end", None))
        self.assertIsNone(self.service.getModelConfig("last_session_label", None))

    def test_database_backup(self):
        """Test that backup method creates backups and prunes old ones."""
        from pathlib import Path
        import shutil

        # 1. Create a backup
        backup_path_str = self.db.backup()
        self.assertIsNotNone(backup_path_str)
        backup_path = Path(backup_path_str)
        self.assertTrue(backup_path.exists())
        self.assertEqual(backup_path.parent.name, "backups")

        # 2. Simulate having 12 backups
        backups_dir = backup_path.parent
        # Clean any existing files first
        for f in backups_dir.glob("*.db"):
            f.unlink()

        # Create 12 dummy files
        for i in range(12):
            dummy = backups_dir / f"test_modelservice_{i:02d}.db"
            dummy.touch()
            # Set mtime so they have predictable creation order
            import os
            os.utime(dummy, (time.time() - (100 - i), time.time() - (100 - i)))

        # Run backup again, it should prune the oldest 3 (leaving exactly 10)
        self.db.backup()
        remaining_backups = sorted(backups_dir.glob("*.db"), key=lambda x: x.stat().st_mtime)
        self.assertEqual(len(remaining_backups), 10)

        # Clean up backups dir
        shutil.rmtree(backups_dir)

if __name__ == '__main__':
    unittest.main()

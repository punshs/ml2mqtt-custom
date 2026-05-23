import unittest
import os
import math
from ModelStore import ModelStore

class TestModelStore(unittest.TestCase):
    def setUp(self):
        self.db_path = "test_model.db"
        try:
            os.remove(self.db_path)
            os.remove(self.db_path + "-shm")
            os.remove(self.db_path + "-wal")
        except FileNotFoundError:
            pass
        self.db = ModelStore(self.db_path)

    def test_retrieveSimpleObservation(self):
        self.db.addObservation("test_label1", { "basement": 123.0, "livingroom": 456.0 })
        self.db.addObservation("test_label2", { "basement": 726.0 })
        self.db.addObservation("test_label3", { "basement": 700.0, "bedroom": 326.0 })

        observations = self.db.getObservations()
        self.assertEqual(len(observations), 3)
        
        # Check first observation (latest since order is DESC)
        self.assertEqual(observations[0].label, "test_label3")
        self.assertEqual(observations[0].sensorValues["basement"], 700.0)
        self.assertEqual(observations[0].sensorValues["bedroom"], 326.0)
        # livingroom is not in observations[0].sensorValues since it was added after observation 1/2
        # but sortEntityValues or the model preprocessors will default it.
        
        # Check label observation count method
        self.assertEqual(self.db.getLabelObservationCount("test_label1"), 1)
        self.assertEqual(self.db.getLabelObservationCount("test_label2"), 1)
        self.assertEqual(self.db.getLabelObservationCount("nonexistent"), 0)

    def test_addSensorOnTheFly(self):
        self.db.addObservation("test_label1", { "basement": 123.0 })
        self.assertEqual(self.db.getLabelObservationCount("test_label1"), 1)
        
        # Add new sensor
        self.db.addSensor("attic", "float")
        
        # Entity keys should contain attic
        keys = [k.name for k in self.db.getEntityKeys()]
        self.assertIn("attic", keys)
        
        # Retrieving the old observation should return None for the new sensor
        obs = self.db.getObservations()
        self.assertEqual(len(obs), 1)
        self.assertNotIn("attic", obs[0].sensorValues) # None values are filtered out in getObservations()
        
        # Add new observation with new sensor
        self.db.addObservation("test_label1", { "basement": 200.0, "attic": 45.0 })
        obs2 = self.db.getObservations()
        self.assertEqual(len(obs2), 2)
        self.assertEqual(obs2[0].sensorValues["attic"], 45.0)

    def tearDown(self):
        self.db.close()
        try:
            os.remove(self.db_path)
            os.remove(self.db_path + "-shm")
            os.remove(self.db_path + "-wal")
        except FileNotFoundError:
            pass

if __name__ == '__main__':
    unittest.main()
import unittest
from typing import Dict, Any
from preprocessors.sensor_dropoff import SensorDropoffHandler

class SensorDropoffHandlerTest(unittest.TestCase):
    def setUp(self):
        # Initialize handler with dummy dbId
        self.handler = SensorDropoffHandler(dbId=1, params={'sensor': [{'SELECT_ALL': True}]})

    def test_short_name_generation(self):
        """Test extraction of short sensor names from long entity IDs."""
        tests = {
            "sensor.bermuda_ec_0e_36_57_a4_69_unfiltered_distance_to_office_sensor": "office_sensor",
            "sensor.sam_whoop_distance_to_bedroom_sensor_2": "bedroom_sensor_2",
            "sensor.kitchen_temperature": "kitchen_temperature",  # fallback case
        }
        for entity, expected in tests.items():
            self.assertEqual(self.handler._short_name(entity), expected)

    def test_process_distinguishes_unknown_unavailable(self):
        """Test that unknown -> count, and unavailable -> count + flag."""
        observation = {
            "sensor.distance_to_office_sensor": 3.28,
            "sensor.distance_to_bedroom_sensor": "unknown",
            "sensor.distance_to_living_room_sensor": "unknown",
            "sensor.distance_to_kitchen_sensor": "unavailable",
            "sensor.distance_to_hallway_sensor": "Unavailable"  # case-insensitive test
        }
        
        result = self.handler.process(observation, {})
        
        # Original values should be preserved
        self.assertEqual(result["sensor.distance_to_office_sensor"], 3.28)
        self.assertEqual(result["sensor.distance_to_bedroom_sensor"], "unknown")
        
        # New count features should be added
        self.assertEqual(result["_unknown_count"], 2.0)
        self.assertEqual(result["_unavailable_count"], 2.0)
        
        # Per-sensor binary flags for unavailable sensors only
        self.assertEqual(result["_kitchen_sensor_offline"], 1.0)
        self.assertEqual(result["_hallway_sensor_offline"], 1.0)
        self.assertNotIn("_bedroom_sensor_offline", result)
        self.assertNotIn("_office_sensor_offline", result)

    def test_process_all_valid_floats(self):
        """Test when all sensors are working normally."""
        observation = {
            "sensor.distance_to_office_sensor": 3.28,
            "sensor.distance_to_bedroom_sensor": 15.5,
        }
        
        result = self.handler.process(observation, {})
        self.assertEqual(result["_unknown_count"], 0.0)
        self.assertEqual(result["_unavailable_count"], 0.0)
        self.assertFalse(any(k.endswith("_offline") for k in result))

if __name__ == '__main__':
    unittest.main()

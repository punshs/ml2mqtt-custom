import unittest
from classifiers.RandomForest import RandomForest
from ModelStore import ModelObservation

class TestRandomForest(unittest.TestCase):
    def test_simple_prediction(self):
        rf = RandomForest()
        observations = [
            ModelObservation(1710000000, "test_label3", {"basement": 700.0, "livingroom": 326.0}),
            ModelObservation(1710000000, "test_label1", {"basement": 123.0, "livingroom": 456.0}),
            ModelObservation(1710000000, "test_label1", {"basement": 123.0, "livingroom": 456.0}),
            ModelObservation(1710000000, "test_label1", {"basement": 123.0, "livingroom": 456.0}),
            ModelObservation(1710000000, "test_label3", {"basement": 700.0, "livingroom": 326.0}),
        ]
        rf.populateDataframe(observations)
        prediction, confidence = rf.predictLabel({"basement": 123.0, "livingroom": 456.0})
        self.assertEqual(prediction, "test_label1")


if __name__ == '__main__':
    unittest.main()
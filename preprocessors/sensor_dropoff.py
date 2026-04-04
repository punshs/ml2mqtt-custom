from typing import Dict, Any, ClassVar
from .base import BasePreprocessor


class SensorDropoffHandler(BasePreprocessor):
    """Distinguishes 'unknown' (Whoop out of range) from 'unavailable' (proxy offline).
    
    Adds count features and per-sensor binary flags BEFORE TypeCaster/NullHandler
    convert these string states to None/float. This gives the model meaningful
    signals about sensor dropout patterns.
    
    Features added:
    - _unknown_count: Number of sensors reporting 'unknown'
    - _unavailable_count: Number of sensors reporting 'unavailable'
    - _<sensor_short>_offline: Binary flag (1.0) for each unavailable sensor
    """

    name: ClassVar[str] = "Sensor Dropoff Handler"
    type: ClassVar[str] = "sensor_dropoff"
    description: ClassVar[str] = (
        "Distinguishes 'unknown' (out of range) from 'unavailable' (proxy offline). "
        "Adds _unknown_count, _unavailable_count, and per-sensor _offline flags as ML features."
    )

    UNKNOWN_VALUES = {"unknown"}
    UNAVAILABLE_VALUES = {"unavailable"}

    def __init__(self, dbId: int, **kwargs):
        super().__init__(dbId, **kwargs)

    def _short_name(self, sensor_key: str) -> str:
        """Create a short identifier from a long entity_id for the binary flag."""
        # e.g. sensor.bermuda_ec_0e_36_57_a4_69_unfiltered_distance_to_office_sensor -> office_sensor
        parts = sensor_key.replace("sensor.", "").split("_")
        # Find 'to' and take everything after it
        if "to" in parts:
            idx = parts.index("to")
            return "_".join(parts[idx + 1:])
        # Fallback: last 2 parts
        return "_".join(parts[-2:]) if len(parts) >= 2 else parts[-1]

    def process(self, observation: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        result = observation.copy()
        unknown_count = 0
        unavailable_count = 0

        for entity, value in observation.items():
            if not self.canConsume(entity):
                continue

            if isinstance(value, str):
                val_lower = value.lower()
                if val_lower in self.UNKNOWN_VALUES:
                    unknown_count += 1
                elif val_lower in self.UNAVAILABLE_VALUES:
                    unavailable_count += 1
                    short = self._short_name(entity)
                    result[f"_{short}_offline"] = 1.0

        result["_unknown_count"] = float(unknown_count)
        result["_unavailable_count"] = float(unavailable_count)

        return result

    def configToString(self) -> str:
        return "Adds _unknown_count, _unavailable_count, and per-sensor _offline flags"

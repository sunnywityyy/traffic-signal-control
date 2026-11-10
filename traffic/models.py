from dataclasses import dataclass

LANE_NAMES = ["Lane 1 (North)", "Lane 2 (East)", "Lane 3 (South)", "Lane 4 (West)"]


@dataclass
class LaneState:
    name: str
    vehicle_count: int = 0
    wait_seconds: int = 0

    @property
    def density_label(self) -> str:
        if self.vehicle_count >= 20:
            return "High"
        if self.vehicle_count >= 8:
            return "Medium"
        return "Low"

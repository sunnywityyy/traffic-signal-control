"""Signal-state helpers for the four-way educational junction."""

def signal_for_lane(lane_index: int, current_lane: int, phase: str) -> str:
    if phase == "ALL_RED":
        return "RED"
    if lane_index != current_lane:
        return "RED"
    if phase in ("GREEN", "EMERGENCY_GREEN"):
        return "GREEN"
    if phase == "AMBER":
        return "AMBER"
    return "RED"

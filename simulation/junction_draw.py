from __future__ import annotations

import cv2
import numpy as np

from traffic.models import LANE_NAMES
from traffic.signal_state_machine import signal_for_lane


def draw_junction(current_lane: int, phase: str, emergency_lane: int | None, remaining: int):
    """Draw a simple top-down four-approach junction using OpenCV."""
    canvas = np.full((650, 760, 3), (238, 238, 238), dtype=np.uint8)
    road = (75, 78, 82)
    white = (245, 245, 245)
    cv2.rectangle(canvas, (280, 0), (480, 650), road, -1)
    cv2.rectangle(canvas, (0, 225), (760, 425), road, -1)

    # Lane separators and stop lines
    cv2.line(canvas, (380, 0), (380, 225), white, 2)
    cv2.line(canvas, (380, 425), (380, 650), white, 2)
    cv2.line(canvas, (0, 325), (280, 325), white, 2)
    cv2.line(canvas, (480, 325), (760, 325), white, 2)
    cv2.line(canvas, (280, 205), (480, 205), white, 5)
    cv2.line(canvas, (280, 445), (480, 445), white, 5)
    cv2.line(canvas, (260, 225), (260, 425), white, 5)
    cv2.line(canvas, (500, 225), (500, 425), white, 5)

    # Signal heads: north, east, south, west
    positions = [(520, 150), (600, 450), (210, 500), (150, 180)]
    for idx, pos in enumerate(positions):
        state = signal_for_lane(idx, current_lane, phase)
        color = {"RED": (40, 40, 220), "AMBER": (0, 190, 255), "GREEN": (40, 190, 40)}[state]
        cv2.rectangle(canvas, (pos[0]-42, pos[1]-27), (pos[0]+42, pos[1]+27), (25, 25, 25), -1)
        for j, lamp in enumerate([(pos[0]-25, pos[1]), (pos[0], pos[1]), (pos[0]+25, pos[1])]):
            lamp_color = (45, 45, 45)
            if (state == "RED" and j == 0) or (state == "AMBER" and j == 1) or (state == "GREEN" and j == 2):
                lamp_color = color
            cv2.circle(canvas, lamp, 10, lamp_color, -1)
        cv2.putText(canvas, f"L{idx+1}", (pos[0]-15, pos[1]+48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2)

    # Labels
    cv2.putText(canvas, "NORTH / LANE 1", (275, 35), cv2.FONT_HERSHEY_SIMPLEX, .65, white, 2)
    cv2.putText(canvas, "EAST / LANE 2", (535, 350), cv2.FONT_HERSHEY_SIMPLEX, .65, white, 2)
    cv2.putText(canvas, "SOUTH / LANE 3", (275, 620), cv2.FONT_HERSHEY_SIMPLEX, .65, white, 2)
    cv2.putText(canvas, "WEST / LANE 4", (15, 350), cv2.FONT_HERSHEY_SIMPLEX, .65, white, 2)

    phase_text = phase.replace("_", " ").upper()
    cv2.putText(canvas, f"PHASE: {phase_text}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, .7, (20, 20, 20), 2)
    cv2.putText(canvas, f"REMAINING: {remaining}s", (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, .7, (20, 20, 20), 2)
    if emergency_lane is not None:
        cv2.putText(canvas, f"EMERGENCY PRIORITY: LANE {emergency_lane+1}", (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 220), 2)
    return canvas

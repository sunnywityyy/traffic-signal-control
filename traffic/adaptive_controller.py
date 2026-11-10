from __future__ import annotations

from traffic.models import LaneState, LANE_NAMES


class AdaptiveSignalController:
    """
    Single-conflict-group, four-approach educational controller.

    Phases:
      GREEN -> AMBER -> ALL_RED -> next GREEN
      GREEN -> AMBER -> ALL_RED -> EMERGENCY_GREEN
      EMERGENCY_GREEN -> ALL_RED -> normal GREEN after clear
    """

    def __init__(self):
        self.min_green = 10
        self.max_green = 50
        self.amber_seconds = 3
        self.all_red_seconds = 2
        self.fairness_weight = 0.7
        self.lanes = [LaneState(name=n) for n in LANE_NAMES]
        self.current_lane = 0
        self.phase = "GREEN"
        self.phase_remaining = self.min_green
        self.emergency_lane = None
        self._pending_emergency = None
        self._emergency_released = False
        self._set_current_green_duration()

    def configure(self, min_green, max_green, amber_seconds, all_red_seconds, fairness_weight):
        self.min_green = int(min_green)
        self.max_green = max(int(max_green), self.min_green)
        self.amber_seconds = int(amber_seconds)
        self.all_red_seconds = int(all_red_seconds)
        self.fairness_weight = float(fairness_weight)
        if self.phase == "GREEN":
            self._set_current_green_duration(preserve_elapsed=True)

    def set_lane_count(self, lane_index: int, count: int):
        self.lanes[lane_index].vehicle_count = max(0, int(count))

    def demand_score(self, lane_index: int) -> float:
        lane = self.lanes[lane_index]
        return float(lane.vehicle_count) + self.fairness_weight * lane.wait_seconds

    def next_green_duration(self, lane_index=None) -> int:
        if lane_index is None:
            lane_index = self.current_lane
        scores = [self.demand_score(i) for i in range(len(self.lanes))]
        max_score = max(scores) if scores else 0.0
        score = self.demand_score(lane_index)
        if max_score <= 0:
            return self.min_green
        duration = self.min_green + (score / max_score) * (self.max_green - self.min_green)
        return int(round(max(self.min_green, min(self.max_green, duration))))

    def _set_current_green_duration(self, preserve_elapsed=False):
        duration = self.next_green_duration(self.current_lane)
        if not preserve_elapsed or self.phase_remaining <= 0:
            self.phase_remaining = duration
        else:
            self.phase_remaining = min(self.phase_remaining, duration)

    def request_emergency(self, lane_index: int):
        lane_index = int(lane_index)
        if lane_index < 0 or lane_index >= len(self.lanes):
            raise ValueError("Invalid emergency lane.")
        self._pending_emergency = lane_index
        self.emergency_lane = lane_index
        self._emergency_released = False

        if self.phase == "GREEN" and self.current_lane == lane_index:
            self.phase = "EMERGENCY_GREEN"
            self.phase_remaining = 9999
        elif self.phase == "EMERGENCY_GREEN" and self.current_lane != lane_index:
            # Queue a second emergency request; it will be handled after release.
            self._pending_emergency = lane_index

    def clear_emergency(self):
        if self.phase == "EMERGENCY_GREEN":
            self.phase = "ALL_RED"
            self.phase_remaining = self.all_red_seconds
            self._emergency_released = True
        self.emergency_lane = None
        self._pending_emergency = None

    def _choose_next_lane(self):
        if self._pending_emergency is not None:
            lane = self._pending_emergency
            self._pending_emergency = None
            return lane
        candidates = [i for i in range(len(self.lanes)) if i != self.current_lane]
        if not candidates:
            candidates = list(range(len(self.lanes)))
        # Demand + waiting time; deterministic tie break by longest wait then lane order.
        return max(candidates, key=lambda i: (self.demand_score(i), self.lanes[i].wait_seconds, -i))

    def _start_green(self, lane_index: int):
        self.current_lane = lane_index
        self.phase = "GREEN"
        self.phase_remaining = self.next_green_duration(lane_index)
        self.lanes[lane_index].wait_seconds = 0

    def _advance_one_second(self):
        # Waiting time accrues for all non-served lanes.
        for idx, lane in enumerate(self.lanes):
            if idx != self.current_lane:
                lane.wait_seconds += 1

        if self.phase == "GREEN":
            if self._pending_emergency is not None and self._pending_emergency != self.current_lane:
                self.phase = "AMBER"
                self.phase_remaining = self.amber_seconds
            elif self.phase_remaining <= 1:
                self.phase = "AMBER"
                self.phase_remaining = self.amber_seconds
            else:
                self.phase_remaining -= 1

        elif self.phase == "AMBER":
            if self.phase_remaining <= 1:
                self.phase = "ALL_RED"
                self.phase_remaining = self.all_red_seconds
            else:
                self.phase_remaining -= 1

        elif self.phase == "ALL_RED":
            if self.phase_remaining <= 1:
                if self._emergency_released:
                    self._emergency_released = False
                    self._start_green(self._choose_next_lane())
                elif self._pending_emergency is not None:
                    lane = self._pending_emergency
                    self._pending_emergency = None
                    self.current_lane = lane
                    self.phase = "EMERGENCY_GREEN"
                    self.phase_remaining = 9999
                    self.lanes[lane].wait_seconds = 0
                else:
                    self._start_green(self._choose_next_lane())
            else:
                self.phase_remaining -= 1

        elif self.phase == "EMERGENCY_GREEN":
            # Holds until the emergency vehicle is explicitly marked as cleared.
            self.phase_remaining = 9999

    def advance(self, seconds: int = 1):
        for _ in range(max(0, int(seconds))):
            self._advance_one_second()

    def reset(self):
        for lane in self.lanes:
            lane.wait_seconds = 0
        self.current_lane = 0
        self.phase = "GREEN"
        self.phase_remaining = self.next_green_duration(0)
        self.emergency_lane = None
        self._pending_emergency = None
        self._emergency_released = False

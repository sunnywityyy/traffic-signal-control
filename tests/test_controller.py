import unittest
from traffic.adaptive_controller import AdaptiveSignalController


class ControllerTests(unittest.TestCase):
    def test_higher_demand_gets_longer_green(self):
        c = AdaptiveSignalController()
        c.set_lane_count(0, 30)
        c.set_lane_count(1, 5)
        self.assertGreater(c.next_green_duration(0), c.next_green_duration(1))

    def test_emergency_uses_safe_transition(self):
        c = AdaptiveSignalController()
        c.request_emergency(2)
        self.assertEqual(c.phase, "GREEN")
        c.advance(1)
        self.assertEqual(c.phase, "AMBER")
        c.advance(c.amber_seconds)
        self.assertEqual(c.phase, "ALL_RED")
        c.advance(c.all_red_seconds)
        self.assertEqual(c.phase, "EMERGENCY_GREEN")
        self.assertEqual(c.current_lane, 2)

    def test_emergency_release_returns_to_normal(self):
        c = AdaptiveSignalController()
        c.request_emergency(2)
        c.advance(1 + c.amber_seconds + c.all_red_seconds)
        self.assertEqual(c.phase, "EMERGENCY_GREEN")
        c.clear_emergency()
        self.assertEqual(c.phase, "ALL_RED")
        c.advance(c.all_red_seconds)
        self.assertNotEqual(c.phase, "EMERGENCY_GREEN")


if __name__ == "__main__":
    unittest.main()

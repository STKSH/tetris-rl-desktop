import unittest
import tetris_rl as tr


class SpeedHelpersTest(unittest.TestCase):
    def test_min_speed_constant_is_one(self):
        self.assertEqual(tr.MIN_SPEED_MS, 1)

    def test_clamp_speed_respects_min_max(self):
        self.assertEqual(tr.clamp_speed(20, -50, 1, 1000), 1)
        self.assertEqual(tr.clamp_speed(980, 50, 1, 1000), 1000)
        self.assertEqual(tr.clamp_speed(100, -10, 1, 1000), 90)


if __name__ == "__main__":
    unittest.main()

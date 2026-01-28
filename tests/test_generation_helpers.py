import unittest
import tetris_rl as tr


class GenerationHelpersTest(unittest.TestCase):
    def test_all_game_over_true_when_all_states_over(self):
        states = [tr.GameState.new() for _ in range(3)]
        for state in states:
            state.game_over = True
        self.assertTrue(tr.all_game_over(states))

    def test_all_game_over_false_when_any_active(self):
        states = [tr.GameState.new() for _ in range(2)]
        states[0].game_over = True
        states[1].game_over = False
        self.assertFalse(tr.all_game_over(states))

    def test_compute_generation_stats(self):
        states = [tr.GameState.new() for _ in range(2)]
        states[0].score = 100
        states[1].score = 300
        states[0].lines_cleared = 1
        states[1].lines_cleared = 4
        stats = tr.compute_generation_stats(states)
        self.assertEqual(stats["avg_score"], 200)
        self.assertEqual(stats["max_score"], 300)
        self.assertEqual(stats["total_lines"], 5)


if __name__ == "__main__":
    unittest.main()

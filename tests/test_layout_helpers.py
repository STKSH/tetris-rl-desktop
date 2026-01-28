import unittest
import tetris_rl as tr


class LayoutHelpersTest(unittest.TestCase):
    def test_grid_constants_are_5x2(self):
        self.assertEqual(tr.GRID_COLUMNS, 5)
        self.assertEqual(tr.GRID_ROWS, 2)

    def test_get_board_origins_5x2(self):
        origins = tr.get_board_origins(
            columns=5,
            rows=2,
            board_width_px=120,
            board_height_px=240,
            padding=8,
            top_offset=40,
        )
        self.assertEqual(len(origins), 10)
        self.assertEqual(origins[0], (8, 48))
        self.assertEqual(origins[1], (136, 48))
        self.assertEqual(origins[2], (264, 48))
        self.assertEqual(origins[5], (8, 296))


if __name__ == "__main__":
    unittest.main()

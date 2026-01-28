# Tetris RL 10-Board Generation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run 10 parallel game states per generation, render them in a 2x5 grid, and speed up the minimum tick delay to 5ms while keeping .bat compatibility.

**Architecture:** Keep a single shared `weights` for each generation and maintain a list of 10 `GameState` objects. A generation ends when all 10 states are game over, at which point stats are aggregated and weights are mutated once before reinitializing all 10 boards. Rendering uses a grid layout with a compact HUD at the top.

**Tech Stack:** Python, Pygame (existing), unittest (standard library).

### Task 1: Generation helpers + tests

**Files:**
- Create: `tests/test_generation_helpers.py`
- Modify: `tetris_rl.py`

**Step 1: Write the failing test**

```python
# tests/test_generation_helpers.py
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
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_generation_helpers.py -v`
Expected: FAIL with `AttributeError: module 'tetris_rl' has no attribute 'all_game_over'` (or similar for missing helpers).

**Step 3: Write minimal implementation**

```python
# tetris_rl.py (add near other helper functions)

def init_generation(count: int) -> List[GameState]:
    return [GameState.new() for _ in range(count)]


def all_game_over(states: List[GameState]) -> bool:
    return all(state.game_over for state in states)


def compute_generation_stats(states: List[GameState]) -> Dict[str, int]:
    if not states:
        return {"avg_score": 0, "max_score": 0, "total_lines": 0}
    scores = [state.score for state in states]
    total_lines = sum(state.lines_cleared for state in states)
    avg_score = int(sum(scores) / len(scores))
    max_score = max(scores)
    return {"avg_score": avg_score, "max_score": max_score, "total_lines": total_lines}
```

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_generation_helpers.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_generation_helpers.py tetris_rl.py
git commit -m "test: add generation helper tests"
```

(If no git repo is initialized, skip this step.)

### Task 2: Grid layout helper + tests

**Files:**
- Create: `tests/test_layout_helpers.py`
- Modify: `tetris_rl.py`

**Step 1: Write the failing test**

```python
# tests/test_layout_helpers.py
import unittest
import tetris_rl as tr


class LayoutHelpersTest(unittest.TestCase):
    def test_get_board_origins_2x5(self):
        origins = tr.get_board_origins(
            columns=2,
            rows=5,
            board_width_px=120,
            board_height_px=240,
            padding=8,
            top_offset=40,
        )
        self.assertEqual(len(origins), 10)
        self.assertEqual(origins[0], (8, 48))
        self.assertEqual(origins[1], (136, 48))
        self.assertEqual(origins[2], (8, 296))


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_layout_helpers.py -v`
Expected: FAIL with `AttributeError: module 'tetris_rl' has no attribute 'get_board_origins'`.

**Step 3: Write minimal implementation**

```python
# tetris_rl.py

def get_board_origins(
    columns: int,
    rows: int,
    board_width_px: int,
    board_height_px: int,
    padding: int,
    top_offset: int,
) -> List[Tuple[int, int]]:
    origins: List[Tuple[int, int]] = []
    total = columns * rows
    for idx in range(total):
        col = idx % columns
        row = idx // columns
        x = padding + col * (board_width_px + padding)
        y = top_offset + padding + row * (board_height_px + padding)
        origins.append((x, y))
    return origins
```

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_layout_helpers.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_layout_helpers.py tetris_rl.py
git commit -m "test: add layout helper tests"
```

(If no git repo is initialized, skip this step.)

### Task 3: Multi-board loop, HUD, speed clamp

**Files:**
- Modify: `tetris_rl.py`

**Step 1: Write the failing test**

```python
# tests/test_speed_helpers.py
import unittest
import tetris_rl as tr


class SpeedHelpersTest(unittest.TestCase):
    def test_clamp_speed_respects_min_max(self):
        self.assertEqual(tr.clamp_speed(20, -50, 5, 1000), 5)
        self.assertEqual(tr.clamp_speed(980, 50, 5, 1000), 1000)
        self.assertEqual(tr.clamp_speed(100, -10, 5, 1000), 90)


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_speed_helpers.py -v`
Expected: FAIL with `AttributeError: module 'tetris_rl' has no attribute 'clamp_speed'`.

**Step 3: Write minimal implementation**

```python
# tetris_rl.py

def clamp_speed(speed_ms: int, delta: int, min_speed: int, max_speed: int) -> int:
    return max(min_speed, min(max_speed, speed_ms + delta))
```

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_speed_helpers.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_speed_helpers.py tetris_rl.py
git commit -m "test: add speed clamp helper test"
```

(If no git repo is initialized, skip this step.)

### Task 4: Wire parallel boards and render grid

**Files:**
- Modify: `tetris_rl.py`

**Step 1: Update constants and layout settings**

```python
# tetris_rl.py (top-level constants)
CELL_SIZE = 12
GRID_COLUMNS = 2
GRID_ROWS = 5
BOARD_COUNT = GRID_COLUMNS * GRID_ROWS
GRID_PADDING = 8
HUD_HEIGHT = 40
```

**Step 2: Update run() state setup**

```python
# tetris_rl.py (inside run())
states = init_generation(BOARD_COUNT)
active = False
speed_ms = 100
min_speed_ms = 5
max_speed_ms = 1000
generation = 1
```

**Step 3: Replace per-board loop with parallel logic**

```python
# tetris_rl.py (inside main loop, when active)
if all_game_over(states):
    stats = compute_generation_stats(states)
    generation += 1
    weights = mutate_weights(weights)
    save_weights(weights)
    states = init_generation(BOARD_COUNT)
else:
    for state in states:
        if state.game_over:
            continue
        if state.current_piece is None:
            spawn_piece(state, weights)
        else:
            step_game(state, weights)
```

**Step 4: Update speed control to use clamp_speed and min 5ms**

```python
# tetris_rl.py (key handling)
elif event.key in (pygame.K_MINUS, pygame.K_UNDERSCORE, pygame.K_LEFTBRACKET):
    speed_ms = clamp_speed(speed_ms, 20, min_speed_ms, max_speed_ms)
elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_RIGHTBRACKET):
    speed_ms = clamp_speed(speed_ms, -20, min_speed_ms, max_speed_ms)
```

**Step 5: Render 2x5 grid + compact HUD**

```python
# tetris_rl.py (inside run())
board_width_px = BOARD_WIDTH * CELL_SIZE
board_height_px = BOARD_HEIGHT * CELL_SIZE
window_width = (GRID_COLUMNS * board_width_px) + ((GRID_COLUMNS + 1) * GRID_PADDING)
window_height = HUD_HEIGHT + (GRID_ROWS * board_height_px) + ((GRID_ROWS + 1) * GRID_PADDING)
screen = pygame.display.set_mode((window_width, window_height))

# draw hud at top
def draw_hud():
    stats = compute_generation_stats(states)
    lines = [
        f"Gen: {generation}",
        f"Speed: {speed_ms} ms",
        f"Running: {'Yes' if active else 'No'}",
        f"Avg: {stats['avg_score']}",
        f"Max: {stats['max_score']}",
    ]
    x = GRID_PADDING
    y = 8
    for text in lines:
        screen.blit(font.render(text, True, (220, 220, 230)), (x, y))
        x += 140

# draw boards
def draw_grid():
    origins = get_board_origins(
        GRID_COLUMNS,
        GRID_ROWS,
        board_width_px,
        board_height_px,
        GRID_PADDING,
        HUD_HEIGHT,
    )
    for idx, (ox, oy) in enumerate(origins):
        state = states[idx]
        draw_board(screen, state.board, (ox, oy))
        if state.current_piece:
            draw_piece(screen, state.current_piece, state.current_position, (ox, oy))
        label = small_font.render(f"#{idx+1} {state.score}", True, (180, 180, 200))
        screen.blit(label, (ox, oy - 16))
```

**Step 6: Manual verification**

Run: `python tetris_rl.py`
Expected:
- 10 boards in 2x5 grid
- Generation increments only after all 10 end
- Speed can go down to 5ms with `+`
- `.bat` launcher still runs

**Step 7: Commit**

```bash
git add tetris_rl.py
git commit -m "feat: add 10-board generation view"
```

(If no git repo is initialized, skip this step.)

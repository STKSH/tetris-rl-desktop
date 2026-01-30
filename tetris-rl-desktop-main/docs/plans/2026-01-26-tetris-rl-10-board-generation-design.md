# Tetris RL: 10-Board Generation View

## Goals
- Run 10 game states in parallel with shared weights per generation.
- Treat one phase (generation) as the full lifecycle where all 10 boards reach game over.
- Visualize 10 boards in a 2-column x 5-row grid.
- Increase max speed (min delay down to 5ms) while keeping current key bindings.
- Preserve current .bat workflow: only modify `tetris_rl.py`.

## Non-Goals
- No new dependencies or additional executables.
- No new training algorithm (no crossover or multi-weight population).

## Constraints
- Must remain compatible with the existing Windows `.bat` launcher.
- Keep Pygame-based UI; no external assets required.

## Proposed Approach (Recommended)
- Use a single shared `weights` for all 10 boards in the same generation.
- Maintain `states: List[GameState]` length 10.
- A generation ends when all 10 boards are game over.
- On generation end: compute stats (avg, max, total lines), mutate weights once, save, increment generation, reinitialize 10 states.

## Layout & Rendering
- Grid: 2 columns x 5 rows.
- Cell size: 12px.
- Remove sidebar; instead, render a compact HUD at the top (generation, speed, running, avg, max).
- Each board gets a small label with index and score.

## Data Flow
1. Init: load weights, create 10 GameStates, reset generation stats.
2. Tick: for each active state, spawn/step using existing logic.
3. If all game over: aggregate stats, mutate weights once, save, reset 10 states, increment generation.

## Error Handling
- Use existing weight load/save guards.
- Avoid invalid draw positions by computing window size from grid layout.

## Testing Plan (Manual)
- Confirm 10 boards render in 2x5 grid.
- Verify generation increments only after all boards end.
- Verify speed can reach 5ms.
- Confirm `.bat` launcher still runs unchanged.

## Risks
- Higher CPU usage from 10 parallel boards; acceptable for intended training view.

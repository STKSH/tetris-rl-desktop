from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple

# 이 파일은 테트리스 시뮬레이터와 선형 가치함수 기반의 간단한 강화학습 루프를 포함한다.
# 휴리스틱 점수를 그대로 고정하지 않고, TD(0) 업데이트로 가중치를 학습하도록 구성되어 있다.

# 보드 셀 단위 크기와 렌더링 레이아웃을 정의한다.
BOARD_WIDTH = 10
BOARD_HEIGHT = 20
CELL_SIZE = 12
GRID_COLUMNS = 5
GRID_ROWS = 2
BOARD_COUNT = GRID_COLUMNS * GRID_ROWS
GRID_PADDING = 8
HUD_HEIGHT = 40
MIN_SPEED_MS = 1

# 강화학습 관련 하이퍼파라미터를 정의한다.
LEARNING_RATE = 0.005
DISCOUNT_FACTOR = 0.9
EPSILON_START = 0.2
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.98

# 테트리스 7종 블록의 모양과 색상을 정의한다(1이 실제 블록 셀).
PIECES = [
    {
        "name": "I",
        "shape": [
            [1, 1, 1, 1],
        ],
        "color": (0, 245, 255),
    },
    {
        "name": "O",
        "shape": [
            [1, 1],
            [1, 1],
        ],
        "color": (255, 215, 0),
    },
    {
        "name": "T",
        "shape": [
            [0, 1, 0],
            [1, 1, 1],
        ],
        "color": (155, 89, 182),
    },
    {
        "name": "S",
        "shape": [
            [0, 1, 1],
            [1, 1, 0],
        ],
        "color": (46, 204, 113),
    },
    {
        "name": "Z",
        "shape": [
            [1, 1, 0],
            [0, 1, 1],
        ],
        "color": (231, 76, 60),
    },
    {
        "name": "J",
        "shape": [
            [1, 0, 0],
            [1, 1, 1],
        ],
        "color": (52, 152, 219),
    },
    {
        "name": "L",
        "shape": [
            [0, 0, 1],
            [1, 1, 1],
        ],
        "color": (243, 156, 18),
    },
]

# 블록 이름을 보드 값(1~7)으로 매핑해 렌더링/충돌 계산에 사용한다.
PIECE_INDEX_BY_NAME = {piece["name"]: idx + 1 for idx, piece in enumerate(PIECES)}

# 보드 렌더링에 사용할 색상 팔레트를 정의한다(0은 빈칸).
COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (26, 26, 46),
    1: (0, 245, 255),
    2: (255, 215, 0),
    3: (155, 89, 182),
    4: (46, 204, 113),
    5: (231, 76, 60),
    6: (52, 152, 219),
    7: (243, 156, 18),
}

# 휴리스틱 가중치를 저장하는 파일명과 기본 가중치 세트를 정의한다.
WEIGHTS_FILENAME = "weights.json"
DEFAULT_WEIGHTS: Dict[str, float] = {
    "height": -0.510066,
    "lines": 0.760666,
    "holes": -0.35663,
    "bumpiness": -0.184483,
}


@dataclass
class GameState:
    # 각 보드(에이전트)의 상태를 하나의 데이터 묶음으로 관리한다.
    board: List[List[int]]
    current_piece: Optional[dict]
    current_position: Dict[str, int]
    next_piece: dict
    score: int
    lines_cleared: int
    level: int
    game_over: bool
    move_queue: Optional[Dict[str, int]]
    pending_features: Optional[Dict[str, float]]

    @staticmethod
    def new() -> "GameState":
        # 빈 보드와 랜덤 다음 블록을 가진 초기 상태를 생성한다.
        return GameState(
            board=create_empty_board(),
            current_piece=None,
            current_position={"x": 0, "y": 0},
            next_piece=random_piece(),
            score=0,
            lines_cleared=0,
            level=1,
            game_over=False,
            move_queue=None,
            pending_features=None,
        )


def create_empty_board() -> List[List[int]]:
    # 0으로 채워진 보드 행렬을 만든다.
    return [[0 for _ in range(BOARD_WIDTH)] for _ in range(BOARD_HEIGHT)]


def init_generation(count: int) -> List[GameState]:
    # 여러 개의 게임 보드를 한 세대로 초기화한다.
    return [GameState.new() for _ in range(count)]


def all_game_over(states: List[GameState]) -> bool:
    # 모든 보드가 종료 상태인지 여부를 반환한다.
    return all(state.game_over for state in states)


def compute_generation_stats(states: List[GameState]) -> Dict[str, int]:
    # 평균/최고 점수와 총 클리어 라인을 계산해 HUD에 제공한다.
    if not states:
        return {"avg_score": 0, "max_score": 0, "total_lines": 0}
    scores = [state.score for state in states]
    total_lines = sum(state.lines_cleared for state in states)
    avg_score = int(sum(scores) / len(scores))
    max_score = max(scores)
    return {"avg_score": avg_score, "max_score": max_score, "total_lines": total_lines}


def get_board_origins(
    columns: int,
    rows: int,
    board_width_px: int,
    board_height_px: int,
    padding: int,
    top_offset: int,
) -> List[Tuple[int, int]]:
    # 여러 보드를 화면에 격자로 배치하기 위한 좌상단 좌표를 계산한다.
    origins: List[Tuple[int, int]] = []
    total = columns * rows
    for idx in range(total):
        col = idx % columns
        row = idx // columns
        x = padding + col * (board_width_px + padding)
        y = top_offset + padding + row * (board_height_px + padding)
        origins.append((x, y))
    return origins


def clamp_speed(speed_ms: int, delta: int, min_speed: int, max_speed: int) -> int:
    # 속도 변경 요청이 최소/최대 범위를 넘지 않도록 보정한다.
    return max(min_speed, min(max_speed, speed_ms + delta))


def rotate_piece(shape: List[List[int]]) -> List[List[int]]:
    # 블록의 2차원 모양을 시계 방향 90도 회전한 결과를 반환한다.
    return [list(reversed(col)) for col in zip(*shape)]


def place_piece(
    board: List[List[int]], piece: dict, pos: Dict[str, int]
) -> List[List[int]]:
    # 현재 블록을 보드에 고정한 새 보드를 만들어 반환한다.
    new_board = [row[:] for row in board]
    piece_index = PIECE_INDEX_BY_NAME[piece["name"]]
    for r, row in enumerate(piece["shape"]):
        for c, cell in enumerate(row):
            if cell and pos["y"] + r >= 0:
                new_board[pos["y"] + r][pos["x"] + c] = piece_index
    return new_board


def clear_lines(board: List[List[int]]) -> Tuple[List[List[int]], int]:
    # 완전히 채워진 줄을 제거하고 위에서 빈 줄을 보충한다.
    new_board = [row for row in board if any(cell == 0 for cell in row)]
    lines_cleared = BOARD_HEIGHT - len(new_board)
    while len(new_board) < BOARD_HEIGHT:
        new_board.insert(0, [0] * BOARD_WIDTH)
    return new_board, lines_cleared


def random_piece() -> dict:
    # 다음에 등장할 블록을 무작위로 선택한다.
    return random.choice(PIECES)


def is_valid_position(
    board: List[List[int]], shape: List[List[int]], pos: Dict[str, int]
) -> bool:
    # 블록이 보드 경계 안에 있고 기존 블록과 충돌하지 않는지 검사한다.
    for r, row in enumerate(shape):
        for c, cell in enumerate(row):
            if not cell:
                continue
            new_x = pos["x"] + c
            new_y = pos["y"] + r
            if new_x < 0 or new_x >= BOARD_WIDTH or new_y >= BOARD_HEIGHT:
                return False
            if new_y >= 0 and board[new_y][new_x]:
                return False
    return True


def rotate_n(shape: List[List[int]], times: int) -> List[List[int]]:
    # 블록을 0~3회 회전한 결과를 계산한다.
    rotated = shape
    for _ in range(times % 4):
        rotated = rotate_piece(rotated)
    return rotated


def place_shape(
    board: List[List[int]],
    shape: List[List[int]],
    piece: dict,
    pos: Dict[str, int],
) -> List[List[int]]:
    # 주어진 모양(회전 적용된 shape)을 보드에 고정한 결과를 만든다.
    new_board = [row[:] for row in board]
    piece_index = PIECE_INDEX_BY_NAME[piece["name"]]
    for r, row in enumerate(shape):
        for c, cell in enumerate(row):
            if cell and pos["y"] + r >= 0:
                new_board[pos["y"] + r][pos["x"] + c] = piece_index
    return new_board


def get_column_heights(board: List[List[int]]) -> List[int]:
    # 각 열에서 가장 높은 블록까지의 높이를 계산한다.
    heights: List[int] = []
    for col in range(BOARD_WIDTH):
        height = 0
        for row in range(BOARD_HEIGHT):
            if board[row][col]:
                height = BOARD_HEIGHT - row
                break
        heights.append(height)
    return heights


def get_holes(board: List[List[int]]) -> int:
    # 위에 블록이 있고 아래가 빈 칸인 구멍 수를 센다.
    holes = 0
    for col in range(BOARD_WIDTH):
        block_found = False
        for row in range(BOARD_HEIGHT):
            if board[row][col]:
                block_found = True
            elif block_found:
                holes += 1
    return holes


def get_bumpiness(heights: List[int]) -> int:
    # 인접한 열 간 높이 차이를 합산해 울퉁불퉁함을 계산한다.
    bumpiness = 0
    for i in range(len(heights) - 1):
        bumpiness += abs(heights[i] - heights[i + 1])
    return bumpiness


def extract_features(board: List[List[int]], lines_cleared: int = 0) -> Dict[str, float]:
    # 보드에서 휴리스틱 특성을 추출해 선형 가치함수 입력으로 사용한다.
    heights = get_column_heights(board)
    return {
        "height": float(sum(heights)),
        "lines": float(lines_cleared),
        "holes": float(get_holes(board)),
        "bumpiness": float(get_bumpiness(heights)),
    }


def evaluate_board(
    board: List[List[int]], lines_cleared: int, weights: Dict[str, float]
) -> float:
    # 현재 보드의 특성과 가중치를 곱해 선형 가치함수를 계산한다.
    features = extract_features(board, lines_cleared)
    return sum(weights[key] * value for key, value in features.items())


def update_weights(
    weights: Dict[str, float],
    features: Dict[str, float],
    reward: float,
    next_value: float,
    learning_rate: float,
    discount: float,
) -> None:
    # TD(0) 방식으로 가중치를 업데이트해 가치 추정을 개선한다.
    current_value = sum(weights[key] * value for key, value in features.items())
    td_target = reward + discount * next_value
    td_error = td_target - current_value
    for key, value in features.items():
        weights[key] += learning_rate * td_error * value


def find_best_move(
    board: List[List[int]],
    piece: dict,
    weights: Dict[str, float],
    epsilon: float = 0.0,
) -> Optional[dict]:
    # 가능한 모든 회전과 위치를 탐색해 가치가 높은 수를 고르되 확률적으로 탐험한다.
    candidates: List[dict] = []
    for rotation in range(4):
        shape = rotate_n(piece["shape"], rotation)
        piece_width = len(shape[0])
        for x in range(-2, BOARD_WIDTH - piece_width + 3):
            y = -len(shape)
            while is_valid_position(board, shape, {"x": x, "y": y + 1}):
                y += 1
            if not is_valid_position(board, shape, {"x": x, "y": y}):
                continue
            new_board = place_shape(board, shape, piece, {"x": x, "y": y})
            cleared_board, lines = clear_lines(new_board)
            score = evaluate_board(cleared_board, lines, weights)
            candidates.append({"rotation": rotation, "x": x, "score": score})

    if not candidates:
        return None
    if epsilon > 0.0 and random.random() < epsilon:
        return random.choice(candidates)
    return max(candidates, key=lambda candidate: candidate["score"])


def spawn_piece(state: GameState, weights: Dict[str, float], epsilon: float) -> None:
    # 다음 블록을 현재 블록으로 가져오고 자동 이동 계획을 준비한다.
    if state.pending_features is None:
        state.pending_features = extract_features(state.board)
    new_piece = state.next_piece
    start_x = (BOARD_WIDTH - len(new_piece["shape"][0])) // 2
    start_y = -1
    if not is_valid_position(
        state.board, new_piece["shape"], {"x": start_x, "y": start_y + 1}
    ):
        if state.pending_features:
            update_weights(
                weights,
                state.pending_features,
                reward=-1.0,
                next_value=0.0,
                learning_rate=LEARNING_RATE,
                discount=DISCOUNT_FACTOR,
            )
        state.game_over = True
        return
    best_move = find_best_move(state.board, new_piece, weights, epsilon)
    if best_move:
        state.move_queue = {
            "rotations": best_move["rotation"],
            "target_x": best_move["x"],
        }
    state.current_piece = new_piece
    state.current_position = {"x": start_x, "y": start_y}
    state.next_piece = random_piece()


def step_game(state: GameState, weights: Dict[str, float]) -> None:
    # 자동 이동 계획을 처리하고 블록을 한 칸 내려 게임을 진행한다.
    if state.game_over or not state.current_piece:
        return

    new_piece = state.current_piece
    new_pos = {"x": state.current_position["x"], "y": state.current_position["y"]}

    if state.move_queue:
        rotations = state.move_queue["rotations"]
        target_x = state.move_queue["target_x"]

        if rotations > 0:
            test_shape = new_piece["shape"]
            for _ in range(rotations):
                test_shape = rotate_piece(test_shape)
            if is_valid_position(state.board, test_shape, new_pos):
                new_piece = {**new_piece, "shape": test_shape}
            state.move_queue["rotations"] = 0

        if new_pos["x"] < target_x:
            if is_valid_position(
                state.board,
                new_piece["shape"],
                {"x": new_pos["x"] + 1, "y": new_pos["y"]},
            ):
                new_pos["x"] += 1
        elif new_pos["x"] > target_x:
            if is_valid_position(
                state.board,
                new_piece["shape"],
                {"x": new_pos["x"] - 1, "y": new_pos["y"]},
            ):
                new_pos["x"] -= 1

        if new_pos["x"] == target_x:
            state.move_queue = None

    next_pos = {"x": new_pos["x"], "y": new_pos["y"] + 1}
    if is_valid_position(state.board, new_piece["shape"], next_pos):
        state.current_piece = new_piece
        state.current_position = next_pos
        return

    new_board = place_shape(state.board, new_piece["shape"], new_piece, new_pos)
    cleared_board, lines_cleared = clear_lines(new_board)
    line_scores = [0, 100, 300, 500, 800]
    score_gain = line_scores[lines_cleared] * state.level

    reward = float(lines_cleared)
    next_value = evaluate_board(cleared_board, 0, weights)
    if state.pending_features:
        update_weights(
            weights,
            state.pending_features,
            reward,
            next_value,
            LEARNING_RATE,
            DISCOUNT_FACTOR,
        )
    state.pending_features = None

    state.board = cleared_board
    state.current_piece = None
    state.current_position = {"x": 0, "y": 0}
    state.score += score_gain
    state.lines_cleared += lines_cleared
    state.level = state.lines_cleared // 10 + 1


def init_population_weights(
    base_weights: Dict[str, float], count: int
) -> List[Dict[str, float]]:
    # 기본 가중치를 복제해 여러 보드가 공유하도록 초기화한다.
    return [base_weights.copy() for _ in range(max(count, 0))]


def average_weights(population_weights: List[Dict[str, float]]) -> Dict[str, float]:
    # 여러 보드의 가중치를 평균내어 저장용 대표값을 만든다.
    if not population_weights:
        return DEFAULT_WEIGHTS.copy()
    avg = {key: 0.0 for key in DEFAULT_WEIGHTS}
    for weights in population_weights:
        for key in avg:
            avg[key] += weights.get(key, 0.0)
    for key in avg:
        avg[key] /= len(population_weights)
    return avg


def weights_file_path(path: Optional[Path] = None) -> Path:
    # 가중치 파일 경로를 결정하며, 지정이 없으면 스크립트 위치를 사용한다.
    if path is None:
        return Path(__file__).resolve().parent / WEIGHTS_FILENAME
    return Path(path)


def load_weights(path: Optional[Path] = None) -> Dict[str, float]:
    # 가중치 파일을 읽고 문제 발생 시 기본값을 사용한다.
    file_path = weights_file_path(path)
    try:
        data = json.loads(file_path.read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return DEFAULT_WEIGHTS.copy()
    if not isinstance(data, dict):
        return DEFAULT_WEIGHTS.copy()
    weights = DEFAULT_WEIGHTS.copy()
    for key in weights:
        value = data.get(key)
        if isinstance(value, (int, float)):
            weights[key] = float(value)
    return weights


def save_weights(weights: Dict[str, float], path: Optional[Path] = None) -> None:
    # 가중치 파일에 저장하되 실패 시에는 무시한다.
    file_path = weights_file_path(path)
    try:
        file_path.write_text(json.dumps(weights, indent=2, sort_keys=True))
    except OSError:
        pass


def init_pygame():
    # pygame 모듈을 초기화하고 반환한다.
    import pygame

    pygame.init()
    return pygame


def draw_board(
    screen, board: List[List[int]], origin: Tuple[int, int] = (0, 0)
) -> None:
    # 보드 배경과 고정된 블록을 화면에 그린다.
    import pygame

    ox, oy = origin
    for r in range(BOARD_HEIGHT):
        for c in range(BOARD_WIDTH):
            cell = board[r][c]
            color = COLORS.get(cell, COLORS[0])
            rect = pygame.Rect(
                ox + c * CELL_SIZE,
                oy + r * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE,
            )
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (30, 30, 50), rect, 1)


def draw_piece(
    screen, piece: dict, pos: Dict[str, int], origin: Tuple[int, int] = (0, 0)
) -> None:
    # 현재 낙하 중인 블록을 렌더링하고 화면 밖(y<0) 부분은 건너뛴다.
    import pygame

    ox, oy = origin
    color = piece["color"]
    for r, row in enumerate(piece["shape"]):
        for c, cell in enumerate(row):
            if not cell:
                continue
            y = pos["y"] + r
            if y < 0:
                continue
            rect = pygame.Rect(
                ox + (pos["x"] + c) * CELL_SIZE,
                oy + y * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE,
            )
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (30, 30, 50), rect, 1)


def draw_next_piece(screen, piece: dict, origin: Tuple[int, int]) -> None:
    # 다음 블록 미리보기 영역을 간단한 크기로 그린다.
    import pygame

    ox, oy = origin
    size = 16
    for r, row in enumerate(piece["shape"]):
        for c, cell in enumerate(row):
            if not cell:
                continue
            rect = pygame.Rect(ox + c * size, oy + r * size, size, size)
            pygame.draw.rect(screen, piece["color"], rect)
            pygame.draw.rect(screen, (30, 30, 50), rect, 1)


def run():
    # pygame 루프를 실행해 시뮬레이션과 렌더링을 수행한다.
    pygame = init_pygame()

    board_width_px = BOARD_WIDTH * CELL_SIZE
    board_height_px = BOARD_HEIGHT * CELL_SIZE
    window_width = (GRID_COLUMNS * board_width_px) + ((GRID_COLUMNS + 1) * GRID_PADDING)
    window_height = (
        HUD_HEIGHT + (GRID_ROWS * board_height_px) + ((GRID_ROWS + 1) * GRID_PADDING)
    )
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Tetris RL (Pygame)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 14)
    small_font = pygame.font.SysFont("monospace", 12)

    base_weights = load_weights()
    population_weights = init_population_weights(base_weights, BOARD_COUNT)
    states = init_generation(BOARD_COUNT)
    is_running = False
    speed_ms = 100
    min_speed_ms = MIN_SPEED_MS
    max_speed_ms = 1000
    generation = 1
    accumulator = 0.0
    high_record = 0
    epsilon = EPSILON_START
    origins = get_board_origins(
        GRID_COLUMNS,
        GRID_ROWS,
        board_width_px,
        board_height_px,
        GRID_PADDING,
        HUD_HEIGHT,
    )

    def draw_hud() -> None:
        # 상단 HUD에 세대/속도/점수 통계를 출력한다.
        nonlocal high_record
        stats = compute_generation_stats(states)
        high_record = max(high_record, stats["max_score"])
        line1 = (
            f"Gen: {generation}  Speed: {speed_ms}ms  Run: {'Yes' if is_running else 'No'}"
        )
        line2 = (
            "Avg: {avg}  Max: {max}  High: {high}  Eps: {eps:.2f}".format(
                avg=stats["avg_score"],
                max=stats["max_score"],
                high=high_record,
                eps=epsilon,
            )
        )
        screen.blit(font.render(line1, True, (220, 220, 230)), (GRID_PADDING, 8))
        screen.blit(small_font.render(line2, True, (200, 200, 210)), (GRID_PADDING, 24))

    def draw_grid() -> None:
        # 모든 보드를 격자 형태로 렌더링하고 각 보드 점수를 표시한다.
        for idx, (ox, oy) in enumerate(origins):
            state = states[idx]
            draw_board(screen, state.board, (ox, oy))
            if state.current_piece:
                draw_piece(
                    screen, state.current_piece, state.current_position, (ox, oy)
                )
            label = small_font.render(
                f"#{idx + 1} {state.score}", True, (180, 180, 200)
            )
            screen.blit(label, (ox + 4, oy + 4))

    running = True
    while running:
        dt = clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    is_running = not is_running
                elif event.key == pygame.K_r:
                    states = init_generation(BOARD_COUNT)
                    accumulator = 0.0
                elif event.key in (
                    pygame.K_MINUS,
                    pygame.K_UNDERSCORE,
                    pygame.K_LEFTBRACKET,
                ):
                    speed_ms = clamp_speed(speed_ms, 20, min_speed_ms, max_speed_ms)
                elif event.key in (
                    pygame.K_EQUALS,
                    pygame.K_PLUS,
                    pygame.K_RIGHTBRACKET,
                ):
                    speed_ms = clamp_speed(speed_ms, -20, min_speed_ms, max_speed_ms)

        if is_running:
            accumulator += dt
            while accumulator >= speed_ms:
                if all_game_over(states):
                    generation += 1
                    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
                    save_weights(average_weights(population_weights))
                    states = init_generation(BOARD_COUNT)
                else:
                    for idx, state in enumerate(states):
                        if state.game_over:
                            continue
                        weights = population_weights[idx]
                        if state.current_piece is None:
                            spawn_piece(state, weights, epsilon)
                        else:
                            step_game(state, weights)
                accumulator -= speed_ms

        screen.fill((10, 10, 20))
        draw_grid()
        draw_hud()
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    run()

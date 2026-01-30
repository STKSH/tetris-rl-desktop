from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple

# 이 파일은 간단한 테트리스 시뮬레이터와 휴리스틱 기반 자동 플레이를 한 파일에 담고 있다.
# "학습"이라기보다는 가중치(휴리스틱)를 무작위로 변이시키며 성능 변화를 관찰하는 구조다.

# 보드 크기(셀 단위)와 렌더링 관련 설정값들
BOARD_WIDTH = 10
BOARD_HEIGHT = 20
CELL_SIZE = 12
GRID_COLUMNS = 5
GRID_ROWS = 2
BOARD_COUNT = GRID_COLUMNS * GRID_ROWS
GRID_PADDING = 8
HUD_HEIGHT = 40
MIN_SPEED_MS = 1

# 테트리스 7종 블록 정의 (shape의 1이 실제 블록 칸)
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
    }
]

# 보드에 표시할 숫자 인덱스(0은 빈칸) 매핑
PIECE_INDEX_BY_NAME = {piece["name"]: idx + 1 for idx, piece in enumerate(PIECES)}

# 렌더링용 색상 테이블
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

# 가중치 파일 및 기본 휴리스틱 파라미터
# 점수 계산 = (-0.5 X 높이) + (0.7 X 줄) + (-0.3 X 구멍) + (-0.15 X 굴곡)
WEIGHTS_FILENAME = "weights.json"
DEFAULT_WEIGHTS: Dict[str, float] = {
    "height": -0.5,
    "lines": 0.7,
    "holes": -0.3,
    "bumpiness": -0.15
}


@dataclass
class GameState:
    # 단일 보드(에이전트)의 상태 묶음
    # board: 보드 상태(0은 빈칸, 1~7은 블록 종류)
    # current_piece/current_position: 현재 낙하 중인 블록과 좌상단 기준 좌표
    # next_piece: 다음에 등장할 블록
    # score/lines_cleared/level: 점수 및 진행 상황
    # game_over: 게임 종료 여부
    # move_queue: 자동 플레이가 계산한 회전/이동 목표
    board: List[List[int]]
    current_piece: Optional[dict]
    current_position: Dict[str, int]
    next_piece: dict
    score: int
    lines_cleared: int
    level: int
    game_over: bool
    move_queue: Optional[Dict[str, int]]

    @staticmethod
    def new() -> "GameState":
        # 새로운 게임 상태 초기화
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
        )


def create_empty_board() -> List[List[int]]:
    # 0으로 가득 찬 빈 보드 생성
    return [[0 for _ in range(BOARD_WIDTH)] for _ in range(BOARD_HEIGHT)]


def init_generation(count: int) -> List[GameState]:
    # 여러 보드를 한 세대(동시 시뮬레이션)로 초기화
    return [GameState.new() for _ in range(count)]


def all_game_over(states: List[GameState]) -> bool:
    # 모든 보드가 종료되었는지 확인
    return all(state.game_over for state in states)


def compute_generation_stats(states: List[GameState]) -> Dict[str, int]:
    # 세대 전체 통계를 계산 (평균/최고 점수, 총 클리어 라인)
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
    # 여러 보드를 격자 형태로 그리기 위한 좌상단 좌표 목록 생성
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
    # 속도 변경 시 최소/최대 범위를 벗어나지 않게 보정
    return max(min_speed, min(max_speed, speed_ms + delta))


def rotate_piece(shape: List[List[int]]) -> List[List[int]]:
    # 2D 배열을 시계 방향 90도 회전
    return [list(reversed(col)) for col in zip(*shape)]


def place_piece(
    board: List[List[int]], piece: dict, pos: Dict[str, int]
) -> List[List[int]]:
    # 현재 블록을 보드에 실제로 "고정"한 상태의 보드를 생성
    new_board = [row[:] for row in board]
    piece_index = PIECE_INDEX_BY_NAME[piece["name"]]
    for r, row in enumerate(piece["shape"]):
        for c, cell in enumerate(row):
            if cell and pos["y"] + r >= 0:
                new_board[pos["y"] + r][pos["x"] + c] = piece_index
    return new_board


def clear_lines(board: List[List[int]]) -> Tuple[List[List[int]], int]:
    # 한 줄이 가득 찬 행을 제거하고 위에서 빈 줄을 채우는 로직
    new_board = [row for row in board if any(cell == 0 for cell in row)]
    lines_cleared = BOARD_HEIGHT - len(new_board)
    while len(new_board) < BOARD_HEIGHT:
        new_board.insert(0, [0] * BOARD_WIDTH)
    return new_board, lines_cleared


def random_piece() -> dict:
    # 다음 블록을 무작위로 선택
    return random.choice(PIECES)


def is_valid_position(
    board: List[List[int]], shape: List[List[int]], pos: Dict[str, int]
) -> bool:
    # 블록이 보드 범위 내에 있고 기존 블록과 충돌하지 않는지 검사
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
    # 블록을 times회 90도 회전 (4회 주기)
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
    # 특정 shape(회전 적용된 모양)을 보드에 고정한 결과를 생성
    new_board = [row[:] for row in board]
    piece_index = PIECE_INDEX_BY_NAME[piece["name"]]
    for r, row in enumerate(shape):
        for c, cell in enumerate(row):
            if cell and pos["y"] + r >= 0:
                new_board[pos["y"] + r][pos["x"] + c] = piece_index
    return new_board


def get_column_heights(board: List[List[int]]) -> List[int]:
    # 각 열의 높이(최상단 블록 기준)를 계산
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
    # 구멍(위에 블록이 있고 아래가 빈칸인 상태)의 개수 계산
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
    # 인접 열 높이 차이의 합 (울퉁불퉁함)
    bumpiness = 0
    for i in range(len(heights) - 1):
        bumpiness += abs(heights[i] - heights[i + 1])
    return bumpiness


def evaluate_board(
    board: List[List[int]], lines_cleared: int, weights: Dict[str, float]
) -> float:
    # 보드 상태를 휴리스틱으로 평가
    # aggregate_height: 전체 높이 합(낮을수록 좋음)
    # lines_cleared: 이번 수로 지워진 라인 수(클수록 좋음)
    # holes: 구멍 개수(적을수록 좋음)
    # bumpiness: 열 간 높이 차이(적을수록 좋음)
    heights = get_column_heights(board)
    aggregate_height = sum(heights)
    holes = get_holes(board)
    bumpiness = get_bumpiness(heights)
    return (
        weights["height"] * aggregate_height
        + weights["lines"] * lines_cleared
        + weights["holes"] * holes
        + weights["bumpiness"] * bumpiness
    )


def find_best_move(
    board: List[List[int]], piece: dict, weights: Dict[str, float]
) -> Optional[dict]:
    # 현재 보드와 블록에 대해 가능한 모든 회전/가로 위치를 탐색
    # 각 후보 위치에서 블록을 "떨어뜨린" 결과를 평가하여 최고 점수 선택
    best_move: Optional[dict] = None
    best_score = float("-inf")
    for rotation in range(4):
        shape = rotate_n(piece["shape"], rotation)
        piece_width = len(shape[0])
        for x in range(-2, BOARD_WIDTH - piece_width + 3):
            # 보드 위쪽(음수 y)에서 시작해 바닥까지 내리기
            y = -len(shape)
            while is_valid_position(board, shape, {"x": x, "y": y + 1}):
                y += 1
            if not is_valid_position(board, shape, {"x": x, "y": y}):
                continue
            new_board = place_shape(board, shape, piece, {"x": x, "y": y})
            cleared_board, lines = clear_lines(new_board)
            score = evaluate_board(cleared_board, lines, weights)
            if score > best_score:
                best_score = score
                best_move = {"rotation": rotation, "x": x, "score": score}
    return best_move


def spawn_piece(state: GameState, weights: Dict[str, float]) -> None:
    # 다음 블록을 현재 블록으로 가져와 스폰하고, 자동 이동 계획을 계산
    new_piece = state.next_piece
    start_x = (BOARD_WIDTH - len(new_piece["shape"][0])) // 2
    start_y = -1
    if not is_valid_position(
        state.board, new_piece["shape"], {"x": start_x, "y": start_y + 1}
    ):
        # 시작 위치가 막혀 있으면 게임 오버
        state.game_over = True
        return
    best_move = find_best_move(state.board, new_piece, weights)
    if best_move:
        state.move_queue = {
            "rotations": best_move["rotation"],
            "target_x": best_move["x"],
        }
    state.current_piece = new_piece
    state.current_position = {"x": start_x, "y": start_y}
    state.next_piece = random_piece()


def step_game(state: GameState, weights: Dict[str, float]) -> None:
    # 한 프레임(틱)에서 게임 상태를 한 단계 진행
    if state.game_over or not state.current_piece:
        return

    new_piece = state.current_piece
    new_pos = {"x": state.current_position["x"], "y": state.current_position["y"]}

    if state.move_queue:
        # 자동 이동: 회전 우선 처리 후 좌우 이동을 한 칸씩 수행
        rotations = state.move_queue["rotations"]
        target_x = state.move_queue["target_x"]

        if rotations > 0:
            test_shape = new_piece["shape"]
            for _ in range(rotations):
                test_shape = rotate_piece(test_shape)
            if is_valid_position(state.board, test_shape, new_pos):
                new_piece = {**new_piece, "shape": test_shape}
            # 회전은 한 번에 반영하고 move_queue에서 제거
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
            # 목표 x에 도달하면 이동 큐 해제
            state.move_queue = None

    next_pos = {"x": new_pos["x"], "y": new_pos["y"] + 1}
    if is_valid_position(state.board, new_piece["shape"], next_pos):
        # 아래로 내려갈 수 있으면 위치만 갱신
        state.current_piece = new_piece
        state.current_position = next_pos
        return

    # 더 내려갈 수 없으면 블록을 고정하고 라인 정리
    new_board = place_shape(state.board, new_piece["shape"], new_piece, new_pos)
    cleared_board, lines_cleared = clear_lines(new_board)
    line_scores = [0, 100, 300, 500, 800]
    score_gain = line_scores[lines_cleared] * state.level

    state.board = cleared_board
    state.current_piece = None
    state.current_position = {"x": 0, "y": 0}
    state.score += score_gain
    state.lines_cleared += lines_cleared
    state.level = state.lines_cleared // 10 + 1


def mutate_weights(weights: Dict[str, float]) -> Dict[str, float]:
    # 가중치에 작은 랜덤 변이를 적용
    # 현재는 성능 기반 선택 없이 무작위로만 바뀐다.
    return {
        "height": weights["height"] + (random.random() - 0.5) * 0.1,
        "lines": weights["lines"] + (random.random() - 0.5) * 0.1,
        "holes": weights["holes"] + (random.random() - 0.5) * 0.1,
        "bumpiness": weights["bumpiness"] + (random.random() - 0.5) * 0.1,
    }


def weights_file_path(path: Optional[Path] = None) -> Path:
    # 가중치 파일 경로를 결정 (기본은 스크립트와 같은 폴더)
    if path is None:
        return Path(__file__).resolve().parent / WEIGHTS_FILENAME
    return Path(path)


def load_weights(path: Optional[Path] = None) -> Dict[str, float]:
    # 가중치 파일을 읽고, 문제가 있으면 기본값 사용
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
    # 가중치 파일 저장 (실패 시 무시)
    file_path = weights_file_path(path)
    try:
        file_path.write_text(json.dumps(weights, indent=2, sort_keys=True))
    except OSError:
        pass


def init_pygame():
    # pygame 초기화
    import pygame

    pygame.init()
    return pygame


def draw_board(
    screen, board: List[List[int]], origin: Tuple[int, int] = (0, 0)
) -> None:
    # 보드 배경과 고정된 블록을 렌더링
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
    # 현재 낙하 중인 블록 렌더링 (y<0인 부분은 화면 밖이라 생략)
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
    # 다음 블록 미리보기 렌더링
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
    # 메인 루프: pygame 초기화 → 입력 처리 → 시뮬레이션 → 렌더링
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

    weights = load_weights()
    states = init_generation(BOARD_COUNT)
    is_running = False
    speed_ms = 100
    min_speed_ms = MIN_SPEED_MS
    max_speed_ms = 1000
    generation = 1
    accumulator = 0.0
    high_record = 0
    origins = get_board_origins(
        GRID_COLUMNS,
        GRID_ROWS,
        board_width_px,
        board_height_px,
        GRID_PADDING,
        HUD_HEIGHT,
    )

    def draw_hud() -> None:
        # 상단 HUD(세대/속도/평균/최고/누적 최고) 출력
        nonlocal high_record
        stats = compute_generation_stats(states)
        high_record = max(high_record, stats["max_score"])
        line1 = f"Gen: {generation}  Speed: {speed_ms}ms  Run: {'Yes' if is_running else 'No'}"
        line2 = (
            f"Avg: {stats['avg_score']}  Max: {stats['max_score']}  High: {high_record}"
        )
        screen.blit(font.render(line1, True, (220, 220, 230)), (GRID_PADDING, 8))
        screen.blit(small_font.render(line2, True, (200, 200, 210)), (GRID_PADDING, 24))

    def draw_grid() -> None:
        # 여러 보드(에이전트)를 격자 형태로 렌더링
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
                    # 스페이스: 실행/정지 토글
                    is_running = not is_running
                elif event.key == pygame.K_r:
                    # R: 현재 세대 초기화
                    states = init_generation(BOARD_COUNT)
                    accumulator = 0.0
                elif event.key in (
                    pygame.K_MINUS,
                    pygame.K_UNDERSCORE,
                    pygame.K_LEFTBRACKET,
                ):
                    # - 또는 [: 속도 느리게(딜레이 증가)
                    speed_ms = clamp_speed(speed_ms, 20, min_speed_ms, max_speed_ms)
                elif event.key in (
                    pygame.K_EQUALS,
                    pygame.K_PLUS,
                    pygame.K_RIGHTBRACKET,
                ):
                    # + 또는 ]: 속도 빠르게(딜레이 감소)
                    speed_ms = clamp_speed(speed_ms, -20, min_speed_ms, max_speed_ms)

        if is_running:
            accumulator += dt
            while accumulator >= speed_ms:
                if all_game_over(states):
                    # 모든 보드가 끝나면 다음 세대로 넘어감
                    generation += 1
                    weights = mutate_weights(weights)
                    save_weights(weights)
                    states = init_generation(BOARD_COUNT)
                else:
                    for state in states:
                        if state.game_over:
                            continue
                        if state.current_piece is None:
                            # 새로운 블록을 스폰하고 자동 이동 계획 계산
                            spawn_piece(state, weights)
                        else:
                            # 현재 블록을 한 칸씩 진행
                            step_game(state, weights)
                accumulator -= speed_ms

        screen.fill((10, 10, 20))
        draw_grid()
        draw_hud()
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    run()

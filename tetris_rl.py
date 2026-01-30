from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple

# ==========================================
# [설정] 강화학습 하이퍼파라미터 & 게임 설정
# ==========================================
BOARD_WIDTH = 10
BOARD_HEIGHT = 20
CELL_SIZE = 12
GRID_COLUMNS = 5
GRID_ROWS = 2
BOARD_COUNT = GRID_COLUMNS * GRID_ROWS
GRID_PADDING = 8
HUD_HEIGHT = 40
MIN_SPEED_MS = 1

# RL 학습률 및 할인율
LEARNING_RATE = 0.001       # 학습 속도 (너무 빠르면 불안정하므로 0.001 추천)
DISCOUNT_FACTOR = 0.95      # 미래 가치 반영 비율
EPSILON_START = 0.2         # 초기 탐험 확률
EPSILON_MIN = 0.01          # 최소 탐험 확률
EPSILON_DECAY = 0.995       # 탐험 감소 비율

# 테트리스 블록 정의
PIECES = [
    {"name": "I", "shape": [[1, 1, 1, 1]], "color": (0, 245, 255)},
    {"name": "O", "shape": [[1, 1], [1, 1]], "color": (255, 215, 0)},
    {"name": "T", "shape": [[0, 1, 0], [1, 1, 1]], "color": (155, 89, 182)},
    {"name": "S", "shape": [[0, 1, 1], [1, 1, 0]], "color": (46, 204, 113)},
    {"name": "Z", "shape": [[1, 1, 0], [0, 1, 1]], "color": (231, 76, 60)},
    {"name": "J", "shape": [[1, 0, 0], [1, 1, 1]], "color": (52, 152, 219)},
    {"name": "L", "shape": [[0, 0, 1], [1, 1, 1]], "color": (243, 156, 18)},
]

PIECE_INDEX_BY_NAME = {piece["name"]: idx + 1 for idx, piece in enumerate(PIECES)}
COLORS = {
    0: (26, 26, 46), 1: (0, 245, 255), 2: (255, 215, 0), 3: (155, 89, 182),
    4: (46, 204, 113), 5: (231, 76, 60), 6: (52, 152, 219), 7: (243, 156, 18),
}

WEIGHTS_FILENAME = "weights.json"
# 초기 가중치 (이 값에서 시작해 학습으로 변해감)
DEFAULT_WEIGHTS: Dict[str, float] = {
    "height": -0.51,
    "lines": 0.76,
    "holes": -0.36,
    "bumpiness": -0.18
}

@dataclass
class GameState:
    board: List[List[int]]
    current_piece: Optional[dict]
    current_position: Dict[str, int]
    next_piece: dict
    score: int
    lines_cleared: int
    level: int
    game_over: bool
    move_queue: Optional[Dict[str, int]]
    # 강화학습용: 직전 행동의 특성값들을 저장해두는 변수
    pending_features: Optional[Dict[str, float]]

    @staticmethod
    def new() -> "GameState":
        return GameState(
            board=[[0] * BOARD_WIDTH for _ in range(BOARD_HEIGHT)],
            current_piece=None,
            current_position={"x": 0, "y": 0},
            next_piece=random.choice(PIECES),
            score=0,
            lines_cleared=0,
            level=1,
            game_over=False,
            move_queue=None,
            pending_features=None,
        )

# ==========================================
# [핵심] 강화학습 로직 (RL Logic)
# ==========================================

def extract_features(board: List[List[int]], lines_cleared: int = 0) -> Dict[str, float]:
    # 보드에서 특징 4가지를 추출
    heights = []
    for col in range(BOARD_WIDTH):
        h = 0
        for row in range(BOARD_HEIGHT):
            if board[row][col]:
                h = BOARD_HEIGHT - row
                break
        heights.append(h)
    
    holes = 0
    for col in range(BOARD_WIDTH):
        block_found = False
        for row in range(BOARD_HEIGHT):
            if board[row][col]:
                block_found = True
            elif block_found:
                holes += 1
    
    bumpiness = sum(abs(heights[i] - heights[i+1]) for i in range(len(heights)-1))
    
    return {
        "height": float(sum(heights)),
        "lines": float(lines_cleared),
        "holes": float(holes),
        "bumpiness": float(bumpiness)
    }

def evaluate_board(board: List[List[int]], lines_cleared: int, weights: Dict[str, float]) -> float:
    # 현재 가중치로 보드 점수 계산 (Linear Value Function)
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
    # [RL 핵심] TD Learning 업데이트
    current_value = sum(weights[key] * val for key, val in features.items())
    
    # TD Target: 실제 보상 + 미래 예측값
    td_target = reward + discount * next_value
    td_error = td_target - current_value
    
    # [안전장치 1] 학습 폭발 방지 (Clipping)
    td_error = max(-10.0, min(10.0, td_error))
    
    for key, value in features.items():
        weights[key] += learning_rate * td_error * value
        
        # [안전장치 2] 가중치 부호 강제 (탑 쌓기 방지)
        if key == "lines":
            # 줄 지우기는 무조건 좋은 것 (+)
            weights[key] = max(0.001, weights[key])
        else:
            # 높이, 구멍, 굴곡은 무조건 나쁜 것 (-)
            weights[key] = min(-0.001, weights[key])

def find_best_move(
    board: List[List[int]],
    piece: dict,
    weights: Dict[str, float],
    epsilon: float = 0.0
) -> Optional[dict]:
    # 최적의 수 찾기 + 탐험(Epsilon)
    candidates = []
    
    for rotation in range(4):
        shape = rotate_n(piece["shape"], rotation)
        piece_width = len(shape[0])
        
        for x in range(-2, BOARD_WIDTH - piece_width + 3):
            y = -len(shape)
            # 바닥까지 내리기
            while is_valid_position(board, shape, {"x": x, "y": y + 1}):
                y += 1
            
            if not is_valid_position(board, shape, {"x": x, "y": y}):
                continue
            
            # 가상으로 둬보기
            new_board = place_shape(board, shape, piece, {"x": x, "y": y})
            cleared_board, lines = clear_lines(new_board)
            score = evaluate_board(cleared_board, lines, weights)
            
            # [안전장치 3] 점수가 정상적인 숫자일 때만 추가
            if math.isfinite(score):
                candidates.append({"rotation": rotation, "x": x, "score": score})
    
    if not candidates:
        return None

    # [탐험] 랜덤 행동
    if epsilon > 0.0 and random.random() < epsilon:
        return random.choice(candidates)
    
    # [활용] 최고 점수 행동 (동점자 랜덤 처리 -> 왼쪽 쏠림 해결)
    max_score = max(candidates, key=lambda c: c["score"])["score"]
    best_moves = [c for c in candidates if c["score"] == max_score]
    
    if not best_moves:
        return max(candidates, key=lambda c: c["score"])
        
    return random.choice(best_moves)

# ==========================================
# 게임 로직 (Helper Functions)
# ==========================================
def rotate_n(shape, times):
    res = shape
    for _ in range(times % 4):
        res = [list(reversed(col)) for col in zip(*res)]
    return res

def is_valid_position(board, shape, pos):
    for r, row in enumerate(shape):
        for c, cell in enumerate(row):
            if not cell: continue
            nx, ny = pos["x"] + c, pos["y"] + r
            if nx < 0 or nx >= BOARD_WIDTH or ny >= BOARD_HEIGHT:
                return False
            if ny >= 0 and board[ny][nx]:
                return False
    return True

def place_shape(board, shape, piece, pos):
    new_board = [row[:] for row in board]
    idx = PIECE_INDEX_BY_NAME[piece["name"]]
    for r, row in enumerate(shape):
        for c, cell in enumerate(row):
            if cell and pos["y"] + r >= 0:
                new_board[pos["y"] + r][pos["x"] + c] = idx
    return new_board

def clear_lines(board):
    new_board = [row for row in board if any(c == 0 for c in row)]
    lines = BOARD_HEIGHT - len(new_board)
    while len(new_board) < BOARD_HEIGHT:
        new_board.insert(0, [0] * BOARD_WIDTH)
    return new_board, lines

def spawn_piece(state: GameState, weights: Dict[str, float], epsilon: float):
    # 직전 수에 대한 보상 계산을 위해 현재 상태 특성 저장
    if state.pending_features is None:
        state.pending_features = extract_features(state.board)
        
    new_piece = state.next_piece
    start_x = (BOARD_WIDTH - len(new_piece["shape"][0])) // 2
    start_y = -1
    
    # 스폰하자마자 죽는 경우 (Game Over)
    if not is_valid_position(state.board, new_piece["shape"], {"x": start_x, "y": start_y + 1}):
        # [수정] 죽었을 때 벌점을 -10 -> -100으로 대폭 강화!
        if state.pending_features:
            update_weights(
                weights, state.pending_features, 
                reward=-100.0,  # <-- 더 강력한 처벌
                next_value=0.0, 
                learning_rate=LEARNING_RATE, discount=DISCOUNT_FACTOR
            )
        state.game_over = True
        return

    # 최적의 수 찾기
    best = find_best_move(state.board, new_piece, weights, epsilon)
    if best:
        state.move_queue = {"rotations": best["rotation"], "target_x": best["x"]}
    
    state.current_piece = new_piece
    state.current_position = {"x": start_x, "y": start_y}
    state.next_piece = random.choice(PIECES)

    # 최적의 수 찾기
    best = find_best_move(state.board, new_piece, weights, epsilon)
    if best:
        state.move_queue = {"rotations": best["rotation"], "target_x": best["x"]}
    
    state.current_piece = new_piece
    state.current_position = {"x": start_x, "y": start_y}
    state.next_piece = random.choice(PIECES)

def step_game(state: GameState, weights: Dict[str, float]):
    if state.game_over or not state.current_piece:
        return

    # 자동 이동 실행
    new_piece = state.current_piece
    new_pos = state.current_position.copy()
    
    if state.move_queue:
        # 회전
        if state.move_queue["rotations"] > 0:
            test_shape = rotate_n(new_piece["shape"], 1)
            if is_valid_position(state.board, test_shape, new_pos):
                new_piece = {**new_piece, "shape": test_shape}
            state.move_queue["rotations"] -= 1
        
        # 좌우 이동
        tx = state.move_queue["target_x"]
        if new_pos["x"] < tx:
            if is_valid_position(state.board, new_piece["shape"], {"x": new_pos["x"]+1, "y": new_pos["y"]}):
                new_pos["x"] += 1
        elif new_pos["x"] > tx:
            if is_valid_position(state.board, new_piece["shape"], {"x": new_pos["x"]-1, "y": new_pos["y"]}):
                new_pos["x"] -= 1
        
        if new_pos["x"] == tx and state.move_queue["rotations"] <= 0:
            state.move_queue = None
# 중력 (아래로 이동)
    next_pos = {"x": new_pos["x"], "y": new_pos["y"] + 1}
    if is_valid_position(state.board, new_piece["shape"], next_pos):
        state.current_piece = new_piece
        state.current_position = next_pos
    else:
        # 바닥에 닿음 -> 고정 (Lock)
        final_board = place_shape(state.board, new_piece["shape"], new_piece, new_pos)
        cleared_board, lines = clear_lines(final_board)
        
        # 점수 계산
        score_add = [0, 100, 300, 500, 800][lines] * state.level
        state.score += score_add
        state.lines_cleared += lines
        state.level = state.lines_cleared // 10 + 1
        
        # [수정] 보상 체계 개선 (Reward Shaping)
        # 1. 줄 지우기 보상
        reward = float(lines * lines * 20)
        
        # 2. [핵심] 높이 페널티 추가! (높을수록 실시간으로 감점)
        # 현재 보드의 높이 특성을 가져와서 벌점을 매김
        current_features = extract_features(cleared_board, lines)
        total_height = current_features["height"]
        total_holes = current_features["holes"]
        
        # "높이가 높거나 구멍이 많으면 점수를 깎는다"
        reward -= (total_height * 0.1)  # 높이 1칸당 -0.1점
        reward -= (total_holes * 0.5)   # 구멍 1개당 -0.5점
        
        next_value = evaluate_board(cleared_board, 0, weights)
        
        if state.pending_features:
            update_weights(
                weights, state.pending_features,
                reward, next_value, LEARNING_RATE, DISCOUNT_FACTOR
            )
        
        # 상태 업데이트
        state.board = cleared_board
        state.current_piece = None
        state.pending_features = None

# ==========================================
# 파일 입출력 및 메인 루프
# ==========================================
def weights_file_path():
    return Path(__file__).resolve().parent / WEIGHTS_FILENAME

def load_weights():
    try:
        data = json.loads(weights_file_path().read_text())
        return {k: float(v) for k, v in data.items()}
    except:
        return DEFAULT_WEIGHTS.copy()

def save_weights(weights):
    try:
        weights_file_path().write_text(json.dumps(weights, indent=2))
    except:
        pass

def run():
    import pygame
    pygame.init()
    
    bw_px = BOARD_WIDTH * CELL_SIZE
    bh_px = BOARD_HEIGHT * CELL_SIZE
    win_w = (GRID_COLUMNS * bw_px) + ((GRID_COLUMNS + 1) * GRID_PADDING)
    win_h = HUD_HEIGHT + (GRID_ROWS * bh_px) + ((GRID_ROWS + 1) * GRID_PADDING)
    
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("Tetris Pure RL (No GA)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 14)
    
    # 공유 가중치 (하나의 뇌를 모든 보드가 공유)
    shared_weights = load_weights()
    states = [GameState.new() for _ in range(BOARD_COUNT)]
    
    origins = []
    for idx in range(BOARD_COUNT):
        c, r = idx % GRID_COLUMNS, idx // GRID_COLUMNS
        origins.append((
            GRID_PADDING + c * (bw_px + GRID_PADDING),
            HUD_HEIGHT + GRID_PADDING + r * (bh_px + GRID_PADDING)
        ))
    
    running = True
    is_paused = False
    speed_ms = 50
    generation = 1
    epsilon = EPSILON_START
    
    accumulator = 0.0
    high_score = 0
    
    while running:
        dt = clock.tick(60)
        
        for e in pygame.event.get():
            if e.type == pygame.QUIT: running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE: running = False
                elif e.key == pygame.K_SPACE: is_paused = not is_paused
                elif e.key == pygame.K_r:
                    # 리셋 시 가중치는 유지, 게임만 초기화
                    states = [GameState.new() for _ in range(BOARD_COUNT)]
                    generation += 1
                elif e.key in (pygame.K_MINUS, pygame.K_LEFTBRACKET):
                    speed_ms = min(1000, speed_ms + 10)
                elif e.key in (pygame.K_EQUALS, pygame.K_RIGHTBRACKET):
                    speed_ms = max(MIN_SPEED_MS, speed_ms - 10)
        
        if not is_paused:
            accumulator += dt
            while accumulator >= speed_ms:
                all_dead = True
                for state in states:
                    if not state.game_over:
                        all_dead = False
                        if state.current_piece is None:
                            spawn_piece(state, shared_weights, epsilon)
                        else:
                            step_game(state, shared_weights)
                
                if all_dead:
                    generation += 1
                    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
                    save_weights(shared_weights)
                    states = [GameState.new() for _ in range(BOARD_COUNT)]
                    print(f"Gen {generation} | Eps: {epsilon:.3f} | Weights: {shared_weights}")
                
                accumulator -= speed_ms

        # 렌더링
        screen.fill((20, 20, 30))
        
        # HUD
        curr_max = max(s.score for s in states)
        high_score = max(high_score, curr_max)
        hud_text = f"Gen: {generation}  Speed: {speed_ms}ms  Eps: {epsilon:.3f}  Max: {curr_max}  High: {high_score}"
        screen.blit(font.render(hud_text, True, (200, 200, 200)), (10, 10))
        
        w_text = f"H:{shared_weights['height']:.2f} L:{shared_weights['lines']:.2f} S:{shared_weights['holes']:.2f} B:{shared_weights['bumpiness']:.2f}"
        screen.blit(font.render(w_text, True, (150, 255, 150)), (10, 26))

        # 보드 그리기
        for idx, (ox, oy) in enumerate(origins):
            s = states[idx]
            # 배경
            for r in range(BOARD_HEIGHT):
                for c in range(BOARD_WIDTH):
                    color = COLORS.get(s.board[r][c], COLORS[0])
                    pygame.draw.rect(screen, color, (ox + c*CELL_SIZE, oy + r*CELL_SIZE, CELL_SIZE, CELL_SIZE))
                    pygame.draw.rect(screen, (40,40,50), (ox + c*CELL_SIZE, oy + r*CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)
            
            # 현재 블록
            if s.current_piece:
                pc = s.current_piece
                for r, row in enumerate(pc["shape"]):
                    for c, val in enumerate(row):
                        if val:
                            px, py = s.current_position["x"] + c, s.current_position["y"] + r
                            if py >= 0:
                                pygame.draw.rect(screen, pc["color"], (ox + px*CELL_SIZE, oy + py*CELL_SIZE, CELL_SIZE, CELL_SIZE))
            
            # 점수
            screen.blit(font.render(f"#{idx+1} {s.score}", True, (255, 255, 255)), (ox, oy - 15))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    run()
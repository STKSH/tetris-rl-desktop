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
# {점수} = (-0.51 X 높이) + (0.76 지운 줄) + (-0.35 X 구멍) + (-0.18 X 굴곡)
WEIGHTS_FILENAME = "weights.json"
DEFAULT_WEIGHTS: Dict[str, float] = {
    "height": -0.51,
    "lines": 0.7,
    "holes": -0.35,
    "bumpiness": -0.18
}


@dataclass
class GameState:
    # 각 보드(에이전트)의 상태를 하나의 데이터 묶음으로 관리한다.
    board: List[List[int]] # board -> 현재 보드 상태 (2차원 리스트)
    current_piece: Optional[dict] # current_piece -> 현재 낙하 중인 블록 (없을 수도 있음)
    current_position: Dict[str, int] # current_position -> 현재 블록의 위치 (x, y 좌표)
    next_piece: dict # next_piece -> 다음에 등장할 블록 (딕셔너리 형태)
    score: int # score -> 현재 점수
    lines_cleared: int # lines_cleared -> 지금까지 클리어한 줄 수
    level: int # level -> 현재 난이도 레벨
    game_over: bool # game_over -> 게임 종료 여부
    move_queue: Optional[Dict[str, int]] # move_queue -> 자동 이동 계획 (없을 수도 있음) 
    pending_features: Optional[Dict[str, float]] # pending_features -> 보드 특성 추출 대기 상태 (없을 수도 있음)

    @staticmethod
    def new() -> "GameState":
        # 빈 보드와 랜덤 다음 블록을 가진 초기 상태를 생성한다.
        return GameState(
            board=create_empty_board(), # create_empty_board -> 텅 빈 상태의 보드 생성
            current_piece=None, # current_piece -> 현재 낙하 중인 블록 (없음)
            current_position={"x": 0, "y": 0}, # current_position -> 현재 블록의 위치 (초기값)
            next_piece=random_piece(), # next_piece -> 다음에 등장할 블록 (random_piece 함수로 무작위 선택)
            score=0, # score -> 현재 점수 (0으로 초기화)
            lines_cleared=0, # lines_cleared -> 지금까지 클리어한 줄 수 (0으로 초기화)
            level=1, # level -> 현재 난이도 레벨 (1로 초기화)
            game_over=False, # game_over -> 게임 종료 여부 (False로 초기화)
            move_queue=None, # move_queue -> 자동 이동 계획 (없음)
            pending_features=None, # pending_features -> 보드 특성 추출 대기 상태 (없음)
        )


def create_empty_board() -> List[List[int]]:
    # 0으로 채워진 보드 행렬을 만든다.
    return [[0 for _ in range(BOARD_WIDTH)] for _ in range(BOARD_HEIGHT)] # _의 역할 -> 반복문에서 인덱스가 필요 없을 때 사용되는 변수


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
    scores = [state.score for state in states] # scores -> 각 보드의 점수 리스트 생성 
    total_lines = sum(state.lines_cleared for state in states) # total_lines -> 모든 보드의 클리어한 줄 수 합산
    avg_score = int(sum(scores) / len(scores)) # avg_score -> 평균 점수 계산
    max_score = max(scores) # max_score -> 최고 점수 계산
    return {"avg_score": avg_score, "max_score": max_score, "total_lines": total_lines} # 결과 딕셔너리 반환


def get_board_origins(
    columns: int, # columns -> 격자 형태의 열 수
    rows: int, # rows -> 격자 형태의 행 수
    board_width_px: int, # board_width_px -> 각 보드의 가로 픽셀 크기
    board_height_px: int, # board_height_px -> 각 보드의 세로 픽셀 크기 
    padding: int, # padding -> 보드 간의 간격
    top_offset: int, # top_offset -> 상단 HUD 높이
) -> List[Tuple[int, int]]: 
    # 여러 보드를 화면에 격자로 배치하기 위한 좌상단 좌표를 계산한다.
    origins: List[Tuple[int, int]] = [] # origins -> 각 보드의 좌상단 좌표 리스트
    total = columns * rows # total -> 전체 보드 수 계산
    for idx in range(total): # idx -> 현재 보드의 인덱스
        col = idx % columns # col -> 현재 보드가 속한 열 인덱스 
        row = idx // columns # row -> 현재 보드가 속한 행 인덱스
        x = padding + col * (board_width_px + padding) # x -> 현재 보드의 좌상단 x 좌표 계산
        y = top_offset + padding + row * (board_height_px + padding) # y -> 현재 보드의 좌상단 y 좌표 계산
        origins.append((x, y)) # origins 리스트에 좌표 추가
    return origins # 계산된 좌표 리스트 반환


def clamp_speed(speed_ms: int, delta: int, min_speed: int, max_speed: int) -> int:
    # 속도 변경 요청이 최소/최대 범위를 넘지 않도록 보정한다.
    return max(min_speed, min(max_speed, speed_ms + delta)) # 괄호 안의 수식에 대해 설명 -> speed_ms에 delta를 더한 값을 min_speed와 max_speed 사이로 제한


def rotate_piece(shape: List[List[int]]) -> List[List[int]]:
    # 블록의 2차원 모양을 시계 방향 90도 회전한 결과를 반환한다.
    return [list(reversed(col)) for col in zip(*shape)] # zip(*shape)의 역할 -> 2차원 리스트 shape의 행과 열을 뒤바꾸는 역할을 함


def place_piece(
    board: List[List[int]], piece: dict, pos: Dict[str, int]
) -> List[List[int]]: # List 안에 List[int]가 또 있는 이유 -> 2차원 리스트를 나타내기 위해 사용됨
    # 현재 블록을 보드에 고정한 새 보드를 만들어 반환한다.
    new_board = [row[:] for row in board] # 보드 복사 -> 기존 보드를 변경하지 않기 위해 새로운 보드 생성
    piece_index = PIECE_INDEX_BY_NAME[piece["name"]] # 블록 이름에 해당하는 보드 값을 가져옴
    for r, row in enumerate(piece["shape"]): # 블록 모양을 순회하며
        for c, cell in enumerate(row): # 각 셀을 확인
            if cell and pos["y"] + r >= 0: # 셀이 채워져 있고 보드 내에 위치할 때
                new_board[pos["y"] + r][pos["x"] + c] = piece_index # 보드에 블록 값을 설정
    return new_board # 수정된 보드 반환


def clear_lines(board: List[List[int]]) -> Tuple[List[List[int]], int]:
    # 완전히 채워진 줄을 제거하고 위에서 빈 줄을 보충한다.
    new_board = [row for row in board if any(cell == 0 for cell in row)] # 빈 칸이 있는 줄만 남김
    lines_cleared = BOARD_HEIGHT - len(new_board) # 제거된 줄 수 계산
    while len(new_board) < BOARD_HEIGHT: # 보드 높이를 유지하기 위해
        new_board.insert(0, [0] * BOARD_WIDTH) # 위에 빈 줄 추가
    return new_board, lines_cleared # 수정된 보드와 제거된 줄 수 반환


def random_piece() -> dict:
    # 다음에 등장할 블록을 무작위로 선택한다.
    return random.choice(PIECES)


def is_valid_position(
    board: List[List[int]], shape: List[List[int]], pos: Dict[str, int]
) -> bool:
    # 블록이 보드 경계 안에 있고 기존 블록과 충돌하지 않는지 검사한다.
    for r, row in enumerate(shape): # 블록 모양을 순회하며
        for c, cell in enumerate(row): # 각 셀을 확인
            if not cell: # 셀이 비어있으면 건너뜀
                continue 
            new_x = pos["x"] + c # 블록의 보드 내 x 좌표 계산
            new_y = pos["y"] + r # 블록의 보드 내 y 좌표 계산
            if new_x < 0 or new_x >= BOARD_WIDTH or new_y >= BOARD_HEIGHT: # 보드 경계를 벗어나면
                return False # False 반환
            if new_y >= 0 and board[new_y][new_x]: # 보드 내에 있고 기존 블록과 충돌하면
                return False # False 반환
    return True # 모든 셀이 유효하면 True 반환


def rotate_n(shape: List[List[int]], times: int) -> List[List[int]]:
    # 블록을 0~3회 회전한 결과를 계산한다.
    rotated = shape # 초기 모양 설정
    for _ in range(times % 4): # 지정된 회전 횟수만큼 반복
        rotated = rotate_piece(rotated) # 블록 회전
    return rotated # 최종 회전된 모양 반환


def place_shape( # place_shape 함수의 역할 -> 주어진 모양을 보드에 고정한 결과를 반환
    board: List[List[int]], # board -> 현재 보드 상태 (2차원 리스트)
    shape: List[List[int]], # shape -> 회전이 적용된 블록 모양 (2차원 리스트)
    piece: dict, # piece -> 블록 정보 (딕셔너리 형태)
    pos: Dict[str, int], # pos -> 블록의 위치 (x, y 좌표)
) -> List[List[int]]: #
    # 주어진 모양(회전 적용된 shape)을 보드에 고정한 결과를 만든다.
    new_board = [row[:] for row in board] # 보드 복사
    piece_index = PIECE_INDEX_BY_NAME[piece["name"]] # 블록 이름에 해당하는 보드 값 가져오기
    for r, row in enumerate(shape): # 모양을 순회하며
        for c, cell in enumerate(row): # 각 셀을 확인
            if cell and pos["y"] + r >= 0: # 셀이 채워져 있고 보드 내에 위치할 때
                new_board[pos["y"] + r][pos["x"] + c] = piece_index # 보드에 블록 값 설정
    return new_board # 수정된 보드 반환


def get_column_heights(board: List[List[int]]) -> List[int]:
    # 각 열에서 가장 높은 블록까지의 높이를 계산한다.
    heights: List[int] = [] # heights -> 각 열의 높이 리스트
    for col in range(BOARD_WIDTH): # 각 열을 순회하며
        height = 0 # 해당 열의 높이 초기화
        for row in range(BOARD_HEIGHT): # 각 행을 위에서부터 순회
            if board[row][col]: # 블록이 채워진 셀을 찾으면
                height = BOARD_HEIGHT - row # 높이 계산
                break # 더 이상 확인할 필요 없음
        heights.append(height) # 계산된 높이를 리스트에 추가
    return heights # 높이 리스트 반환


def get_holes(board: List[List[int]]) -> int:
    # 위에 블록이 있고 아래가 빈 칸인 구멍 수를 센다.
    holes = 0 # holes -> 구멍 수 초기화
    for col in range(BOARD_WIDTH): # 각 열을 순회하며
        block_found = False # 블록 발견 여부 초기화
        for row in range(BOARD_HEIGHT): # 각 행을 위에서부터 순회
            if board[row][col]: # 블록이 채워진 셀을 찾으면
                block_found = True # 블록 발견 표시
            elif block_found: # 블록이 발견된 이후에 빈 칸이 나오면
                holes += 1 # 구멍 수 증가
    return holes # 구멍 수 반환


def get_bumpiness(heights: List[int]) -> int:
    # 인접한 열 간 높이 차이를 합산해 울퉁불퉁함을 계산한다.
    bumpiness = 0 # bumpiness -> 울퉁불퉁함 초기화
    for i in range(len(heights) - 1): # 각 열을 순회하며
        bumpiness += abs(heights[i] - heights[i + 1]) # 인접 열 간 높이 차이 누적
    return bumpiness # 울퉁불퉁함 반환


def extract_features(board: List[List[int]], lines_cleared: int = 0) -> Dict[str, float]:
    # 보드에서 휴리스틱 특성을 추출해 선형 가치함수 입력으로 사용한다.
    heights = get_column_heights(board) # 각 열의 높이 계산
    return { # 보드 특성 딕셔너리 반환
        "height": float(sum(heights)), # 전체 높이 합산
        "lines": float(lines_cleared), # 클리어된 줄 수
        "holes": float(get_holes(board)), # 구멍 수 계산
        "bumpiness": float(get_bumpiness(heights)) # 울퉁불퉁함 계산
    }


def evaluate_board( # evaluate_board 함수의 역할 -> 현재 보드 상태와 가중치를 기반으로 선형 가치함수를 계산
    board: List[List[int]], lines_cleared: int, weights: Dict[str, float] 
) -> float: # -> float 반환 타입은 실수형 값을 반환함을 나타냄
    # 현재 보드의 특성과 가중치를 곱해 선형 가치함수를 계산한다.
    features = extract_features(board, lines_cleared) # 보드 특성 추출
    return sum(weights[key] * value for key, value in features.items()) # 선형 가치함수 계산 및 반환


def update_weights( # update_weights 함수의 역할 -> TD(0) 방식으로 가중치를 업데이트하여 가치 추정을 개선
    weights: Dict[str, float], 
    features: Dict[str, float],
    reward: float,
    next_value: float,
    learning_rate: float,
    discount: float,
) -> None:
    # TD(0) 방식으로 가중치를 업데이트해 가치 추정을 개선한다.
    current_value = sum(weights[key] * value for key, value in features.items()) # 현재 가치 계산
    td_target = reward + discount * next_value # TD 타겟 계산
    td_error = td_target - current_value # TD 오차 계산
    for key, value in features.items(): # 각 특성에 대해
        weights[key] += learning_rate * td_error * value # 가중치 업데이트
# TD(Temporal Difference) 학습 -> 강화학습에서 사용되는 방법으로, 에이전트가 환경과 상호작용하면서 얻은 경험을 바탕으로 가치 함수를 업데이트하는 기법입니다. TD 학습은 미래의 보상에 대한 예측을 현재의 가치 추정에 반영하여 점진적으로 학습을 진행합니다. TD(0)는 가장 기본적인 형태로, 현재 상태에서 얻은 보상과 다음 상태의 가치 추정을 사용하여 현재 상태의 가치를 업데이트합니다.

def find_best_move( # find_best_move 함수의 역할 -> 현재 보드 상태와 블록 정보를 기반으로 가능한 모든 이동을 평가하여 가장 높은 가치를 가진 이동을 선택
    board: List[List[int]], #
    piece: dict,
    weights: Dict[str, float],
    epsilon: float = 0.0 # epsilon -> 탐험 확률 (0.0이면 탐험 없음)
) -> Optional[dict]:
    # 가능한 모든 회전과 위치를 탐색해 가치가 높은 수를 고르되 확률적으로 탐험한다.
    candidates: List[dict] = [] # 후보 이동 리스트 초기화
    for rotation in range(4): # 0~3회 회전 시도
        shape = rotate_n(piece["shape"], rotation) # 회전된 블록 모양 계산
        piece_width = len(shape[0]) # 블록의 가로 너비 계산
        for x in range(-2, BOARD_WIDTH - piece_width + 3): # 가능한 x 위치 탐색
            y = -len(shape) # 블록이 보드 위에 시작하도록 y 초기화
            while is_valid_position(board, shape, {"x": x, "y": y + 1}): # 가능한 최대 y 위치까지 내림
                y += 1 # y 위치 증가
            if not is_valid_position(board, shape, {"x": x, "y": y}): # 유효한 위치가 아니면 건너뜀
                continue 
            new_board = place_shape(board, shape, piece, {"x": x, "y": y}) # 블록을 보드에 고정
            cleared_board, lines = clear_lines(new_board) # 줄 제거
            score = evaluate_board(cleared_board, lines, weights) # 보드 평가
            candidates.append({"rotation": rotation, "x": x, "score": score}) # 후보 이동 추가

    if not candidates: # 후보 이동이 없으면
        return None # None 반환
    if epsilon > 0.0 and random.random() < epsilon: # 확률적으로 탐험할 경우
        return random.choice(candidates) # 무작위 후보 이동 반환
    return max(candidates, key=lambda candidate: candidate["score"]) # 최고 점수 후보 이동 반환
# 근데 탐험할 확률을이 0.0 인데 이건 언제 증가함? -> epsilon 값은 강화학습에서 탐험(exploration)과 활용(exploitation) 사이의 균형을 조절하는 하이퍼파라미터입니다. 일반적으로 학습 초기에는 높은 탐험 확률을 설정하여 다양한 상태와 행동을 시도하게 하고, 학습이 진행됨에 따라 탐험 확률을 점차 감소시켜 더 안정적인 정책을 따르도록 합니다. 이 코드에서는 epsilon 값이 시간이 지남에 따라 감소하는 메커니즘이 포함되어 있지 않지만, 외부에서 epsilon 값을 조절하여 탐험 확률을 변경할 수 있습니다. 예를 들어, 학습 세대가 진행될수록 epsilon 값을 감소시키는 로직을 추가할 수 있습니다.
# 그게 현재 코드에 있는거야? -> 현재 코드에는 epsilon 값을 감소시키는 로직이 포함되어 있지 않습니다. epsilon 값은 run() 함수 내에서 초기화되며, 이후 변경되지 않습니다. 만약 epsilon 값을 점진적으로 감소시키고 싶다면, run() 함수 내에서 세대가 진행될 때마다 epsilon 값을 업데이트하는 코드를 추가해야 합니다.
# 그럼 탐험이 선언 되어있는 이유가 지금은 없는거 아니야? -> 네, 현재 코드에서는 epsilon 값을 감소시키는 로직이 없기 때문에 탐험 확률이 고정되어 있습니다. 따라서 탐험이 선언되어 있지만 실제로는 변화하지 않는 상태입니다. 탐험 확률을 동적으로 조절하려면 추가적인 로직이 필요합니다.

def spawn_piece(state: GameState, weights: Dict[str, float], epsilon: float) -> None:
    # 다음 블록을 현재 블록으로 가져오고 자동 이동 계획을 준비한다.
    if state.pending_features is None: # pending_features가 없으면
        state.pending_features = extract_features(state.board) # 보드 특성 추출, pending_features -> 보드 특성 추출 대기 상태
    new_piece = state.next_piece # 다음 블록을 현재 블록으로 설정
    start_x = (BOARD_WIDTH - len(new_piece["shape"][0])) // 2 # 블록이 중앙에 오도록 시작 x 위치 계산
    start_y = -1 # 시작 y 위치 설정
    if not is_valid_position( # 유효한 시작 위치인지 검사
        state.board, new_piece["shape"], {"x": start_x, "y": start_y + 1} # y + 1인 이유 -> 블록이 보드 위에 시작하도록 하기 위해 y 위치를 1 증가시킨 상태에서 유효성을 검사
    ):
        if state.pending_features: # pending_features가 있으면
            update_weights( # 가중치 업데이트
                weights, # weights -> 현재 가중치 딕셔너리
                state.pending_features, # features -> 보드 특성 딕셔너리
                reward=-1.0, # reward -> 게임 오버에 대한 음수 보상 
                next_value=0.0, # next_value -> 다음 상태의 가치 (게임 오버이므로 0) 
                learning_rate=LEARNING_RATE, # learning_rate -> 학습률 상수
                discount=DISCOUNT_FACTOR, # discount -> 할인율 상수
            )
        state.game_over = True # 게임 오버 설정
        return
    best_move = find_best_move(state.board, new_piece, weights, epsilon) # 최적 이동 찾기
    if best_move: # 최적 이동이 있으면
        state.move_queue = { # 자동 이동 계획 설정 
            "rotations": best_move["rotation"], # rotations -> 회전 횟수
            "target_x": best_move["x"], # target_x -> 목표 x 위치
        }
    state.current_piece = new_piece # 현재 블록 설정
    state.current_position = {"x": start_x, "y": start_y} # 현재 블록 위치 설정
    state.next_piece = random_piece() # 다음 블록 무작위 선택
 

def step_game(state: GameState, weights: Dict[str, float]) -> None: 
    # 자동 이동 계획을 처리하고 블록을 한 칸 내려 게임을 진행한다.
    if state.game_over or not state.current_piece: # 게임 오버이거나 현재 블록이 없으면
        return # 함수 종료

    new_piece = state.current_piece # 현재 블록 복사
    new_pos = {"x": state.current_position["x"], "y": state.current_position["y"]} # 현재 블록 위치 복사

    if state.move_queue: # 자동 이동 계획이 있으면
        rotations = state.move_queue["rotations"] # rotations -> 남은 회전 횟수
        target_x = state.move_queue["target_x"] # target_x -> 목표 x 위치

        if rotations > 0: # 회전이 남아있으면
            test_shape = new_piece["shape"]  # 현재 블록 모양 복사
            for _ in range(rotations): # 남은 회전 횟수만큼
                test_shape = rotate_piece(test_shape) # 블록 회전
            if is_valid_position(state.board, test_shape, new_pos): # 유효한 위치인지 검사
                new_piece = {**new_piece, "shape": test_shape} # 블록 모양 업데이트
            state.move_queue["rotations"] = 0 # 회전 완료 처리

        if new_pos["x"] < target_x: # 목표 위치보다 왼쪽에 있으면
            if is_valid_position( # 오른쪽으로 이동이 유효한지 검사
                state.board, # 현재 보드 상태가 들어간 이유 -> 이동 후의 위치가 유효한지 검사하기 위해 현재 보드 상태를 사용
                new_piece["shape"], # 현재 블록 모양
                {"x": new_pos["x"] + 1, "y": new_pos["y"]}, # 이동 후 위치
            ):
                new_pos["x"] += 1 # 오른쪽으로 이동
        elif new_pos["x"] > target_x: # 목표 위치보다 오른쪽에 있으면 
            if is_valid_position( # 왼쪽으로 이동이 유효한지 검사
                state.board, # 현재 보드 상태가 들어간 이유 -> 이동 후의 위치가 유효한지 검사하기 위해 현재 보드 상태를 사용
                new_piece["shape"], # 현재 블록 모양
                {"x": new_pos["x"] - 1, "y": new_pos["y"]}, # 이동 후 위치
            ):
                new_pos["x"] -= 1 # 왼쪽으로 이동

        if new_pos["x"] == target_x: # 목표 위치에 도달했으면
            state.move_queue = None # 자동 이동 계획 완료 처리

    next_pos = {"x": new_pos["x"], "y": new_pos["y"] + 1} # 다음 y 위치 계산
    if is_valid_position(state.board, new_piece["shape"], next_pos): # 유효한 위치이면
        state.current_piece = new_piece # 현재 블록 업데이트
        state.current_position = next_pos # 현재 블록 위치 업데이트 # 현재 블록 위치 업데이트 이유가 새로운 블록에 타겟팅 하기 위해서인가? -> 아니요, 현재 블록의 위치를 한 칸 아래로 이동시키기 위해 업데이트하는 것입니다. next_pos는 현재 블록이 한 칸 아래로 이동한 위치를 나타내며, 이 위치가 유효한지 검사한 후에 현재 블록의 위치를 업데이트합니다.
        return

    new_board = place_shape(state.board, new_piece["shape"], new_piece, new_pos) # 블록을 보드에 고정
    cleared_board, lines_cleared = clear_lines(new_board) # 줄 제거
    line_scores = [0, 100, 300, 500, 800] # 클리어된 줄 수에 따른 점수
    score_gain = line_scores[lines_cleared] * state.level # 획득 점수 계산

    reward = float(lines_cleared) # 보상은 클리어된 줄 수에 비례
    next_value = evaluate_board(cleared_board, 0, weights) # 다음 상태 가치 계산
    if state.pending_features: # pending_features가 있으면
        update_weights( # 가중치 업데이트
            weights, # weights -> 현재 가중치 딕셔너리
            state.pending_features, # features -> 보드 특성 딕셔너리
            reward, # reward -> 클리어된 줄 수에 비례하는 보상
            next_value, # next_value -> 다음 상태의 가치
            LEARNING_RATE, # learning_rate -> 학습률 상수
            DISCOUNT_FACTOR # discount -> 할인율 상수
        )
    state.pending_features = None # pending_features 초기화

    state.board = cleared_board # 보드 상태 업데이트
    state.current_piece = None # 현재 블록 초기화
    state.current_position = {"x": 0, "y": 0} # 현재 블록 위치 초기화
    state.score += score_gain # 점수 업데이트
    state.lines_cleared += lines_cleared # 클리어된 줄 수 업데이트
    state.level = state.lines_cleared // 10 + 1 # 레벨 업데이트


def init_population_weights( # init_population_weights 함수의 역할 -> 여러 보드가 공유할 기본 가중치 세트를 복제하여 초기화
    base_weights: Dict[str, float], count: int # -> base_weights -> 기본 가중치 딕셔너리, count -> 보드 수
) -> List[Dict[str, float]]:
    # 기본 가중치를 복제해 여러 보드가 공유하도록 초기화한다.
    return [base_weights.copy() for _ in range(max(count, 0))] # 기본 가중치 복제 리스트 반환


def average_weights(population_weights: List[Dict[str, float]]) -> Dict[str, float]:
    # 여러 보드의 가중치를 평균내어 저장용 대표값을 만든다.
    if not population_weights: # 가중치 리스트가 비어있으면
        return DEFAULT_WEIGHTS.copy() # 기본 가중치 반환
    avg = {key: 0.0 for key in DEFAULT_WEIGHTS} # 평균 가중치 초기화
    for weights in population_weights: # 각 보드의 가중치에 대해
        for key in avg: # 각 가중치 키에 대해
            avg[key] += weights.get(key, 0.0) # 가중치 누적
    for key in avg: # 각 가중치 키에 대해
        avg[key] /= len(population_weights) # 평균 계산
    return avg # 평균 가중치 반환


def weights_file_path(path: Optional[Path] = None) -> Path: # weights_file_path 함수의 역할 -> 가중치 파일의 경로를 결정하며, 지정된 경로가 없으면 기본 위치를 사용
    # 가중치 파일 경로를 결정하며, 지정이 없으면 스크립트 위치를 사용한다.
    if path is None: # 경로가 지정되지 않았으면
        return Path(__file__).resolve().parent / WEIGHTS_FILENAME # 스크립트 위치에 기본 파일명 추가
    return Path(path) # 지정된 경로 반환


def load_weights(path: Optional[Path] = None) -> Dict[str, float]: # load_weights 함수의 역할 -> 가중치 파일에서 가중치를 읽어오며, 실패 시 기본 가중치를 반환
    # 가중치 파일을 읽고 문제 발생 시 기본값을 사용한다.
    file_path = weights_file_path(path) # 가중치 파일 경로 결정
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

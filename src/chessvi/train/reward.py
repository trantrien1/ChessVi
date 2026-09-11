"""Hàm reward cho GRPO.

Tách hẳn khỏi ``grpo.py`` để test được mà không cần GPU, không cần trl. Đây là
phần dễ sai nhất và rẻ nhất để test trong cả pipeline RL: reward sai dấu hoặc
sai thang thì RL vẫn "chạy" mà model càng học càng tệ.

Thang điểm (nước đi và format là hai trục riêng, cộng vào nhau):

==========================================  ======
nước đi khớp lời giải puzzle                 +1.0
hợp lệ nhưng sai                             +0.1
không hợp lệ hoặc không parse được           -1.0
bonus riêng cho đúng format output           +0.2
==========================================  ======
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import chess

from chessvi.data.mask import find_move_tokens, parse_move

logger = logging.getLogger(__name__)

__all__ = [
    "ANSWER_RE",
    "ANSWER_TEMPLATE",
    "REWARD_CORRECT",
    "REWARD_FORMAT_BONUS",
    "REWARD_ILLEGAL",
    "REWARD_LEGAL_BUT_WRONG",
    "RewardBreakdown",
    "extract_move",
    "has_valid_format",
    "puzzle_reward",
    "score_completion",
]

REWARD_CORRECT = 1.0
REWARD_LEGAL_BUT_WRONG = 0.1
REWARD_ILLEGAL = -1.0
REWARD_FORMAT_BONUS = 0.2

#: Format bắt buộc của output: lý giải tự do, rồi đúng một dòng kết luận.
ANSWER_TEMPLATE = "Nước đi: {move}"
ANSWER_RE = re.compile(r"Nước đi:\s*(\S+)", re.IGNORECASE)


@dataclass(frozen=True)
class RewardBreakdown:
    """Chi tiết điểm của một completion, để debug khi reward trông lạ."""

    total: float
    move_reward: float
    format_bonus: float
    move: chess.Move | None
    is_correct: bool
    is_legal: bool
    has_format: bool


def has_valid_format(completion: str) -> bool:
    """Output có đúng một dòng kết luận ``Nước đi: ...``."""
    return len(ANSWER_RE.findall(completion)) == 1


def extract_move(completion: str, board: chess.Board) -> chess.Move | None:
    """Lấy nước đi model đề xuất, xác thực bằng python-chess.

    Ưu tiên phần nằm sau ``Nước đi:``; không có thì lấy nước hợp lệ cuối cùng
    xuất hiện trong text. Luôn đi qua :func:`~chessvi.data.mask.parse_move` nên
    không bao giờ trả về nước không hợp lệ, dù regex có bắt nhầm gì.
    """
    matches = ANSWER_RE.findall(completion)
    if matches:
        move = parse_move(board, matches[-1].strip().strip(".,;:)"))
        if move is not None:
            return move

    found: chess.Move | None = None
    for token in find_move_tokens(completion):
        move = parse_move(board, token.text)
        if move is not None:
            found = move
    return found


def _solution_move(board: chess.Board, solution: Sequence[str] | str) -> chess.Move | None:
    """Nước đầu tiên của lời giải puzzle, dạng ``chess.Move``."""
    if isinstance(solution, str):
        moves = solution.split()
    else:
        moves = list(solution)
    if not moves:
        return None
    return parse_move(board, moves[0])


def score_completion(
    completion: str,
    fen: str,
    solution: Sequence[str] | str,
) -> RewardBreakdown:
    """Chấm điểm một completion trên một puzzle."""
    try:
        board = chess.Board(fen)
    except ValueError:
        logger.warning("FEN hỏng trong dataset RL: %r", fen)
        return RewardBreakdown(
            total=REWARD_ILLEGAL,
            move_reward=REWARD_ILLEGAL,
            format_bonus=0.0,
            move=None,
            is_correct=False,
            is_legal=False,
            has_format=False,
        )

    has_format = has_valid_format(completion)
    format_bonus = REWARD_FORMAT_BONUS if has_format else 0.0

    move = extract_move(completion, board)
    expected = _solution_move(board, solution)

    if move is None:
        move_reward = REWARD_ILLEGAL
        is_correct = is_legal = False
    elif expected is not None and move == expected:
        move_reward = REWARD_CORRECT
        is_correct = is_legal = True
    else:
        move_reward = REWARD_LEGAL_BUT_WRONG
        is_correct = False
        is_legal = True

    return RewardBreakdown(
        total=move_reward + format_bonus,
        move_reward=move_reward,
        format_bonus=format_bonus,
        move=move,
        is_correct=is_correct,
        is_legal=is_legal,
        has_format=has_format,
    )


def _as_text(completion: Any) -> str:
    """TRL đưa completion dạng str hoặc list message tuỳ chế độ."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        return "\n".join(
            str(turn.get("content", "")) for turn in completion if isinstance(turn, dict)
        )
    return str(completion)


def puzzle_reward(
    completions: Sequence[Any],
    fen: Sequence[str],
    solution: Sequence[Sequence[str] | str],
    **kwargs: Any,
) -> list[float]:
    """Adapter cho ``trl.GRPOTrainer``.

    TRL gọi ``reward_func(prompts=..., completions=..., **cột_dataset)`` nên
    ``fen`` và ``solution`` phải là tên cột trong train_dataset.
    """
    return [
        score_completion(_as_text(completion), position, answer).total
        for completion, position, answer in zip(completions, fen, solution, strict=True)
    ]

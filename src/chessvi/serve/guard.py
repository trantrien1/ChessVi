"""Chốt chặn output: không cho một nước đi không hợp lệ nào lọt ra ngoài.

Prompt tốt đến mấy thì model vẫn có lúc bịa nước đi. Đây là lớp kiểm tra cứng
sau cùng: trích mọi nước đi trong text bằng chính regex của ``data/mask.py``
rồi xác thực từng nước bằng python-chess.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import chess

from chessvi.data.mask import MoveContext, find_move_tokens
from chessvi.engine.facts import PositionFacts
from chessvi.serve.prompt import fallback_answer

logger = logging.getLogger(__name__)

__all__ = ["GuardResult", "check_output", "guarded_generate"]


@dataclass(frozen=True)
class GuardResult:
    """Kết quả soi một câu trả lời."""

    ok: bool
    #: Nước đi xuất hiện trong text nhưng không hợp lệ.
    violations: tuple[str, ...]
    #: Mọi nước đi tìm thấy trong text, kể cả hợp lệ.
    checked: tuple[str, ...]

    def __bool__(self) -> bool:
        return self.ok


def check_output(text: str, board: chess.Board) -> GuardResult:
    """Xác thực mọi nước đi xuất hiện trong ``text`` với ``board``.

    Một nước được chấp nhận khi hợp lệ với thế hiện tại, hoặc hợp lệ với thế
    đang chạy nếu câu trả lời đang kể một biến theo thứ tự (``1.e4 e5 2.Nf3``).
    Mọi thứ còn lại là vi phạm.
    """
    context = MoveContext(board.fen())
    violations: list[str] = []
    checked: list[str] = []
    # require_chess_context=False: thà bắt nhầm còn hơn để lọt nước bịa.
    for token in find_move_tokens(text, require_chess_context=False):
        checked.append(token.text)
        if not context.accepts(token):
            violations.append(token.text)
    if violations:
        logger.warning("Guard bắt được nước không hợp lệ: %s", violations)
    return GuardResult(not violations, tuple(violations), tuple(checked))


def guarded_generate(
    generate: Callable[[int], str],
    board: chess.Board,
    facts: PositionFacts,
    *,
    max_regenerations: int = 2,
) -> str:
    """Sinh câu trả lời, regenerate khi guard bắt lỗi, cuối cùng lùi về fact.

    ``generate(attempt)`` được gọi với số lần thử (bắt đầu từ 0) để caller có
    thể tăng nhiệt độ hoặc thêm lời nhắc ở lần thử sau. Sau
    ``max_regenerations`` lần vẫn hỏng thì trả về
    :func:`~chessvi.serve.prompt.fallback_answer` - câu trả lời chỉ gồm fact.
    """
    for attempt in range(max_regenerations + 1):
        text = generate(attempt)
        result = check_output(text, board)
        if result.ok:
            return text
        logger.info(
            "Lần %d/%d bị guard chặn (%s), sinh lại",
            attempt + 1,
            max_regenerations + 1,
            ", ".join(result.violations),
        )
    logger.warning("Hết lượt sinh lại, lùi về câu trả lời chỉ gồm fact")
    return fallback_answer(facts)

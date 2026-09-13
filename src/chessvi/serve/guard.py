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

from chessvi.data.mask import MoveContext, find_move_tokens, is_unambiguous_move
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

    Chỉ **kết tội** token chắc chắn là nước đi. Nước tốt trần (``c4``, ``e6``)
    trùng hệt cú pháp với tên ô, mà văn giải thích cờ tiếng Việt thì đầy tên ô —
    "xe trên e1", "tốt ở d5" — nên soi chúng là báo nhầm gần như mọi câu.

    Nhưng vẫn phải **đưa chúng qua ngữ cảnh**. Nước tốt trần là mắt xích của
    biến: bỏ qua hẳn ``c4`` thì cây không bao giờ tới thế sau ``c4``, và
    ``dxc4`` ngay sau đó bị kết tội oan. Đúng câu trả lời Gambit Hậu nào cũng
    dính. Nhận hay không cũng mở nhánh; chỉ là không tính vào vi phạm.

    Cái giá: model bịa một nước tốt trần (``chơi e5`` khi e5 không đi được) sẽ
    lọt. Nước có chữ quân, nước ăn quân và UCI vẫn bị soi đủ.
    """
    context = MoveContext(board.fen())
    violations: list[str] = []
    checked: list[str] = []
    # require_chess_context=False: thà bắt nhầm còn hơn để lọt nước bịa.
    for token in find_move_tokens(text, require_chess_context=False):
        accepted = context.accepts(token)  # gọi cả với token mờ: để mở nhánh
        if not is_unambiguous_move(token):
            continue
        checked.append(token.text)
        if not accepted:
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

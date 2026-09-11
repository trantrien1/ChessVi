"""Dựng prompt cho model phục vụ.

Block fact của T3 nằm ở **vị trí cố định**, có nhãn rõ ràng, kèm chỉ thị rằng
đó là sự thật đã xác minh và model không được mâu thuẫn. Đây là nửa đầu của
lớp chống hallucination; nửa sau là ``guard.py`` kiểm tra lại output.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from chessvi.engine.facts import SIDE_VI, PositionFacts, facts_to_prompt

__all__ = [
    "SYSTEM_PROMPT",
    "ChatTurn",
    "build_messages",
    "build_prompt",
    "fallback_answer",
]

Role = Literal["user", "assistant"]

SYSTEM_PROMPT = """Bạn là trợ lý cờ vua tiếng Việt.

Quy tắc bắt buộc:
1. Khối [SỰ THẬT ĐÃ XÁC MINH] là dữ liệu do python-chess và Stockfish tính ra. \
Nó luôn đúng. Không được nói bất cứ điều gì mâu thuẫn với nó.
2. Chỉ được nhắc tới nước đi có trong danh sách nước đi hợp lệ. Không bịa nước đi.
3. Không tự tính lại thế cờ, không tự đoán eval. Cần số liệu nào thì lấy từ khối \
sự thật; khối đó không có thì nói thẳng là không biết.
4. Trả lời bằng tiếng Việt, ngắn gọn, dùng đúng thuật ngữ cờ vua tiếng Việt \
(đòn đôi, ghim, xiên, đòn mở, chiếu hết hàng cuối, bắt tốt qua đường, nhập thành, \
zugzwang, sai lầm nghiêm trọng, quân bỏ ngỏ).
5. Giữ nguyên ký hiệu nước đi dạng SAN như trong khối sự thật, không dịch sang \
tiếng Việt (viết Nf3, không viết "Mã f3")."""


@dataclass(frozen=True)
class ChatTurn:
    """Một lượt trong lịch sử hội thoại."""

    role: Role
    content: str


def _history_block(history: Sequence[ChatTurn]) -> str:
    lines: list[str] = []
    for turn in history:
        speaker = "Người dùng" if turn.role == "user" else "Trợ lý"
        lines.append(f"{speaker}: {turn.content}")
    return "\n".join(lines)


def build_prompt(
    facts: PositionFacts,
    user_question: str,
    history: Sequence[ChatTurn] = (),
) -> str:
    """Prompt dạng text thuần cho backend không có chat template.

    Thứ tự cố định: hệ thống -> lịch sử -> **khối sự thật** -> câu hỏi. Khối
    sự thật luôn nằm ngay trước câu hỏi để model không "quên" nó.
    """
    parts = [SYSTEM_PROMPT]
    if history:
        parts.append("[LỊCH SỬ HỘI THOẠI]\n" + _history_block(history))
    parts.append(facts_to_prompt(facts))
    parts.append(f"[CÂU HỎI]\n{user_question}")
    parts.append("[TRẢ LỜI]")
    return "\n\n".join(parts)


def build_messages(
    facts: PositionFacts,
    user_question: str,
    history: Sequence[ChatTurn] = (),
) -> list[dict[str, Any]]:
    """Dạng messages cho backend có chat template. Cùng thứ tự với text thuần."""
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history:
        messages.append({"role": turn.role, "content": turn.content})
    messages.append(
        {"role": "user", "content": f"{facts_to_prompt(facts)}\n\n[CÂU HỎI]\n{user_question}"}
    )
    return messages


def fallback_answer(facts: PositionFacts) -> str:
    """Câu trả lời chỉ gồm fact, dùng khi guard bắt lỗi quá số lần cho phép.

    Thà trả lời khô khan mà đúng còn hơn trả lời trôi chảy mà bịa nước đi.
    """
    side = SIDE_VI[facts.side_to_move]
    lines = [
        "Tôi chưa đưa ra được lời giải thích đáng tin cho thế cờ này, "
        "nên chỉ nêu các dữ kiện đã xác minh:",
        f"- Lượt đi: {side}.",
        f"- Cán cân lực lượng: Trắng {facts.material['white']} - Đen {facts.material['black']}.",
        f"- Đang bị chiếu: {'có' if facts.in_check else 'không'}.",
    ]
    if facts.hanging:
        hanging = ", ".join(f"{symbol} ở {square}" for square, symbol in facts.hanging)
        lines.append(f"- Quân bỏ ngỏ: {hanging}.")
    if facts.checks_available:
        lines.append(f"- Nước chiếu khả dụng: {', '.join(facts.checks_available)}.")
    if facts.best_move:
        lines.append(f"- Nước tốt nhất theo Stockfish: {facts.best_move}.")
    return "\n".join(lines)

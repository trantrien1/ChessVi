"""Bảng thuật ngữ cờ vua EN -> VI. Chuẩn duy nhất của toàn dự án.

Bổ sung thuật ngữ thì sửa :data:`GLOSSARY` (và bảng trong CLAUDE.md) trước,
không bao giờ tự chế cách dịch trong prompt hay trong dữ liệu.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "GLOSSARY",
    "GlossaryViolation",
    "VARIANTS",
    "canonical",
    "enforce_glossary",
    "glossary_violations",
]

#: Thuật ngữ tiếng Anh -> cách dịch tiếng Việt duy nhất được chấp nhận.
GLOSSARY: dict[str, str] = {
    "fork": "đòn đôi",
    "pin": "ghim",
    "skewer": "xiên",
    "discovered attack": "đòn mở",
    "back rank mate": "chiếu hết hàng cuối",
    "en passant": "bắt tốt qua đường",
    "castling": "nhập thành",
    "zugzwang": "zugzwang",
    "blunder": "sai lầm nghiêm trọng",
    "hanging piece": "quân bỏ ngỏ",
}

#: Cách dịch sai/biến thể hay gặp -> khoá tiếng Anh tương ứng trong GLOSSARY.
#: Dùng để ép nhất quán và để T6 phát hiện dữ liệu dịch lệch chuẩn.
VARIANTS: dict[str, str] = {
    # fork
    "nĩa": "fork",
    "đòn nĩa": "fork",
    "chĩa đôi": "fork",
    "đòn chĩa": "fork",
    "tấn công đôi": "fork",
    # pin
    "ghim chặt": "pin",
    "đòn ghim": "pin",
    "sự ghim": "pin",
    # skewer
    "đòn xiên": "skewer",
    "xiên que": "skewer",
    # discovered attack
    "tấn công mở": "discovered attack",
    "đòn tấn công mở": "discovered attack",
    "khám phá tấn công": "discovered attack",
    # back rank mate
    "chiếu bí hàng cuối": "back rank mate",
    "chiếu hết hàng ngang cuối": "back rank mate",
    "chiếu hết hàng ngang cuối cùng": "back rank mate",
    # en passant
    "bắt tốt qua đường đi": "en passant",
    "ăn tốt qua đường": "en passant",
    "bắt qua đường": "en passant",
    # castling
    "nhập thành trì": "castling",
    "nhập cung": "castling",
    "đổi thành": "castling",
    # blunder
    "nước đi tồi": "blunder",
    "sai lầm lớn": "blunder",
    "lỗi nghiêm trọng": "blunder",
    # hanging piece
    "quân treo": "hanging piece",
    "quân lơ lửng": "hanging piece",
    "quân không được bảo vệ": "hanging piece",
}


@dataclass(frozen=True)
class GlossaryViolation:
    """Một chỗ trong text dùng sai/lệch chuẩn thuật ngữ."""

    term_en: str
    found: str
    expected: str


def canonical(term_en: str) -> str:
    """Cách dịch chuẩn của một thuật ngữ tiếng Anh."""
    key = term_en.strip().lower()
    if key not in GLOSSARY:
        raise KeyError(f"{term_en!r} chưa có trong bảng thuật ngữ; thêm vào GLOSSARY trước")
    return GLOSSARY[key]


def _pattern(phrase: str) -> re.Pattern[str]:
    """Regex khớp cụm từ theo biên từ, chịu được khoảng trắng thừa."""
    parts = [re.escape(word) for word in phrase.split()]
    return re.compile(r"(?<!\w)" + r"\s+".join(parts) + r"(?!\w)", re.IGNORECASE)


# Cụm dài khớp trước cụm ngắn ("hanging piece" trước "pin"); chuẩn bị sẵn một lần.
_EN_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (_pattern(en), vi) for en, vi in sorted(GLOSSARY.items(), key=lambda kv: -len(kv[0]))
]
_VARIANT_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    (_pattern(variant), en, GLOSSARY[en])
    for variant, en in sorted(VARIANTS.items(), key=lambda kv: -len(kv[0]))
]


def enforce_glossary(text: str) -> str:
    """Ép mọi thuật ngữ trong ``text`` về đúng cách dịch chuẩn.

    Thay cả thuật ngữ còn sót tiếng Anh lẫn biến thể tiếng Việt lệch chuẩn.
    Không đụng tới placeholder dạng ``<M0>``/``<F0>`` vì không thuật ngữ nào
    khớp được chúng.
    """
    result = text
    for pattern, _en, vi in _VARIANT_PATTERNS:
        result = pattern.sub(vi, result)
    for pattern, vi in _EN_PATTERNS:
        result = pattern.sub(vi, result)
    return result


def glossary_violations(text: str) -> list[GlossaryViolation]:
    """Liệt kê thuật ngữ dùng sai chuẩn trong ``text`` (không sửa gì)."""
    found: list[GlossaryViolation] = []
    for pattern, en, vi in _VARIANT_PATTERNS:
        for match in pattern.finditer(text):
            found.append(GlossaryViolation(term_en=en, found=match.group(0), expected=vi))
    for pattern, vi in _EN_PATTERNS:
        en = next(k for k, v in GLOSSARY.items() if v == vi)
        # "zugzwang" giữ nguyên tiếng Anh nên không tính là vi phạm.
        if en == vi:
            continue
        for match in pattern.finditer(text):
            found.append(GlossaryViolation(term_en=en, found=match.group(0), expected=vi))
    return found

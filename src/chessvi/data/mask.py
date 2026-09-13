"""Che ký hiệu cờ trước khi dịch, khôi phục sau khi dịch.

Nguyên tắc 3 trong CLAUDE.md: ký hiệu cờ không bao giờ đi qua máy dịch. Máy
dịch sẽ biến ``Nf3`` thành ``Mã f3``, ``Bxc6`` thành ``Bxc6.`` hoặc nuốt luôn
``O-O``, và toàn bộ label thành rác mà không ai phát hiện ra.

Hai cạm bẫy chính, cả hai đều được xử lý ở đây:

1. **Bỏ sót** - ký hiệu lọt qua máy dịch.
2. **Bắt nhầm** - từ tiếng Anh bình thường bị coi là nước đi. ``Be careful``,
   ``Bed``, ``a4 paper size`` đều trông giống SAN nếu regex lỏng.

Với nhóm 2, nước tốt của quân tốt (``e4``, ``a4``) là chỗ nguy hiểm nhất vì
trùng hoàn toàn với từ/ký hiệu đời thường. Chiến lược:

- regex CHẶT cho mọi lớp ký hiệu;
- nước tốt trần chỉ được coi là nước đi khi trong text có ít nhất một ký hiệu
  cờ *chắc chắn* khác (nước có tên quân, ăn quân, phong cấp, nhập thành, UCI,
  FEN, số thứ tự nước đi);
- nếu truyền ``context_fen`` thì bỏ hẳn suy đoán: chỉ mask khi
  ``board.parse_san()`` / ``Move.from_uci()`` thành công.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import chess

logger = logging.getLogger(__name__)

__all__ = [
    "FEN_RE",
    "MAX_VARIATION_LINES",
    "MOVE_NUMBER_RE",
    "PLACEHOLDER_RE",
    "SAN_RE",
    "UCI_RE",
    "MoveContext",
    "Token",
    "find_move_tokens",
    "find_tokens",
    "is_unambiguous_move",
    "parse_move",
    "mask",
    "unmask",
]

TokenKind = Literal["fen", "uci", "san", "movenum"]

#: Tiền tố placeholder theo loại token.
_PREFIX: dict[TokenKind, str] = {"fen": "F", "uci": "M", "san": "M", "movenum": "N"}

#: Ưu tiên khi hai regex chồng nhau - số nhỏ thắng.
_PRIORITY: dict[TokenKind, int] = {"fen": 0, "uci": 1, "san": 2, "movenum": 3}

_NOT_WORD_BEFORE = r"(?<![0-9A-Za-z_])"
_NOT_WORD_AFTER = r"(?![0-9A-Za-z_])"

#: FEN đầy đủ 6 trường.
FEN_RE = re.compile(
    _NOT_WORD_BEFORE
    + r"[rnbqkpRNBQKP1-8]{1,8}(?:/[rnbqkpRNBQKP1-8]{1,8}){7}"
    + r"\s+[wb]\s+(?:-|K?Q?k?q?)\s+(?:-|[a-h][36])\s+\d{1,3}\s+\d{1,4}"
    + _NOT_WORD_AFTER
)

#: UCI: e2e4, a7a8q.
UCI_RE = re.compile(_NOT_WORD_BEFORE + r"[a-h][1-8][a-h][1-8][qrbnQRBN]?" + _NOT_WORD_AFTER)

_CASTLE = r"(?:O-O-O|O-O|0-0-0|0-0)"
_PIECE_MOVE = r"[KQRBN][a-h]?[1-8]?x?[a-h][1-8]"
_PAWN_CAPTURE = r"[a-h]x[a-h][1-8]"
_PAWN_MOVE = r"[a-h][1-8]"

#: SAN: Nf3, Bxc6+, O-O, exd5, a8=Q#, Rfe1, kèm dấu bình luận !? nếu có.
SAN_RE = re.compile(
    _NOT_WORD_BEFORE
    + rf"(?:{_CASTLE}|(?:{_PIECE_MOVE}|{_PAWN_CAPTURE}|{_PAWN_MOVE})(?:=[QRBN])?)"
    + r"[+#]?[!?]{0,2}"
    + _NOT_WORD_AFTER
)

#: Số thứ tự nước đi: "1." "23...". Chỉ tính khi ngay sau đó là một nước đi,
#: nếu không thì "anh ấy thắng 3." cũng bị bắt.
MOVE_NUMBER_RE = re.compile(
    r"(?<!\d)\d{1,3}\.(?:\.\.)?(?!\d)(?=\s*(?:[KQRBNO0]|[a-h][1-8]))"
)

#: Placeholder đã sinh ra: <M0>, <F1>, <N2>.
PLACEHOLDER_RE = re.compile(r"<([MFN])(\d+)>")

#: Lỏng hơn, để vá lại placeholder bị máy dịch làm xộc xệch: "< M0 >", "<m0>",
#: dấu ngoặc nhọn full-width.
_LENIENT_PLACEHOLDER_RE = re.compile(r"[<＜]\s*([MFNmfn])\s*(\d+)\s*[>＞]")


@dataclass(frozen=True)
class Token:
    """Một ký hiệu cờ tìm thấy trong text."""

    start: int
    end: int
    text: str
    kind: TokenKind


def _is_bare_pawn_move(text: str) -> bool:
    """``e4`` có, ``exd5``/``Nf3``/``a8=Q`` không."""
    return re.fullmatch(_PAWN_MOVE + r"[+#]?[!?]{0,2}", text) is not None


def is_unambiguous_move(token: Token) -> bool:
    """Token chắc chắn là nước đi, không thể là thứ khác.

    ``Nf3``, ``e2e4``, ``exd5``, ``O-O`` thì chắc chắn. ``e4`` thì **không**:
    nước tốt SAN trùng hệt cú pháp với tên ô, mà văn giải thích cờ thì đầy tên
    ô ("vua trắng ở h2", "các tốt f2 và h3"). Ai đối chiếu tính hợp lệ của
    từng token đều phải lọc bằng hàm này trước, nếu không sẽ loại nhầm gần như
    mọi mẫu.
    """
    return _is_strong(token)


def _is_strong(token: Token) -> bool:
    """Ký hiệu đủ đặc trưng để không thể là từ tiếng Anh bình thường."""
    if token.kind in ("fen", "uci", "movenum"):
        return True
    if token.text.startswith("0"):  # "0-0" cũng là tỷ số bóng đá
        return False
    return not _is_bare_pawn_move(token.text)


def _candidates(text: str) -> Iterator[Token]:
    for pattern, kind in (
        (FEN_RE, "fen"),
        (UCI_RE, "uci"),
        (SAN_RE, "san"),
        (MOVE_NUMBER_RE, "movenum"),
    ):
        for match in pattern.finditer(text):
            yield Token(match.start(), match.end(), match.group(0), kind)  # type: ignore[arg-type]


def _resolve_overlaps(tokens: list[Token]) -> list[Token]:
    """Giữ token không chồng nhau: ưu tiên loại mạnh hơn, rồi khớp dài hơn."""
    ordered = sorted(
        tokens,
        key=lambda t: (t.start, _PRIORITY[t.kind], -(t.end - t.start)),
    )
    kept: list[Token] = []
    last_end = -1
    for token in ordered:
        if token.start < last_end:
            continue
        kept.append(token)
        last_end = token.end
    return kept


def _adjacent_to_hyphen(text: str, token: Token) -> bool:
    before = text[token.start - 1] if token.start > 0 else ""
    after = text[token.end] if token.end < len(text) else ""
    return before == "-" or after == "-"


def parse_move(board: chess.Board, text: str) -> chess.Move | None:
    """Parse một chuỗi SAN hoặc UCI thành nước đi hợp lệ trên ``board``.

    Nguyên tắc 2: không tin chuỗi thô. Trả ``None`` nếu không parse được hoặc
    parse được nhưng không hợp lệ — không bao giờ trả nước ảo.
    """
    cleaned = text.strip().rstrip("!?")
    try:
        return board.parse_san(cleaned)
    except ValueError:
        pass
    try:
        move = chess.Move.from_uci(cleaned.lower())
    except ValueError:
        return None
    return move if move in board.legal_moves else None


#: Trần số thế cờ giữ trong cây biến của :class:`MoveContext`.
MAX_VARIATION_LINES = 32


class MoveContext:
    """Xác thực token bằng FEN ngữ cảnh.

    Giữ một **cây biến**, không phải một ván đơn. Văn giải thích cờ luôn rẽ
    nhánh - "sau c4, đen có dxc4 hoặc e6, còn nếu b5 thì a4" - nên một bàn cờ
    đang chạy duy nhất không biểu diễn nổi. Với một bàn duy nhất, nhánh thứ hai
    trở đi bị kết tội oan toàn bộ.

    Một nước được nhận khi hợp lệ với **bất kỳ** thế nào đã tới được (thế gốc,
    hoặc thế sinh ra từ các nước đã nhận trước đó). Nhận xong thì thế mới được
    thêm vào cây. Thử thế mới nhất trước, thế gốc sau cùng, để văn kể tuần tự
    vẫn đi đúng đường.

    Cây bị chặn ở ``max_lines`` thế: cây càng rộng thì càng dễ có nước bịa vô
    tình hợp lệ ở một nhánh nào đó. Đầy thì bỏ nhánh cũ nhất, thế gốc giữ lại.

    Trần 32 chọn theo số đo, trên thế sau 1.d4 d5 với 307 ký hiệu có chữ quân
    không hợp lệ, chèn vào một câu giải thích Gambit Hậu dài có rẽ nhánh:

        trần   bắt được nước bịa   câu đúng bị báo nhầm
           2               95.8%                      5
           8               93.8%                      1
          16               89.6%                      1
          32               89.3%                      1
          64               89.3%                      1

    Từ 12 trở lên đường cong phẳng, nên trần thấp chỉ mua thêm báo nhầm. Trong
    câu ngắn một nước ("Nên đi Qh5") thì vẫn bắt đủ **100%** - cây chưa kịp
    rộng. Phần 10% lọt ở câu dài phần lớn là nước hợp lệ thật trong một biến mà
    chính câu đó vừa dựng ra, không phải nước bịa.

    Giới hạn còn lại: văn **lược** nước. "Nếu đen chơi b5 thì a4 bxc4" bỏ mất
    ``c4`` ở đầu, nên không thế nào tới được ``bxc4`` và nó bị kết tội oan.
    Không có cách nào biết nước bị lược là nước gì.
    """

    def __init__(self, context_fen: str, *, max_lines: int = MAX_VARIATION_LINES) -> None:
        self._origin = chess.Board(context_fen)
        self._reachable = [self._origin]
        self._seen = {self._origin.epd()}
        self._max_lines = max(1, max_lines)

    def accepts(self, token: Token) -> bool:
        if token.kind == "movenum":
            return True
        if token.kind == "fen":
            try:
                chess.Board(token.text)
            except ValueError:
                return False
            return True
        return self._accepts_move(token.text)

    def _accepts_move(self, text: str) -> bool:
        """Mở nhánh ở **mọi** thế mà nước này hợp lệ, không chỉ thế gần nhất.

        Một ký hiệu như ``b5`` thường hợp lệ ở nhiều nhánh, và không cách nào
        biết văn bản đang nói nhánh nào. Chọn đại một nhánh là hỏng: ``b5``
        bám nhầm vào nhánh vừa ăn mất tốt c4 thì ``bxc4`` ngay sau đó không
        còn thế nào để hợp lệ, và bị kết tội oan.
        """
        found = [
            (board, move)
            for board in reversed(self._reachable)
            if (move := parse_move(board, text)) is not None
        ]
        for board, move in found:
            self._extend(board, move)
        return bool(found)

    def _extend(self, board: chess.Board, move: chess.Move) -> None:
        """Thêm thế sau ``move`` vào cây, trừ khi đã có."""
        nxt = board.copy(stack=False)
        nxt.push(move)
        key = nxt.epd()
        if key in self._seen:
            return
        if len(self._reachable) >= self._max_lines:
            if len(self._reachable) < 2:
                return  # trần bằng 1: chỉ còn chỗ cho thế gốc
            self._seen.discard(self._reachable.pop(1).epd())  # thế gốc ở lại
        self._reachable.append(nxt)
        self._seen.add(key)


def find_tokens(
    text: str,
    context_fen: str | None = None,
    *,
    require_chess_context: bool = True,
) -> list[Token]:
    """Mọi ký hiệu cờ trong ``text``, đã khử chồng lấn, theo thứ tự xuất hiện.

    Không có ``context_fen`` thì nước tốt trần (``e4``, ``a5``) chỉ được nhận
    khi trong text có ít nhất một ký hiệu cờ chắc chắn khác — nếu không thì
    "a4 paper size" cũng thành nước đi.

    ``require_chess_context=False`` tắt cái chốt đó: dùng khi *đã biết chắc*
    text là văn bản cờ và thà bắt nhầm còn hơn bỏ sót (``serve/guard.py``:
    bỏ sót một nước bịa là lọt thẳng ra người dùng).
    """
    tokens = _resolve_overlaps(list(_candidates(text)))
    if context_fen is not None:
        validator = MoveContext(context_fen)
        return [token for token in tokens if validator.accepts(token)]

    if not require_chess_context:
        return tokens

    has_strong = any(_is_strong(token) for token in tokens)
    kept: list[Token] = []
    for token in tokens:
        if _is_strong(token):
            kept.append(token)
            continue
        if not has_strong:
            continue
        if _adjacent_to_hyphen(text, token):  # "a4-paper"
            continue
        kept.append(token)
    return kept


def find_move_tokens(
    text: str,
    context_fen: str | None = None,
    *,
    require_chess_context: bool = True,
) -> list[Token]:
    """Chỉ các token là nước đi (SAN/UCI), bỏ FEN và số thứ tự nước.

    ``guard.py`` và ``data/validate.py`` dùng hàm này để không phải viết lại
    regex - một bản regex duy nhất cho cả dự án.
    """
    tokens = find_tokens(text, context_fen, require_chess_context=require_chess_context)
    return [token for token in tokens if token.kind in ("san", "uci")]


def _next_free_index(text: str) -> dict[str, int]:
    """Bắt đầu đánh số trên mọi placeholder đã có sẵn trong text.

    Nếu text gốc vô tình chứa ``<M0>`` thì ta không được sinh trùng, nếu không
    ``unmask`` sẽ phá hỏng chính chuỗi gốc đó.
    """
    start = {"M": 0, "F": 0, "N": 0}
    for match in PLACEHOLDER_RE.finditer(text):
        prefix, number = match.group(1), int(match.group(2))
        start[prefix] = max(start[prefix], number + 1)
    return start


def mask(text: str, context_fen: str | None = None) -> tuple[str, dict[str, str]]:
    """Thay mọi ký hiệu cờ bằng placeholder.

    Trả về ``(text_đã_mask, mapping)`` với ``mapping`` là placeholder -> chuỗi
    gốc. Cùng một ký hiệu xuất hiện nhiều lần dùng chung một placeholder.

    ``unmask(mask(text)[0], mapping)`` luôn bằng ``text``.
    """
    tokens = find_tokens(text, context_fen)
    if not tokens:
        return text, {}

    counters = _next_free_index(text)
    mapping: dict[str, str] = {}
    seen: dict[tuple[str, str], str] = {}
    pieces: list[str] = []
    cursor = 0
    for token in tokens:
        prefix = _PREFIX[token.kind]
        key = (prefix, token.text)
        placeholder = seen.get(key)
        if placeholder is None:
            placeholder = f"<{prefix}{counters[prefix]}>"
            counters[prefix] += 1
            seen[key] = placeholder
            mapping[placeholder] = token.text
        pieces.append(text[cursor : token.start])
        pieces.append(placeholder)
        cursor = token.end
    pieces.append(text[cursor:])
    return "".join(pieces), mapping


def unmask(text: str, mapping: dict[str, str], *, lenient: bool = True) -> str:
    """Khôi phục ký hiệu cờ từ placeholder.

    ``lenient`` vá thêm các placeholder bị máy dịch làm xộc xệch (``< M0 >``,
    ``<m0>``, ngoặc full-width). Placeholder không có trong ``mapping`` được
    giữ nguyên - T6 sẽ bắt chúng như lỗi unmask.
    """
    result = text
    for placeholder, original in mapping.items():
        result = result.replace(placeholder, original)
    if not lenient:
        return result

    def repair(match: re.Match[str]) -> str:
        canonical = f"<{match.group(1).upper()}{match.group(2)}>"
        replacement = mapping.get(canonical)
        if replacement is None:
            return match.group(0)
        logger.debug("Vá placeholder hỏng %r -> %r", match.group(0), replacement)
        return replacement

    return _LENIENT_PLACEHOLDER_RE.sub(repair, result)

"""Trích xuất fact kiểm chứng được từ một thế cờ.

Đây là lớp chống hallucination: mọi điều LLM được phép khẳng định về bàn cờ
đều phải xuất phát từ đây. Trừ ``eval_cp``/``eval_mate`` (hỏi Stockfish), tất
cả các trường đều tính thuần bằng ``python-chess``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import chess

from chessvi.engine.stockfish import StockfishEngine

logger = logging.getLogger(__name__)

__all__ = [
    "OPENING_BOOK",
    "PIECE_VALUES",
    "PositionFacts",
    "extract_facts",
    "facts_to_prompt",
]

#: Quy ước quy đổi lực lượng. Vua không tính.
PIECE_VALUES: dict[chess.PieceType, int] = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}

PIECE_NAMES_VI: dict[chess.PieceType, str] = {
    chess.PAWN: "tốt",
    chess.KNIGHT: "mã",
    chess.BISHOP: "tượng",
    chess.ROOK: "xe",
    chess.QUEEN: "hậu",
    chess.KING: "vua",
}

#: Các biến khai cuộc phổ biến, khai báo bằng chuỗi SAN cho dễ đọc.
_OPENING_LINES: dict[tuple[str, ...], str] = {
    ("e4",): "Khai cuộc tốt vua (1.e4)",
    ("d4",): "Khai cuộc tốt hậu (1.d4)",
    ("c4",): "Khai cuộc Anh",
    ("Nf3",): "Khai cuộc Réti",
    ("e4", "c5"): "Phòng thủ Sicilia",
    ("e4", "e6"): "Phòng thủ Pháp",
    ("e4", "c6"): "Phòng thủ Caro-Kann",
    ("e4", "d5"): "Phòng thủ Scandinavia",
    ("e4", "d6"): "Phòng thủ Pirc",
    ("e4", "g6"): "Phòng thủ hiện đại",
    ("e4", "Nf6"): "Phòng thủ Alekhine",
    ("e4", "e5"): "Ván cờ mở",
    ("e4", "e5", "f4"): "Gambit vua",
    ("e4", "e5", "Nf3", "d6"): "Phòng thủ Philidor",
    ("e4", "e5", "Nf3", "Nf6"): "Phòng thủ Nga (Petrov)",
    ("e4", "e5", "Nf3", "Nc6"): "Ván cờ mở, 2...Mc6",
    ("e4", "e5", "Nf3", "Nc6", "Bb5"): "Khai cuộc Tây Ban Nha (Ruy Lopez)",
    ("e4", "e5", "Nf3", "Nc6", "Bc4"): "Khai cuộc Ý",
    ("e4", "e5", "Nf3", "Nc6", "d4"): "Ván cờ Scotch",
    ("e4", "e5", "Nc3"): "Ván cờ Viên",
    ("d4", "d5"): "Khai cuộc tốt hậu, 1...d5",
    ("d4", "d5", "c4"): "Gambit hậu",
    ("d4", "d5", "c4", "e6"): "Gambit hậu từ chối",
    ("d4", "d5", "c4", "dxc4"): "Gambit hậu chấp nhận",
    ("d4", "Nf6"): "Phòng thủ Ấn Độ",
    ("d4", "Nf6", "c4", "g6"): "Phòng thủ Ấn Độ cổ",
    ("d4", "Nf6", "c4", "e6"): "Hệ thống Ấn Độ hậu",
    ("d4", "Nf6", "c4", "e6", "Nc3", "Bb4"): "Phòng thủ Nimzo-Ấn Độ",
    ("d4", "f5"): "Phòng thủ Hà Lan",
}


def _build_opening_book(lines: Mapping[tuple[str, ...], str]) -> dict[str, str]:
    """Đổi từ chuỗi SAN sang tra cứu theo EPD (bỏ bộ đếm nước)."""
    book: dict[str, str] = {}
    for sans, name in lines.items():
        board = chess.Board()
        try:
            for san in sans:
                board.push_san(san)
        except ValueError:  # pragma: no cover - lỗi gõ trong bảng trên
            logger.error("Biến khai cuộc không hợp lệ: %s", sans)
            continue
        book[board.epd()] = name
    return book


#: EPD -> tên khai cuộc tiếng Việt.
OPENING_BOOK: dict[str, str] = _build_opening_book(_OPENING_LINES)


@dataclass(frozen=True)
class PositionFacts:
    """Sự thật đã xác minh về một thế cờ.

    ``material_diff`` tính theo góc nhìn **Trắng**: dương nghĩa là Trắng hơn
    quân. ``eval_cp``/``eval_mate`` theo góc nhìn **bên đang đi** (xem
    :class:`~chessvi.engine.stockfish.EngineAnalysis`) và là None khi không
    truyền engine vào.
    """

    fen: str
    side_to_move: str  # "white" hoặc "black"
    legal_moves: list[str]
    material: dict[str, int]
    material_diff: int
    hanging: list[tuple[str, str]]
    in_check: bool
    checks_available: list[str]
    captures_available: list[str]
    eval_cp: int | None = None
    eval_mate: int | None = None
    best_move: str | None = None
    opening_name: str | None = None
    is_checkmate: bool = False
    is_stalemate: bool = False
    fullmove_number: int = 1
    #: Quân của bên đang đi đang bị bỏ ngỏ - tập con của ``hanging``.
    hanging_own: list[tuple[str, str]] = field(default_factory=list)

    @property
    def is_game_over(self) -> bool:
        return self.is_checkmate or self.is_stalemate


def _material(board: chess.Board) -> dict[str, int]:
    totals = {"white": 0, "black": 0}
    for piece in board.piece_map().values():
        key = "white" if piece.color == chess.WHITE else "black"
        totals[key] += PIECE_VALUES[piece.piece_type]
    return totals


def _hanging(board: chess.Board) -> list[tuple[str, str]]:
    """Quân bị tấn công mà không có quân cùng màu bảo vệ.

    Bỏ qua vua: vua không bao giờ bị ăn, thế bị chiếu đã có ``in_check``.
    """
    out: list[tuple[str, str]] = []
    for square in sorted(board.piece_map()):
        piece = board.piece_at(square)
        if piece is None or piece.piece_type == chess.KING:
            continue
        if not board.attackers(not piece.color, square):
            continue
        if board.attackers(piece.color, square):
            continue
        out.append((chess.square_name(square), piece.symbol()))
    return out


def _lookup_opening(board: chess.Board, book: Mapping[str, str]) -> str | None:
    """Tên khai cuộc, ưu tiên biến sâu nhất khớp được.

    Nếu ``board`` còn giữ ``move_stack`` thì đi lại từ đầu để bắt được cả
    trường hợp thế hiện tại đã vượt ra ngoài sách.
    """
    name = book.get(board.epd())
    if name is not None:
        return name
    if not board.move_stack:
        return None
    replay = chess.Board()
    deepest: str | None = None
    for move in board.move_stack:
        replay.push(move)
        found = book.get(replay.epd())
        if found is not None:
            deepest = found
    return deepest


def extract_facts(
    board: chess.Board,
    engine: StockfishEngine | None = None,
    opening_book: Mapping[str, str] | None = None,
    *,
    depth: int | None = None,
) -> PositionFacts:
    """Tính toàn bộ fact của ``board``.

    ``engine`` là tuỳ chọn: không có thì ``eval_cp``/``eval_mate``/``best_move``
    để None chứ không đoán. ``opening_book`` mặc định là :data:`OPENING_BOOK`.
    """
    book = OPENING_BOOK if opening_book is None else opening_book
    legal = list(board.legal_moves)

    checks: list[str] = []
    captures: list[str] = []
    legal_san: list[str] = []
    for move in legal:
        san = board.san(move)
        legal_san.append(san)
        if board.is_capture(move):
            captures.append(san)
        if board.gives_check(move):
            checks.append(san)

    material = _material(board)
    hanging = _hanging(board)
    own_is_upper = board.turn == chess.WHITE
    hanging_own = [
        (square, symbol) for square, symbol in hanging if symbol.isupper() == own_is_upper
    ]

    eval_cp: int | None = None
    eval_mate: int | None = None
    best_move: str | None = None
    if engine is not None:
        analysis = engine.analyse(board, depth=depth)
        eval_cp = analysis.cp
        eval_mate = analysis.mate
        if analysis.best_move is not None:
            best_move = board.san(analysis.best_move)

    return PositionFacts(
        fen=board.fen(),
        side_to_move="white" if board.turn == chess.WHITE else "black",
        legal_moves=legal_san,
        material=material,
        material_diff=material["white"] - material["black"],
        hanging=hanging,
        in_check=board.is_check(),
        checks_available=checks,
        captures_available=captures,
        eval_cp=eval_cp,
        eval_mate=eval_mate,
        best_move=best_move,
        opening_name=_lookup_opening(board, book),
        is_checkmate=board.is_checkmate(),
        is_stalemate=board.is_stalemate(),
        fullmove_number=board.fullmove_number,
        hanging_own=hanging_own,
    )


# -- render sang prompt ---------------------------------------------------

SIDE_VI: dict[str, str] = {"white": "Trắng", "black": "Đen"}


def _describe_piece(symbol: str) -> str:
    piece = chess.Piece.from_symbol(symbol)
    side = SIDE_VI["white"] if piece.color == chess.WHITE else SIDE_VI["black"]
    return f"{PIECE_NAMES_VI[piece.piece_type]} {side}"


def _format_hanging(hanging: Sequence[tuple[str, str]]) -> str:
    if not hanging:
        return "không có"
    return ", ".join(f"{_describe_piece(symbol)} ở {square}" for square, symbol in hanging)


def _format_moves(moves: Sequence[str], limit: int = 12) -> str:
    if not moves:
        return "không có"
    shown = ", ".join(moves[:limit])
    if len(moves) > limit:
        shown += f", ... (tổng {len(moves)})"
    return shown


def _format_material(facts: PositionFacts) -> str:
    diff = facts.material_diff
    if diff == 0:
        balance = "cân bằng"
    else:
        leader = SIDE_VI["white"] if diff > 0 else SIDE_VI["black"]
        balance = f"{leader} hơn {abs(diff)} điểm quân"
    white, black = facts.material["white"], facts.material["black"]
    return f"Trắng {white} - Đen {black} ({balance})"


def _format_eval(facts: PositionFacts) -> str | None:
    side = SIDE_VI[facts.side_to_move]
    if facts.eval_mate is not None:
        if facts.eval_mate > 0:
            return f"{side} có chiếu hết sau {facts.eval_mate} nước"
        if facts.eval_mate < 0:
            return f"{side} bị chiếu hết sau {abs(facts.eval_mate)} nước"
        return f"{side} đã bị chiếu hết"
    if facts.eval_cp is None:
        return None
    pawns = facts.eval_cp / 100
    if abs(facts.eval_cp) < 30:
        verdict = "thế cân bằng"
    elif facts.eval_cp > 0:
        verdict = f"{side} đang ưu thế"
    else:
        verdict = f"{side} đang bất lợi"
    return f"{pawns:+.2f} tốt theo góc nhìn {side} ({verdict})"


def facts_to_prompt(facts: PositionFacts, *, move_limit: int = 20) -> str:
    """Render fact thành block tiếng Việt để nhét vào prompt.

    Dùng thuật ngữ trong :mod:`chessvi.data.glossary`. Block này là *sự thật
    đã xác minh*; prompt sẽ dặn model không được mâu thuẫn với nó.

    ``move_limit`` cắt danh sách nước hợp lệ cho đỡ tốn token khi phục vụ.
    Nâng lên khi cần liệt kê đủ: prompt dặn model chỉ được nhắc nước có trong
    danh sách, nên danh sách cắt mất nước cần nói là chặn trần chính model.
    """
    side = SIDE_VI[facts.side_to_move]
    in_check_vi = "có" if facts.in_check else "không"
    lines: list[str] = [
        "[SỰ THẬT ĐÃ XÁC MINH - tính bằng python-chess/Stockfish]",
        f"FEN: {facts.fen}",
        f"Lượt đi: {side} (nước thứ {facts.fullmove_number})",
        f"Cán cân lực lượng: {_format_material(facts)}",
        f"Đang bị chiếu: {in_check_vi}",
        f"Quân bỏ ngỏ: {_format_hanging(facts.hanging)}",
        f"Nước ăn quân khả dụng: {_format_moves(facts.captures_available)}",
        f"Nước chiếu khả dụng: {_format_moves(facts.checks_available)}",
        f"Nước đi hợp lệ: {_format_moves(facts.legal_moves, limit=move_limit)}",
    ]
    if facts.opening_name:
        lines.append(f"Khai cuộc: {facts.opening_name}")
    evaluation = _format_eval(facts)
    if evaluation is not None:
        lines.append(f"Đánh giá của Stockfish: {evaluation}")
    if facts.best_move:
        lines.append(f"Nước tốt nhất theo Stockfish: {facts.best_move}")
    if facts.is_checkmate:
        lines.append(f"Ván đã kết thúc: {side} bị chiếu hết.")
    elif facts.is_stalemate:
        lines.append("Ván đã kết thúc: hết nước đi (hòa).")
    return "\n".join(lines)

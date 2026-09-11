"""Test trích xuất fact. Mỗi trường có FEN dựng riêng, khẳng định chính xác."""

from __future__ import annotations

import chess
import pytest

from chessvi.engine.facts import (
    OPENING_BOOK,
    PositionFacts,
    extract_facts,
    facts_to_prompt,
)
from chessvi.engine.stockfish import StockfishEngine
from tests.conftest import FEN_CHECKMATED, FEN_IN_CHECK, FEN_MATE_IN_1, FEN_STALEMATE, FEN_START

#: Mã đen e5 bị xe e1 tấn công, không quân nào bảo vệ. Trắng đi.
FEN_HANGING_KNIGHT = "4k3/8/8/4n3/8/8/8/4R1K1 w - - 0 1"
#: Cùng thế nhưng Đen đi - quân bỏ ngỏ giờ là quân của bên đang đi.
FEN_HANGING_OWN = "4k3/8/8/4n3/8/8/8/4R1K1 b - - 0 1"


def _facts(fen: str) -> PositionFacts:
    return extract_facts(chess.Board(fen))


# -- từng trường ----------------------------------------------------------


def test_fen_va_ben_di_khop_board() -> None:
    facts = _facts(FEN_START)
    assert facts.fen == FEN_START
    assert facts.side_to_move == "white"
    assert facts.fullmove_number == 1
    assert _facts(FEN_IN_CHECK).side_to_move == "black"


def test_legal_moves_la_san_va_parse_lai_duoc() -> None:
    board = chess.Board(FEN_START)
    facts = extract_facts(board)
    assert len(facts.legal_moves) == board.legal_moves.count() == 20
    # Nguyên tắc 2: mọi chuỗi nước đi phải parse lại được.
    for san in facts.legal_moves:
        assert board.parse_san(san) in board.legal_moves


def test_material_theo_quy_uoc_1_3_3_5_9() -> None:
    facts = _facts(FEN_START)
    # 8*1 + 2*3 + 2*3 + 2*5 + 9 = 39
    assert facts.material == {"white": 39, "black": 39}
    assert facts.material_diff == 0

    lech = _facts(FEN_HANGING_KNIGHT)
    assert lech.material == {"white": 5, "black": 3}
    assert lech.material_diff == 2, "dương nghĩa là Trắng hơn quân"


def test_hanging_bat_dung_quan_bo_ngo() -> None:
    facts = _facts(FEN_HANGING_KNIGHT)
    assert facts.hanging == [("e5", "n")]
    assert facts.hanging_own == [], "Trắng đi, quân bỏ ngỏ là của Đen"

    facts_black = _facts(FEN_HANGING_OWN)
    assert facts_black.hanging_own == [("e5", "n")]


def test_hanging_bo_qua_quan_duoc_bao_ve() -> None:
    # Tốt f7 bị Tc4 và Hf3 tấn công nhưng vua e8 bảo vệ - không phải bỏ ngỏ.
    facts = _facts(FEN_MATE_IN_1)
    assert ("f7", "p") not in facts.hanging


def test_hanging_khong_bao_gio_tinh_vua() -> None:
    facts = _facts(FEN_IN_CHECK)
    assert facts.in_check is True
    assert all(symbol.lower() != "k" for _square, symbol in facts.hanging)


def test_in_check() -> None:
    assert _facts(FEN_IN_CHECK).in_check is True
    assert _facts(FEN_START).in_check is False


def test_checks_available() -> None:
    facts = _facts(FEN_MATE_IN_1)
    assert set(facts.checks_available) == {"Bxf7+", "Qxf7#"}
    assert _facts(FEN_START).checks_available == []


def test_captures_available_gom_ca_bat_tot_qua_duong() -> None:
    # Trắng có thể bắt tốt qua đường: exd6.
    facts = _facts("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3")
    assert "exd6" in facts.captures_available


def test_eval_none_khi_khong_truyen_engine() -> None:
    facts = _facts(FEN_START)
    assert facts.eval_cp is None
    assert facts.eval_mate is None
    assert facts.best_move is None


def test_opening_name_tra_cuu_duoc() -> None:
    board = chess.Board()
    for san in ("e4", "e5", "Nf3", "Nc6", "Bb5"):
        board.push_san(san)
    assert extract_facts(board).opening_name == "Khai cuộc Tây Ban Nha (Ruy Lopez)"


def test_opening_name_tu_fen_thuan_khong_can_move_stack() -> None:
    fen = "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2"
    assert _facts(fen).opening_name == "Phòng thủ Sicilia"


def test_opening_name_lay_bien_sau_nhat_khi_ra_khoi_sach() -> None:
    board = chess.Board()
    for san in ("e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6"):
        board.push_san(san)
    assert board.epd() not in OPENING_BOOK
    assert extract_facts(board).opening_name == "Khai cuộc Tây Ban Nha (Ruy Lopez)"


def test_opening_name_none_khi_khong_biet() -> None:
    assert _facts(FEN_HANGING_KNIGHT).opening_name is None


def test_the_ket_thuc() -> None:
    mated = _facts(FEN_CHECKMATED)
    assert mated.is_checkmate and mated.is_game_over
    assert mated.legal_moves == []

    stalemated = _facts(FEN_STALEMATE)
    assert stalemated.is_stalemate and not stalemated.is_checkmate


def test_opening_book_tuy_chinh_duoc() -> None:
    board = chess.Board()
    board.push_san("e4")
    facts = extract_facts(board, opening_book={board.epd(): "Sách riêng"})
    assert facts.opening_name == "Sách riêng"


# -- render prompt --------------------------------------------------------


def test_facts_to_prompt_dung_thuat_ngu_chuan() -> None:
    text = facts_to_prompt(_facts(FEN_HANGING_KNIGHT))
    assert "SỰ THẬT ĐÃ XÁC MINH" in text
    assert "quân bỏ ngỏ" in text.lower(), "phải dùng thuật ngữ trong bảng glossary"
    assert "mã Đen ở e5" in text
    assert FEN_HANGING_KNIGHT in text


def test_facts_to_prompt_khong_bia_eval_khi_khong_co_engine() -> None:
    text = facts_to_prompt(_facts(FEN_START))
    assert "Stockfish" not in text.split("\n", 1)[1], "không được bịa eval"


def test_facts_to_prompt_noi_ro_van_da_ket_thuc() -> None:
    assert "bị chiếu hết" in facts_to_prompt(_facts(FEN_CHECKMATED))
    assert "hòa" in facts_to_prompt(_facts(FEN_STALEMATE))


def test_facts_to_prompt_cat_bot_danh_sach_dai() -> None:
    facts = _facts(FEN_MATE_IN_1)  # 42 nước hợp lệ, vượt ngưỡng hiển thị 20
    text = facts_to_prompt(facts)
    assert f"(tổng {len(facts.legal_moves)})" in text
    # Danh sách 20 nước của thế khởi đầu thì in đủ, không cắt.
    assert "tổng" not in facts_to_prompt(_facts(FEN_START))


# -- cần binary (tự skip) -------------------------------------------------


def test_eval_lay_tu_stockfish(stockfish: StockfishEngine) -> None:
    board = chess.Board(FEN_START)
    facts = extract_facts(board, engine=stockfish, depth=10)
    assert facts.eval_cp is not None
    assert facts.eval_mate is None
    assert board.parse_san(facts.best_move or "") in board.legal_moves


def test_eval_mate_duoc_ghi_nhan(stockfish: StockfishEngine) -> None:
    facts = extract_facts(chess.Board(FEN_MATE_IN_1), engine=stockfish, depth=10)
    assert facts.eval_mate == 1
    assert facts.eval_cp is None
    assert "chiếu hết sau 1 nước" in facts_to_prompt(facts)


def test_eval_the_ket_thuc_khong_crash(stockfish: StockfishEngine) -> None:
    facts = extract_facts(chess.Board(FEN_STALEMATE), engine=stockfish)
    assert facts.eval_cp is None
    assert facts.best_move is None


@pytest.mark.parametrize("fen", [FEN_START, FEN_IN_CHECK, FEN_HANGING_KNIGHT])
def test_moi_nuoc_trong_prompt_deu_hop_le(fen: str, stockfish: StockfishEngine) -> None:
    """Block prompt không được chứa nước đi nào không hợp lệ."""
    board = chess.Board(fen)
    facts = extract_facts(board, engine=stockfish, depth=8)
    for san in facts.legal_moves + facts.checks_available + facts.captures_available:
        assert board.parse_san(san) in board.legal_moves

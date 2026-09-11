"""Test wrapper Stockfish. Test cần binary tự skip, không fail."""

from __future__ import annotations

import chess
import pytest

from chessvi.engine.stockfish import EngineAnalysis, EngineNotOpenError, StockfishEngine
from tests.conftest import FEN_CHECKMATED, FEN_MATE_IN_1, FEN_STALEMATE, FEN_START

# -- không cần binary -----------------------------------------------------


def test_the_ket_thuc_khong_goi_engine() -> None:
    """Thế đã kết thúc phải trả kết quả rỗng mà không chạm tới process."""
    engine = StockfishEngine("khong-ton-tai")  # cố tình không mở
    for fen in (FEN_STALEMATE, FEN_CHECKMATED):
        result = engine.analyse(chess.Board(fen))
        assert result.best_move is None
        assert result.cp is None
        assert result.pv == []
        assert result.is_terminal


def test_chieu_het_co_mate_0_hoa_thi_khong() -> None:
    engine = StockfishEngine("khong-ton-tai")
    assert engine.analyse(chess.Board(FEN_CHECKMATED)).mate == 0
    assert engine.analyse(chess.Board(FEN_STALEMATE)).mate is None


def test_chua_mo_thi_bao_loi_ro_rang() -> None:
    engine = StockfishEngine("khong-ton-tai")
    with pytest.raises(EngineNotOpenError):
        engine.analyse(chess.Board(FEN_START))


@pytest.mark.parametrize("level", [-1, 21, 100])
def test_skill_level_ngoai_khoang_bi_tu_choi(level: int) -> None:
    engine = StockfishEngine("khong-ton-tai")
    with pytest.raises(ValueError):
        engine.configure_skill(level)


def test_engine_analysis_mac_dinh_pv_rong() -> None:
    analysis = EngineAnalysis(best_move=None, cp=None, mate=None)
    assert analysis.pv == []
    assert not analysis.is_mate


# -- cần binary (tự skip) -------------------------------------------------


def test_the_khoi_dau_tra_nuoc_hop_le(stockfish: StockfishEngine) -> None:
    board = chess.Board(FEN_START)
    result = stockfish.analyse(board, depth=10)
    assert result.best_move in board.legal_moves
    assert result.mate is None
    assert result.cp is not None
    # Thế khởi đầu cân bằng; cho biên rộng vì depth thấp.
    assert abs(result.cp) < 150
    assert result.pv and result.pv[0] == result.best_move


def test_mate_in_1(stockfish: StockfishEngine) -> None:
    board = chess.Board(FEN_MATE_IN_1)
    result = stockfish.analyse(board, depth=10)
    assert result.mate == 1, "phải thấy chiếu hết sau 1 nước"
    assert result.cp is None
    assert board.san(result.best_move) == "Qxf7#"


def test_stalemate_khong_crash(stockfish: StockfishEngine) -> None:
    result = stockfish.analyse(chess.Board(FEN_STALEMATE), depth=10)
    assert result == EngineAnalysis(best_move=None, cp=None, mate=None, pv=[])


def test_process_dong_sach_sau_context_manager(stockfish_path: str) -> None:
    with StockfishEngine(stockfish_path) as engine:
        assert engine.is_open
    assert not engine.is_open
    with pytest.raises(EngineNotOpenError):
        engine.analyse(chess.Board(FEN_START))


def test_process_dong_sach_khi_co_exception(stockfish_path: str) -> None:
    engine = StockfishEngine(stockfish_path)
    with pytest.raises(ZeroDivisionError):
        with engine:
            1 / 0
    assert not engine.is_open


def test_play_ton_trong_skill_level(stockfish_path: str) -> None:
    board = chess.Board(FEN_START)
    with StockfishEngine(stockfish_path) as engine:
        engine.configure_skill(3)
        move = engine.play(board, depth=6)
    assert move in board.legal_moves


def test_play_tra_none_khi_het_van(stockfish: StockfishEngine) -> None:
    assert stockfish.play(chess.Board(FEN_STALEMATE)) is None

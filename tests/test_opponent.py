"""Test lớp đối thủ. Phần routing thuần không cần binary."""

from __future__ import annotations

import chess
import pytest

from chessvi.config import EngineConfig, Paths
from chessvi.engine.opponent import Opponent, depth_for_elo, route_engine, skill_for_elo
from tests.conftest import FEN_STALEMATE, FEN_START

ELO_LEVELS = [1000, 1500, 2000, 2500]


# -- routing thuần --------------------------------------------------------


@pytest.mark.parametrize(
    ("elo", "expected"),
    [(800, "maia"), (1500, "maia"), (2200, "maia"), (2201, "stockfish"), (2600, "stockfish")],
)
def test_route_theo_nguong_2200(elo: int, expected: str) -> None:
    assert route_engine(elo) == expected


def test_skill_va_depth_tang_dan_theo_elo() -> None:
    skills = [skill_for_elo(e) for e in range(2200, 2601, 50)]
    depths = [depth_for_elo(e) for e in range(2200, 2601, 50)]
    assert skills == sorted(skills)
    assert depths == sorted(depths)
    assert skills[0] == 10 and skills[-1] == 20
    assert depths[-1] == 18


@pytest.mark.parametrize("elo", [-100, 0, 5000])
def test_skill_luon_trong_khoang_uci(elo: int) -> None:
    assert 0 <= skill_for_elo(elo) <= 20
    assert depth_for_elo(elo) >= 1


def test_config_tuy_chinh_duoc_nguong() -> None:
    cfg = EngineConfig(maia_elo_ceiling=1900)
    assert route_engine(2000, cfg) == "stockfish"
    assert route_engine(1800, cfg) == "maia"


def test_fen_hong_bi_tu_choi() -> None:
    with Opponent() as bot:
        with pytest.raises(ValueError):
            bot.get_move("khong-phai-fen", 1500)


def test_van_da_ket_thuc_thi_bao_loi() -> None:
    with Opponent() as bot:
        with pytest.raises(ValueError, match="kết thúc"):
            bot.get_move(FEN_STALEMATE, 1500)


def test_thieu_maia_thi_bao_loi_ro_rang(monkeypatch: pytest.MonkeyPatch) -> None:
    """Không được im lặng hạ cấp sang Stockfish khi chưa cho phép."""
    monkeypatch.setattr(Paths, "resolve_maia", lambda self: None)
    with Opponent() as bot:
        with pytest.raises(FileNotFoundError, match="Maia"):
            bot.get_move(FEN_START, 1500)


# -- cần binary (tự skip) -------------------------------------------------


@pytest.mark.parametrize("elo", ELO_LEVELS)
def test_get_move_hop_le_o_moi_muc_elo(elo: int, paths: Paths) -> None:
    if route_engine(elo) == "maia":
        if paths.resolve_maia() is None:
            pytest.skip(f"không tìm thấy Maia ({paths.maia_bin}) — bỏ qua")
    elif paths.resolve_stockfish() is None:
        pytest.skip(f"không tìm thấy Stockfish ({paths.stockfish_bin}) — bỏ qua")

    board = chess.Board(FEN_START)
    with Opponent() as bot:
        move = bot.get_move(FEN_START, elo)
    assert move in board.legal_moves


def test_choi_lien_tiep_dung_lai_mot_process(stockfish_path: str) -> None:
    """Nhiều nước liên tiếp không được rò rỉ process mới mỗi lần."""
    board = chess.Board()
    with Opponent() as bot:
        for _ in range(4):
            move = bot.get_move(board.fen(), 2500)
            assert move in board.legal_moves
            board.push(move)
    assert board.fullmove_number >= 2

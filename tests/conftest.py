"""Fixture dùng chung. Test phải chạy được không GPU, không mạng."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from chessvi.config import Paths

# FEN cố định dùng xuyên suốt test — đặt tên để không phải đọc chuỗi.
FEN_START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
#: Trắng đi Qh5# hoặc Qf7# — mate in 1 (Scholar's mate đã dàn xong).
FEN_MATE_IN_1 = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
#: Đen bị stalemate, đến lượt đen đi.
FEN_STALEMATE = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"
#: Trắng đã bị chiếu hết (fool's mate), đến lượt trắng.
FEN_CHECKMATED = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
#: Đen đang bị chiếu bởi tượng b5, còn đường gỡ.
FEN_IN_CHECK = "rnbqkbnr/ppp2ppp/3p4/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 1 4"


@pytest.fixture(scope="session")
def paths() -> Paths:
    return Paths()


@pytest.fixture(scope="session")
def stockfish_path(paths: Paths) -> str:
    """Đường dẫn Stockfish, skip cả test nếu máy không có binary."""
    found = paths.resolve_stockfish()
    if found is None:
        pytest.skip(f"không tìm thấy Stockfish ({paths.stockfish_bin}) — bỏ qua")
    return found


@pytest.fixture(scope="session")
def maia_path(paths: Paths) -> str:
    """Đường dẫn Maia, skip cả test nếu máy không có binary."""
    found = paths.resolve_maia()
    if found is None:
        pytest.skip(f"không tìm thấy Maia ({paths.maia_bin}) — bỏ qua")
    return found


@pytest.fixture
def stockfish(stockfish_path: str) -> Iterator[object]:
    """StockfishEngine đã mở sẵn, đóng sạch sau test."""
    from chessvi.engine.stockfish import StockfishEngine

    with StockfishEngine(stockfish_path) as engine:
        yield engine

"""Test web demo. Không GPU, không mạng ngoài — server chạy trên localhost.

Trọng tâm là biên I/O: chuỗi từ trình duyệt không được tin, mọi nước đi phải
qua parser rồi mới được đi (nguyên tắc 2).
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from http.server import ThreadingHTTPServer
from typing import Any

import chess
import pytest

from chessvi.serve.web import (
    _Bot,
    _make_handler,
    build_parser,
    build_state,
    parse_move,
)
from tests.conftest import FEN_CHECKMATED, FEN_IN_CHECK, FEN_STALEMATE, FEN_START


# -- parse_move: biên không tin được -------------------------------------


def test_nuoc_hop_le_thi_di_duoc() -> None:
    board = chess.Board(FEN_START)
    assert parse_move(board, "e2", "e4") == chess.Move.from_uci("e2e4")


def test_nuoc_khong_hop_le_bi_tu_choi() -> None:
    """Nguyên tắc 2: không tin chuỗi thô từ trình duyệt."""
    board = chess.Board(FEN_START)
    with pytest.raises(ValueError, match="không hợp lệ"):
        parse_move(board, "e2", "e5")


def test_chuoi_rac_khong_lam_no_server() -> None:
    board = chess.Board(FEN_START)
    with pytest.raises(ValueError, match="không đọc được"):
        parse_move(board, "zz", "99")


def test_tot_len_hang_cuoi_mac_dinh_phong_hau() -> None:
    """Bàn cờ chỉ có hai ô được bấm, không có chỗ hỏi thêm quân phong cấp."""
    board = chess.Board("8/P6k/8/8/8/8/8/K7 w - - 0 1")
    assert parse_move(board, "a7", "a8") == chess.Move.from_uci("a7a8q")


def test_phong_cap_quan_khac_van_ton_trong() -> None:
    board = chess.Board("8/P6k/8/8/8/8/8/K7 w - - 0 1")
    assert parse_move(board, "a7", "a8", "n") == chess.Move.from_uci("a7a8n")


# -- build_state ----------------------------------------------------------


def test_state_co_du_thu_de_ve_lai_ban_co() -> None:
    state = build_state(chess.Board(FEN_START))
    assert state["fen"] == FEN_START
    assert state["turn"] == "white"
    assert state["turn_vi"] == "Trắng"
    assert state["game_over"] is False
    assert state["result"] is None
    assert len(state["legal"]["e2"]) == 2, "tốt e2 đi được e3 và e4"
    assert state["facts"]["legal_count"] == 20


def test_state_bao_dang_bi_chieu() -> None:
    state = build_state(chess.Board(FEN_IN_CHECK))
    assert state["facts"]["in_check"] is True


def test_state_bao_chieu_het_kem_ly_do() -> None:
    state = build_state(chess.Board(FEN_CHECKMATED))
    assert state["game_over"] is True
    assert state["result"] == "Đen thắng — chiếu hết"
    assert state["legal"] == {}


def test_state_bao_hoa_do_het_nuoc_di() -> None:
    state = build_state(chess.Board(FEN_STALEMATE))
    assert state["game_over"] is True
    assert state["result"] is not None
    assert state["result"].startswith("Hoà")


def test_state_kem_khoi_fact_tho() -> None:
    """Panel hiển thị đúng khối sẽ nhét vào prompt chatbot — nguyên tắc 1."""
    block = build_state(chess.Board(FEN_START))["facts_block"]
    assert "[SỰ THẬT ĐÃ XÁC MINH" in block
    assert FEN_START in block


# -- HTTP -----------------------------------------------------------------


@pytest.fixture
def base_url() -> Iterator[str]:
    """Server thật trên cổng tự chọn. Opponent mở engine lười nên chưa tốn gì."""
    bot = _Bot(allow_stockfish_fallback=True)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(bot))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        bot.close()


def _post(url: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_van_moi_tra_ve_the_khoi_dau(base_url: str) -> None:
    status, data = _post(f"{base_url}/api/new", {})
    assert status == 200
    assert data["fen"] == FEN_START


def test_di_mot_nuoc_qua_http(base_url: str) -> None:
    status, data = _post(f"{base_url}/api/move", {"fen": FEN_START, "from": "e2", "to": "e4"})
    assert status == 200
    assert data["last_move"] == "e4"
    assert data["turn"] == "black"


def test_di_nhieu_nuoc_lien_tiep(base_url: str) -> None:
    """Hồi quy: quân đã tiến lên rồi đi tiếp, ở thế ngoài sách khai cuộc.

    Mỗi request dựng board mới từ FEN, nên ``move_stack`` chỉ có một nước. Chừng
    nào nước đó xuất phát từ ô còn quân ở thế khởi đầu thì không lộ gì — phải
    để một quân tiến lên rồi đi tiếp mới thấy.
    """
    fen = FEN_START
    for src, dst in (("e2", "e4"), ("a7", "a6"), ("e4", "e5")):
        status, data = _post(f"{base_url}/api/move", {"fen": fen, "from": src, "to": dst})
        assert status == 200, data
        fen = data["fen"]
    assert data["last_move"] == "e5"


def test_nuoc_khong_hop_le_tra_400_chu_khong_sap_server(base_url: str) -> None:
    status, data = _post(f"{base_url}/api/move", {"fen": FEN_START, "from": "e2", "to": "e5"})
    assert status == 400
    assert "không hợp lệ" in data["error"]
    # Server vẫn phục vụ request tiếp theo.
    assert _post(f"{base_url}/api/new", {})[0] == 200


def test_fen_hong_tra_400(base_url: str) -> None:
    status, data = _post(f"{base_url}/api/move", {"fen": "khong-phai-fen", "from": "e2", "to": "e4"})
    assert status == 400
    assert data["error"]


def test_endpoint_la_khong_tra_404(base_url: str) -> None:
    assert _post(f"{base_url}/api/khong-co", {})[0] == 404


def test_trang_chu_phuc_vu_duoc(base_url: str) -> None:
    with urllib.request.urlopen(f"{base_url}/", timeout=10) as response:
        body = response.read().decode("utf-8")
    assert response.status == 200
    assert "<title>chess-vi" in body


def test_khong_cho_thoat_ra_ngoai_thu_muc_static(base_url: str) -> None:
    """``..`` trong đường dẫn không được đọc file ngoài static/."""
    request = urllib.request.Request(f"{base_url}/../../pyproject.toml")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            body = response.read().decode("utf-8", "replace")
        assert "[project]" not in body
    except urllib.error.HTTPError as error:
        assert error.code == 404


# -- CLI ------------------------------------------------------------------


def test_cli_mac_dinh_cho_stockfish_thay_maia() -> None:
    """Demo phải đi được trên máy chưa tải binary Maia."""
    assert build_parser().parse_args([]).strict_maia is False
    assert build_parser().parse_args(["--strict-maia"]).strict_maia is True

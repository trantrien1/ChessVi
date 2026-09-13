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
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from typing import Any

import chess
import pytest

from chessvi.config import ServeConfig
from chessvi.serve.chat import LocalLLM
from chessvi.serve.colab_api import build_parser as colab_parser
from chessvi.serve.colab_api import extract_prompt
from chessvi.serve.web import (
    _Analyst,
    _Bot,
    _make_handler,
    build_llm,
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


@contextmanager
def _running(llm: LocalLLM | None) -> Iterator[str]:
    """Server thật trên cổng tự chọn. Engine mở lười nên chưa tốn gì."""
    bot = _Bot(allow_stockfish_fallback=True)
    analyst = _Analyst()
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), _make_handler(bot, analyst, llm, ServeConfig())
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        bot.close()
        analyst.close()


@pytest.fixture
def base_url() -> Iterator[str]:
    with _running(None) as url:
        yield url


class _FakeLLM(LocalLLM):
    """Trả lần lượt các câu đã dựng sẵn, ghi lại prompt và nhiệt độ."""

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.prompts: list[str] = []
        self.temperatures: list[float] = []

    def generate(self, prompt: str, *, temperature: float) -> str:
        self.prompts.append(prompt)
        self.temperatures.append(temperature)
        return self.answers[min(len(self.prompts) - 1, len(self.answers) - 1)]


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


# -- chatbot --------------------------------------------------------------


def test_chua_bat_chatbot_thi_bao_503_kem_cach_bat(base_url: str) -> None:
    status, data = _post(f"{base_url}/api/explain", {"fen": FEN_START})
    assert status == 503
    assert "--llm remote" in data["error"], "báo lỗi phải nói luôn cách sửa"


def test_giai_thich_tra_ve_cau_tra_loi() -> None:
    llm = _FakeLLM("Trắng nên phát triển quân. Nf3 là nước tự nhiên.")
    with _running(llm) as url:
        status, data = _post(f"{url}/api/explain", {"fen": FEN_START})
    assert status == 200
    assert data["answer"].startswith("Trắng nên phát triển")
    assert data["question"]


def test_prompt_gui_cho_model_co_khoi_fact() -> None:
    """Nguyên tắc 1: model chỉ được diễn giải fact, không tự suy trạng thái."""
    llm = _FakeLLM("Nước Nf3 phát triển mã.")
    with _running(llm) as url:
        _post(f"{url}/api/explain", {"fen": FEN_START, "question": "Nên đi gì?"})
    prompt = llm.prompts[0]
    assert "[SỰ THẬT ĐÃ XÁC MINH" in prompt
    assert FEN_START in prompt
    assert "Nên đi gì?" in prompt


def test_nuoc_bia_bi_guard_chan_roi_lui_ve_fact() -> None:
    """Model bịa nước không hợp lệ thì thà trả lời khô khan mà đúng."""
    llm = _FakeLLM("Hãy chơi Qh8, nước đó thắng ngay.")
    with _running(llm) as url:
        status, data = _post(f"{url}/api/explain", {"fen": FEN_START})

    assert status == 200
    assert "Qh8" not in data["answer"], "câu bịa không được lọt ra ngoài"
    assert "dữ kiện đã xác minh" in data["answer"]
    assert len(llm.prompts) == 3, "sinh lần đầu + 2 lần sinh lại"
    assert llm.temperatures[0] < llm.temperatures[-1], "nhiệt độ phải tăng dần"
    assert "LƯU Ý" in llm.prompts[1], "lần sinh lại phải kèm lời nhắc"


def test_cau_tra_loi_sach_thi_khong_sinh_lai() -> None:
    llm = _FakeLLM("Nước Nf3 phát triển mã và kiểm soát trung tâm.")
    with _running(llm) as url:
        _post(f"{url}/api/explain", {"fen": FEN_START})
    assert len(llm.prompts) == 1


# -- build_llm ------------------------------------------------------------


def test_build_llm_none_tra_ve_none() -> None:
    assert build_llm("none") is None


def test_build_llm_remote_thieu_api_base_thi_bao_ngay() -> None:
    with pytest.raises(ValueError, match="--api-base"):
        build_llm("remote")


def test_build_llm_backend_la_liet_ke_duoc() -> None:
    with pytest.raises(ValueError, match="none, remote, gguf"):
        build_llm("khong-co")


# -- endpoint phía Colab --------------------------------------------------


class _FakeTokenizer:
    """Ghi lại messages nhận được, trả chuỗi dễ kiểm."""

    def __init__(self) -> None:
        self.seen: list[Any] = []

    def apply_chat_template(
        self, turns: Any, *, tokenize: bool, add_generation_prompt: bool
    ) -> str:
        self.seen.append(turns)
        assert tokenize is False
        assert add_generation_prompt is True
        return "|".join(f"{t['role']}:{t['content']}" for t in turns)


def test_extract_prompt_dung_chat_template_cua_model() -> None:
    """T8 tokenize bằng apply_chat_template, nên nối chuỗi tay là sai khuôn."""
    tokenizer = _FakeTokenizer()
    prompt = extract_prompt(
        [{"role": "system", "content": "luật"}, {"role": "user", "content": "hỏi"}],
        tokenizer,
    )
    assert prompt == "system:luật|user:hỏi"
    assert len(tokenizer.seen) == 1


def test_extract_prompt_bo_phan_tu_khong_phai_dict() -> None:
    tokenizer = _FakeTokenizer()
    assert extract_prompt([{"role": "user", "content": "a"}, "rác"], tokenizer) == "user:a"


def test_extract_prompt_messages_rong_thi_bao_loi() -> None:
    with pytest.raises(ValueError, match="rỗng"):
        extract_prompt([], _FakeTokenizer())


def test_colab_api_cli_co_mac_dinh_hop_ly() -> None:
    args = colab_parser().parse_args(["--model-path", "Qwen/Qwen3-14B"])
    assert args.port == 8001
    # Phải nghe mọi interface, không thì tunnel không vào được.
    assert args.host == "0.0.0.0"  # noqa: S104
    assert args.adapter is None


def test_bao_cho_giao_dien_biet_khi_bi_guard_chan() -> None:
    """Im lặng đưa bản chỉ-fact thì người dùng tưởng trợ lý vốn nói cụt thế."""
    with _running(_FakeLLM("Chơi Qh8 là thắng.")) as url:
        _, blocked = _post(f"{url}/api/explain", {"fen": FEN_START})
    with _running(_FakeLLM("Nf3 phát triển mã, kiểm soát trung tâm.")) as url:
        _, clean = _post(f"{url}/api/explain", {"fen": FEN_START})

    assert blocked["blocked"] is True
    assert clean["blocked"] is False

"""Test lớp serving: prompt, guard, chat loop.

Nghiệm thu T10: guard phải bắt được **100%** nước đi không hợp lệ trong output.
Test ``test_bat_100_phan_tram_nuoc_khong_hop_le`` sinh toàn bộ chuỗi SAN đúng
dạng nhưng không hợp lệ với thế cờ rồi khẳng định không cái nào lọt.
"""

from __future__ import annotations

import chess
import pytest

from chessvi.config import ServeConfig
from chessvi.data.mask import parse_move
from chessvi.engine.facts import extract_facts
from chessvi.serve.chat import ChatSession, LocalLLM, _handle_command, run_loop
from chessvi.serve.guard import check_output, guarded_generate
from chessvi.serve.prompt import (
    SYSTEM_PROMPT,
    ChatTurn,
    build_messages,
    build_prompt,
    fallback_answer,
)
from tests.conftest import FEN_IN_CHECK, FEN_MATE_IN_1, FEN_START


class _ScriptedLLM(LocalLLM):
    """Trả lần lượt các câu đã soạn sẵn, ghi lại prompt nhận được."""

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.prompts: list[str] = []
        self.temperatures: list[float] = []

    def generate(self, prompt: str, *, temperature: float) -> str:
        self.prompts.append(prompt)
        self.temperatures.append(temperature)
        return self.answers[min(len(self.prompts) - 1, len(self.answers) - 1)]


# -- prompt ---------------------------------------------------------------


def test_prompt_dat_khoi_fact_ngay_truoc_cau_hoi() -> None:
    facts = extract_facts(chess.Board(FEN_START))
    prompt = build_prompt(facts, "Nên đi nước nào?")
    assert prompt.index(SYSTEM_PROMPT) == 0
    assert prompt.index("[SỰ THẬT ĐÃ XÁC MINH") < prompt.index("[CÂU HỎI]")
    assert prompt.rstrip().endswith("[TRẢ LỜI]")


def test_prompt_noi_ro_fact_la_su_that_da_xac_minh() -> None:
    assert "Nó luôn đúng" in SYSTEM_PROMPT
    assert "Không được nói bất cứ điều gì mâu thuẫn" in SYSTEM_PROMPT
    assert "Không bịa nước đi" in SYSTEM_PROMPT


def test_prompt_dan_giu_nguyen_ky_hieu_san() -> None:
    assert "Nf3" in SYSTEM_PROMPT, "phải dặn không dịch ký hiệu sang tiếng Việt"


def test_prompt_co_lich_su_hoi_thoai() -> None:
    facts = extract_facts(chess.Board(FEN_START))
    history = [ChatTurn("user", "Khai cuộc gì?"), ChatTurn("assistant", "Chưa đi nước nào.")]
    prompt = build_prompt(facts, "Còn bây giờ?", history)
    assert "[LỊCH SỬ HỘI THOẠI]" in prompt
    assert "Người dùng: Khai cuộc gì?" in prompt
    # rindex: chuỗi "[SỰ THẬT ĐÃ XÁC MINH]" còn xuất hiện trong SYSTEM_PROMPT.
    assert prompt.index("Khai cuộc gì?") < prompt.rindex("[SỰ THẬT ĐÃ XÁC MINH")


def test_build_messages_cung_thu_tu() -> None:
    facts = extract_facts(chess.Board(FEN_START))
    messages = build_messages(facts, "Hỏi gì đó", [ChatTurn("user", "Trước đó")])
    assert [m["role"] for m in messages] == ["system", "user", "user"]
    assert "[SỰ THẬT ĐÃ XÁC MINH" in messages[-1]["content"]


def test_fallback_chi_gom_fact() -> None:
    facts = extract_facts(chess.Board(FEN_IN_CHECK))
    answer = fallback_answer(facts)
    assert "Đen" in answer
    assert "Đang bị chiếu: có" in answer
    # Không được chứa nước đi nào không hợp lệ.
    assert check_output(answer, chess.Board(FEN_IN_CHECK)).ok


# -- guard ----------------------------------------------------------------


def test_guard_cho_qua_cau_tra_loi_dung() -> None:
    board = chess.Board(FEN_START)
    result = check_output("Nên đi Nf3 hoặc e4, cả hai đều phát triển.", board)
    assert result.ok
    assert set(result.checked) == {"Nf3", "e4"}
    assert bool(result) is True


def test_guard_bat_nuoc_bia() -> None:
    board = chess.Board(FEN_START)
    result = check_output("Nước mạnh nhất là Qh8, sau đó Nf3.", board)
    assert not result.ok
    assert result.violations == ("Qh8",)


def test_guard_bat_ca_nuoc_tot_tran_dung_mot_minh() -> None:
    """Không có ký hiệu cờ nào khác trong câu vẫn phải bắt được."""
    board = chess.Board(FEN_START)
    assert check_output("Cứ đi a5.", board).violations == ("a5",)


def test_guard_cho_phep_ke_lai_mot_bien_theo_thu_tu() -> None:
    board = chess.Board(FEN_START)
    result = check_output("Sau 1. e4 e5 2. Nf3 Nc6 thì thế cân bằng.", board)
    assert result.ok, result.violations


def test_guard_bat_nuoc_sai_giua_mot_bien() -> None:
    board = chess.Board(FEN_START)
    result = check_output("Sau 1. e4 e5 2. Qh8 thì hỏng.", board)
    assert result.violations == ("Qh8",)
    assert "e4" in result.checked, "các nước trước đó vẫn được soi"


def test_bat_100_phan_tram_nuoc_khong_hop_le() -> None:
    """Sinh mọi chuỗi đúng dạng SAN mà không trỏ tới nước hợp lệ nào.

    "Không hợp lệ" định nghĩa theo python-chess chứ không theo so khớp chuỗi:
    ``Qf7`` (thiếu dấu x) vẫn trỏ đúng một nước hợp lệ nên không phải bịa.
    """
    for fen in (FEN_START, FEN_IN_CHECK, FEN_MATE_IN_1):
        board = chess.Board(fen)
        candidates = [
            f"{piece}{file}{rank}"
            for piece in ("", "K", "Q", "R", "B", "N")
            for file in "abcdefgh"
            for rank in "12345678"
        ]
        illegal = [san for san in candidates if parse_move(board, san) is None]
        assert len(illegal) > 300, "phải có đủ nhiều case âm để test có ý nghĩa"

        missed = [
            san
            for san in illegal
            if san not in check_output(f"Nên đi {san}.", board).violations
        ]
        assert missed == [], f"{fen}: lọt {len(missed)} nước, ví dụ {missed[:5]}"


def test_guard_khong_bao_dong_nham_voi_nuoc_hop_le() -> None:
    for fen in (FEN_START, FEN_IN_CHECK, FEN_MATE_IN_1):
        board = chess.Board(fen)
        for move in board.legal_moves:
            san = board.san(move)
            result = check_output(f"Nên đi {san}.", board)
            assert result.ok, f"{fen}: báo nhầm {san} ({result.violations})"


def test_guard_text_khong_co_nuoc_nao() -> None:
    result = check_output("Thế cờ này cân bằng.", chess.Board(FEN_START))
    assert result.ok
    assert result.checked == ()


# -- regenerate + fallback ------------------------------------------------


def test_regenerate_roi_tra_ve_cau_dung() -> None:
    board = chess.Board(FEN_START)
    facts = extract_facts(board)
    attempts: list[int] = []

    def generate(attempt: int) -> str:
        attempts.append(attempt)
        return "Nên đi Qh8." if attempt == 0 else "Nên đi Nf3."

    answer = guarded_generate(generate, board, facts, max_regenerations=2)
    assert answer == "Nên đi Nf3."
    assert attempts == [0, 1]


def test_het_luot_thi_lui_ve_fallback() -> None:
    board = chess.Board(FEN_START)
    facts = extract_facts(board)
    calls = 0

    def generate(attempt: int) -> str:
        nonlocal calls
        calls += 1
        return "Nên đi Qh8."

    answer = guarded_generate(generate, board, facts, max_regenerations=2)
    assert calls == 3, "1 lần đầu + 2 lần sinh lại"
    assert answer == fallback_answer(facts)
    assert check_output(answer, board).ok, "fallback không bao giờ được vi phạm"


# -- chat session ---------------------------------------------------------


def test_ask_di_qua_guard_va_luu_lich_su() -> None:
    llm = _ScriptedLLM("Nên đi Nf3 để phát triển.")
    session = ChatSession(llm)
    answer = session.ask("Nên đi nước nào?")
    assert answer == "Nên đi Nf3 để phát triển."
    assert [turn.role for turn in session.history] == ["user", "assistant"]
    assert "[SỰ THẬT ĐÃ XÁC MINH" in llm.prompts[0]


def test_ask_sinh_lai_khi_model_bia_nuoc() -> None:
    llm = _ScriptedLLM("Nên đi Qh8.", "Nên đi Nf3.")
    session = ChatSession(llm, serve_config=ServeConfig(max_regenerations=2))
    assert session.ask("Nước nào?") == "Nên đi Nf3."
    assert len(llm.prompts) == 2
    assert "LƯU Ý" in llm.prompts[1], "lần sinh lại phải nhắc model"
    assert llm.temperatures[1] > llm.temperatures[0]


def test_ask_lui_ve_fact_khi_model_bia_mai() -> None:
    llm = _ScriptedLLM("Nên đi Qh8.")
    session = ChatSession(llm, serve_config=ServeConfig(max_regenerations=2))
    answer = session.ask("Nước nào?")
    assert answer.startswith("Tôi chưa đưa ra được lời giải thích đáng tin")


def test_play_tu_choi_nuoc_khong_hop_le() -> None:
    session = ChatSession(_ScriptedLLM("ok"))
    with pytest.raises(ValueError, match="không phải nước đi hợp lệ"):
        session.play("Qh8")
    assert session.board.fen() == FEN_START, "bàn cờ không được đổi"


def test_play_nhan_ca_san_lan_uci() -> None:
    session = ChatSession(_ScriptedLLM("ok"))
    assert session.play("e4").uci() == "e2e4"
    assert session.play("e7e5").uci() == "e7e5"


def test_play_opponent_khong_co_doi_thu() -> None:
    session = ChatSession(_ScriptedLLM("ok"), opponent=None)
    with pytest.raises(RuntimeError, match="đối thủ"):
        session.play_opponent()


def test_status_mo_ta_dung_trang_thai() -> None:
    assert "Lượt Trắng" in ChatSession(_ScriptedLLM("ok")).status
    session = ChatSession(_ScriptedLLM("ok"), board=chess.Board(FEN_IN_CHECK))
    assert "đang bị chiếu" in session.status


def test_reset_xoa_ca_lich_su() -> None:
    session = ChatSession(_ScriptedLLM("Nên đi Nf3."))
    session.ask("Hỏi")
    session.play("e4")
    session.reset()
    assert session.history == []
    assert session.board.fen() == FEN_START


# -- CLI loop -------------------------------------------------------------


def test_lenh_di_va_fen(caplog: pytest.LogCaptureFixture) -> None:
    session = ChatSession(_ScriptedLLM("ok"))
    with caplog.at_level("INFO"):
        assert _handle_command(session, "/di e4") is True
        assert _handle_command(session, "/fen") is True
    assert "Bạn đi e2e4" in caplog.text
    assert session.board.fen().startswith("rnbqkbnr/pppppppp/8/8/4P3")


def test_lenh_di_sai_khong_lam_sap_vong_lap(caplog: pytest.LogCaptureFixture) -> None:
    session = ChatSession(_ScriptedLLM("ok"))
    with caplog.at_level("INFO"):
        assert _handle_command(session, "/di Qh8") is True
    assert "Lỗi:" in caplog.text


def test_lenh_thoat() -> None:
    assert _handle_command(ChatSession(_ScriptedLLM("ok")), "/thoat") is False


def test_lenh_la_thi_bao_khong_biet(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("INFO"):
        _handle_command(ChatSession(_ScriptedLLM("ok")), "/xyz")
    assert "Không biết lệnh" in caplog.text


def test_run_loop_xen_ke_choi_va_hoi(caplog: pytest.LogCaptureFixture) -> None:
    # Sau /di e4 là lượt Đen, nên câu trả lời phải là nước của Đen.
    llm = _ScriptedLLM("Nên đi Nf6.")
    session = ChatSession(llm)
    with caplog.at_level("INFO"):
        run_loop(session, ["/di e4", "", "Thế này ai hơn?", "/moi", "/thoat", "/di e4"])
    assert "Bạn đi e2e4" in caplog.text
    assert "Nên đi Nf6." in caplog.text
    assert "Ván mới." in caplog.text
    assert session.board.fen() == FEN_START, "/thoat phải dừng trước lệnh cuối"

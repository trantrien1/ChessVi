"""Test eval harness: accuracy puzzle và đo hallucination."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path

import chess
import pytest

from chessvi.engine.facts import extract_facts
from chessvi.eval.hallucination import (
    check_answer,
    evaluate_answers,
)
from chessvi.eval.hallucination import main as hallucination_main
from chessvi.eval.predictor import EchoPredictor, Predictor, build_predictor
from chessvi.eval.puzzle_acc import (
    build_facts_prompt,
    build_parser,
    build_puzzle_prompt,
    evaluate_puzzles,
    write_csv,
)
from chessvi.eval.puzzle_acc import main as puzzle_main
from chessvi.train.reward import ANSWER_TEMPLATE
from tests.conftest import FEN_IN_CHECK, FEN_MATE_IN_1, FEN_START

#: Thế khởi đầu, lời giải là Nf3.
PUZZLE_A = {
    "puzzle_id": "A",
    "fen": FEN_START,
    "solution": ["g1f3", "b8c6"],
    "rating_bucket": "1200-1599",
    "primary_theme": "fork",
}
#: Cùng bucket khác theme.
PUZZLE_B = {**PUZZLE_A, "puzzle_id": "B", "primary_theme": "pin"}
#: Khác bucket.
PUZZLE_C = {**PUZZLE_A, "puzzle_id": "C", "rating_bucket": "1600-1999"}


class _ByIdPredictor(Predictor):
    """Trả câu trả lời theo thứ tự đã soạn."""

    name = "by-id"

    def __init__(self, *answers: str) -> None:
        self._answers = list(answers)
        self._index = 0

    def predict(self, prompt: str) -> str:
        answer = self._answers[min(self._index, len(self._answers) - 1)]
        self._index += 1
        return answer


class _BatchPredictor(Predictor):
    """Ghi lại kích thước từng batch để kiểm tra harness có batch thật không."""

    name = "batch"

    def __init__(self, answer: str) -> None:
        self._answer = answer
        self.batch_sizes: list[int] = []

    def predict(self, prompt: str) -> str:  # pragma: no cover - không dùng
        return self._answer

    def predict_batch(self, prompts: Sequence[str]) -> list[str]:
        self.batch_sizes.append(len(prompts))
        return [self._answer] * len(prompts)


# -- predictor ------------------------------------------------------------


def test_backend_khong_biet_bi_tu_choi() -> None:
    with pytest.raises(ValueError, match="Backend"):
        build_predictor("gpt-4")


def test_echo_predictor_ghi_lai_prompt() -> None:
    predictor = EchoPredictor("Nước đi: Nf3")
    assert predictor.predict_batch(["a", "b"]) == ["Nước đi: Nf3"] * 2
    assert predictor.prompts == ["a", "b"]


def test_prompt_puzzle_co_fen_va_yeu_cau_format() -> None:
    prompt = build_puzzle_prompt(FEN_START)
    assert FEN_START in prompt
    # Chốt vào ANSWER_TEMPLATE chứ không chép lại chuỗi: nhãn lệch giữa prompt
    # và phép trích xuất là lỗi đã xảy ra một lần, đừng để nó âm thầm quay lại.
    assert ANSWER_TEMPLATE.format(move="<nước đi>") in prompt


# -- prompt có fact: cấu hình sẽ ship -------------------------------------


def test_prompt_co_fact_dung_lai_khoi_cua_production() -> None:
    """Không dựng format thứ ba: eval và serving phải cùng một prompt."""
    prompt = build_facts_prompt(FEN_IN_CHECK)
    assert "[SỰ THẬT ĐÃ XÁC MINH" in prompt
    assert FEN_IN_CHECK in prompt
    assert "Đang bị chiếu: có" in prompt


def test_prompt_co_fact_khong_lo_dap_an() -> None:
    """``extract_facts`` không kèm engine, nếu không model chỉ việc chép lại.

    Lộ ``best_move`` vào prompt là biến bài đo thành bài chép, con số sẽ vô
    nghĩa theo chiều ngược lại với FEN trần.
    """
    prompt = build_facts_prompt(FEN_MATE_IN_1)
    assert "Nước tốt nhất theo Stockfish" not in prompt
    assert "Đánh giá của Stockfish" not in prompt


def test_prompt_co_fact_liet_ke_du_nuoc_hop_le() -> None:
    """Prompt dặn chỉ được nhắc nước trong danh sách; cắt danh sách là chặn trần."""
    board = chess.Board(FEN_MATE_IN_1)
    assert len(list(board.legal_moves)) > 20, "FEN này phải vượt mức cắt mặc định"
    prompt = build_facts_prompt(FEN_MATE_IN_1)
    assert "... (tổng" not in prompt
    for move in board.legal_moves:
        assert board.san(move) in prompt


def test_prompt_co_fact_doi_dung_nhan_ket_luan() -> None:
    assert ANSWER_TEMPLATE.format(move="<nước đi>") in build_facts_prompt(FEN_START)


def test_evaluate_dung_prompt_builder_duoc_truyen() -> None:
    predictor = EchoPredictor("FINAL_ANSWER: Nf3")
    evaluate_puzzles([PUZZLE_A], predictor, prompt_builder=build_facts_prompt)
    assert "[SỰ THẬT ĐÃ XÁC MINH" in predictor.prompts[0]


def test_fen_hong_bi_bo_qua_chu_khong_giet_ca_luot() -> None:
    """Prompt có fact parse FEN *trước khi* sinh, nên phải chặn ngay ở vòng lặp."""
    puzzles = [{**PUZZLE_A, "puzzle_id": "hong", "fen": "khong-phai-fen"}, PUZZLE_A]
    report = evaluate_puzzles(
        puzzles, EchoPredictor("FINAL_ANSWER: Nf3"), prompt_builder=build_facts_prompt
    )
    assert report.total == 1


def test_cli_with_facts_mac_dinh_tat() -> None:
    """Mặc định vẫn là FEN trần để so sánh được với các lượt chạy đã có."""
    assert build_parser().parse_args(["--puzzles", "x.parquet"]).with_facts is False
    assert build_parser().parse_args(["--puzzles", "x.parquet", "--with-facts"]).with_facts


# -- accuracy -------------------------------------------------------------


def test_accuracy_dem_dung() -> None:
    predictor = _ByIdPredictor("Nước đi: Nf3", "Nước đi: e4", "Nước đi: Qh8")
    report = evaluate_puzzles([PUZZLE_A, PUZZLE_B, PUZZLE_C], predictor, batch_size=1)
    assert report.total == 3
    assert report.correct == 1
    assert report.accuracy == pytest.approx(1 / 3)
    # Nf3 đúng, e4 hợp lệ nhưng sai, Qh8 không hợp lệ.
    assert report.legal_rate == pytest.approx(2 / 3)


def test_accuracy_tach_theo_bucket_va_theme() -> None:
    predictor = _ByIdPredictor("Nước đi: Nf3", "Nước đi: e4", "Nước đi: Nf3")
    report = evaluate_puzzles([PUZZLE_A, PUZZLE_B, PUZZLE_C], predictor, batch_size=1)
    assert report.by_bucket() == {"1200-1599": (1, 2), "1600-1999": (1, 1)}
    assert report.by_theme() == {"fork": (2, 2), "pin": (0, 1)}


def test_harness_goi_model_theo_batch() -> None:
    predictor = _BatchPredictor("Nước đi: Nf3")
    evaluate_puzzles([PUZZLE_A] * 5, predictor, batch_size=2)
    assert predictor.batch_sizes == [2, 2, 1]


def test_limit_duoc_ton_trong() -> None:
    report = evaluate_puzzles([PUZZLE_A] * 10, EchoPredictor("Nước đi: Nf3"), limit=3)
    assert report.total == 3


def test_puzzle_thieu_fen_bi_bo_qua() -> None:
    report = evaluate_puzzles(
        [PUZZLE_A, {"puzzle_id": "X"}], EchoPredictor("Nước đi: Nf3"), batch_size=1
    )
    assert report.total == 1


def test_bao_cao_rong_khong_chia_cho_0() -> None:
    report = evaluate_puzzles([], EchoPredictor(""))
    assert report.accuracy == 0.0
    assert report.legal_rate == 0.0


def test_ghi_csv_co_dong_tong_va_dong_tach(tmp_path: Path) -> None:
    predictor = _ByIdPredictor("Nước đi: Nf3", "Nước đi: e4", "Nước đi: Nf3")
    report = evaluate_puzzles([PUZZLE_A, PUZZLE_B, PUZZLE_C], predictor, batch_size=1)
    target = tmp_path / "acc.csv"
    write_csv(report, target, "model-thu-nghiem")

    with target.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    overall = [r for r in rows if r["slice_type"] == "overall"]
    assert len(overall) == 1
    assert overall[0]["accuracy"] == "0.6667"
    assert {r["slice_type"] for r in rows} == {"overall", "rating_bucket", "theme"}
    assert all(r["model"] == "model-thu-nghiem" for r in rows)


def test_cli_puzzle_acc(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "test.jsonl"
    source.write_text(
        "\n".join(json.dumps(p) for p in [PUZZLE_A, PUZZLE_B]), encoding="utf-8"
    )
    out = tmp_path / "acc.csv"
    assert puzzle_main(["--puzzles", str(source), "--backend", "echo", "--out", str(out)]) == 0
    assert "Accuracy tổng" in capsys.readouterr().out
    assert out.exists()


def test_cli_puzzle_acc_bao_loi_khi_thieu_model_path(tmp_path: Path) -> None:
    source = tmp_path / "test.jsonl"
    source.write_text(json.dumps(PUZZLE_A), encoding="utf-8")
    assert puzzle_main(["--puzzles", str(source), "--backend", "gguf"]) == 2


def test_cli_puzzle_acc_bao_loi_khi_tap_rong(tmp_path: Path) -> None:
    source = tmp_path / "empty.jsonl"
    source.write_text("", encoding="utf-8")
    assert puzzle_main(["--puzzles", str(source)]) == 1


# -- hallucination --------------------------------------------------------


def _kinds(text: str, fen: str = FEN_START) -> dict[str, list[bool]]:
    out: dict[str, list[bool]] = {}
    for claim in check_answer(text, fen):
        out.setdefault(claim.kind, []).append(claim.is_true)
    return out


def test_bat_khang_dinh_nuoc_di_sai() -> None:
    assert _kinds("Nên đi Qh8.")["move_legal"] == [False]
    assert _kinds("Nên đi Nf3.")["move_legal"] == [True]


def test_bat_khang_dinh_vi_tri_quan() -> None:
    assert _kinds("Mã Trắng ở g1 đang bảo vệ.")["piece_square"] == [True]
    assert _kinds("Mã Trắng ở g4 đang bảo vệ.")["piece_square"] == [False]
    assert _kinds("Hậu Đen ở d1.")["piece_square"] == [False], "d1 là hậu Trắng"


def test_vi_tri_quan_khong_bi_dem_thanh_nuoc_di() -> None:
    """"ở g4" là mô tả ô, không phải nước đi - không được tính là bịa nước."""
    assert "move_legal" not in _kinds("Mã Trắng ở g4.")


def test_bat_khang_dinh_can_can_luc_luong() -> None:
    assert _kinds("Cán cân lực lượng cân bằng.")["material"] == [True]
    assert _kinds("Trắng đang hơn quân.")["material"] == [False]
    assert _kinds("Trắng hơn 3 điểm quân.")["material"] == [False]


def test_bat_khang_dinh_dang_bi_chieu() -> None:
    assert _kinds("Bên đang đi không bị chiếu.")["in_check"] == [True]
    assert _kinds("Bên đang đi đang bị chiếu.")["in_check"] == [False]
    assert _kinds("Đen đang bị chiếu.", FEN_IN_CHECK)["in_check"] == [True]


def test_bat_khang_dinh_den_luot_ai() -> None:
    assert _kinds("Đến lượt Trắng.")["side_to_move"] == [True]
    assert _kinds("Đến lượt Đen.")["side_to_move"] == [False]


def test_cau_tra_loi_khong_co_khang_dinh_nao() -> None:
    assert check_answer("Thế cờ này thú vị.", FEN_START) == []


def test_truyen_san_facts_thi_khong_tinh_lai() -> None:
    facts = extract_facts(chess.Board(FEN_START))
    claims = check_answer("Đến lượt Trắng.", FEN_START, facts)
    assert claims[0].is_true


def test_ty_le_hallucination_tren_ca_tap() -> None:
    items = [
        {"fen": FEN_START, "answer": "Đến lượt Trắng. Nên đi Nf3."},
        {"fen": FEN_START, "answer": "Đến lượt Đen. Nên đi Qh8."},
    ]
    report = evaluate_answers(items)
    assert report.answers == 2
    assert report.total_claims == 4
    assert report.false_claims == 2
    assert report.hallucination_rate == pytest.approx(0.5)
    assert report.answers_with_error == 1
    assert report.answer_error_rate == pytest.approx(0.5)
    assert report.by_kind() == {"move_legal": (1, 2), "side_to_move": (1, 2)}


def test_bo_qua_mau_hong() -> None:
    items = [
        {"fen": FEN_START, "answer": "Nên đi Nf3."},
        {"fen": "khong-phai-fen", "answer": "Nên đi Nf3."},
        {"fen": FEN_START},
    ]
    assert evaluate_answers(items).answers == 1


def test_bao_cao_hallucination_render() -> None:
    report = evaluate_answers([{"fen": FEN_START, "answer": "Nên đi Qh8."}])
    text = report.render("model-thu-nghiem")
    assert "model-thu-nghiem" in text
    assert "Khẳng định SAI: 1" in text


def test_cli_hallucination_tu_file_answers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "answers.jsonl"
    source.write_text(
        json.dumps({"fen": FEN_START, "answer": "Nên đi Qh8."}, ensure_ascii=False),
        encoding="utf-8",
    )
    out = tmp_path / "hallu.csv"
    assert hallucination_main(["--answers", str(source), "--out", str(out)]) == 0
    assert "Khẳng định SAI" in capsys.readouterr().out
    assert out.exists()


def test_cli_hallucination_chay_model_tren_puzzle(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "test.jsonl"
    source.write_text(json.dumps(PUZZLE_A), encoding="utf-8")
    assert hallucination_main(["--puzzles", str(source), "--backend", "echo"]) == 0
    assert "Số câu trả lời: 1" in capsys.readouterr().out


def test_cli_hallucination_tap_rong(tmp_path: Path) -> None:
    source = tmp_path / "empty.jsonl"
    source.write_text("", encoding="utf-8")
    assert hallucination_main(["--answers", str(source)]) == 1

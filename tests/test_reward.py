"""Test hàm reward GRPO. Không cần GPU, không cần trl.

Reward sai dấu hoặc sai thang thì RL vẫn "chạy" mà model càng học càng tệ -
đây là chỗ rẻ nhất để bắt lỗi đó.
"""

from __future__ import annotations

import chess
import pytest

from chessvi.train.reward import (
    ANSWER_TEMPLATE,
    REWARD_CORRECT,
    REWARD_FORMAT_BONUS,
    REWARD_ILLEGAL,
    REWARD_LEGAL_BUT_WRONG,
    answer_is_complete,
    extract_move,
    has_valid_format,
    puzzle_reward,
    score_completion,
)
from tests.conftest import FEN_START

SOLUTION = ["g1f3", "b8c6"]  # Nf3 là nước phải đoán đúng


# -- bốn nhánh reward -----------------------------------------------------


def test_nhanh_1_dung_loi_giai() -> None:
    result = score_completion("Nước đi: Nf3", FEN_START, SOLUTION)
    assert result.move_reward == REWARD_CORRECT
    assert result.is_correct
    assert result.total == pytest.approx(REWARD_CORRECT + REWARD_FORMAT_BONUS)


def test_nhanh_2_hop_le_nhung_sai() -> None:
    result = score_completion("Nước đi: e4", FEN_START, SOLUTION)
    assert result.move_reward == REWARD_LEGAL_BUT_WRONG
    assert result.is_legal and not result.is_correct
    assert result.total == pytest.approx(REWARD_LEGAL_BUT_WRONG + REWARD_FORMAT_BONUS)


def test_nhanh_3_khong_hop_le() -> None:
    """Qh8 đúng dạng SAN nhưng không đi được từ thế khởi đầu."""
    result = score_completion("Nước đi: Qh8", FEN_START, SOLUTION)
    assert result.move_reward == REWARD_ILLEGAL
    assert not result.is_legal
    assert result.move is None


def test_nhanh_3_khong_parse_duoc() -> None:
    result = score_completion("Nước đi: xin lỗi tôi không biết", FEN_START, SOLUTION)
    assert result.move_reward == REWARD_ILLEGAL
    assert result.move is None


def test_nhanh_4_bonus_format_la_truc_rieng() -> None:
    co_format = score_completion("Nước đi: Nf3", FEN_START, SOLUTION)
    khong_format = score_completion("Tôi nghĩ nên chơi Nf3.", FEN_START, SOLUTION)

    assert co_format.format_bonus == REWARD_FORMAT_BONUS
    assert khong_format.format_bonus == 0.0
    # Cùng nước đúng, chỉ khác format -> chênh đúng bằng bonus.
    assert co_format.total - khong_format.total == pytest.approx(REWARD_FORMAT_BONUS)
    assert khong_format.move_reward == REWARD_CORRECT


def test_bonus_format_van_tinh_ca_khi_nuoc_sai() -> None:
    result = score_completion("Nước đi: Qh8", FEN_START, SOLUTION)
    assert result.has_format
    assert result.total == pytest.approx(REWARD_ILLEGAL + REWARD_FORMAT_BONUS)


# -- format ---------------------------------------------------------------


def test_format_phai_co_dung_mot_dong_ket_luan() -> None:
    assert has_valid_format("Suy nghĩ...\nNước đi: Nf3")
    assert not has_valid_format("Không có kết luận")
    assert not has_valid_format("Nước đi: Nf3\nNước đi: e4"), "hai dòng là sai format"


def test_format_khong_phan_biet_hoa_thuong() -> None:
    assert has_valid_format("nước đi: Nf3")


# -- trích nước đi --------------------------------------------------------


def test_trich_nuoc_di_uu_tien_dong_ket_luan() -> None:
    board = chess.Board(FEN_START)
    completion = "Có thể chơi e4 hoặc d4, nhưng tốt nhất là phát triển.\nNước đi: Nf3"
    assert extract_move(completion, board) == board.parse_san("Nf3")


def test_trich_nuoc_di_chap_nhan_uci() -> None:
    board = chess.Board(FEN_START)
    assert extract_move("Nước đi: g1f3", board) == chess.Move.from_uci("g1f3")


def test_trich_nuoc_di_bo_dau_cau_thua() -> None:
    board = chess.Board(FEN_START)
    assert extract_move("Nước đi: Nf3.", board) == board.parse_san("Nf3")


def test_trich_nuoc_di_khi_khong_co_format_thi_lay_nuoc_cuoi() -> None:
    board = chess.Board(FEN_START)
    assert extract_move("Đầu tiên e4, sau đó có thể Nf3.", board) == board.parse_san("Nf3")


def test_khong_bao_gio_tra_ve_nuoc_khong_hop_le() -> None:
    """Nguyên tắc 2: parse bằng python-chess, không tin chuỗi thô."""
    board = chess.Board(FEN_START)
    assert extract_move("Nước đi: Kh9", board) is None
    assert extract_move("Nước đi: zzzz", board) is None
    assert extract_move("Nước đi: Qh8", board) is None


def test_dong_ket_luan_hong_thi_lui_ve_nuoc_hop_le_trong_than_bai() -> None:
    board = chess.Board(FEN_START)
    completion = "Tôi cân nhắc Nf3.\nNước đi: Qh8"
    assert extract_move(completion, board) == board.parse_san("Nf3")


# -- lời giải và FEN hỏng -------------------------------------------------


def test_loi_giai_dang_chuoi_cung_doc_duoc() -> None:
    assert score_completion("Nước đi: Nf3", FEN_START, "g1f3 b8c6").is_correct


def test_loi_giai_rong_thi_khong_the_dung() -> None:
    result = score_completion("Nước đi: Nf3", FEN_START, [])
    assert result.is_legal
    assert result.move_reward == REWARD_LEGAL_BUT_WRONG


def test_fen_hong_thi_phat_khong_crash() -> None:
    result = score_completion("Nước đi: Nf3", "khong-phai-fen", SOLUTION)
    assert result.total == REWARD_ILLEGAL


# -- adapter cho TRL ------------------------------------------------------


def test_puzzle_reward_tra_ve_dung_so_luong() -> None:
    rewards = puzzle_reward(
        completions=["Nước đi: Nf3", "Nước đi: e4", "chịu"],
        fen=[FEN_START] * 3,
        solution=[SOLUTION] * 3,
    )
    assert rewards == pytest.approx(
        [
            REWARD_CORRECT + REWARD_FORMAT_BONUS,
            REWARD_LEGAL_BUT_WRONG + REWARD_FORMAT_BONUS,
            REWARD_ILLEGAL,
        ]
    )


def test_puzzle_reward_nhan_completion_dang_message() -> None:
    rewards = puzzle_reward(
        completions=[[{"role": "assistant", "content": "Nước đi: Nf3"}]],
        fen=[FEN_START],
        solution=[SOLUTION],
    )
    assert rewards[0] == pytest.approx(REWARD_CORRECT + REWARD_FORMAT_BONUS)


def test_puzzle_reward_bo_qua_kwargs_la_cua_trl() -> None:
    rewards = puzzle_reward(
        completions=["Nước đi: Nf3"],
        fen=[FEN_START],
        solution=[SOLUTION],
        prompts=["bất kỳ"],
        completion_ids=[[1, 2, 3]],
        trainer_state=object(),
    )
    assert rewards == pytest.approx([REWARD_CORRECT + REWARD_FORMAT_BONUS])


def test_puzzle_reward_lech_do_dai_thi_bao_loi() -> None:
    with pytest.raises(ValueError):
        puzzle_reward(completions=["a", "b"], fen=[FEN_START], solution=[SOLUTION])


# -- nhãn kết luận khớp với dữ liệu T8 ------------------------------------


def test_nhan_final_answer_doc_duoc() -> None:
    """Nhãn model thật sự xuất ra sau T8, vì C1 kết thúc bằng chuỗi này."""
    board = chess.Board(FEN_START)
    assert extract_move("Phát triển mã.\nFINAL_ANSWER: g1f3", board) == board.parse_san("Nf3")


def test_template_dung_nhan_ma_t8_da_hoc() -> None:
    """Prompt T9 phải đòi đúng format T8 đã học, không dạy nhãn thứ hai.

    Lệch nhãn thì `has_valid_format` luôn False nên GRPO vĩnh viễn mất
    ``REWARD_FORMAT_BONUS``, và cổng T8 chấm bằng đường lui thay vì bằng câu
    trả lời của model.
    """
    assert ANSWER_TEMPLATE.format(move="e2e4") == "FINAL_ANSWER: e2e4"


def test_doan_lap_o_duoi_khong_quyet_dinh_diem() -> None:
    """Model chưa dừng đúng lúc lặp lại cả nhãn; bản lặp không phải kết luận."""
    board = chess.Board(FEN_START)
    completion = (
        "Phát triển mã là tốt nhất.\nFINAL_ANSWER: g1f3\n"
        "Phát triển mã là tốt nhất.\nFINAL_ANSWER: e2e4"
    )
    assert extract_move(completion, board) == board.parse_san("Nf3")
    assert not has_valid_format(completion), "hai dòng kết luận là sai format"


def test_has_valid_format_nhan_final_answer() -> None:
    assert has_valid_format("Lý giải.\nFINAL_ANSWER: e2e4")
    assert has_valid_format("Lý giải.\nfinal_answer: e2e4")


# -- điều kiện dừng lúc sinh ----------------------------------------------


def test_answer_is_complete_can_ky_tu_ket_thuc_nuoc_di() -> None:
    """``e2`` là tiền tố của ``e2e4``: dừng ở đó là chấm sai nước hẳn."""
    assert not answer_is_complete("FINAL_ANSWER: e2")
    assert answer_is_complete("FINAL_ANSWER: e2e4\n")
    assert answer_is_complete("FINAL_ANSWER: e2e4 còn lại")


def test_answer_is_complete_chua_ket_luan_thi_false() -> None:
    assert not answer_is_complete("Vua trắng ở a6 và tốt b6 tạo áp lực")
    assert not answer_is_complete("")


def test_answer_is_complete_nhan_ca_nhan_phu() -> None:
    assert answer_is_complete("Nước đi: g1f3\n")

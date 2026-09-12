"""Test chuẩn hoá C1-data.

Trọng tâm là các đường HỎNG: schema thật của C1-data giấu FEN trong câu văn và
giấu nhãn sau ``FINAL_ANSWER:``, nên mọi phép rút trích đều có thể trượt. Trượt
mà im lặng thì sinh ra record rác, và rác đó đi thẳng vào tập train.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from chessvi.data.c1 import (
    QUESTION_EN,
    build_parser,
    extract_fen,
    extract_label,
    main,
    normalize_record,
    normalize_rows,
)
from tests.conftest import FEN_START

# Đúng khuôn instruction thật của C1-data, rút gọn phần danh sách cho dễ đọc.
INSTRUCTION = (
    "You are given a chess position in FEN: {fen}. "
    "Piece positions: White King: ['e1'], White Queen: ['d1']. "
    "Legal moves: e2e4, d2d4. "
    "Find the best move for the side to play. Analyze step by step and explain "
    "your reasoning. Finish with a single line formatted EXACTLY as: "
    "FINAL_ANSWER: <answer> Use UCI notation (e.g., e2e4, c2b1q)."
)


def _row(fen: str = FEN_START, uci: str = "e2e4", **overrides: Any) -> dict[str, Any]:
    row = {
        "instruction": INSTRUCTION.format(fen=fen),
        "input": "",
        "output": f"The centre is the priority here.\nFINAL_ANSWER: {uci}",
    }
    row.update(overrides)
    return row


# -- rút FEN --------------------------------------------------------------


def test_extract_fen_lay_dung_sau_dau_hai_cham() -> None:
    assert extract_fen(INSTRUCTION.format(fen=FEN_START)) == FEN_START


def test_extract_fen_khong_nuot_dau_cham_cau() -> None:
    """FEN kết thúc bằng số rồi liền dấu chấm — regex không được ôm dấu chấm."""
    fen = extract_fen(INSTRUCTION.format(fen=FEN_START))
    assert not fen.endswith(".")
    assert fen.split()[-1] == "1"


def test_extract_fen_thieu_thi_nem() -> None:
    with pytest.raises(ValueError, match="không tìm thấy FEN"):
        extract_fen("Find the best move for the side to play.")


def test_extract_fen_hong_thi_nem() -> None:
    """Đủ 6 trường về hình thức nhưng bàn cờ không dựng được."""
    with pytest.raises(ValueError):
        extract_fen("in FEN: 9999999/8/8/8/8/8/8/8 w - - 0 1. Piece positions:")


# -- rút nhãn -------------------------------------------------------------


def test_extract_label_tra_ve_uci_chuan_hoa() -> None:
    assert extract_label("FINAL_ANSWER: E2E4", FEN_START) == "e2e4"


def test_extract_label_lay_khop_cuoi_cung() -> None:
    """Phần hướng dẫn có thể nhắc lại nhãn; đáp án thật luôn ở cuối."""
    text = "Format: FINAL_ANSWER: e2e4 ... after analysis\nFINAL_ANSWER: d2d4"
    assert extract_label(text, FEN_START) == "d2d4"


def test_extract_label_thieu_thi_nem() -> None:
    with pytest.raises(ValueError, match="không tìm thấy FINAL_ANSWER"):
        extract_label("I think the knight move is strong.", FEN_START)


def test_extract_label_nuoc_khong_hop_le_thi_nem() -> None:
    """Case bắt buộc: nhãn là nước KHÔNG đi được ở thế đó.

    e2e5 đúng cú pháp UCI nhưng tốt không nhảy ba ô từ thế ban đầu. Nếu lọt,
    ta dạy model một lời giải thích dẫn tới nước không tồn tại.
    """
    with pytest.raises(ValueError, match="không hợp lệ"):
        extract_label("FINAL_ANSWER: e2e5", FEN_START)


def test_extract_label_cu_phap_uci_sai_thi_nem() -> None:
    with pytest.raises(ValueError, match="không tìm thấy FINAL_ANSWER"):
        extract_label("FINAL_ANSWER: xx99", FEN_START)


# -- chuẩn hoá cả record --------------------------------------------------


def test_normalize_record_du_truong_pipeline_can() -> None:
    record = normalize_record(_row())
    assert set(record) == {"id", "fen", "question", "answer", "label"}
    assert record["fen"] == FEN_START
    assert record["label"] == "e2e4"
    assert record["question"] == QUESTION_EN


def test_normalize_record_giu_nguyen_phan_giai_thich() -> None:
    """``answer`` là thứ T5 sẽ dịch — không được cắt xén."""
    record = normalize_record(_row())
    assert "The centre is the priority here." in record["answer"]
    assert "FINAL_ANSWER: e2e4" in record["answer"]


def test_normalize_record_sai_schema_thi_noi_ro() -> None:
    """Config 'rl'/'test' có prompt + reward_model, không có instruction."""
    row = {"prompt": [{"role": "user", "content": "..."}], "reward_model": {}}
    with pytest.raises(ValueError, match="config 'sft'"):
        normalize_record(row)


# -- gom cả luồng ---------------------------------------------------------


def test_normalize_rows_dem_ly_do_bo_qua() -> None:
    rows = [
        _row(),
        _row(uci="e2e5"),  # nước không hợp lệ
        {"instruction": "no fen here", "output": "FINAL_ANSWER: e2e4"},
        {"prompt": []},  # sai schema
    ]
    records, skipped = normalize_rows(rows)
    assert len(records) == 1
    assert skipped["illegal_label"] == 1
    assert skipped["missing_fen"] == 1
    assert skipped["other"] == 1


def test_normalize_rows_limit_dem_theo_mau_lay_duoc() -> None:
    """--limit đếm mẫu GIỮ LẠI, không phải row đã đọc."""
    rows = [_row(uci="e2e5"), _row(), _row(), _row()]
    records, _ = normalize_rows(rows, limit=2)
    assert len(records) == 2


# -- CLI ------------------------------------------------------------------


def test_cli_doc_jsonl_local_va_ghi_ra_file(tmp_path: Path) -> None:
    src = tmp_path / "raw.jsonl"
    src.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in (_row(), _row(uci="d2d4"))),
        encoding="utf-8",
    )
    out = tmp_path / "c1_sft.jsonl"

    assert main(["--input-jsonl", str(src), "--out", str(out)]) == 0

    written = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["label"] for r in written] == ["e2e4", "d2d4"]


def test_cli_khong_mau_nao_thi_tra_ve_1(tmp_path: Path) -> None:
    src = tmp_path / "raw.jsonl"
    src.write_text(json.dumps({"instruction": "rác", "output": "rác"}), encoding="utf-8")
    assert main(["--input-jsonl", str(src), "--out", str(tmp_path / "o.jsonl")]) == 1


def test_cli_tu_choi_split_khong_ho_tro() -> None:
    """'rl' và 'test' schema khác hẳn — chặn ngay ở argparse."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--split", "rl"])

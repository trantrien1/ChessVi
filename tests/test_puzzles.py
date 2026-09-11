"""Test chuẩn bị puzzle set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chess
import pytest

from chessvi.data.puzzles import (
    UNKNOWN_THEME,
    Puzzle,
    balanced_sample,
    main,
    prepare_puzzle,
    rating_bucket,
    render_distribution,
    write_parquet,
)
from tests.conftest import FEN_START

FIXTURE = Path(__file__).parent / "fixtures" / "puzzles_sample.jsonl"


def _rows() -> list[dict[str, Any]]:
    with FIXTURE.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _row(**overrides: Any) -> dict[str, Any]:
    base = {
        "PuzzleId": "P1",
        "FEN": FEN_START,
        "Moves": "e2e4 e7e5 g1f3",
        "Rating": 1500,
        "Themes": "fork short",
    }
    base.update(overrides)
    return base


# -- cạm bẫy nước dẫn -----------------------------------------------------


def test_nuoc_dau_tien_la_nuoc_dan_khong_phai_loi_giai() -> None:
    """Lỗi kinh điển của dataset: Moves[0] là nước của đối thủ."""
    puzzle = prepare_puzzle(_row())

    expected = chess.Board(FEN_START)
    expected.push_uci("e2e4")

    assert puzzle.setup_fen == FEN_START
    assert puzzle.opponent_move == "e2e4"
    assert puzzle.fen == expected.fen(), "FEN phải là thế SAU nước dẫn"
    assert puzzle.fen != puzzle.setup_fen
    assert puzzle.solution == ["e7e5", "g1f3"]
    assert puzzle.first_move == "e7e5"


def test_loi_giai_chay_duoc_tu_fen_da_ap_nuoc_dan() -> None:
    puzzle = prepare_puzzle(_row())
    board = chess.Board(puzzle.fen)
    for uci in puzzle.solution:
        move = chess.Move.from_uci(uci)
        assert move in board.legal_moves
        board.push(move)


def test_moi_puzzle_trong_fixture_deu_hop_le() -> None:
    for row in _rows():
        puzzle = prepare_puzzle(row)
        board = chess.Board(puzzle.fen)
        for uci in puzzle.solution:
            assert chess.Move.from_uci(uci) in board.legal_moves
            board.push_uci(uci)


# -- case nước đi không hợp lệ --------------------------------------------


def test_nuoc_dan_khong_hop_le_thi_bao_loi() -> None:
    with pytest.raises(ValueError, match="nước dẫn"):
        prepare_puzzle(_row(Moves="e2e5 e7e5"))


def test_nuoc_loi_giai_khong_hop_le_thi_bao_loi() -> None:
    with pytest.raises(ValueError, match="lời giải"):
        prepare_puzzle(_row(Moves="e2e4 e7e5 g1f6"))


def test_chuoi_uci_rac_thi_bao_loi() -> None:
    with pytest.raises(ValueError):
        prepare_puzzle(_row(Moves="zzzz yyyy"))


def test_fen_hong_thi_bao_loi() -> None:
    with pytest.raises(ValueError):
        prepare_puzzle(_row(FEN="khong-phai-fen"))


def test_thieu_nuoc_thi_bao_loi() -> None:
    with pytest.raises(ValueError, match="ít nhất 2 nước"):
        prepare_puzzle(_row(Moves="e2e4"))


def test_thieu_fen_thi_bao_loi() -> None:
    row = _row()
    del row["FEN"]
    with pytest.raises(ValueError, match="FEN"):
        prepare_puzzle(row)


# -- bucket và theme ------------------------------------------------------


@pytest.mark.parametrize(
    ("rating", "expected"),
    [(0, "0-399"), (399, "0-399"), (400, "400-799"), (1500, "1200-1599"), (2999, "2800-3199")],
)
def test_rating_bucket(rating: int, expected: str) -> None:
    assert rating_bucket(rating) == expected


def test_rating_bucket_width_khac() -> None:
    assert rating_bucket(1500, width=500) == "1500-1999"
    with pytest.raises(ValueError):
        rating_bucket(1500, width=0)


def test_theme_rong_thi_dung_nhan_mac_dinh() -> None:
    assert prepare_puzzle(_row(Themes="")).primary_theme == UNKNOWN_THEME


def test_doc_duoc_ten_cot_viet_thuong() -> None:
    row = {"id": "x", "fen": FEN_START, "moves": "e2e4 e7e5", "rating": 900, "themes": "pin"}
    puzzle = prepare_puzzle(row)
    assert puzzle.rating == 900
    assert puzzle.themes == ["pin"]


def test_themes_dang_list_cung_doc_duoc() -> None:
    assert prepare_puzzle(_row(Themes=["pin", "fork"])).themes == ["fork", "pin"]


# -- lấy mẫu cân bằng -----------------------------------------------------


def test_train_test_khong_giao_nhau() -> None:
    split, _skipped = balanced_sample(_rows(), train_size=200, test_size=40, per_cell=6)
    split.assert_disjoint()
    assert len(split.train) == 200
    assert len(split.test) == 40


def test_phan_phoi_can_bang_theo_bucket() -> None:
    split, _skipped = balanced_sample(_rows(), train_size=200, test_size=40, per_cell=6)
    counts = [
        sum(1 for p in split.train if p.rating_bucket == bucket)
        for bucket in {p.rating_bucket for p in split.train}
    ]
    assert max(counts) - min(counts) <= max(counts) * 0.5, "không ô nào áp đảo"


def test_tran_moi_o_duoc_ton_trong() -> None:
    split, skipped = balanced_sample(_rows(), train_size=10_000, test_size=0, per_cell=2)
    cells: dict[tuple[str, str], int] = {}
    for puzzle in split.train:
        key = (puzzle.primary_theme, puzzle.rating_bucket)
        cells[key] = cells.get(key, 0) + 1
    assert max(cells.values()) <= 2
    assert skipped["cell_full"] > 0


def test_cung_seed_thi_ket_qua_giong_nhau() -> None:
    first, _ = balanced_sample(_rows(), train_size=50, test_size=10, per_cell=4, seed=7)
    second, _ = balanced_sample(_rows(), train_size=50, test_size=10, per_cell=4, seed=7)
    assert [p.puzzle_id for p in first.train] == [p.puzzle_id for p in second.train]
    assert [p.puzzle_id for p in first.test] == [p.puzzle_id for p in second.test]


def test_row_hong_bi_bo_qua_va_dem_lai() -> None:
    rows = [_row(PuzzleId="ok"), _row(PuzzleId="bad", Moves="e2e5 e7e5"), _row(PuzzleId="ok")]
    split, skipped = balanced_sample(rows, train_size=10, test_size=0, per_cell=10)
    assert len(split.train) == 1, "một mẫu hỏng bị loại, một mẫu trùng id bị loại"
    assert skipped["ValueError"] == 1
    assert skipped["duplicate"] == 1


def test_max_scan_gioi_han_so_row_quet() -> None:
    split, _skipped = balanced_sample(_rows(), train_size=999, test_size=0, per_cell=99, max_scan=10)
    assert len(split.train) == 10


def test_test_size_0_thi_khong_lay_gi() -> None:
    split, _skipped = balanced_sample(_rows(), train_size=20, test_size=0, per_cell=5)
    assert split.test == []


# -- báo cáo và ghi file --------------------------------------------------


def test_render_distribution_in_ca_bucket_va_theme() -> None:
    split, _skipped = balanced_sample(_rows(), train_size=60, test_size=10, per_cell=4)
    text = render_distribution(split.train, "TRAIN")
    assert "TRAIN: 60 puzzle" in text
    assert "Theo rating bucket:" in text
    assert "Theo theme" in text
    assert "%" in text


def test_render_distribution_tap_rong() -> None:
    assert "0 puzzle" in render_distribution([], "TRỐNG")


def test_ghi_parquet_doc_lai_duoc(tmp_path: Path) -> None:
    import pyarrow.parquet as pq

    puzzles: list[Puzzle] = [prepare_puzzle(_row())]
    target = tmp_path / "train.parquet"
    write_parquet(puzzles, target)
    rows = pq.read_table(target).to_pylist()
    assert rows[0]["fen"] == puzzles[0].fen
    assert rows[0]["solution"] == ["e7e5", "g1f3"]


def test_cli_dry_run_khong_ghi_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "--input-jsonl", str(FIXTURE),
            "--train-size", "50",
            "--test-size", "10",
            "--per-cell", "4",
            "--out-dir", str(tmp_path),
            "--dry-run",
        ]
    )
    assert code == 0
    assert "TRAIN: 50 puzzle" in capsys.readouterr().out
    assert list(tmp_path.rglob("*")) == []


def test_cli_ghi_train_va_test(tmp_path: Path) -> None:
    code = main(
        [
            "--input-jsonl", str(FIXTURE),
            "--train-size", "50",
            "--test-size", "10",
            "--per-cell", "4",
            "--out-dir", str(tmp_path),
        ]
    )
    assert code == 0
    assert (tmp_path / "train.parquet").exists()
    assert (tmp_path / "test.parquet").exists()


def test_cli_bao_loi_khi_khong_co_puzzle(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert main(["--input-jsonl", str(empty), "--out-dir", str(tmp_path)]) == 1

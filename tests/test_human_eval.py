"""Test xuất form chấm tay và tính độ đồng thuận."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from chessvi.eval.human_eval import (
    AXES,
    FORM_FIELDS,
    collect_ratings,
    export_samples,
    main,
    quadratic_weighted_kappa,
    write_form_csv,
)
from tests.conftest import FEN_START


def _records(count: int = 250) -> list[dict[str, Any]]:
    return [
        {
            "puzzle_id": f"P{index}",
            "fen": FEN_START,
            "question": "Nên đi nước nào?",
            "answer": f"Nên đi Nf3 vì lý do số {index}.",
        }
        for index in range(count)
    ]


def _write_ratings(path: Path, scores: dict[str, tuple[int, int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FORM_FIELDS))
        writer.writeheader()
        for sample_id, (natural, term) in scores.items():
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "fen": FEN_START,
                    "question": "",
                    "answer": "",
                    "naturalness": natural,
                    "terminology": term,
                    "notes": "",
                }
            )


# -- xuất form ------------------------------------------------------------


def test_hai_truc_cham_dung_yeu_cau_t12() -> None:
    assert set(AXES) == {"naturalness", "terminology"}
    assert "1" in AXES["naturalness"] and "5" in AXES["naturalness"]
    assert "thuật ngữ" in AXES["terminology"]


def test_xuat_dung_100_mau() -> None:
    rows = export_samples(_records())
    assert len(rows) == 100
    assert len({row["sample_id"] for row in rows}) == 100, "không được trùng mẫu"


def test_cot_diem_de_trong_cho_nguoi_cham_dien() -> None:
    row = export_samples(_records(), size=1)[0]
    assert row["naturalness"] == ""
    assert row["terminology"] == ""
    assert row["answer"].startswith("Nên đi Nf3")


def test_it_mau_hon_yeu_cau_thi_lay_het() -> None:
    assert len(export_samples(_records(7), size=100)) == 7


def test_cung_seed_thi_cung_bo_mau() -> None:
    first = [r["sample_id"] for r in export_samples(_records(), seed=5)]
    second = [r["sample_id"] for r in export_samples(_records(), seed=5)]
    assert first == second


def test_khong_co_cau_tra_loi_thi_bao_loi() -> None:
    with pytest.raises(ValueError, match="câu trả lời"):
        export_samples([{"fen": FEN_START, "question": "Hỏi"}])


def test_ghi_csv_dung_cot(tmp_path: Path) -> None:
    target = tmp_path / "form.csv"
    write_form_csv(export_samples(_records(), size=5), target)
    with target.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(FORM_FIELDS)
        assert len(list(reader)) == 5


# -- kappa ----------------------------------------------------------------


def test_kappa_bang_1_khi_cham_giong_het() -> None:
    scores = [1, 2, 3, 4, 5, 3, 2]
    assert quadratic_weighted_kappa(scores, scores) == pytest.approx(1.0)


def test_kappa_giam_khi_lech_nhieu() -> None:
    a = [1, 2, 3, 4, 5]
    gan = [1, 2, 3, 4, 4]
    xa = [5, 4, 3, 2, 1]
    assert quadratic_weighted_kappa(a, gan) > quadratic_weighted_kappa(a, xa)
    assert quadratic_weighted_kappa(a, xa) < 0, "ngược hẳn thì tệ hơn ngẫu nhiên"


def test_kappa_tu_choi_dau_vao_hong() -> None:
    with pytest.raises(ValueError, match="lệch nhau"):
        quadratic_weighted_kappa([1, 2], [1])
    with pytest.raises(ValueError, match="không có điểm"):
        quadratic_weighted_kappa([], [])


# -- gom kết quả ----------------------------------------------------------


def test_gom_ket_qua_hai_nguoi_cham(tmp_path: Path) -> None:
    _write_ratings(tmp_path / "rater_a.csv", {"P1": (5, 5), "P2": (3, 4), "P3": (1, 2)})
    _write_ratings(tmp_path / "rater_b.csv", {"P1": (5, 4), "P2": (3, 4), "P3": (2, 2)})

    report = collect_ratings([tmp_path / "rater_a.csv", tmp_path / "rater_b.csv"])
    assert report.raters == ["rater_a", "rater_b"]
    assert report.shared_samples == 3
    assert report.means["naturalness"]["rater_a"] == pytest.approx(3.0)
    assert report.means["naturalness"]["rater_b"] == pytest.approx(10 / 3)
    assert 0.0 < report.kappa["naturalness"] <= 1.0
    assert report.within_one["naturalness"] == pytest.approx(1.0)


def test_chi_tinh_tren_mau_ca_nhom_cung_cham(tmp_path: Path) -> None:
    _write_ratings(tmp_path / "a.csv", {"P1": (5, 5), "P2": (3, 3), "P9": (1, 1)})
    _write_ratings(tmp_path / "b.csv", {"P1": (5, 5), "P2": (3, 3)})
    report = collect_ratings([tmp_path / "a.csv", tmp_path / "b.csv"])
    assert report.shared_samples == 2


def test_bo_qua_diem_hong_hoac_ngoai_thang(tmp_path: Path) -> None:
    path = tmp_path / "a.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FORM_FIELDS))
        writer.writeheader()
        writer.writerow({"sample_id": "P1", "naturalness": "5", "terminology": "5"})
        writer.writerow({"sample_id": "P2", "naturalness": "tốt", "terminology": "9"})
        writer.writerow({"sample_id": "P3", "naturalness": "4", "terminology": "4"})
    _write_ratings(tmp_path / "b.csv", {"P1": (5, 5), "P2": (3, 3), "P3": (4, 4)})

    report = collect_ratings([path, tmp_path / "b.csv"])
    assert report.shared_samples == 2, "P2 bị loại vì điểm hỏng"


def test_mot_nguoi_cham_thi_bao_loi(tmp_path: Path) -> None:
    _write_ratings(tmp_path / "a.csv", {"P1": (5, 5)})
    with pytest.raises(ValueError, match="ít nhất 2 người chấm"):
        collect_ratings([tmp_path / "a.csv"])


# -- CLI ------------------------------------------------------------------


def test_cli_export(tmp_path: Path) -> None:
    source = tmp_path / "answers.jsonl"
    source.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in _records(150)), encoding="utf-8"
    )
    out = tmp_path / "form.csv"
    assert main(["export", "--input", str(source), "--out", str(out), "--n", "100"]) == 0
    with out.open(encoding="utf-8-sig", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 100


def test_cli_collect(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_ratings(tmp_path / "a.csv", {"P1": (5, 5), "P2": (3, 3)})
    _write_ratings(tmp_path / "b.csv", {"P1": (4, 5), "P2": (3, 4)})
    out = tmp_path / "agreement.csv"
    code = main(
        [
            "collect",
            "--ratings", str(tmp_path / "a.csv"), str(tmp_path / "b.csv"),
            "--out", str(out),
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert "Người chấm" in printed
    assert "kappa" in printed
    with out.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {r["metric"] for r in rows} >= {"mean", "quadratic_weighted_kappa"}

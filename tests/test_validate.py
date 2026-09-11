"""Test bước validate dữ liệu đã dịch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from chessvi.data.validate import (
    RejectReason,
    ValidationReport,
    main,
    validate_dataset,
    validate_record,
)

from tests.conftest import FEN_START

GOOD_ANSWER = "Nước tốt nhất là Nf3 vì nó phát triển quân và kiểm soát ô trung tâm."


def _record(**overrides: Any) -> dict[str, Any]:
    base = {
        "fen": FEN_START,
        "question": "Nước nào tốt nhất ở thế này?",
        "answer": GOOD_ANSWER,
        "label": "Nf3",
    }
    base.update(overrides)
    return base


# -- từng tiêu chí --------------------------------------------------------


def test_mau_sach_thi_pass() -> None:
    result = validate_record(_record())
    assert result.ok, result.details
    assert result.reasons == ()


def test_bat_nuoc_di_khong_hop_le() -> None:
    """Tiêu chí 1: Qh8 đúng dạng SAN nhưng không đi được từ thế khởi đầu."""
    result = validate_record(_record(answer="Nước mạnh nhất là Qh8 ngay lập tức."))
    assert not result.ok
    assert RejectReason.INVALID_MOVE in result.reasons
    assert "Qh8" in " ".join(result.details)


def test_nuoc_ket_luan_khop_label() -> None:
    assert validate_record(_record(label="Nf3")).ok
    # Label ghi bằng UCI cũng phải khớp được.
    assert validate_record(_record(label="g1f3")).ok


def test_nuoc_ket_luan_lech_label() -> None:
    result = validate_record(_record(label="e4"))
    assert RejectReason.LABEL_MISMATCH in result.reasons


def test_label_goc_khong_parse_duoc() -> None:
    result = validate_record(_record(label="Qz9"))
    assert RejectReason.BAD_LABEL in result.reasons


def test_khong_co_label_thi_bo_qua_tieu_chi_2() -> None:
    record = _record()
    del record["label"]
    assert validate_record(record).ok


def test_bat_placeholder_con_sot() -> None:
    """Tiêu chí 3: <M0> lọt ra ngoài nghĩa là unmask hỏng."""
    result = validate_record(_record(answer="Nước tốt nhất là <M0> nhé.", label="Nf3"))
    assert RejectReason.LEFTOVER_PLACEHOLDER in result.reasons


def test_bat_khuc_tieng_anh_dai() -> None:
    result = validate_record(
        _record(
            answer=(
                "The best continuation is the move that keeps the balance in "
                "this position and Nf3 is fine."
            )
        )
    )
    assert RejectReason.ENGLISH_CHUNK in result.reasons


def test_khong_bao_dong_nham_voi_tieng_viet_binh_thuong() -> None:
    result = validate_record(_record())
    assert RejectReason.ENGLISH_CHUNK not in result.reasons


def test_bat_thuat_ngu_lech_chuan() -> None:
    result = validate_record(
        _record(answer="Nf3 tạo ra một cái nĩa và để lại quân treo cho đối thủ.")
    )
    assert RejectReason.GLOSSARY in result.reasons
    assert "nĩa" in " ".join(result.details)


def test_thieu_fen_hoac_fen_hong() -> None:
    no_fen = _record()
    del no_fen["fen"]
    assert validate_record(no_fen).reasons == (RejectReason.MISSING_FEN,)
    assert validate_record(_record(fen="khong-phai-fen")).reasons == (
        RejectReason.MISSING_FEN,
    )


def test_mot_mau_co_the_dinh_nhieu_ly_do() -> None:
    result = validate_record(
        _record(answer="Qh8 tạo ra một cái nĩa <M0>.", label="Nf3")
    )
    assert set(result.reasons) >= {
        RejectReason.INVALID_MOVE,
        RejectReason.GLOSSARY,
        RejectReason.LEFTOVER_PLACEHOLDER,
    }


def test_ke_lai_van_co_nhieu_nuoc_van_hop_le() -> None:
    """Nước đi theo thứ tự của một ván thật thì đều hợp lệ."""
    result = validate_record(
        _record(answer="Sau 1. e4 e5 2. Nf3 Nc6 thì thế cân bằng.", label="e4")
    )
    assert RejectReason.INVALID_MOVE not in result.reasons


# -- báo cáo --------------------------------------------------------------


def test_bao_cao_dem_dung_va_co_ty_le() -> None:
    records = [_record(), _record(), _record(answer="Qh8 là nước duy nhất.")]
    report = validate_dataset(records)
    assert report.total == 3
    assert report.passed == 2
    assert report.rejected == 1
    assert report.by_reason[str(RejectReason.INVALID_MOVE)] == 1
    assert report.reject_rate == pytest.approx(1 / 3)
    rendered = report.render()
    assert "invalid_move" in rendered
    assert "33.33%" in rendered


def test_bao_cao_rong_khong_chia_cho_0() -> None:
    report = ValidationReport()
    assert report.reject_rate == 0.0
    assert "Tổng số mẫu" in report.render()


def test_ghi_mau_bi_loai_ra_jsonl(tmp_path: Path) -> None:
    out = tmp_path / "rejected.jsonl"
    validate_dataset([_record(), _record(answer="Qh8 thôi.")], rejected_path=out)
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1, "chỉ mẫu bị loại mới được ghi"
    assert str(RejectReason.INVALID_MOVE) in rows[0]["reason"]
    assert "Qh8" in rows[0]["reason_detail"][0]
    assert rows[0]["fen"] == FEN_START, "giữ nguyên mẫu gốc để soi tay"


# -- CLI ------------------------------------------------------------------


def test_cli_chay_doc_lap_tren_jsonl(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "in.jsonl"
    source.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in [_record(), _record()]),
        encoding="utf-8",
    )
    rejected = tmp_path / "rejected.jsonl"
    assert main(["--input", str(source), "--rejected", str(rejected)]) == 0
    out = capsys.readouterr().out
    assert "Tổng số mẫu" in out
    assert "Pass" in out
    assert rejected.exists()


def test_cli_canh_bao_to_khi_ty_le_loai_qua_cao(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = [_record(answer="Qh8 thôi.") for _ in range(4)]
    source = tmp_path / "in.jsonl"
    source.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in bad), encoding="utf-8"
    )
    main(["--input", str(source), "--rejected", str(tmp_path / "r.jsonl")])
    assert "PIPELINE DỊCH CÓ VẤN ĐỀ" in capsys.readouterr().out


def test_cli_bao_loi_khi_khong_doc_duoc_mau_nao(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert main(["--input", str(empty), "--rejected", str(tmp_path / "r.jsonl")]) == 1


# -- xuất tập đã pass -----------------------------------------------------


def test_ghi_tap_da_pass_ra_parquet(tmp_path: Path) -> None:
    """Đầu vào của T8: phải có file chứa đúng những mẫu đã pass."""
    import pyarrow.parquet as pq

    clean = tmp_path / "validated" / "sft"
    validate_dataset(
        [_record(), _record(answer="Qh8 thôi."), _record()], clean_path=clean
    )
    rows = pq.read_table(clean / "part-00000.parquet").to_pylist()
    assert len(rows) == 2, "chỉ mẫu pass mới được ghi"
    assert all(row["answer"] == GOOD_ANSWER for row in rows)


def test_ghi_tap_da_pass_ra_jsonl(tmp_path: Path) -> None:
    clean = tmp_path / "clean.jsonl"
    validate_dataset([_record(), _record(answer="Qh8 thôi.")], clean_path=clean)
    rows = [json.loads(line) for line in clean.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert "reason" not in rows[0], "tập sạch không kèm trường chẩn đoán"


def test_khong_truyen_clean_path_thi_khong_ghi_gi(tmp_path: Path) -> None:
    validate_dataset([_record()], rejected_path=tmp_path / "r.jsonl")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["r.jsonl"]


def test_cli_out_clean(tmp_path: Path) -> None:
    source = tmp_path / "in.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(r, ensure_ascii=False)
            for r in [_record(), _record(answer="Qh8 thôi."), _record()]
        ),
        encoding="utf-8",
    )
    clean = tmp_path / "validated"
    code = main(
        [
            "--input", str(source),
            "--rejected", str(tmp_path / "r.jsonl"),
            "--out-clean", str(clean),
        ]
    )
    assert code == 0
    assert (clean / "part-00000.parquet").exists()

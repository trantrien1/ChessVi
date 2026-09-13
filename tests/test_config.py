"""Test cấu hình, trọng tâm là phần nạp `.env`.

Nơi này dễ sai âm thầm: parse hỏng một dòng thì biến không được đặt, và lỗi
hiện ra ở chỗ hoàn toàn khác — "không tìm thấy Stockfish" chẳng hạn.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chessvi.config import Paths, _parse_dotenv, load_dotenv


# -- parse ----------------------------------------------------------------


def test_doc_cap_key_value() -> None:
    assert _parse_dotenv("A=1\nB=hai") == {"A": "1", "B": "hai"}


def test_bo_qua_dong_trong_va_chu_thich() -> None:
    text = "\n# chú thích\n\n  # thụt vào vẫn là chú thích\nA=1\n"
    assert _parse_dotenv(text) == {"A": "1"}


def test_bo_nhay_bao_quanh() -> None:
    """Nháy là cách viết đường dẫn có dấu cách, không phải nội dung."""
    parsed = _parse_dotenv('A="C:/Program Files/sf.exe"\nB=\'x\'')
    assert parsed == {"A": "C:/Program Files/sf.exe", "B": "x"}


def test_chap_nhan_tien_to_export() -> None:
    assert _parse_dotenv("export A=1") == {"A": "1"}


def test_gia_tri_co_dau_bang_giu_nguyen() -> None:
    """Token và URL hay có dấu '=' bên trong; chỉ tách ở dấu đầu tiên."""
    assert _parse_dotenv("A=x=y=z") == {"A": "x=y=z"}


def test_dong_rac_khong_lam_hong_ca_file() -> None:
    assert _parse_dotenv("rác không có dấu bằng\nA=1\n=thiếu key") == {"A": "1"}


def test_gia_tri_rong_van_duoc_dat() -> None:
    assert _parse_dotenv("A=") == {"A": ""}


# -- load -----------------------------------------------------------------


def test_nap_tu_file_chi_dinh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CHESSVI_STOCKFISH_BIN", raising=False)
    env = tmp_path / ".env"
    env.write_text("CHESSVI_STOCKFISH_BIN=/opt/sf\n", encoding="utf-8")

    assert load_dotenv(env) == 1
    assert Paths().stockfish_bin == "/opt/sf"


def test_bien_moi_truong_that_thang_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Đặt ở shell phải đè được .env, nếu không thì không thử nhanh được gì."""
    monkeypatch.setenv("CHESSVI_STOCKFISH_BIN", "/tu-shell")
    env = tmp_path / ".env"
    env.write_text("CHESSVI_STOCKFISH_BIN=/tu-file\n", encoding="utf-8")

    assert load_dotenv(env) == 0
    assert Paths().stockfish_bin == "/tu-shell"


def test_override_thi_file_thang(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHESSVI_STOCKFISH_BIN", "/tu-shell")
    env = tmp_path / ".env"
    env.write_text("CHESSVI_STOCKFISH_BIN=/tu-file\n", encoding="utf-8")

    assert load_dotenv(env, override=True) == 1
    assert Paths().stockfish_bin == "/tu-file"


def test_khong_co_file_thi_im_lang(tmp_path: Path) -> None:
    """Thiếu .env là chuyện bình thường, không phải lỗi."""
    assert load_dotenv(tmp_path / "khong-ton-tai") == 0

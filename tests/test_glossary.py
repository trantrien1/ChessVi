"""Test bảng thuật ngữ: ép nhất quán và phát hiện cách dịch lệch chuẩn."""

from __future__ import annotations

import pytest

from chessvi.data.glossary import (
    GLOSSARY,
    VARIANTS,
    canonical,
    enforce_glossary,
    glossary_violations,
)


def test_bang_thuat_ngu_khop_claude_md() -> None:
    assert GLOSSARY["fork"] == "đòn đôi"
    assert GLOSSARY["pin"] == "ghim"
    assert GLOSSARY["skewer"] == "xiên"
    assert GLOSSARY["discovered attack"] == "đòn mở"
    assert GLOSSARY["back rank mate"] == "chiếu hết hàng cuối"
    assert GLOSSARY["en passant"] == "bắt tốt qua đường"
    assert GLOSSARY["castling"] == "nhập thành"
    assert GLOSSARY["zugzwang"] == "zugzwang"
    assert GLOSSARY["blunder"] == "sai lầm nghiêm trọng"
    assert GLOSSARY["hanging piece"] == "quân bỏ ngỏ"


def test_canonical_bao_loi_voi_thuat_ngu_chua_khai_bao() -> None:
    with pytest.raises(KeyError):
        canonical("windmill")


@pytest.mark.parametrize(("term_en", "term_vi"), sorted(GLOSSARY.items()))
def test_thuat_ngu_tieng_anh_con_sot_bi_thay(term_en: str, term_vi: str) -> None:
    assert enforce_glossary(f"Nước này tạo ra {term_en} ngay lập tức.") == (
        f"Nước này tạo ra {term_vi} ngay lập tức."
    )


@pytest.mark.parametrize(("variant", "term_en"), sorted(VARIANTS.items()))
def test_bien_the_lech_chuan_bi_ep_ve_chuan(variant: str, term_en: str) -> None:
    assert enforce_glossary(f"Đây là {variant}.") == f"Đây là {GLOSSARY[term_en]}."


def test_khong_con_hai_cach_dich_cho_cung_mot_thuat_ngu() -> None:
    """Nghiệm thu T5: sau enforce, mỗi thuật ngữ chỉ còn đúng một cách dịch."""
    text = (
        "Nước đi tạo ra một cái nĩa và một đòn nĩa, đồng thời ghim chặt và đòn ghim "
        "quân hậu, để lại quân treo lẫn quân lơ lửng. This is also a fork and a pin."
    )
    cleaned = enforce_glossary(text)
    assert glossary_violations(cleaned) == []
    assert cleaned.count("đòn đôi") == 3
    assert "nĩa" not in cleaned
    assert "quân treo" not in cleaned


def test_zugzwang_giu_nguyen_khong_bi_coi_la_vi_pham() -> None:
    assert enforce_glossary("thế zugzwang") == "thế zugzwang"
    assert glossary_violations("thế zugzwang") == []


def test_khong_bat_nham_giua_tu() -> None:
    # "forklift" không phải "fork", "pint" không phải "pin".
    assert enforce_glossary("a forklift and a pint") == "a forklift and a pint"


def test_khong_dung_toi_placeholder_va_ky_hieu_co() -> None:
    text = "<M0> tạo ra a fork, xem <F0> và Nf3."
    assert enforce_glossary(text) == "<M0> tạo ra a đòn đôi, xem <F0> và Nf3."


def test_khong_dung_toi_nuoc_di_khong_hop_le() -> None:
    """Case nước đi không hợp lệ: chuỗi rác vẫn phải đi qua nguyên vẹn."""
    text = "Nước Nf9 và Qz0 không hợp lệ nhưng vẫn là a blunder."
    assert enforce_glossary(text) == (
        "Nước Nf9 và Qz0 không hợp lệ nhưng vẫn là a sai lầm nghiêm trọng."
    )


def test_glossary_violations_chi_bao_cao_khong_sua() -> None:
    text = "một cái nĩa"
    violations = glossary_violations(text)
    assert [(v.term_en, v.found, v.expected) for v in violations] == [
        ("fork", "nĩa", "đòn đôi")
    ]
    assert text == "một cái nĩa"

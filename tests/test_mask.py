"""Test masking ký hiệu cờ.

Corpus ``tests/fixtures/c1_sample.txt`` mô phỏng đúng các dạng câu của C1-data
(FEN, game score, danh sách nước ứng viên, UCI, số thứ tự nước) và được sinh
bằng python-chess nên mọi ký hiệu trong đó đều là ký hiệu cờ thật. Không tải
dữ liệu qua mạng trong test.
"""

from __future__ import annotations

from pathlib import Path

import chess
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from chessvi.data.mask import (
    PLACEHOLDER_RE,
    MoveContext,
    Token,
    find_tokens,
    mask,
    unmask,
)
from tests.conftest import FEN_START

CORPUS_PATH = Path(__file__).parent / "fixtures" / "c1_sample.txt"
CORPUS: list[str] = CORPUS_PATH.read_text(encoding="utf-8").splitlines()

#: Câu tiếng Anh bình thường - tuyệt đối không được coi là nước đi.
NEGATIVE_CASES = [
    "Be careful with that.",
    "Bed and breakfast",
    "Print it on a4 paper size.",
    "She will be there at 5.",
    "Bedrock is beneath the soil.",
    "a4 paper size",
    "Be brief, be bright, be gone.",
    "The bed is made.",
]


# -- round-trip -----------------------------------------------------------


def test_corpus_du_lon() -> None:
    assert len(CORPUS) >= 500, "cần ít nhất 500 dòng để test có ý nghĩa"


def test_round_trip_tren_toan_corpus() -> None:
    for index, line in enumerate(CORPUS):
        masked, mapping = mask(line)
        assert unmask(masked, mapping) == line, f"dòng {index + 1} không round-trip"


def test_corpus_thuc_su_co_ky_hieu_de_mask() -> None:
    """Nếu regex hỏng thành không bắt gì thì round-trip vẫn pass - chặn ở đây."""
    masked_lines = sum(1 for line in CORPUS if mask(line)[1])
    assert masked_lines > len(CORPUS) * 0.8


def test_text_sau_mask_khong_con_ky_hieu_co() -> None:
    for line in CORPUS:
        masked, _mapping = mask(line)
        leftover = [
            token for token in find_tokens(masked) if not PLACEHOLDER_RE.fullmatch(token.text)
        ]
        assert leftover == [], f"còn sót {leftover} trong {masked!r}"


# -- case âm --------------------------------------------------------------


@pytest.mark.parametrize("text", NEGATIVE_CASES)
def test_khong_bat_nham_tu_tieng_anh(text: str) -> None:
    masked, mapping = mask(text)
    assert mapping == {}
    assert masked == text


def test_nuoc_tot_tran_chi_duoc_nhan_khi_co_ngu_canh_co() -> None:
    assert mask("a4 paper size")[1] == {}
    # Có Nf3 làm chứng cứ ngữ cảnh thì a4 mới được coi là nước đi.
    _masked, mapping = mask("After Nf3 White plays a4.")
    assert sorted(mapping.values()) == ["Nf3", "a4"]


def test_khong_bat_ky_hieu_dinh_gach_noi() -> None:
    _masked, mapping = mask("Print Nf3 on a4-paper.")
    assert list(mapping.values()) == ["Nf3"]


# -- case dương -----------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("e2e4", ["e2e4"]),
        ("a7a8q", ["a7a8q"]),
        ("Nf3", ["Nf3"]),
        ("Bxc6+", ["Bxc6+"]),
        ("O-O", ["O-O"]),
        ("O-O-O", ["O-O-O"]),
        ("Nf3 exd5", ["Nf3", "exd5"]),
        ("Nf3 a8=Q#", ["Nf3", "a8=Q#"]),
        ("Nf3 Rfe1", ["Nf3", "Rfe1"]),
        ("Nf3!?", ["Nf3!?"]),
    ],
)
def test_bat_dung_tung_lop_ky_hieu(text: str, expected: list[str]) -> None:
    _masked, mapping = mask(text)
    assert list(mapping.values()) == expected


def test_bat_fen_day_du_6_truong() -> None:
    text = f"The position is {FEN_START} right now."
    masked, mapping = mask(text)
    assert mapping == {"<F0>": FEN_START}
    assert masked == "The position is <F0> right now."


def test_bat_so_thu_tu_nuoc_di() -> None:
    _masked, mapping = mask("1. e4 e5 2. Nf3")
    assert "1." in mapping.values()
    assert "2." in mapping.values()


def test_so_thu_tu_chi_tinh_khi_theo_sau_la_nuoc_di() -> None:
    assert mask("He won 3 games. The score was 2.")[1] == {}


def test_ky_hieu_trung_nhau_dung_chung_placeholder() -> None:
    masked, mapping = mask("Nf3 then Nf3 again")
    assert len(mapping) == 1
    assert masked.count("<M0>") == 2


def test_placeholder_khong_dung_do_voi_placeholder_co_san() -> None:
    """Text gốc đã chứa <M0> thì không được sinh trùng, nếu không unmask phá text."""
    text = "Literal <M0> plus the move Nf3."
    masked, mapping = mask(text)
    assert "<M0>" not in mapping
    assert unmask(masked, mapping) == text


# -- xác thực bằng FEN ngữ cảnh -------------------------------------------


def test_context_fen_loai_nuoc_khong_hop_le() -> None:
    """Case nước đi không hợp lệ: Bb4/Ke7 không đi được, tốt e7 chắn đường."""
    _masked, mapping = mask("Nf3 and Bb4 and Ke7", context_fen=FEN_START)
    assert list(mapping.values()) == ["Nf3"], "chỉ Nf3 hợp lệ với FEN ngữ cảnh"


def test_context_fen_loai_nuoc_sai_luot() -> None:
    """Nf6 là nước của Đen; đứng một mình với FEN lượt Trắng thì bị loại."""
    assert mask("Nf6", context_fen=FEN_START)[1] == {}


def test_context_fen_cho_phep_ke_lai_van_co_theo_thu_tu() -> None:
    _masked, mapping = mask("1. e4 e5 2. Nf3 Nc6", context_fen=FEN_START)
    assert sorted(v for v in mapping.values() if not v.endswith(".")) == [
        "Nc6",
        "Nf3",
        "e4",
        "e5",
    ]


def test_context_fen_cho_phep_bien_re_nhanh() -> None:
    """Văn cờ rẽ nhánh: "sau c4, đen có dxc4 hoặc e6" là hai nhánh một thế.

    Một bàn cờ đang chạy duy nhất chỉ đi được một nhánh; mọi nhánh sau đó bị
    loại oan. Đây đúng là case làm guard chặn sạch câu trả lời Gambit Hậu.
    """
    board = chess.Board()
    board.push_san("d4")
    board.push_san("d5")
    text = "c4 dxc4, hoặc c4 e6 rồi cxd5 exd5, hoặc c4 b5 a4 bxc4"
    _masked, mapping = mask(text, context_fen=board.fen())
    for san in ("dxc4", "cxd5", "exd5", "bxc4"):
        assert san in mapping.values(), f"{san} bị loại oan"


def test_context_fen_chan_so_nhanh() -> None:
    """Cây biến có trần: cây càng rộng thì nước bịa càng dễ hợp lệ ở đâu đó.

    ``max_lines=1`` là trường hợp cực đoan - chỉ còn thế gốc - nên nước thứ hai
    của một biến không còn thế nào để bám vào.
    """
    context = MoveContext(FEN_START, max_lines=1)
    assert context.accepts(Token(0, 2, "e4", "san"))
    assert not context.accepts(Token(3, 5, "e5", "san"))


def test_context_fen_nhan_nuoc_tot_tran_ma_khong_can_ngu_canh_khac() -> None:
    _masked, mapping = mask("a4", context_fen=FEN_START)
    assert list(mapping.values()) == ["a4"]


def test_context_fen_loai_uci_khong_hop_le() -> None:
    _masked, mapping = mask("e2e4 then e2e5", context_fen=FEN_START)
    assert list(mapping.values()) == ["e2e4"]


def test_context_fen_loai_fen_hong() -> None:
    text = "Bad FEN: 9999999/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 here."
    _masked, mapping = mask(text, context_fen=FEN_START)
    assert all(not value.count("/") for value in mapping.values())


# -- unmask ---------------------------------------------------------------

def test_unmask_va_placeholder_bi_may_dich_lam_xoc_xech() -> None:
    mapping = {"<M0>": "Nf3", "<F0>": FEN_START}
    assert unmask("Nước < M0 > tốt", mapping) == "Nước Nf3 tốt"
    assert unmask("Nước <m0> tốt", mapping) == "Nước Nf3 tốt"
    assert unmask("Nước <M0> tốt", mapping, lenient=False) == "Nước Nf3 tốt"


def test_unmask_giu_nguyen_placeholder_la() -> None:
    """Placeholder không có trong mapping là lỗi - phải để nguyên cho T6 bắt."""
    assert unmask("còn sót <M9>", {"<M0>": "Nf3"}) == "còn sót <M9>"


def test_unmask_text_khong_co_placeholder() -> None:
    assert unmask("không có gì", {}) == "không có gì"


# -- property test --------------------------------------------------------

_ALPHABET = "abcdefgh12345678KQRBNOxX=+#-. <>MFNqrbn\n"


@settings(max_examples=300, deadline=None)
@given(st.text(alphabet=_ALPHABET, max_size=120))
def test_round_trip_identity_voi_text_bat_ky(text: str) -> None:
    masked, mapping = mask(text)
    assert unmask(masked, mapping) == text


@settings(max_examples=100, deadline=None)
@given(st.text(alphabet=_ALPHABET, max_size=120))
def test_mapping_luon_anh_xa_dung_placeholder(text: str) -> None:
    masked, mapping = mask(text)
    for placeholder in mapping:
        assert PLACEHOLDER_RE.fullmatch(placeholder)
        assert placeholder in masked


# -- phân biệt nước đi với tên ô ------------------------------------------


def test_is_unambiguous_move_loai_nuoc_tot_tran() -> None:
    """``e4`` trùng cú pháp với tên ô, mà văn giải thích cờ thì đầy tên ô."""
    from chessvi.data.mask import find_move_tokens, is_unambiguous_move

    text = "Vua trắng ở g1 và xe ở e1, nhưng Nf3 rồi e2e4 mới là nước mạnh."
    kept = [t.text for t in find_move_tokens(text) if is_unambiguous_move(t)]
    assert kept == ["Nf3", "e2e4"]


def test_is_unambiguous_move_giu_nuoc_co_dac_trung() -> None:
    from chessvi.data.mask import find_move_tokens, is_unambiguous_move

    text = "Sau exd5 thì O-O và Rxe1 đều được, e2e4 cũng vậy."
    kept = [t.text for t in find_move_tokens(text) if is_unambiguous_move(t)]
    assert kept == ["exd5", "O-O", "Rxe1", "e2e4"]

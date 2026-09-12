"""Test predictor eval. Không GPU, không mạng — thay torch và model bằng đồ giả.

Trọng tâm là ``HFPredictor.predict_batch``: nó phải sinh cả batch trong một
lời gọi ``generate`` và phải pad BÊN TRÁI. Pad bên phải vẫn chạy, vẫn trả về
chuỗi trông hợp lý, nhưng nội dung là rác — model decoder-only sinh tiếp từ
token cuối cùng, mà token đó là ``<pad>``. Loại lỗi im lặng đó chỉ test mới
bắt được.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Sequence
from typing import Any

import pytest

from chessvi.eval.predictor import EchoPredictor, HFPredictor, build_predictor


def _fake_torch() -> types.ModuleType:
    """Module torch giả, chỉ đủ cho ``with torch.no_grad()``."""
    module = types.ModuleType("torch")

    class _NoGrad:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *exc: object) -> bool:
            return False

    module.no_grad = _NoGrad  # type: ignore[attr-defined]
    return module


class _FakeBatch(dict[str, Any]):
    """Thứ tokenizer trả về: dict có .to() và ["input_ids"] có .shape."""

    def to(self, device: str) -> _FakeBatch:
        self["device"] = device
        return self


class _Ids(list[list[int]]):
    @property
    def shape(self) -> tuple[int, int]:
        return (len(self), len(self[0]))


class _FakeTokenizer:
    """Pad tới độ dài lớn nhất, tôn trọng padding_side."""

    pad_token = "<pad>"
    pad_token_id = 0
    eos_token = "</s>"

    def __init__(self) -> None:
        self.padding_side = "right"
        self.seen_side: list[str] = []

    def __call__(
        self, prompts: Sequence[str], *, return_tensors: str, padding: bool
    ) -> _FakeBatch:
        self.seen_side.append(self.padding_side)
        # Mỗi "token" là một ký tự, cho dễ kiểm.
        rows = [[ord(c) for c in text] for text in prompts]
        width = max(len(row) for row in rows)
        padded = [
            [self.pad_token_id] * (width - len(row)) + row
            if self.padding_side == "left"
            else row + [self.pad_token_id] * (width - len(row))
            for row in rows
        ]
        return _FakeBatch(input_ids=_Ids(padded), attention_mask=_Ids(padded))

    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool) -> str:
        return "".join(chr(i) for i in ids if i != self.pad_token_id)


class _FakeModel:
    def __init__(self) -> None:
        self.calls = 0
        self.last_kwargs: dict[str, Any] = {}

    def generate(self, **kwargs: Any) -> list[list[int]]:
        self.calls += 1
        self.last_kwargs = kwargs
        # Phần "sinh ra" là input của chính hàng đó, đảo ngược: mỗi hàng ra một
        # kết quả khác nhau nên test bắt được lỗi lệch thứ tự, và nếu phép lát
        # bỏ sót pad thì ký tự pad sẽ lộ ra trong chuỗi trả về.
        rows: list[list[int]] = []
        for row in kwargs["input_ids"]:
            content = [i for i in row if i != 0]
            rows.append(list(row) + list(reversed(content)))
        return rows


def _predictor() -> HFPredictor:
    """Dựng instance không chạy __init__ (init cần transformers + torch)."""
    predictor = HFPredictor.__new__(HFPredictor)
    predictor._tokenizer = _FakeTokenizer()
    predictor._model = _FakeModel()
    predictor._device = "cpu"
    predictor._max_new_tokens = 16
    predictor._temperature = 0.0
    # None = không dựng được StoppingCriteria (transformers không có ở đây).
    predictor._stopper_factory = None
    return predictor


@pytest.fixture(autouse=True)
def _torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())


def test_predict_batch_chi_goi_generate_mot_lan() -> None:
    """Cả điểm của việc này: 8 chuỗi song song gần như cùng giá 1 chuỗi."""
    predictor = _predictor()
    out = predictor.predict_batch(["abc", "de", "f"])

    assert out == ["cba", "ed", "f"], "chỉ trả token mới, đã cắt phần prompt"
    assert predictor._model.calls == 1


def test_predict_batch_pad_ben_trai() -> None:
    """Pad bên phải thì model sinh tiếp từ <pad> và ra rác — lỗi im lặng."""
    predictor = _predictor()
    predictor.predict_batch(["abc", "de"])
    assert predictor._tokenizer.seen_side == ["left"]


def test_predict_batch_tra_padding_side_ve_nhu_cu() -> None:
    """Đừng để lại trạng thái đã đổi trên tokenizer dùng chung."""
    predictor = _predictor()
    predictor.predict_batch(["abc"])
    assert predictor._tokenizer.padding_side == "right"


def test_predict_batch_rong_thi_khong_goi_model() -> None:
    predictor = _predictor()
    assert predictor.predict_batch([]) == []
    assert predictor._model.calls == 0


def test_predict_batch_giu_dung_thu_tu() -> None:
    """Kết quả phải khớp 1-1 với đầu vào; lệch là chấm sai puzzle."""
    predictor = _predictor()
    assert predictor.predict_batch(["ab", "cd", "ef"]) == ["ba", "dc", "fe"]


# -- hợp đồng của lớp cơ sở -----------------------------------------------


def test_predict_batch_mac_dinh_chay_tuan_tu() -> None:
    echo = EchoPredictor("đáp")
    assert echo.predict_batch(["a", "b"]) == ["đáp", "đáp"]
    assert echo.prompts == ["a", "b"]


def test_build_predictor_backend_la_liet_ke_duoc() -> None:
    with pytest.raises(ValueError, match="echo, gguf, hf"):
        build_predictor("khong-co")


# -- điều kiện dừng -------------------------------------------------------


def test_stopper_nhan_dung_do_rong_prompt() -> None:
    """Sai chỗ này là sinh ra chuỗi rỗng, không phải chậm.

    ``SYSTEM_PROMPT`` của T9 được dựng từ ``ANSWER_TEMPLATE``, nên bản thân
    prompt đã chứa ``FINAL_ANSWER: <nước đi>``. Nếu stopper soi cả prompt thì
    nó khớp ngay bước đầu và generate dừng với 0 token mới. Phần prompt phải
    bị cắt đúng ``prompt_width``.
    """
    predictor = _predictor()
    seen: list[int] = []

    def factory(tokenizer: Any, prompt_width: int) -> str:
        seen.append(prompt_width)
        return "stopper"

    predictor._stopper_factory = factory
    predictor.predict_batch(["abc", "de"])

    assert seen == [3], "độ rộng sau khi pad trái, không phải độ dài prompt gốc"
    assert predictor._model.last_kwargs["stopping_criteria"] == "stopper"


def test_khong_co_stopper_thi_truyen_none() -> None:
    """``generate`` nhận ``stopping_criteria=None`` như mặc định của nó."""
    predictor = _predictor()
    predictor.predict_batch(["abc"])
    assert predictor._model.last_kwargs["stopping_criteria"] is None


def test_predict_batch_khong_lan_pad_vao_ket_qua() -> None:
    """Chuỗi ngắn nhất bị pad nhiều nhất — phép lát phải bỏ đúng toàn bộ."""
    predictor = _predictor()
    out = predictor.predict_batch(["a", "bbbbbbbb"])
    assert out == ["a", "bbbbbbbb"]
    assert "<pad>" not in "".join(out)

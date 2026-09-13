"""Test phần SFT chạy được không cần GPU: dựng mẫu và parse tham số.

Bản thân vòng train không test ở đây — acceptance của T8 là smoke test
``--limit 50 --max-steps 5`` trên CPU, chạy tay/CI riêng vì cần torch.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chessvi.train.dataset import (
    DEFAULT_SYSTEM_PROMPT,
    build_example,
    iter_records,
    load_examples,
)
from chessvi.train.sft import (
    ATTENTION_ONLY_TARGETS,
    LORA_ALPHA,
    LORA_RANK,
    _lora_targets,
    build_parser,
    settings_from_args,
)
from tests.conftest import FEN_START

FIXTURE = Path(__file__).parent / "fixtures" / "c1_sample.jsonl"


# -- dựng mẫu -------------------------------------------------------------


def test_build_example_ghep_fen_vao_cau_hoi() -> None:
    example = build_example(
        {"fen": FEN_START, "question": "Nên đi nước nào?", "answer": "Đi Nf3."}
    )
    assert example is not None
    system, user, assistant = example["messages"]
    assert system["role"] == "system" and system["content"] == DEFAULT_SYSTEM_PROMPT
    assert user["content"].startswith(f"FEN: {FEN_START}")
    assert "Nên đi nước nào?" in user["content"]
    assert assistant == {"role": "assistant", "content": "Đi Nf3."}


def test_build_example_khong_co_fen_van_chay() -> None:
    example = build_example({"question": "Khai cuộc Ý là gì?", "answer": "Là..."})
    assert example is not None
    assert example["prompt"] == "Khai cuộc Ý là gì?"


def test_build_example_thieu_truong_thi_tra_none() -> None:
    assert build_example({"fen": FEN_START, "question": "Hỏi"}) is None
    assert build_example({"fen": FEN_START, "answer": "Đáp"}) is None
    assert build_example({"question": "  ", "answer": "Đáp"}) is None


def test_build_example_doc_duoc_ten_truong_khac() -> None:
    example = build_example({"prompt": "Hỏi gì đó", "completion": "Trả lời"})
    assert example is not None
    assert example["completion"] == "Trả lời"


def test_system_prompt_tuy_chinh_duoc() -> None:
    example = build_example(
        {"question": "Hỏi", "answer": "Đáp"}, system_prompt="Bạn là bot cờ."
    )
    assert example is not None
    assert example["messages"][0]["content"] == "Bạn là bot cờ."


# -- nạp dữ liệu ----------------------------------------------------------


def test_load_examples_tu_jsonl() -> None:
    examples = load_examples(FIXTURE)
    assert len(examples) == 25
    assert all(len(e["messages"]) == 3 for e in examples)


def test_load_examples_ton_trong_limit() -> None:
    assert len(load_examples(FIXTURE, limit=7)) == 7


def test_load_examples_bo_qua_mau_thieu_truong(tmp_path: Path) -> None:
    source = tmp_path / "in.jsonl"
    rows = [
        {"fen": FEN_START, "question": "Hỏi", "answer": "Đáp"},
        {"fen": FEN_START, "question": "Chỉ có hỏi"},
    ]
    source.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
    )
    assert len(load_examples(source)) == 1


def test_iter_records_thu_muc_khong_co_parquet(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="parquet"):
        list(iter_records(tmp_path))


def test_iter_records_doc_duoc_parquet(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist([{"fen": FEN_START, "question": "Hỏi", "answer": "Đáp"}])
    pq.write_table(table, tmp_path / "part-00000.parquet")
    assert len(load_examples(tmp_path)) == 1


# -- tham số --------------------------------------------------------------


def test_cau_hinh_lora_dung_yeu_cau_t8() -> None:
    assert LORA_RANK == 32
    assert LORA_ALPHA == 64


class _Config:
    """Đủ giống config của transformers cho phép chọn target module."""

    def __init__(self, **fields: object) -> None:
        self.__dict__.update(fields)


def test_lora_bam_all_linear_tren_model_dense() -> None:
    assert _lora_targets(_Config(hidden_size=2560)) == "all-linear"


def test_lora_bo_qua_expert_tren_model_moe() -> None:
    """all-linear trên Qwen3-30B-A3B ra ~1,7 tỷ tham số LoRA — không dùng được.

    128 expert x 3 phép chiếu x 48 lớp = 18.432 linear, mà router chỉ kích
    hoạt 8/128 expert mỗi token nên hầu hết adapter không nhận gradient.
    """
    targets = _lora_targets(_Config(hidden_size=2048, num_experts=128))
    assert targets == ATTENTION_ONLY_TARGETS
    assert not any("expert" in name for name in targets)
    assert "gate_proj" not in targets and "down_proj" not in targets


def test_mac_dinh_bat_push_to_hub() -> None:
    args = build_parser().parse_args(["--data", "d"])
    settings = settings_from_args(args)
    assert settings.push_to_hub is True
    assert settings.base_model == "Qwen/Qwen3-4B"
    assert settings.max_length == 2048
    assert settings.save_steps == 200
    assert settings.load_in_4bit is None, "None = tự quyết theo việc có GPU hay không"


def test_co_smoke_test_tat_push_va_doi_model() -> None:
    args = build_parser().parse_args(
        [
            "--data", "d",
            "--limit", "50",
            "--max-steps", "5",
            "--base-model", "Qwen/Qwen3-0.6B",
            "--no-push",
            "--no-4bit",
        ]
    )
    settings = settings_from_args(args)
    assert settings.limit == 50
    assert settings.max_steps == 5
    assert settings.base_model == "Qwen/Qwen3-0.6B"
    assert settings.push_to_hub is False
    assert settings.load_in_4bit is False


def test_co_resume() -> None:
    args = build_parser().parse_args(["--data", "d", "--resume"])
    assert settings_from_args(args).resume is True


# -- chốt quyền ghi Hub ----------------------------------------------------


def _fake_hub(error: Exception | None) -> object:
    """Module huggingface_hub giả, create_repo ném lỗi hoặc ghi lại lời gọi."""
    import types

    module = types.ModuleType("huggingface_hub")
    calls: list[dict[str, object]] = []

    def create_repo(repo_id: str, **kwargs: object) -> None:
        calls.append({"repo_id": repo_id, **kwargs})
        if error is not None:
            raise error

    module.create_repo = create_repo  # type: ignore[attr-defined]
    module.calls = calls  # type: ignore[attr-defined]
    return module


def test_chot_quyen_ghi_hub_tao_repo_private(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    from chessvi.train.sft import _verify_push_access

    hub = _fake_hub(None)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    _verify_push_access("ai-do/chessvi-4b-sft")

    (call,) = hub.calls  # type: ignore[attr-defined]
    assert call["repo_id"] == "ai-do/chessvi-4b-sft"
    assert call["private"] is True
    assert call["exist_ok"] is True


def test_chot_quyen_ghi_hub_noi_ro_ca_hai_nguyen_nhan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """403 của Hub không phân biệt token Read với namespace sai — ta phải nói."""
    import sys

    from chessvi.train.sft import _verify_push_access

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", _fake_hub(PermissionError("403 Forbidden"))
    )
    with pytest.raises(RuntimeError) as caught:
        _verify_push_access("ai-do/chessvi-4b-sft")

    message = str(caught.value)
    assert "403 Forbidden" in message, "giữ nguyên lỗi gốc để còn tra"
    assert "Write" in message
    assert "namespace" in message.lower()
    assert "--no-push" in message


# -- phân bố độ dài và gom batch ------------------------------------------


def test_log_do_dai_in_phan_vi(caplog: pytest.LogCaptureFixture) -> None:
    """p50 sát p99 thì gom batch vô ích; lệch xa thì đáng bật."""
    import logging

    from chessvi.train.sft import _log_lengths

    with caplog.at_level(logging.INFO, logger="chessvi.train.sft"):
        _log_lengths([300] * 500 + [1900] * 10)

    assert "p50=300" in caplog.text
    assert "max=1900" in caplog.text

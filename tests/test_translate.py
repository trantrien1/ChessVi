"""Test pipeline dịch. Không mạng, không GPU - dùng EchoTranslator."""

from __future__ import annotations

import json
import sys
import types
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from chessvi.data.glossary import glossary_violations
from chessvi.data.glossary import GLOSSARY
from chessvi.data.translate import (
    EchoTranslator,
    LLMTranslator,
    TranslationJob,
    Translator,
    _load_tokenizer,
    _reject_reason,
    build_glossary_block,
    build_translator,
    translate_record,
)
from tests.conftest import FEN_START

FIXTURE = Path(__file__).parent / "fixtures" / "c1_sample.jsonl"
FIELDS = ("question", "answer")


def _records() -> list[dict[str, Any]]:
    with FIXTURE.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class _ShoutingTranslator(Translator):
    """Giả lập máy dịch: viết hoa mọi thứ. Placeholder phải sống sót."""

    name = "shout"

    def translate_batch(self, texts: Sequence[str]) -> list[str]:
        return [text.upper() for text in texts]


class _FlakyTranslator(Translator):
    """Hỏng ``fail_times`` lần đầu rồi mới chạy được - để test retry."""

    name = "flaky"

    def __init__(self, fail_times: int) -> None:
        self.remaining = fail_times
        self.calls = 0

    def translate_batch(self, texts: Sequence[str]) -> list[str]:
        self.calls += 1
        if self.remaining > 0:
            self.remaining -= 1
            raise ConnectionError("Kaggle đứt mạng")
        return list(texts)


class _WrongCountTranslator(Translator):
    name = "wrong-count"

    def translate_batch(self, texts: Sequence[str]) -> list[str]:
        return list(texts)[:-1]


# -- một mẫu --------------------------------------------------------------


def test_backend_khong_biet_bi_tu_choi() -> None:
    with pytest.raises(ValueError, match="Backend"):
        build_translator("google-translate")


def test_echo_giu_nguyen_moi_thu() -> None:
    record = {"question": "Is Nf3 good?", "fen": FEN_START}
    assert translate_record(record, EchoTranslator(), FIELDS) == record


def test_ky_hieu_co_song_sot_qua_may_dich() -> None:
    """Máy dịch viết hoa mọi thứ; Nf3/O-O/FEN vẫn phải nguyên vẹn."""
    record = {
        "question": f"After Nf3 and O-O, from {FEN_START}, is exd5 good?",
        "fen": FEN_START,
    }
    out = translate_record(record, _ShoutingTranslator(), FIELDS)
    assert "Nf3" in out["question"]
    assert "O-O" in out["question"]
    assert FEN_START in out["question"]
    assert "AFTER" in out["question"], "phần văn xuôi thì có đi qua máy dịch"


def test_glossary_duoc_ep_sau_khi_dich() -> None:
    record = {"answer": "This is a fork and a hanging piece.", "fen": FEN_START}
    out = translate_record(record, EchoTranslator(), FIELDS)
    assert out["answer"] == "This is a đòn đôi and a quân bỏ ngỏ."
    assert glossary_violations(out["answer"]) == []


def test_truong_khong_phai_str_duoc_bo_qua() -> None:
    record = {"id": 7, "question": None, "answer": "a fork", "fen": FEN_START}
    out = translate_record(record, EchoTranslator(), FIELDS)
    assert out["id"] == 7
    assert out["question"] is None
    assert out["answer"] == "a đòn đôi"


def test_nuoc_di_khong_hop_le_khong_duoc_mask() -> None:
    """Case nước đi không hợp lệ: Nf9/Bb4 phải đi thẳng qua máy dịch."""
    record = {"question": "Is Nf9 or Bb4 playable after Nf3?", "fen": FEN_START}
    out = translate_record(record, _ShoutingTranslator(), FIELDS)
    assert "NF9" in out["question"], "Nf9 không phải nước đi, không được mask"
    assert "BB4" in out["question"], "Bb4 không hợp lệ với FEN này, không được mask"
    assert "Nf3" in out["question"], "Nf3 hợp lệ nên phải được giữ nguyên"


def test_fen_ngu_canh_hong_thi_lui_ve_mask_khong_ngu_canh() -> None:
    record = {"question": "After Nf3 White is fine.", "fen": "khong-phai-fen"}
    out = translate_record(record, _ShoutingTranslator(), FIELDS)
    assert "Nf3" in out["question"]


def test_backend_tra_sai_so_luong_thi_bao_loi() -> None:
    record = {"question": "Is Nf3 good?", "answer": "Yes.", "fen": FEN_START}
    with pytest.raises(RuntimeError, match="wrong-count"):
        translate_record(record, _WrongCountTranslator(), FIELDS)


# -- job ------------------------------------------------------------------


def test_dry_run_khong_ghi_file(tmp_path: Path) -> None:
    job = TranslationJob(EchoTranslator(), out_dir=tmp_path, split="sft", fields=FIELDS)
    processed = job.run(_records(), limit=20, dry_run=True)
    assert processed == 20
    assert list(tmp_path.rglob("*")) == []


def test_ghi_parquet_va_checkpoint(tmp_path: Path) -> None:
    job = TranslationJob(
        EchoTranslator(),
        out_dir=tmp_path,
        split="sft",
        fields=FIELDS,
        batch_size=4,
        checkpoint_every=8,
    )
    processed = job.run(_records())
    assert processed == 25
    shards = sorted(p.name for p in (tmp_path / "sft").glob("*.parquet"))
    # 25 mẫu, buffer xả mỗi khi đầy 8 -> 8 + 8 + 8 + 1.
    assert shards == [f"part-{i:05d}.parquet" for i in range(4)]
    assert job.read_state() == {"done": 25, "shards": 4}


def test_resume_bo_qua_mau_da_dich(tmp_path: Path) -> None:
    common = {"out_dir": tmp_path, "split": "sft", "fields": FIELDS, "batch_size": 4}
    first = TranslationJob(EchoTranslator(), **common)
    assert first.run(_records(), limit=12) == 12

    second = TranslationJob(EchoTranslator(), **common)
    assert second.run(_records(), resume=True) == 13, "chỉ dịch 13 mẫu còn lại"
    assert second.read_state()["done"] == 25


def test_retry_khi_backend_hong_tam_thoi(tmp_path: Path) -> None:
    flaky = _FlakyTranslator(fail_times=2)
    job = TranslationJob(
        flaky,
        out_dir=tmp_path,
        split="sft",
        fields=FIELDS,
        batch_size=25,
        max_retries=3,
        retry_backoff=1.0,
    )
    assert job.run(_records()) == 25
    assert flaky.calls == 3, "cả batch 25 mẫu vào một lời gọi: 2 lần hỏng + 1 lần được"


def test_het_luot_retry_thi_bao_loi(tmp_path: Path) -> None:
    job = TranslationJob(
        _FlakyTranslator(fail_times=99),
        out_dir=tmp_path,
        split="sft",
        fields=FIELDS,
        batch_size=25,
        max_retries=2,
        retry_backoff=1.0,
    )
    with pytest.raises(RuntimeError, match="thất bại sau 2 lần"):
        job.run(_records())


def test_tu_nhan_truong_text_khi_khong_khai_bao(tmp_path: Path) -> None:
    job = TranslationJob(EchoTranslator(), out_dir=tmp_path, split="sft")
    assert job.run(_records(), limit=4, dry_run=True) == 4


def test_khong_nhan_ra_truong_nao_thi_bao_loi(tmp_path: Path) -> None:
    job = TranslationJob(EchoTranslator(), out_dir=tmp_path, split="sft")
    with pytest.raises(ValueError, match="--fields"):
        job.run([{"foo": 1}], dry_run=True)


# -- nạp tokenizer ---------------------------------------------------------
#
# transformers 5.x dựng T5Tokenizer từ spiece.model và chết với
# VietAI/envit5-translation: "argument 'vocab': 'dict' object cannot be
# converted to 'Sequence'". Repo model có sẵn tokenizer.json nên đọc thẳng file
# đó là qua. Ba test dưới khoá đường lui mà không cần mạng lẫn transformers.


def _fake_transformers(auto_error: Exception | None = None) -> types.ModuleType:
    module = types.ModuleType("transformers")

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(name: str) -> str:
            if auto_error is not None:
                raise auto_error
            return f"auto:{name}"

    class PreTrainedTokenizerFast:
        @staticmethod
        def from_pretrained(name: str) -> str:
            return f"fast:{name}"

    module.AutoTokenizer = AutoTokenizer  # type: ignore[attr-defined]
    module.PreTrainedTokenizerFast = PreTrainedTokenizerFast  # type: ignore[attr-defined]
    return module


def test_load_tokenizer_duong_thuong_dung_autotokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers())
    assert _load_tokenizer("m") == "auto:m"


def test_load_tokenizer_typeerror_thi_lui_ve_tokenizer_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Đúng lỗi thật gặp trên Colab với transformers 5.5.4."""
    error = TypeError("argument 'vocab': 'dict' object cannot be converted to 'Sequence'")
    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers(error))
    assert _load_tokenizer("m") == "fast:m"


def test_load_tokenizer_khong_nuot_loi_khac(monkeypatch: pytest.MonkeyPatch) -> None:
    """Model không tồn tại thì phải nổ, không được lặng lẽ thử đường khác."""
    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers(OSError("404")))
    with pytest.raises(OSError, match="404"):
        _load_tokenizer("khong-ton-tai")


# -- backend LLM -----------------------------------------------------------


def test_glossary_block_dung_tu_bang_chuan() -> None:
    """Prompt phải dựng TỪ GLOSSARY, không chép tay — quy định của CLAUDE.md."""
    block = build_glossary_block()
    for en, vi in GLOSSARY.items():
        assert f"- {en} = {vi}" in block


def _bare_llm_translator() -> LLMTranslator:
    """Dựng instance không chạy __init__ (init cần transformers + vLLM)."""
    translator = LLMTranslator.__new__(LLMTranslator)
    translator.rejected = Counter()
    translator._cache = {}
    return translator


def _stub_generation(translator: LLMTranslator, replies: list[list[str]]) -> list[list[str]]:
    """Thay _generate bằng kịch bản dựng sẵn; trả về log các lượt gọi."""
    calls: list[list[str]] = []
    pending = list(replies)

    def fake_generate(prompts: Sequence[str]) -> list[str]:
        calls.append(list(prompts))
        return pending.pop(0)

    translator._render = lambda text, *, hint=None: (f"[{hint}]" if hint else "") + text
    translator._generate = fake_generate
    return calls


def test_finalize_bo_khoi_think_cua_qwen() -> None:
    translator = _bare_llm_translator()
    assert translator._finalize("Play <M0>.", "<think>cân nhắc</think>Đi <M0>.") == (
        "Đi <M0>.",
        None,
    )


def test_finalize_giu_nguyen_ban_goc_khi_mat_placeholder() -> None:
    """Mất ký hiệu cờ mà vẫn trả bản dịch thì T6 không bắt được — phải từ chối.

    Trả lại bản gốc tiếng Anh để T6 loại nó vì english_chunk, thấy được.
    """
    translator = _bare_llm_translator()
    source = "White plays <M0> then <M1>."
    assert translator._finalize(source, "Trắng đi rồi đi tiếp.") == (source, "placeholder")


def test_finalize_bat_ca_khi_placeholder_bi_doi_so() -> None:
    translator = _bare_llm_translator()
    source = "White plays <M0> then <M1>."
    assert translator._finalize(source, "Trắng đi <M0> rồi <M0>.")[0] == source


def test_finalize_chap_nhan_khi_placeholder_khop() -> None:
    translator = _bare_llm_translator()
    assert translator._finalize("White plays <M0>.", "  Trắng đi <M0>.  ") == (
        "Trắng đi <M0>.",
        None,
    )


# -- cổng chất lượng -------------------------------------------------------
#
# Ba case dưới đây lấy nguyên văn từ dry-run 20 mẫu của Qwen3-30B-A3B, rút gọn
# cho vừa test. Hai trong ba KHÔNG bị T6 bắt: chúng là tiếng Việt trôi chảy,
# nước đi hợp lệ, FINAL_ANSWER đúng — chỉ có nội dung là sai.

_EN_LONG = (
    "The position is defined by the exposure of the white king on d1 and the "
    "vulnerability of the undefended bishop on c4. With the black queen active "
    "on a1, the immediate tactical priority is to exploit these weaknesses. "
    "White is forced to address the check with the bishop move, which allows "
    "Black to maintain the initiative and a crushing advantage."
)


def test_reject_bat_ban_chua_dich_du_khong_chep_y_nguyen() -> None:
    """Mẫu 5/20: model chép lại tiếng Anh nhưng rụng một dấu '+'.

    Phép so bằng thuần tuý trượt vì chỉ lệch một ký tự; luật khúc tiếng Anh
    của T6 thì bắt được.
    """
    assert _reject_reason(_EN_LONG + " Play a1d4+.", _EN_LONG + " Play a1d4.") == "chưa dịch"


def test_reject_van_bat_ban_chep_y_nguyen_du_cau_ngan() -> None:
    """Câu ngắn không đủ hư từ cho luật khúc tiếng Anh — cần phép so bằng."""
    assert _reject_reason("White plays <M0>.", "White plays <M0>.") == "chưa dịch"


def test_reject_bat_ban_dich_bi_cut() -> None:
    """Mẫu 10/20: rụng nguyên đoạn mở đầu, phần còn lại vẫn là tiếng Việt tốt."""
    source = "A" * 1104
    assert _reject_reason(source, "Bản dịch tiếng Việt đầy đủ." * 20) == "cụt"


def test_reject_khong_phan_nan_cau_ngan_co_lai() -> None:
    """Câu ngắn co giãn tự nhiên; đừng bắt nhầm."""
    assert _reject_reason("Find the best move.", "Tìm nước tốt nhất.") is None


def test_reject_bat_doi_mau_quan() -> None:
    """Mẫu 17/20: Vua Trắng thành Vua Đen. T6 mù hoàn toàn với lỗi này."""
    source = "The White King is trapped. White is forced to delay the mate."
    flipped = "Vua Đen đang bị vây. Đen buộc phải hoãn nước chiếu hết lại."
    assert _reject_reason(source, flipped) == "sai màu quân"


def test_reject_cho_qua_khi_mau_quan_con_du() -> None:
    source = "The white king is trapped. White cannot defend the black knight."
    fine = "Vua trắng đang bị vây. Trắng không thể phòng thủ trước mã đen."
    assert _reject_reason(source, fine) is None


def test_reject_bo_qua_khi_mau_quan_chi_nhac_mot_lan() -> None:
    """Nhắc một lần lẻ có thể được diễn đạt lại hợp lệ — không đủ để kết luận."""
    source = "The white king is trapped in the corner of the board by the rook."
    assert _reject_reason(source, "Vua đang bị xe vây ở góc bàn cờ hoàn toàn.") is None


def test_backend_khong_biet_liet_ke_ca_llm() -> None:
    with pytest.raises(ValueError, match="llm"):
        build_translator("khong-co")


# -- khử trùng lặp ---------------------------------------------------------


class _CountingTranslator(Translator):
    name = "counting"

    def __init__(self) -> None:
        self.seen: list[str] = []

    def translate_batch(self, texts: Sequence[str]) -> list[str]:
        self.seen.extend(texts)
        return [f"vi:{text}" for text in texts]


def test_translate_records_khu_trung_lap_truoc_khi_goi_backend() -> None:
    """question của C1-data giống hệt ở mọi mẫu — dịch lại từng lần là đốt GPU."""
    from chessvi.data.translate import translate_records

    records = [
        {"question": "Find the best move.", "answer": "First answer."},
        {"question": "Find the best move.", "answer": "Second answer."},
        {"question": "Find the best move.", "answer": "First answer."},
    ]
    translator = _CountingTranslator()
    out = translate_records(records, translator, ["question", "answer"])

    # 6 trường text, nhưng chỉ 3 chuỗi khác nhau.
    assert len(translator.seen) == 3
    assert sorted(translator.seen) == sorted(
        {"Find the best move.", "First answer.", "Second answer."}
    )
    # Kết quả vẫn phải phân phối đúng về từng mẫu.
    assert out[0]["answer"] == "vi:First answer."
    assert out[1]["answer"] == "vi:Second answer."
    assert out[2]["answer"] == "vi:First answer."
    assert all(r["question"] == "vi:Find the best move." for r in out)


def test_translate_batch_cache_xuyen_batch() -> None:
    """Gọi lại cùng một chuỗi thì không sinh lại — `question` lặp ở mọi batch."""
    translator = _bare_llm_translator()
    calls = _stub_generation(translator, [["Đi <M0>."]])

    first = translator.translate_batch(["Play <M0>."])
    second = translator.translate_batch(["Play <M0>.", "Play <M0>."])

    assert first == ["Đi <M0>."]
    assert second == ["Đi <M0>.", "Đi <M0>."]
    assert len(calls) == 1, "lượt thứ hai phải lấy từ cache"


def test_translate_batch_thu_lai_khi_model_chep_nguyen_van() -> None:
    """Quan sát thật: 2/20 mẫu bị Qwen trả về nguyên văn tiếng Anh."""
    translator = _bare_llm_translator()
    source = "White plays <M0>."
    calls = _stub_generation(translator, [[source], ["Trắng đi <M0>."]])

    assert translator.translate_batch([source]) == ["Trắng đi <M0>."]
    assert len(calls) == 2, "phải có một lượt nhắc lại"
    assert translator.rejected == Counter(), "thử lại thành công thì không tính là hỏng"


def test_translate_batch_nhac_dung_loi_khi_thu_lai() -> None:
    """Nhắc trúng lỗi hiệu quả hơn nhắc chung chung."""
    translator = _bare_llm_translator()
    source = "White plays <M0>."
    calls = _stub_generation(translator, [[source], ["Trắng đi <M0>."]])

    translator.translate_batch([source])
    assert "Không chép lại tiếng Anh" in calls[1][0]


def test_translate_batch_dem_lai_khi_thu_lai_van_khong_dich() -> None:
    translator = _bare_llm_translator()
    source = "White plays <M0>."
    _stub_generation(translator, [[source], [source]])

    assert translator.translate_batch([source]) == [source]
    assert translator.rejected["chưa dịch"] == 1


def test_quality_report_noi_ro_con_bao_nhieu_truong_hong() -> None:
    """Chạy 5 tiếng xong mà không biết bao nhiêu mẫu hỏng thì vô dụng."""
    translator = _bare_llm_translator()
    assert "mọi trường đều đạt" in translator.quality_report()

    translator.rejected["cụt"] = 3
    report = translator.quality_report()
    assert "3 trường" in report
    assert "cụt: 3" in report

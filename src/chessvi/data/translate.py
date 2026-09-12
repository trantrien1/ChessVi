"""Pipeline dịch C1-data sang tiếng Việt.

Luồng một mẫu: ``mask -> dịch -> unmask -> enforce_glossary``.

Chạy:

    python -m chessvi.data.translate --split sft --limit 20 --dry-run
    python -m chessvi.data.translate --split sft --dataset <hf_id> --backend hf
    python -m chessvi.data.translate --split sft --dataset <hf_id> --resume

Colab/Kaggle hay đứt giữa chừng nên pipeline ghi checkpoint mỗi
``--checkpoint-every`` mẫu; ``--resume`` đọc lại state và chạy tiếp.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chessvi.config import Paths
from chessvi.data.glossary import GLOSSARY, enforce_glossary
from chessvi.data.mask import PLACEHOLDER_RE, mask, unmask
from chessvi.data.validate import ENGLISH_RUN_THRESHOLD, longest_english_run
from chessvi.logging_setup import configure_logging

logger = logging.getLogger(__name__)

__all__ = [
    "EchoTranslator",
    "HFTranslator",
    "LLMTranslator",
    "TranslationJob",
    "Translator",
    "build_glossary_block",
    "build_translator",
    "translate_record",
    "translate_records",
]

#: Các trường text thường gặp trong C1-data. Dùng khi không truyền --fields.
DEFAULT_TEXT_FIELDS: tuple[str, ...] = (
    "question",
    "answer",
    "prompt",
    "completion",
    "reasoning",
    "explanation",
    "text",
)

DEFAULT_FEN_FIELD = "fen"
DEFAULT_CHECKPOINT_EVERY = 500
DEFAULT_BATCH_SIZE = 16
DEFAULT_HF_MODEL = "VietAI/envit5-translation"
#: Mặc định cho backend llm. **Cần A100 80GB.**
#:
#: Vì sao 30B-A3B chứ không phải 32B, dù 32B to hơn: đây là việc nghẽn ở thông
#: lượng, không phải ở chất lượng. Ước lượng bf16 với gpu_memory_utilization
#: 0.90 trên 80GB (72GB khả dụng, trừ ~2.5GB activation + CUDA graph):
#:
#:     Qwen3-32B      65.6GB weights ->  3.9GB KV  | 32.8B active/token
#:     Qwen3-30B-A3B  61.0GB weights ->  8.5GB KV  |  3.3B active/token
#:     Qwen3-14B      29.6GB weights -> 39.9GB KV  | 14.8B active/token
#:
#: 32B chỉ còn 3.9GB KV nên vLLM chạy được rất ít request song song, chậm hơn
#: cả 14B. 30B-A3B là MoE: chất lượng cỡ 30B mà mỗi token chỉ kích hoạt 3.3B.
#:
#: Chỉ có A100 40GB thì dùng ``--model Qwen/Qwen3-14B``.
DEFAULT_LLM_MODEL = "Qwen/Qwen3-30B-A3B"

#: Bản dịch ngắn hơn ngần này lần bản gốc thì coi là bị cụt.
#:
#: Đo trên 20 mẫu dry-run thật của Qwen3-30B-A3B: bảy bản dịch đạt nằm trong
#: khoảng 0.82-0.92, bản rụng mất nguyên đoạn mở đầu là 0.59. Ngưỡng 0.70 nằm
#: giữa khoảng trống đó nên không đụng vào bản dịch tốt.
MIN_LENGTH_RATIO = 0.70

#: Dưới ngần này ký tự thì bỏ qua phép so độ dài — câu ngắn co giãn tự nhiên.
SHORT_TEXT_CHARS = 200


# -- backend dịch ---------------------------------------------------------


class Translator(ABC):
    """Giao diện backend dịch. Đổi backend không đụng tới pipeline."""

    name: str = "abstract"

    @abstractmethod
    def translate_batch(self, texts: Sequence[str]) -> list[str]:
        """Dịch cả batch. Trả về đúng số phần tử như đầu vào."""

    def translate(self, text: str) -> str:
        return self.translate_batch([text])[0]

    def quality_report(self) -> str | None:
        """Tóm tắt chất lượng cả lần chạy. ``None`` nếu backend không theo dõi."""
        return None


class EchoTranslator(Translator):
    """Không dịch gì, trả nguyên văn.

    Dùng để smoke test pipeline (kể cả trên CI không mạng, không GPU) và để
    kiểm tra riêng phần mask/unmask/glossary mà không lẫn nhiễu từ model.
    """

    name = "echo"

    def translate_batch(self, texts: Sequence[str]) -> list[str]:
        return list(texts)


def _load_tokenizer(model_name: str) -> Any:
    """Nạp tokenizer, có đường lui khi ``AutoTokenizer`` hỏng.

    transformers 5.x dựng ``T5Tokenizer`` bằng cách chuyển đổi ``spiece.model``
    và với ``VietAI/envit5-translation`` thì chết ngay::

        TypeError: argument 'vocab': 'dict' object cannot be converted to 'Sequence'

    Repo của model vốn đã có sẵn ``tokenizer.json``, nên đọc thẳng file đó là
    bỏ qua được đường chuyển đổi sentencepiece đang hỏng. Đã kiểm chứng: cùng
    vocab 50048, đúng eos/pad, và placeholder ``<M0>`` sống sót round-trip —
    điều kiện sống còn của bước mask (nguyên tắc 3 trong CLAUDE.md).
    """
    from transformers import AutoTokenizer, PreTrainedTokenizerFast  # noqa: PLC0415

    try:
        return AutoTokenizer.from_pretrained(model_name)
    except (TypeError, ValueError) as error:
        logger.warning(
            "AutoTokenizer hỏng với %s (%s: %s) — đọc thẳng tokenizer.json",
            model_name,
            type(error).__name__,
            error,
        )
        return PreTrainedTokenizerFast.from_pretrained(model_name)


class HFTranslator(Translator):
    """Dịch bằng model seq2seq trên Hugging Face (chạy ở Kaggle/Colab).

    Import ``transformers`` ở trong ``__init__`` để máy local không cài
    transformers vẫn dùng được phần còn lại của module.
    """

    name = "hf"

    def __init__(
        self,
        model_name: str = DEFAULT_HF_MODEL,
        *,
        device: str | None = None,
        max_new_tokens: int = 512,
        prefix: str = "en: ",
    ) -> None:
        from transformers import AutoModelForSeq2SeqLM  # noqa: PLC0415

        import torch  # noqa: PLC0415

        self._max_new_tokens = max_new_tokens
        self._prefix = prefix
        resolved = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Nạp model dịch %s trên %s", model_name, resolved)
        self._tokenizer = _load_tokenizer(model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(resolved)
        self._device = resolved

    def translate_batch(self, texts: Sequence[str]) -> list[str]:
        if not texts:
            return []
        import torch  # noqa: PLC0415

        batch = self._tokenizer(
            [self._prefix + text for text in texts],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_new_tokens,
        ).to(self._device)
        with torch.no_grad():
            generated = self._model.generate(**batch, max_new_tokens=self._max_new_tokens)
        decoded = self._tokenizer.batch_decode(generated, skip_special_tokens=True)
        # envit5 trả về tiền tố "vi: " ở đầu câu.
        return [line.removeprefix("vi: ").strip() for line in decoded]


def build_glossary_block() -> str:
    """Bảng thuật ngữ dạng text để nhét vào prompt.

    Dựng từ :data:`GLOSSARY` chứ không chép tay, đúng quy định trong CLAUDE.md:
    "Bổ sung thì thêm vào bảng này trước, không tự chế trong prompt."
    """
    return "\n".join(f"- {en} = {vi}" for en, vi in sorted(GLOSSARY.items()))


SYSTEM_PROMPT_VI = """Bạn là dịch giả chuyên ngành cờ vua, dịch Anh sang Việt.

QUY TẮC BẮT BUỘC:
1. Giữ NGUYÊN VĂN mọi ký hiệu dạng <M0>, <F1>, <N2>. Không dịch, không đổi số,
   không thêm, không bớt. Số lượng và thứ tự phải y hệt bản gốc.
2. Giữ nguyên FINAL_ANSWER và dấu hai chấm sau nó nếu bản gốc có.
3. Dùng đúng bảng thuật ngữ dưới đây, không dùng từ đồng nghĩa khác.
4. Chỉ xuất bản dịch. Không lời dẫn, không giải thích, không nhắc lại bản gốc.

BẢNG THUẬT NGỮ:
{glossary}"""


class LLMTranslator(Translator):
    """Dịch bằng LLM instruct, prompt có sẵn bảng thuật ngữ.

    Vì sao cần: model dịch phổ thông (envit5) dịch từ vựng cờ theo nghĩa đời
    thường — rook thành "cổ tay", checkmate thành "giao phối" — và còn bịa sai
    sự thật bàn cờ (tốt thành hậu). Một LLM đủ mạnh, được đưa thẳng bảng thuật
    ngữ và quy tắc giữ placeholder, cho chất lượng khác hẳn.

    Placeholder được kiểm lại sau khi sinh. Lệch là trả về nguyên văn bản gốc
    chứ không trả bản dịch hỏng: mẫu tiếng Anh sẽ bị T6 loại thẳng, còn mẫu mất
    ký hiệu cờ thì lọt qua T6 và âm thầm đầu độc tập train.
    """

    name = "llm"

    def __init__(
        self,
        model_name: str = DEFAULT_LLM_MODEL,
        *,
        use_vllm: bool = True,
        max_new_tokens: int = 1024,
        max_input_tokens: int = 4096,
        temperature: float = 0.0,
        gpu_memory_utilization: float = 0.90,
    ) -> None:
        from transformers import AutoTokenizer  # noqa: PLC0415

        self._model_name = model_name
        self._max_new_tokens = max_new_tokens
        self._max_input_tokens = max_input_tokens
        self._temperature = temperature
        self._system = SYSTEM_PROMPT_VI.format(glossary=build_glossary_block())
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._cache: dict[str, str] = {}
        #: Trường không qua nổi cổng chất lượng kể cả sau khi thử lại, đếm
        #: theo lý do. Được in ra cuối lần chạy qua :meth:`quality_report`.
        self.rejected: Counter[str] = Counter()

        if use_vllm:
            from vllm import LLM, SamplingParams  # noqa: PLC0415

            logger.info("Nạp %s qua vLLM", model_name)
            self._engine = LLM(
                model=model_name,
                max_model_len=max_input_tokens + max_new_tokens,
                gpu_memory_utilization=gpu_memory_utilization,
            )
            self._sampling = SamplingParams(
                temperature=temperature, max_tokens=max_new_tokens
            )
        else:
            import torch  # noqa: PLC0415
            from transformers import AutoModelForCausalLM  # noqa: PLC0415

            logger.info("Nạp %s qua transformers (không vLLM — sẽ chậm)", model_name)
            self._engine = None
            self._torch = torch
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype="auto", device_map="auto"
            )

    def _render(self, text: str, *, hint: str | None = None) -> str:
        content = text
        if hint is not None:
            content = f"DỊCH đoạn dưới sang tiếng Việt. {hint}\n\n{text}"
        messages = [
            {"role": "system", "content": self._system},
            {"role": "user", "content": content},
        ]
        try:
            # Qwen3 mặc định bật "thinking"; tắt đi cho output sạch và nhanh hơn.
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    def translate_batch(self, texts: Sequence[str]) -> list[str]:
        if not texts:
            return []

        # Cache xuyên batch: trường `question` giống hệt ở mọi mẫu, mà khử trùng
        # lặp trong translate_records chỉ hoạt động trong phạm vi một batch.
        todo = [text for text in texts if text not in self._cache]
        if todo:
            self._cache.update(zip(todo, self._translate_checked(todo), strict=True))

        return [self._cache[text] for text in texts]

    def _translate_checked(self, todo: Sequence[str]) -> list[str]:
        """Sinh, soát cổng chất lượng, thử lại đúng một lượt cho mẫu hỏng.

        Hỏng cả lượt hai thì trả về **bản gốc tiếng Anh** chứ không phải bản
        dịch hỏng: mẫu tiếng Anh bị T6 loại thẳng và đếm được, còn mẫu dịch
        sai màu quân hay mất một đoạn thì lọt qua T6 và âm thầm vào tập train.
        """
        outputs = self._generate([self._render(text) for text in todo])
        results: list[str] = []
        bad: list[tuple[int, str]] = []
        for index, (source, output) in enumerate(zip(todo, outputs, strict=True)):
            text, reason = self._finalize(source, output)
            results.append(text)
            if reason is not None:
                bad.append((index, reason))

        if bad:
            tally = Counter(reason for _, reason in bad)
            logger.warning(
                "%d/%d bản dịch không đạt (%s) — thử lại",
                len(bad),
                len(todo),
                ", ".join(f"{reason}: {n}" for reason, n in tally.most_common()),
            )
            retry = self._generate(
                [self._render(todo[i], hint=_RETRY_HINTS[reason]) for i, reason in bad]
            )
            for (index, _), output in zip(bad, retry, strict=True):
                text, reason = self._finalize(todo[index], output)
                results[index] = text
                if reason is not None:
                    self.rejected[reason] += 1

        return results

    def quality_report(self) -> str | None:
        if not self.rejected:
            return "Cổng chất lượng: mọi trường đều đạt."
        detail = ", ".join(f"{reason}: {n}" for reason, n in self.rejected.most_common())
        return (
            f"Cổng chất lượng: {sum(self.rejected.values())} trường giữ nguyên "
            f"tiếng Anh sau khi thử lại ({detail}) — T6 sẽ loại chúng"
        )

    def _generate(self, prompts: Sequence[str]) -> list[str]:
        if self._engine is not None:
            outputs = self._engine.generate(list(prompts), self._sampling)
            return [out.outputs[0].text for out in outputs]

        results: list[str] = []
        for prompt in prompts:
            batch = self._tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=self._max_input_tokens
            ).to(self._model.device)
            with self._torch.no_grad():
                generated = self._model.generate(
                    **batch,
                    max_new_tokens=self._max_new_tokens,
                    do_sample=self._temperature > 0,
                )
            new_tokens = generated[0][batch["input_ids"].shape[1] :]
            results.append(self._tokenizer.decode(new_tokens, skip_special_tokens=True))
        return results

    def _finalize(self, source: str, output: str) -> tuple[str, str | None]:
        """Dọn output rồi soát cổng chất lượng.

        Trả ``(text, lý do hỏng)``; hỏng thì ``text`` là **bản gốc**, không
        phải bản dịch hỏng.
        """
        cleaned = _THINK_RE.sub("", output).strip()
        reason = _reject_reason(source, cleaned)
        if reason is not None:
            logger.debug("Từ chối bản dịch (%s): %.80s", reason, cleaned)
            return source, reason
        return cleaned, None


#: Qwen3 và vài model khác bọc phần suy luận trong <think>...</think>.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

#: Màu quân: cách viết tiếng Anh trong bản gốc -> cách viết tiếng Việt phải có
#: trong bản dịch. Dùng để bắt lỗi đổi bên.
_COLOR_PAIRS: tuple[tuple[re.Pattern[str], re.Pattern[str]], ...] = (
    (
        re.compile(r"(?<!\w)white(?!\w)", re.IGNORECASE),
        re.compile(r"(?<!\w)trắng(?!\w)", re.IGNORECASE),
    ),
    (
        re.compile(r"(?<!\w)black(?!\w)", re.IGNORECASE),
        re.compile(r"(?<!\w)đen(?!\w)", re.IGNORECASE),
    ),
)

#: Lý do hỏng -> câu nhắc thêm khi thử lại. Nhắc đúng lỗi hiệu quả hơn nhắc chung.
_RETRY_HINTS: dict[str, str] = {
    "placeholder": (
        "Chép lại y hệt mọi ký hiệu dạng <M0>, <F0>, <N0> — đúng số lượng, "
        "đúng thứ tự, không thêm không bớt."
    ),
    "chưa dịch": "Không chép lại tiếng Anh. Chỉ xuất bản dịch tiếng Việt.",
    "cụt": "Dịch ĐẦY ĐỦ từ câu đầu tới câu cuối, không bỏ sót câu hay đoạn nào.",
    "sai màu quân": (
        "Dịch đúng màu quân: white = trắng, black = đen. Tuyệt đối không đổi bên."
    ),
}


def _reject_reason(source: str, cleaned: str) -> str | None:
    """Bản dịch có hỏng không? Trả nhãn lý do, hoặc ``None`` nếu đạt.

    Cả bốn lỗi đều quan sát được trong dry-run 20 mẫu của Qwen3-30B-A3B, và
    hai trong số đó **T6 không bắt được** vì chúng cho ra tiếng Việt trôi chảy:

    ``placeholder``   số hoặc thứ tự ký hiệu cờ lệch so với bản gốc.
    ``chưa dịch``     model chép lại tiếng Anh (mẫu 5/20). Dùng đúng luật của
                      T6 nên thứ gì T6 sắp vứt thì được thử lại trước. Bản
                      chép y nguyên bị bắt riêng vì câu ngắn không đủ hư từ
                      cho phép đếm khúc tiếng Anh.
    ``cụt``           rụng mất cả một đoạn (mẫu 10/20 mất nguyên đoạn mở đầu).
    ``sai màu quân``  bản gốc nhắc "white"/"black" nhiều lần mà bản dịch không
                      nhắc "trắng"/"đen" lần nào (mẫu 17/20 biến Vua Trắng
                      thành Vua Đen). Đây là sai sự thật bàn cờ.
    """
    if _placeholders(cleaned) != _placeholders(source):
        return "placeholder"
    if cleaned == source or longest_english_run(cleaned) >= ENGLISH_RUN_THRESHOLD:
        return "chưa dịch"
    if len(source) >= SHORT_TEXT_CHARS and len(cleaned) < MIN_LENGTH_RATIO * len(source):
        return "cụt"
    for en_pattern, vi_pattern in _COLOR_PAIRS:
        # Nhắc >= 2 lần mới xét: một lần lẻ có thể được diễn đạt lại hợp lệ.
        if len(en_pattern.findall(source)) >= 2 and not vi_pattern.search(cleaned):
            return "sai màu quân"
    return None


def _placeholders(text: str) -> list[str]:
    """Danh sách placeholder theo đúng thứ tự xuất hiện."""
    return [match.group(0) for match in PLACEHOLDER_RE.finditer(text)]


def build_translator(
    backend: str, *, model_name: str | None = None, use_vllm: bool = True
) -> Translator:
    """Tạo backend theo tên. Thêm backend mới thì khai báo ở đây."""
    if backend == "echo":
        return EchoTranslator()
    if backend == "hf":
        return HFTranslator(model_name or DEFAULT_HF_MODEL)
    if backend == "llm":
        return LLMTranslator(model_name or DEFAULT_LLM_MODEL, use_vllm=use_vllm)
    raise ValueError(f"Backend không biết: {backend!r} (có: echo, hf, llm)")


# -- dịch ------------------------------------------------------------------


def translate_records(
    records: Sequence[dict[str, Any]],
    translator: Translator,
    fields: Sequence[str],
    *,
    fen_field: str | None = DEFAULT_FEN_FIELD,
) -> list[dict[str, Any]]:
    """Dịch các trường text của cả một batch mẫu, giữ nguyên mọi ký hiệu cờ.

    Toàn bộ text của cả batch đi vào **một** lời gọi ``translate_batch`` — gọi
    từng mẫu một thì GPU ở Kaggle chạy không hết công suất.

    Ký hiệu cờ được mask trước khi gọi model và khôi phục sau đó, nên model
    dịch không bao giờ nhìn thấy ``Nf3`` hay FEN. Mẫu nào có trường FEN thì
    dùng luôn làm ngữ cảnh xác thực nước đi (chỉ mask nước hợp lệ).
    """
    slots: list[tuple[int, str, dict[str, str]]] = []
    masked_texts: list[str] = []
    for index, record in enumerate(records):
        context_fen = None
        if fen_field and isinstance(record.get(fen_field), str):
            context_fen = record[fen_field]
        for name in fields:
            if not isinstance(record.get(name), str):
                continue
            masked_text, mapping = _mask_safely(record[name], context_fen)
            slots.append((index, name, mapping))
            masked_texts.append(masked_text)

    # Khử trùng lặp trước khi gọi backend. Trường `question` của C1-data giống
    # hệt nhau ở cả 39.601 mẫu, dịch lại từng lần là đốt GPU vô ích.
    index_of: dict[str, int] = {}
    unique_texts: list[str] = []
    for text in masked_texts:
        if text not in index_of:
            index_of[text] = len(unique_texts)
            unique_texts.append(text)

    unique_out = translator.translate_batch(unique_texts) if unique_texts else []
    if len(unique_out) != len(unique_texts):
        raise RuntimeError(
            f"Backend {translator.name} trả {len(unique_out)} kết quả cho "
            f"{len(unique_texts)} đầu vào"
        )
    translated = [unique_out[index_of[text]] for text in masked_texts]

    out = [dict(record) for record in records]
    for (index, name, mapping), text in zip(slots, translated, strict=True):
        out[index][name] = enforce_glossary(unmask(text, mapping))
    return out


def translate_record(
    record: dict[str, Any],
    translator: Translator,
    fields: Sequence[str],
    *,
    fen_field: str | None = DEFAULT_FEN_FIELD,
) -> dict[str, Any]:
    """Dịch một mẫu. Bọc mỏng quanh :func:`translate_records`."""
    return translate_records([record], translator, fields, fen_field=fen_field)[0]


def _mask_safely(text: str, context_fen: str | None) -> tuple[str, dict[str, str]]:
    """Mask có ngữ cảnh; FEN hỏng thì lùi về mask không ngữ cảnh."""
    if context_fen is None:
        return mask(text)
    try:
        return mask(text, context_fen)
    except ValueError:
        logger.warning("FEN ngữ cảnh không hợp lệ (%r), mask không ngữ cảnh", context_fen)
        return mask(text)


# -- nạp dữ liệu ----------------------------------------------------------


def _load_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_hf(dataset: str, split: str) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset  # noqa: PLC0415

    logger.info("Nạp dataset %s (split=%s) từ Hugging Face", dataset, split)
    stream = load_dataset(dataset, split=split, streaming=True)
    for row in stream:
        yield dict(row)


def _detect_fields(record: dict[str, Any], requested: Sequence[str] | None) -> list[str]:
    if requested:
        return list(requested)
    detected = [name for name in DEFAULT_TEXT_FIELDS if isinstance(record.get(name), str)]
    if not detected:
        raise ValueError(
            f"Không tự nhận ra trường text nào trong {sorted(record)}; dùng --fields"
        )
    logger.info("Tự nhận trường text: %s", ", ".join(detected))
    return detected


# -- job ------------------------------------------------------------------


@dataclass
class TranslationJob:
    """Một lần chạy dịch, có checkpoint để ngắt giữa chừng chạy tiếp được."""

    translator: Translator
    out_dir: Path
    split: str
    fields: Sequence[str] | None = None
    fen_field: str | None = DEFAULT_FEN_FIELD
    batch_size: int = DEFAULT_BATCH_SIZE
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY
    max_retries: int = 3
    retry_backoff: float = 2.0
    _buffer: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    @property
    def state_path(self) -> Path:
        return self.out_dir / self.split / "_state.json"

    def read_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"done": 0, "shards": 0}
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def _write_state(self, done: int, shards: int) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps({"done": done, "shards": shards}, ensure_ascii=False),
            encoding="utf-8",
        )

    def _translate_with_retry(
        self, records: Sequence[dict[str, Any]], fields: Sequence[str]
    ) -> list[dict[str, Any]]:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return translate_records(
                    records, self.translator, fields, fen_field=self.fen_field
                )
            except Exception as error:  # noqa: BLE001 - backend nào cũng có thể ném
                last_error = error
                wait = self.retry_backoff ** (attempt - 1)
                logger.warning(
                    "Batch lỗi (lần %d/%d): %s — thử lại sau %.1fs",
                    attempt,
                    self.max_retries,
                    error,
                    wait,
                )
                time.sleep(wait)
        raise RuntimeError(f"Batch thất bại sau {self.max_retries} lần") from last_error

    def _flush(self, shard_index: int) -> bool:
        """Ghi buffer ra một shard parquet. Trả về True nếu thật sự có ghi."""
        if not self._buffer:
            return False
        import pyarrow as pa  # noqa: PLC0415
        import pyarrow.parquet as pq  # noqa: PLC0415

        target = self.out_dir / self.split / f"part-{shard_index:05d}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(self._buffer), target)
        logger.info("Ghi %d mẫu vào %s", len(self._buffer), target)
        self._buffer.clear()
        return True

    def run(
        self,
        records: Iterable[dict[str, Any]],
        *,
        limit: int | None = None,
        resume: bool = False,
        dry_run: bool = False,
        dry_run_samples: int = 20,
    ) -> int:
        """Chạy job, trả về số mẫu đã xử lý trong lần chạy này."""
        state = self.read_state() if resume else {"done": 0, "shards": 0}
        skip: int = state["done"]
        shards: int = state["shards"]
        if skip:
            logger.info("Resume: bỏ qua %d mẫu đã dịch", skip)

        fields: Sequence[str] | None = self.fields
        processed = 0
        taken = 0
        shown = 0
        batch: list[dict[str, Any]] = []

        for index, record in enumerate(records):
            if index < skip:
                continue
            if limit is not None and taken >= limit:
                break
            if fields is None:
                fields = _detect_fields(record, self.fields)
            batch.append(record)
            taken += 1
            if len(batch) < self.batch_size and not (limit is not None and taken >= limit):
                continue
            processed, shown, shards = self._handle_batch(
                batch, fields, processed, shown, shards, skip, dry_run, dry_run_samples
            )
            batch = []

        if batch and fields is not None:
            processed, shown, shards = self._handle_batch(
                batch, fields, processed, shown, shards, skip, dry_run, dry_run_samples
            )

        if not dry_run:
            if self._flush(shards):
                shards += 1
            self._write_state(skip + processed, shards)
        report = self.translator.quality_report()
        if report is not None:
            logger.info("%s", report)
        logger.info("Xong %d mẫu%s", processed, " (dry-run, không ghi file)" if dry_run else "")
        return processed

    def _handle_batch(
        self,
        batch: list[dict[str, Any]],
        fields: Sequence[str],
        processed: int,
        shown: int,
        shards: int,
        skip: int,
        dry_run: bool,
        dry_run_samples: int,
    ) -> tuple[int, int, int]:
        translated = self._translate_with_retry(batch, fields)
        if dry_run:
            shown = _show_pairs(batch, translated, fields, shown, dry_run_samples)
        else:
            self._buffer.extend(translated)
        processed += len(batch)
        if not dry_run and len(self._buffer) >= self.checkpoint_every:
            if self._flush(shards):
                shards += 1
            self._write_state(skip + processed, shards)
        return processed, shown, shards


def _show_pairs(
    before: Sequence[dict[str, Any]],
    after: Sequence[dict[str, Any]],
    fields: Sequence[str],
    shown: int,
    limit: int,
) -> int:
    """In cặp before/after ra stdout cho người đọc kiểm tra bằng mắt."""
    for source, result in zip(before, after, strict=True):
        if shown >= limit:
            return shown
        shown += 1
        logger.info("--- mẫu %d ---", shown)
        for name in fields:
            if isinstance(source.get(name), str):
                logger.info("[%s] EN: %s", name, source[name])
                logger.info("[%s] VI: %s", name, result[name])
    return shown


# -- CLI ------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m chessvi.data.translate",
        description="Dịch C1-data sang tiếng Việt, giữ nguyên ký hiệu cờ.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--dataset", help="ID dataset trên Hugging Face (C1-data)")
    source.add_argument("--input-jsonl", type=Path, help="File .jsonl local thay cho HF")
    parser.add_argument("--split", default="sft", help="Split cần dịch (mặc định: sft)")
    parser.add_argument("--limit", type=int, help="Chỉ xử lý N mẫu đầu")
    parser.add_argument("--backend", default="echo", choices=["echo", "hf", "llm"])
    parser.add_argument(
        "--model",
        help=f"Mặc định: {DEFAULT_HF_MODEL} cho hf, {DEFAULT_LLM_MODEL} cho llm",
    )
    parser.add_argument(
        "--no-vllm",
        action="store_true",
        help="Backend llm chạy bằng transformers thay vì vLLM (chậm hơn nhiều)",
    )
    parser.add_argument("--fields", help="Danh sách trường cần dịch, phân tách bằng dấu phẩy")
    parser.add_argument("--fen-field", default=DEFAULT_FEN_FIELD)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, help="Mặc định: <data_dir>/translated")
    parser.add_argument("--resume", action="store_true", help="Chạy tiếp từ checkpoint")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="In 20 cặp before/after ra stdout, không ghi file",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def _configure_logging(verbose: bool) -> None:
    configure_logging(verbose)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)

    if args.input_jsonl is not None:
        records: Iterable[dict[str, Any]] = _load_jsonl(args.input_jsonl)
    elif args.dataset:
        records = _load_hf(args.dataset, args.split)
    else:
        logger.error("Cần --dataset <hf_id> hoặc --input-jsonl <file>")
        return 2

    out_dir = args.out_dir or (Paths().data_dir / "translated")
    job = TranslationJob(
        translator=build_translator(
            args.backend, model_name=args.model, use_vllm=not args.no_vllm
        ),
        out_dir=out_dir,
        split=args.split,
        fields=tuple(args.fields.split(",")) if args.fields else None,
        fen_field=args.fen_field or None,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
        max_retries=args.max_retries,
    )
    job.run(records, limit=args.limit, resume=args.resume, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

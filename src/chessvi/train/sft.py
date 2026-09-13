"""QLoRA SFT cho chess-vi. Chạy trên Colab, KHÔNG chạy local (4GB VRAM).

    # smoke test trên CPU với model tí hon
    python -m chessvi.train.sft --data data/translated/sft --limit 50 \
        --max-steps 5 --base-model Qwen/Qwen3-0.6B --no-push --output-dir /tmp/smoke

    # chạy thật trên Colab
    python -m chessvi.train.sft --data data/validated/sft --hub-model-id <user>/chessvi-4b

Token HF đọc từ biến môi trường ``HF_TOKEN``, không bao giờ hardcode.
Checkpoint được đẩy lên Hub theo ``hub_strategy="checkpoint"`` để Colab ngắt
giữa chừng vẫn resume được từ Hub.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chessvi.config import hf_token
from chessvi.logging_setup import TIMESTAMPED_FORMAT, configure_logging
from chessvi.train.dataset import DEFAULT_SYSTEM_PROMPT, load_examples

logger = logging.getLogger(__name__)

__all__ = ["SFTSettings", "build_parser", "main", "run_sft"]

DEFAULT_BASE_MODEL = "Qwen/Qwen3-4B"
DEFAULT_MAX_LENGTH = 2048
LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05


@dataclass(frozen=True)
class SFTSettings:
    """Toàn bộ tham số một lần chạy SFT."""

    data: Path
    output_dir: Path
    base_model: str = DEFAULT_BASE_MODEL
    max_length: int = DEFAULT_MAX_LENGTH
    limit: int | None = None
    max_steps: int = -1
    epochs: float = 1.0
    batch_size: int = 1
    grad_accum: int = 16
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 200
    seed: int = 20240911
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    push_to_hub: bool = True
    hub_model_id: str | None = None
    #: None = tự quyết theo việc máy có GPU hay không.
    load_in_4bit: bool | None = None
    resume: bool = False
    #: Gom chuỗi dài gần nhau vào cùng batch để bớt pad. Xem :func:`_log_lengths`.


# -- phụ thuộc nặng, import lười ------------------------------------------


def _has_cuda() -> bool:
    import torch  # noqa: PLC0415

    return bool(torch.cuda.is_available())


def _bf16_supported() -> bool:
    """bf16 chỉ thật sự dùng được từ Ampere trở lên (compute capability >= 8).

    T4 của Colab là sm_75. Tuỳ phiên bản, transformers ném thẳng
    "Your setup doesn't support bf16/gpu", hoặc tệ hơn là im lặng chạy bf16 giả
    lập chậm khủng khiếp. Kiểm tra compute capability thay vì hỏi
    ``torch.cuda.is_bf16_supported()`` — hàm đó đổi ngữ nghĩa giữa các bản torch
    (bản mới mặc định tính cả emulation là "được hỗ trợ").
    """
    import torch  # noqa: PLC0415

    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability()
    return major >= 8


def _compute_dtype() -> Any:
    """dtype tính toán khi quantize 4-bit: bf16 nếu GPU đỡ được, không thì fp16."""
    import torch  # noqa: PLC0415

    return torch.bfloat16 if _bf16_supported() else torch.float16


#: LoRA chỉ bám vào attention. Dùng cho model MoE; xem :func:`_lora_targets`.
ATTENTION_ONLY_TARGETS: list[str] = ["q_proj", "k_proj", "v_proj", "o_proj"]


def _lora_targets(config: Any) -> str | list[str]:
    """Chọn module cho LoRA bám vào, theo kiến trúc của base model.

    Model dense thì ``"all-linear"`` là lựa chọn tốt. Model **MoE thì không**:
    Qwen3-30B-A3B có 128 expert × 3 phép chiếu × 48 lớp = 18.432 linear, nên
    ``"all-linear"`` sinh ra ~1,7 tỷ tham số LoRA — nhiều hơn cả một model
    1,5B. Tệ hơn, router chỉ kích hoạt 8/128 expert mỗi token, nên hầu hết
    adapter gần như không nhận gradient: tốn bộ nhớ và thời gian để học được
    rất ít.

    Attention thì mọi token đều đi qua, nên bám vào đó là đủ và ổn định. Với
    Qwen3-30B-A3B ra khoảng 27M tham số huấn luyện.
    """
    if getattr(config, "num_experts", None):
        logger.info(
            "Base model là MoE (%s expert) — LoRA chỉ bám attention, bỏ expert",
            config.num_experts,
        )
        return ATTENTION_ONLY_TARGETS
    return "all-linear"


def _verify_push_access(hub_model_id: str) -> None:
    """Chốt quyền ghi Hub **trước** khi tốn công nạp dữ liệu và model.

    ``Trainer`` chỉ tạo repo lúc dựng, tức sau khi đã tokenize xong cả tập và
    nạp xong base model — vài phút cho 4B, lâu hơn nhiều cho 30B. Mà token chỉ
    có quyền đọc thì kể cả qua được chỗ đó cũng chết lần nữa ở lần lưu
    checkpoint đầu tiên, vì ``hub_strategy="checkpoint"`` đẩy mỗi ``save_steps``.

    Hai kiểu hỏng hay gặp, thông điệp của Hub giống hệt nhau nên phải nói rõ cả
    hai: namespace không phải account của token, và token là loại Read.
    """
    from huggingface_hub import create_repo  # noqa: PLC0415

    try:
        create_repo(hub_model_id, private=True, exist_ok=True, token=hf_token())
    except Exception as error:  # noqa: BLE001 - hub ném nhiều kiểu lỗi khác nhau
        raise RuntimeError(
            f"Không tạo hoặc ghi được repo {hub_model_id!r} trên Hugging Face.\n"
            f"  {type(error).__name__}: {error}\n"
            "Hai nguyên nhân thường gặp:\n"
            "  1. Token là loại Read. Tạo token Write ở "
            "huggingface.co/settings/tokens rồi đặt lại HF_TOKEN.\n"
            "  2. Namespace không phải account của token. Phần trước dấu '/' "
            "phải là tên account Hugging Face, không phải tên GitHub.\n"
            "Chỉ muốn lưu vào --output-dir, không push: thêm --no-push."
        ) from error
    logger.info("Repo Hub %s sẵn sàng", hub_model_id)


def _build_model(settings: SFTSettings, use_4bit: bool) -> Any:
    import torch  # noqa: PLC0415
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: PLC0415
    from transformers import AutoModelForCausalLM  # noqa: PLC0415

    kwargs: dict[str, Any] = {"dtype": _compute_dtype() if use_4bit else torch.float32}
    if use_4bit:
        from transformers import BitsAndBytesConfig  # noqa: PLC0415

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=_compute_dtype(),
        )
        kwargs["device_map"] = "auto"

    logger.info("Nạp base model %s (4-bit: %s)", settings.base_model, use_4bit)
    model = AutoModelForCausalLM.from_pretrained(settings.base_model, **kwargs)
    model.config.use_cache = False  # bắt buộc khi bật gradient checkpointing
    if use_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=_lora_targets(model.config),
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model


def _tokenize(examples: Sequence[dict[str, Any]], tokenizer: Any, max_length: int) -> Any:
    """Tokenize hội thoại, che (mask) phần prompt để chỉ học phần trả lời."""
    from datasets import Dataset  # noqa: PLC0415

    rows: list[dict[str, list[int]]] = []
    for example in examples:
        messages = example["messages"]
        prompt_text = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        full_text = tokenizer.apply_chat_template(messages, tokenize=False)

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(
            full_text, add_special_tokens=False, truncation=True, max_length=max_length
        )["input_ids"]

        labels = list(full_ids)
        for index in range(min(len(prompt_ids), len(labels))):
            labels[index] = -100
        if all(label == -100 for label in labels):
            continue  # prompt dài hơn max_length, mẫu này không học được gì
        rows.append(
            {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}
        )

    if not rows:
        raise ValueError("không dựng được mẫu nào sau khi tokenize")
    logger.info("Tokenize xong %d mẫu (max_length=%d)", len(rows), max_length)
    _log_lengths([len(row["input_ids"]) for row in rows])
    return Dataset.from_list(rows)


def _log_lengths(lengths: Sequence[int]) -> None:
    """In phân bố độ dài token — cho biết pad đang phí bao nhiêu.

    Collator pad động theo chuỗi dài nhất *trong từng micro-batch*, nên chi phí
    mỗi batch tỷ lệ với chuỗi dài nhất của batch đó chứ không phải độ dài trung
    bình. Batch càng to thì càng dễ vớ phải một chuỗi dài, và cả batch phải
    gánh theo.

    Không có cờ gom chuỗi cùng độ dài: ``group_by_length`` đã bị bỏ khỏi
    ``TrainingArguments`` từ transformers 5. Trên dữ liệu này cũng không tiếc:
    đo thật được p50=353 p90=415 p99=495 max=760, phổ rất hẹp nên pad chỉ
    phí cỡ 20-30%. Dữ liệu nào lệch mạnh hơn thì phải tự viết sampler.
    """
    ordered = sorted(lengths)
    last = len(ordered) - 1

    def at(quantile: float) -> int:
        return ordered[min(last, int(quantile * len(ordered)))]

    logger.info(
        "Độ dài token: p50=%d p90=%d p99=%d max=%d", at(0.50), at(0.90), at(0.99), ordered[-1]
    )


def _training_arguments(
    settings: SFTSettings, use_bf16: bool, use_fp16: bool, push: bool
) -> Any:
    from transformers import TrainingArguments  # noqa: PLC0415

    return TrainingArguments(
        output_dir=str(settings.output_dir),
        per_device_train_batch_size=settings.batch_size,
        gradient_accumulation_steps=settings.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=settings.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=settings.warmup_ratio,
        num_train_epochs=settings.epochs,
        max_steps=settings.max_steps,
        bf16=use_bf16,
        fp16=use_fp16,
        optim="paged_adamw_8bit" if (use_bf16 or use_fp16) else "adamw_torch",
        logging_steps=settings.logging_steps,
        save_strategy="steps",
        save_steps=settings.save_steps,
        save_total_limit=2,
        report_to=[],
        seed=settings.seed,
        # Colab hay ngắt giữa chừng: đẩy luôn checkpoint lên Hub để resume được.
        push_to_hub=push,
        hub_strategy="checkpoint",
        hub_private_repo=True,
        hub_model_id=settings.hub_model_id,
        hub_token=hf_token(),
    )


def run_sft(settings: SFTSettings) -> Path:
    """Chạy một lần SFT. Trả về thư mục chứa adapter đã lưu."""
    from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Trainer  # noqa: PLC0415

    use_4bit = settings.load_in_4bit if settings.load_in_4bit is not None else _has_cuda()
    # CPU thì train fp32 cho chắc. Có GPU thì bf16 nếu là Ampere trở lên,
    # còn T4/V100 phải dùng fp16 — bật bf16 ở đó là lỗi cứng, không phải chậm.
    use_bf16 = use_4bit and _bf16_supported()
    use_fp16 = use_4bit and not use_bf16

    push = settings.push_to_hub
    if push and hf_token() is None:
        logger.warning("Không có HF_TOKEN trong môi trường — tắt push_to_hub")
        push = False
    if push and settings.hub_model_id is None:
        logger.warning("Không có --hub-model-id — tắt push_to_hub")
        push = False
    if push:
        _verify_push_access(str(settings.hub_model_id))

    examples = load_examples(
        settings.data, limit=settings.limit, system_prompt=settings.system_prompt
    )
    if not examples:
        raise ValueError(f"không có mẫu nào trong {settings.data}")

    tokenizer = AutoTokenizer.from_pretrained(settings.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = _tokenize(examples, tokenizer, settings.max_length)

    model = _build_model(settings, use_4bit)
    trainer = Trainer(
        model=model,
        args=_training_arguments(settings, use_bf16, use_fp16, push),
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    )

    trainer.train(resume_from_checkpoint=settings.resume or None)
    trainer.save_model(str(settings.output_dir))
    tokenizer.save_pretrained(str(settings.output_dir))
    if push:
        trainer.push_to_hub()
    logger.info("Đã lưu adapter vào %s", settings.output_dir)
    return settings.output_dir


# -- CLI ------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m chessvi.train.sft",
        description="QLoRA SFT trên dữ liệu tiếng Việt đã validate.",
    )
    parser.add_argument("--data", type=Path, required=True, help="File/thư mục dữ liệu")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sft"))
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--limit", type=int, help="Chỉ dùng N mẫu đầu (smoke test)")
    parser.add_argument("--max-steps", type=int, default=-1, help="-1 = theo epoch")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20240911)
    parser.add_argument("--hub-model-id", help="Ví dụ: <user>/chessvi-4b-sft")
    parser.add_argument("--no-push", action="store_true", help="Tắt push_to_hub")
    quant = parser.add_mutually_exclusive_group()
    quant.add_argument("--4bit", dest="four_bit", action="store_true", default=None)
    quant.add_argument("--no-4bit", dest="four_bit", action="store_false")
    parser.add_argument("--resume", action="store_true", help="Resume từ checkpoint")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def settings_from_args(args: argparse.Namespace) -> SFTSettings:
    return SFTSettings(
        data=args.data,
        output_dir=args.output_dir,
        base_model=args.base_model,
        max_length=args.max_length,
        limit=args.limit,
        max_steps=args.max_steps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.learning_rate,
        save_steps=args.save_steps,
        seed=args.seed,
        push_to_hub=not args.no_push,
        hub_model_id=args.hub_model_id,
        load_in_4bit=args.four_bit,
        resume=args.resume,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose, fmt=TIMESTAMPED_FORMAT)
    run_sft(settings_from_args(args))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

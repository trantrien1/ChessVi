"""Interface chung cho mọi model được đánh giá.

Cả ``puzzle_acc.py`` lẫn ``hallucination.py`` chỉ biết tới :class:`Predictor`,
nhờ vậy so sánh được nhiều baseline trên cùng một thước đo: model của mình
(GGUF/HF), C1-4B gốc, Qwen3-4B chưa fine-tune.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from chessvi.config import ServeConfig

logger = logging.getLogger(__name__)

__all__ = [
    "EchoPredictor",
    "GGUFPredictor",
    "HFPredictor",
    "Predictor",
    "build_predictor",
]


class Predictor(ABC):
    """Model sinh câu trả lời từ prompt text thuần."""

    name: str = "abstract"

    @abstractmethod
    def predict(self, prompt: str) -> str:
        """Sinh câu trả lời cho một prompt."""

    def predict_batch(self, prompts: Sequence[str]) -> list[str]:
        """Mặc định chạy tuần tự; backend nào batch được thì override."""
        return [self.predict(prompt) for prompt in prompts]

    def close(self) -> None:
        """Giải phóng tài nguyên. Mặc định không làm gì."""


class EchoPredictor(Predictor):
    """Trả về câu cố định. Dùng để test harness mà không cần model."""

    name = "echo"

    def __init__(self, answer: str = "") -> None:
        self._answer = answer
        self.prompts: list[str] = []

    def predict(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._answer


class GGUFPredictor(Predictor):
    """Model local GGUF Q4 qua llama-cpp-python (4GB VRAM, ctx <= 4096)."""

    name = "gguf"

    def __init__(self, model_path: Path, config: ServeConfig | None = None) -> None:
        from llama_cpp import Llama  # noqa: PLC0415

        self._config = config or ServeConfig()
        if not model_path.is_file():
            raise FileNotFoundError(f"không thấy file GGUF: {model_path}")
        self._llama = Llama(
            model_path=str(model_path),
            n_ctx=self._config.n_ctx,
            n_gpu_layers=self._config.n_gpu_layers,
            type_k=self._config.cache_type_k,
            type_v=self._config.cache_type_v,
            verbose=False,
        )

    def predict(self, prompt: str) -> str:
        output = self._llama(
            prompt,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
            stop=["[CÂU HỎI]", "[SỰ THẬT ĐÃ XÁC MINH"],
        )
        return str(output["choices"][0]["text"]).strip()

    def close(self) -> None:
        self._llama.close()


class HFPredictor(Predictor):
    """Model trên Hugging Face (dùng khi eval baseline trên Colab/Kaggle)."""

    name = "hf"

    def __init__(
        self,
        model_path: str,
        *,
        adapter: str | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        device: str | None = None,
    ) -> None:
        import torch  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Nạp %s trên %s", model_path, self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        # dtype="auto" lấy đúng dtype ghi trong config (bf16 với Qwen3). Không
        # truyền thì transformers nạp fp32: 4B thành 16GB thay vì 8GB, và chậm
        # gấp đôi mà chẳng chính xác hơn — eval chỉ đọc argmax của nước đi.
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype="auto")
        model = model.to(self._device)
        if adapter is not None:
            from peft import PeftModel  # noqa: PLC0415

            model = PeftModel.from_pretrained(model, adapter)
        self._model = model.eval()

    def predict(self, prompt: str) -> str:
        import torch  # noqa: PLC0415

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                temperature=self._temperature,
                do_sample=self._temperature > 0,
            )
        new_tokens = generated[0][inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def predict_batch(self, prompts: Sequence[str]) -> list[str]:
        """Sinh cả batch trong **một** lời gọi ``generate``.

        Chạy tuần tự là bỏ phí gần hết GPU: mỗi bước decode chủ yếu tốn công
        đọc trọng số model từ VRAM, mà đọc một lượt thì phục vụ được cả batch.
        8 chuỗi song song gần như cùng thời gian với 1 chuỗi. Đo trên 300
        puzzle: tuần tự mất khoảng một tiếng.

        Phải pad **bên trái**. Model decoder-only sinh tiếp từ token cuối cùng
        của input; pad bên phải thì token cuối là ``<pad>`` và nó sinh ra rác.
        Pad bên trái cũng làm mọi hàng có cùng độ rộng input, nên cắt phần
        prompt ra khỏi output chỉ là một phép lát duy nhất.
        """
        if not prompts:
            return []
        import torch  # noqa: PLC0415

        previous_side = self._tokenizer.padding_side
        self._tokenizer.padding_side = "left"
        try:
            inputs = self._tokenizer(
                list(prompts), return_tensors="pt", padding=True
            ).to(self._device)
        finally:
            self._tokenizer.padding_side = previous_side

        with torch.no_grad():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                temperature=self._temperature,
                do_sample=self._temperature > 0,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        width = inputs["input_ids"].shape[1]
        return [
            self._tokenizer.decode(row[width:], skip_special_tokens=True).strip()
            for row in generated
        ]


def build_predictor(kind: str, **kwargs: Any) -> Predictor:
    """Tạo predictor theo tên backend. Thêm baseline mới thì khai báo ở đây."""
    if kind == "echo":
        return EchoPredictor(**kwargs)
    if kind == "gguf":
        return GGUFPredictor(Path(kwargs.pop("model_path")), **kwargs)
    if kind == "hf":
        return HFPredictor(**kwargs)
    raise ValueError(f"Backend không biết: {kind!r} (có: echo, gguf, hf)")

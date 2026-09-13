"""Bày model lên một endpoint kiểu OpenAI, để chạy trên Colab.

    python -m chessvi.serve.colab_api \
        --model-path Qwen/Qwen3-14B \
        --adapter /content/drive/MyDrive/chessvi/outputs/sft-14b

Vì sao cần: chatbot bám Qwen3-14B, mà 14B bf16 ngốn ~28GB VRAM — máy local 4GB
không chạy nổi (xem CLAUDE.md). Colab có GPU nhưng không có địa chỉ công khai,
nên notebook mở thêm một tunnel rồi đưa URL cho::

    python -m chessvi.serve.web --llm remote --api-base <url>/v1

Chỉ nói đúng hai endpoint: ``GET /health`` và ``POST /v1/chat/completions``. Đủ
cho :class:`~chessvi.serve.chat.RemoteLLM`, và đủ để thử bằng curl.

Trình duyệt **không** gọi thẳng vào đây — ``serve/web.py`` gọi từ phía Python,
nên không phải lo CORS và không lộ URL Colab ra trang web.
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from chessvi.eval.predictor import HFPredictor
from chessvi.logging_setup import TIMESTAMPED_FORMAT, configure_logging

logger = logging.getLogger(__name__)

__all__ = ["build_parser", "extract_prompt", "main", "serve"]

#: Prompt của trợ lý cờ dài cỡ 1-2KB; để rộng tay cho lịch sử hội thoại sau này.
MAX_BODY = 256 * 1024


def extract_prompt(messages: list[dict[str, Any]], tokenizer: Any) -> str:
    """Đổi ``messages`` kiểu OpenAI thành prompt đúng chat template của model.

    Dùng chính ``apply_chat_template`` của tokenizer chứ không tự nối chuỗi:
    T8 tokenize dữ liệu huấn luyện bằng đúng hàm đó, nên nối tay là cho model
    một khuôn nó chưa từng thấy.
    """
    turns = [
        {"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
        for m in messages
        if isinstance(m, dict)
    ]
    if not turns:
        raise ValueError("messages rỗng")
    return str(
        tokenizer.apply_chat_template(turns, tokenize=False, add_generation_prompt=True)
    )


class _Model:
    """Model dùng chung, có khoá — một GPU thì phải tuần tự hoá."""

    def __init__(self, predictor: HFPredictor, tokenizer: Any, name: str) -> None:
        self._predictor = predictor
        self._tokenizer = tokenizer
        self._lock = threading.Lock()
        self.name = name

    def complete(self, messages: list[dict[str, Any]], temperature: float) -> str:
        prompt = extract_prompt(messages, self._tokenizer)
        with self._lock:
            return self._predictor.predict(prompt, temperature=temperature)


def _make_handler(model: _Model) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "chessvi-api"

        def log_message(self, format: str, *args: Any) -> None:
            logger.debug("%s %s", self.address_string(), format % args)

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - tên do thư viện chuẩn quy định
            if self.path.rstrip("/") in ("/health", ""):
                self._send_json({"status": "ok", "model": model.name})
                return
            self._send_json({"error": "Không có endpoint này"}, status=404)

        def do_POST(self) -> None:  # noqa: N802 - tên do thư viện chuẩn quy định
            # Đọc hết body trước khi phân nhánh: bỏ dở trên kết nối keep-alive
            # thì phần thừa bị đọc thành request kế tiếp.
            try:
                length = int(self.headers.get("Content-Length") or 0)
                if length > MAX_BODY:
                    self.close_connection = True
                    raise ValueError("Request quá lớn")
                data = json.loads(self.rfile.read(length) or b"{}")
            except ValueError as error:
                self._send_json({"error": str(error)}, status=400)
                return

            if self.path.rstrip("/") != "/v1/chat/completions":
                self._send_json({"error": "Không có endpoint này"}, status=404)
                return

            try:
                messages = data["messages"]
                if not isinstance(messages, list):
                    raise ValueError("messages phải là danh sách")
                temperature = float(data.get("temperature", 0.3))
                started = time.monotonic()
                text = model.complete(messages, temperature)
                logger.info("Sinh xong trong %.1fs", time.monotonic() - started)
            except (KeyError, TypeError, ValueError) as error:
                self._send_json({"error": f"Body không hợp lệ: {error}"}, status=400)
                return
            except Exception as error:  # noqa: BLE001 - model ném nhiều kiểu
                logger.exception("Lỗi khi sinh")
                self._send_json({"error": f"{type(error).__name__}: {error}"}, status=500)
                return

            self._send_json(
                {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model.name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": "stop",
                        }
                    ],
                }
            )

    return Handler


def serve(
    model_path: str,
    *,
    adapter: str | None = None,
    host: str = "0.0.0.0",  # noqa: S104 - phải nghe mọi interface để tunnel vào được
    port: int = 8001,
    max_new_tokens: int = 512,
    served_name: str | None = None,
) -> None:
    """Nạp model rồi phục vụ cho tới khi Ctrl+C."""
    from transformers import AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    predictor = HFPredictor(model_path, adapter=adapter, max_new_tokens=max_new_tokens)
    model = _Model(predictor, tokenizer, served_name or (adapter or model_path))

    server = ThreadingHTTPServer((host, port), _make_handler(model))
    logger.info("Sẵn sàng ở cổng %d (model %s)", port, model.name)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Dừng server")
    finally:
        server.server_close()
        predictor.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m chessvi.serve.colab_api",
        description="Endpoint kiểu OpenAI cho model chess-vi, chạy trên Colab.",
    )
    parser.add_argument("--model-path", required=True, help="Base model, vd Qwen/Qwen3-14B")
    parser.add_argument("--adapter", help="Thư mục adapter LoRA sau SFT")
    parser.add_argument("--host", default="0.0.0.0")  # noqa: S104
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--served-name", help="Tên báo ra ngoài, mặc định theo adapter")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose, fmt=TIMESTAMPED_FORMAT)
    serve(
        args.model_path,
        adapter=args.adapter,
        host=args.host,
        port=args.port,
        max_new_tokens=args.max_new_tokens,
        served_name=args.served_name,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

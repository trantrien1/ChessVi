"""Vòng lặp chat CLI: vừa chơi cờ vừa hỏi, mọi câu trả lời đi qua guard.

Inference local là GGUF Q4 qua llama-cpp-python, ``n_ctx=4096``, KV cache q8_0
— VRAM thật chỉ 4GB (xem CLAUDE.md), tràn sang shared memory sẽ chậm 10–50x.

    python -m chessvi.serve.chat --gguf models/chessvi-q4_k_m.gguf --elo 1500

Lệnh trong phiên: ``/di <nước>``, ``/bot``, ``/fen``, ``/moi``, ``/giup``,
``/thoat``. Gõ bất cứ thứ gì khác là đặt câu hỏi cho trợ lý.
"""

from __future__ import annotations

import argparse
import logging
import sys
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

import chess

from chessvi.config import EngineConfig, Paths, ServeConfig
from chessvi.data.mask import parse_move
from chessvi.engine.facts import PositionFacts, extract_facts
from chessvi.engine.opponent import Opponent
from chessvi.engine.stockfish import StockfishEngine
from chessvi.logging_setup import configure_logging
from chessvi.serve.guard import guarded_generate
from chessvi.serve.prompt import ChatTurn, build_prompt

logger = logging.getLogger(__name__)

__all__ = ["ChatSession", "LlamaCppLLM", "LocalLLM", "RemoteLLM", "main"]


class LocalLLM(ABC):
    """Backend sinh text. Tách interface để test không cần model thật."""

    @abstractmethod
    def generate(self, prompt: str, *, temperature: float) -> str:
        """Sinh câu trả lời cho ``prompt``."""

    def close(self) -> None:
        """Giải phóng tài nguyên. Mặc định không làm gì."""


class LlamaCppLLM(LocalLLM):
    """GGUF Q4 chạy local qua llama-cpp-python."""

    def __init__(self, gguf_path: Path, config: ServeConfig | None = None) -> None:
        from llama_cpp import Llama  # noqa: PLC0415

        self._config = config or ServeConfig()
        if not gguf_path.is_file():
            raise FileNotFoundError(f"không thấy file GGUF: {gguf_path}")
        logger.info("Nạp GGUF %s (n_ctx=%d)", gguf_path, self._config.n_ctx)
        self._llama = Llama(
            model_path=str(gguf_path),
            n_ctx=self._config.n_ctx,
            n_gpu_layers=self._config.n_gpu_layers,
            type_k=self._config.cache_type_k,
            type_v=self._config.cache_type_v,
            verbose=False,
        )

    def generate(self, prompt: str, *, temperature: float) -> str:
        output = self._llama(
            prompt,
            max_tokens=self._config.max_tokens,
            temperature=temperature,
            stop=["[CÂU HỎI]", "[SỰ THẬT ĐÃ XÁC MINH"],
        )
        return str(output["choices"][0]["text"]).strip()

    def close(self) -> None:
        self._llama.close()


class RemoteLLM(LocalLLM):
    """Model chạy ở máy khác, gọi qua API kiểu OpenAI.

    Đường này tồn tại vì chatbot bám Qwen3-14B (lý do và số đo trong CLAUDE.md)
    mà 14B không nhét vừa 4GB VRAM. Bất cứ thứ gì nói giao thức đó đều cắm
    được: vLLM, llama.cpp server, LM Studio, Ollama, hay một Colab mở tunnel.

    Dùng ``urllib`` của thư viện chuẩn nên không thêm phụ thuộc nào.

    ``api_base`` là gốc có kèm ``/v1``, ví dụ ``http://localhost:8001/v1``.
    """

    def __init__(
        self,
        api_base: str,
        model: str,
        *,
        api_key: str | None = None,
        config: ServeConfig | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._url = api_base.rstrip("/") + "/chat/completions"
        self._model = model
        self._api_key = api_key
        self._config = config or ServeConfig()
        self._timeout = timeout
        logger.info("Chatbot gọi %s (model %s)", self._url, model)

    def generate(self, prompt: str, *, temperature: float) -> str:
        import json  # noqa: PLC0415
        import urllib.error  # noqa: PLC0415
        import urllib.request  # noqa: PLC0415

        # Gửi cả prompt làm một lượt user. Prompt do build_prompt dựng đã chứa
        # sẵn phần hệ thống, và server bên kia sẽ bọc nó bằng chat template của
        # chính model — đúng khuôn model đã học ở T8.
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": self._config.max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request = urllib.request.Request(
            self._url, data=json.dumps(payload).encode("utf-8"), headers=headers
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                body = json.loads(response.read())
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")[:500]
            raise RuntimeError(f"Model trả lỗi {error.code}: {detail}") from error
        except OSError as error:
            raise RuntimeError(f"Không gọi được {self._url}: {error}") from error
        return str(body["choices"][0]["message"]["content"]).strip()


class ChatSession:
    """Một ván cờ kèm hội thoại. Giữ bàn cờ, engine và lịch sử."""

    def __init__(
        self,
        llm: LocalLLM,
        *,
        board: chess.Board | None = None,
        engine: StockfishEngine | None = None,
        opponent: Opponent | None = None,
        elo: int = 1500,
        serve_config: ServeConfig | None = None,
        analyse_depth: int | None = None,
    ) -> None:
        self.board = board if board is not None else chess.Board()
        self.history: list[ChatTurn] = []
        self._llm = llm
        self._engine = engine
        self._opponent = opponent
        self._elo = elo
        self._config = serve_config or ServeConfig()
        self._depth = analyse_depth

    # -- fact + hỏi đáp ---------------------------------------------------

    def facts(self) -> PositionFacts:
        """Fact của thế cờ hiện tại, có eval nếu có engine."""
        return extract_facts(self.board, engine=self._engine, depth=self._depth)

    def ask(self, question: str) -> str:
        """Hỏi trợ lý về thế cờ hiện tại. Câu trả lời luôn đi qua guard."""
        facts = self.facts()

        def generate(attempt: int) -> str:
            prompt = build_prompt(facts, question, self.history)
            if attempt:
                prompt += (
                    "\nLƯU Ý: câu trả lời trước có nước đi không hợp lệ. "
                    "Chỉ dùng nước trong danh sách nước đi hợp lệ ở trên."
                )
            return self._llm.generate(
                prompt, temperature=self._config.temperature + 0.1 * attempt
            )

        answer = guarded_generate(
            generate,
            self.board,
            facts,
            max_regenerations=self._config.max_regenerations,
        )
        self.history.append(ChatTurn("user", question))
        self.history.append(ChatTurn("assistant", answer))
        return answer

    # -- chơi cờ ----------------------------------------------------------

    def play(self, move_text: str) -> chess.Move:
        """Đi một nước của người chơi. Nguyên tắc 2: parse rồi mới tin."""
        move = parse_move(self.board, move_text)
        if move is None:
            raise ValueError(f"{move_text!r} không phải nước đi hợp lệ ở thế này")
        self.board.push(move)
        return move

    def play_opponent(self) -> chess.Move:
        """Để đối thủ AI đi một nước ở mức Elo đã chọn."""
        if self._opponent is None:
            raise RuntimeError("phiên này không có đối thủ AI")
        move = self._opponent.get_move(self.board.fen(), self._elo)
        self.board.push(move)
        return move

    def reset(self, fen: str | None = None) -> None:
        self.board = chess.Board() if fen is None else chess.Board(fen)
        self.history.clear()

    @property
    def status(self) -> str:
        """Mô tả ngắn trạng thái ván cờ bằng tiếng Việt."""
        if self.board.is_checkmate():
            winner = "Đen" if self.board.turn == chess.WHITE else "Trắng"
            return f"Chiếu hết. {winner} thắng."
        if self.board.is_stalemate():
            return "Hết nước đi. Hòa."
        if self.board.is_insufficient_material():
            return "Không đủ lực chiếu hết. Hòa."
        side = "Trắng" if self.board.turn == chess.WHITE else "Đen"
        check = " (đang bị chiếu)" if self.board.is_check() else ""
        return f"Lượt {side}{check}."


# -- CLI ------------------------------------------------------------------

HELP_TEXT = """Lệnh:
  /di <nước>   đi một nước (SAN hoặc UCI), ví dụ: /di e4
  /bot         để đối thủ AI đi một nước
  /fen         in FEN và bàn cờ hiện tại
  /moi [fen]   ván mới (có thể kèm FEN)
  /giup        in trợ giúp này
  /thoat       thoát
Gõ bất cứ thứ gì khác để hỏi trợ lý về thế cờ."""


def _print(text: str) -> None:
    """In cho người dùng. Dùng logging để tôn trọng quy ước của dự án."""
    logger.info("%s", text)


def _handle_command(session: ChatSession, line: str) -> bool:
    """Xử lý một lệnh ``/...``. Trả về False khi cần thoát."""
    command, _, argument = line.partition(" ")
    argument = argument.strip()

    if command in ("/thoat", "/quit", "/exit"):
        return False
    if command in ("/giup", "/help"):
        _print(HELP_TEXT)
    elif command == "/fen":
        _print(f"{session.board}\nFEN: {session.board.fen()}\n{session.status}")
    elif command in ("/moi", "/new"):
        session.reset(argument or None)
        _print(f"Ván mới.\n{session.status}")
    elif command == "/di":
        try:
            move = session.play(argument)
        except ValueError as error:
            _print(f"Lỗi: {error}")
        else:
            _print(f"Bạn đi {move.uci()}. {session.status}")
    elif command == "/bot":
        try:
            move = session.play_opponent()
        except (RuntimeError, ValueError, FileNotFoundError) as error:
            _print(f"Lỗi: {error}")
        else:
            _print(f"Máy đi {move.uci()}. {session.status}")
    else:
        _print(f"Không biết lệnh {command!r}. Gõ /giup để xem danh sách.")
    return True


def run_loop(session: ChatSession, stream: object = None) -> None:
    """Vòng lặp đọc lệnh. ``stream`` mặc định là stdin, test truyền iterable."""
    source = sys.stdin if stream is None else stream
    _print(HELP_TEXT)
    _print(session.status)
    for raw in source:  # type: ignore[union-attr]
        line = raw.strip()
        if not line:
            continue
        if line.startswith("/"):
            if not _handle_command(session, line):
                break
            continue
        _print(session.ask(line))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m chessvi.serve.chat",
        description="Chat cờ vua tiếng Việt + đối thủ AI, chạy local bằng GGUF Q4.",
    )
    parser.add_argument("--gguf", type=Path, help="Mặc định: theo CHESSVI_GGUF_PATH")
    parser.add_argument("--fen", help="Bắt đầu từ FEN này thay vì thế khởi đầu")
    parser.add_argument("--elo", type=int, default=1500, help="Mức Elo của đối thủ AI")
    parser.add_argument("--depth", type=int, help="Độ sâu Stockfish khi lấy eval")
    parser.add_argument(
        "--no-engine", action="store_true", help="Không dùng Stockfish (fact sẽ không có eval)"
    )
    parser.add_argument(
        "--allow-stockfish-fallback",
        action="store_true",
        help="Thiếu Maia thì tạm dùng Stockfish yếu (độ mạnh KHÔNG giống người)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)

    paths = Paths()
    serve_config = ServeConfig()
    gguf_path = args.gguf or serve_config.gguf_path

    engine: StockfishEngine | None = None
    llm: LocalLLM | None = None
    try:
        llm = LlamaCppLLM(gguf_path, serve_config)
    except (FileNotFoundError, ImportError) as error:
        logger.error("Không nạp được model local: %s", error)
        logger.error('Cài bằng: pip install -e ".[serve]" và tải file GGUF Q4 về.')
        return 1

    stockfish_bin = None if args.no_engine else paths.resolve_stockfish()
    if stockfish_bin is not None:
        engine = StockfishEngine(stockfish_bin, config=EngineConfig()).__enter__()
    elif not args.no_engine:
        logger.warning("Không thấy Stockfish — fact sẽ không có eval")

    opponent = Opponent(
        paths=paths, allow_stockfish_fallback=args.allow_stockfish_fallback
    )
    try:
        session = ChatSession(
            llm,
            board=chess.Board(args.fen) if args.fen else None,
            engine=engine,
            opponent=opponent,
            elo=args.elo,
            serve_config=serve_config,
            analyse_depth=args.depth,
        )
        run_loop(session)
    finally:
        opponent.close()
        if engine is not None:
            engine.close()
        llm.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

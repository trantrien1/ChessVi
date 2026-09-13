"""Web demo: bàn cờ trong trình duyệt, đối thủ theo Elo, fact đã xác minh.

    python -m chessvi.serve.web --port 8000

Không thêm phụ thuộc nào — ``http.server`` của thư viện chuẩn là đủ cho một
demo chạy trên máy mình. Toàn bộ logic cờ nằm ở ``engine/``, lớp này chỉ dịch
HTTP sang lời gọi hàm, nên đổi sang FastAPI sau này là thay mỗi phần routing.

Trạng thái ván **không** giữ ở server: mỗi request mang theo FEN. Nhờ vậy
không cần session, mở nhiều tab không đá nhau, và restart server không mất ván.

Bảng bên phải hiển thị đúng khối ``[SỰ THẬT ĐÃ XÁC MINH]`` mà chatbot nhận,
nên nhìn được bằng mắt model được cho biết những gì — và chỉ những gì đó.

Chatbot bật bằng ``--llm``. Model 14B không nhét vừa 4GB VRAM nên đường
thường dùng là ``--llm remote``: model chạy trên Colab (xem
``notebooks/serve_colab.ipynb``), web chạy ở máy mình, nối nhau qua HTTP. Trình
duyệt không gọi thẳng sang Colab — chính server này gọi, nên không vướng CORS
và URL Colab không lộ ra trang web.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import chess

from chessvi.config import Paths, ServeConfig
from chessvi.engine.facts import PositionFacts, extract_facts, facts_to_prompt
from chessvi.engine.opponent import Opponent
from chessvi.engine.stockfish import StockfishEngine
from chessvi.logging_setup import configure_logging
from chessvi.serve.chat import LlamaCppLLM, LocalLLM, RemoteLLM
from chessvi.serve.guard import guarded_generate
from chessvi.serve.prompt import build_prompt, fallback_answer

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_ELO",
    "DEFAULT_QUESTION",
    "build_llm",
    "build_parser",
    "build_state",
    "main",
    "parse_move",
    "serve",
]

STATIC_DIR = Path(__file__).parent / "static"
DEFAULT_ELO = 1500

#: Hỏi gì khi người dùng bấm nút mà không gõ gì.
DEFAULT_QUESTION = "Phân tích ngắn gọn thế cờ này và cho biết bên đang đi nên chơi nước nào."

#: Thêm vào prompt sau mỗi lần guard chặn. Giống hệt `ChatSession.ask`.
RETRY_HINT = (
    "\nLƯU Ý: câu trả lời trước có nước đi không hợp lệ. "
    "Chỉ dùng nước trong danh sách nước đi hợp lệ ở trên."
)
#: FEN dài nhất cũng dưới 100 byte; lớn hơn mức này là request rác.
MAX_BODY = 8 * 1024

SIDE_VI: dict[str, str] = {"white": "Trắng", "black": "Đen"}

#: Lý do ván kết thúc, theo ``chess.Termination``.
TERMINATION_VI: dict[str, str] = {
    "CHECKMATE": "chiếu hết",
    "STALEMATE": "hết nước đi",
    "INSUFFICIENT_MATERIAL": "không đủ quân để chiếu hết",
    "SEVENTYFIVE_MOVES": "luật 75 nước",
    "FIVEFOLD_REPETITION": "lặp thế 5 lần",
    "FIFTY_MOVES": "luật 50 nước",
    "THREEFOLD_REPETITION": "lặp thế 3 lần",
}


def parse_move(board: chess.Board, src: str, dst: str, promotion: str | None = None) -> chess.Move:
    """Dựng nước đi từ hai ô người dùng bấm, và khẳng định nó hợp lệ.

    Nguyên tắc 2: không tin chuỗi thô. Chuỗi từ trình duyệt luôn đi qua
    ``Move.from_uci`` rồi kiểm ``in board.legal_moves`` trước khi được đi.

    Client không gửi quân phong cấp thì mặc định hậu — bàn cờ chỉ có hai ô
    được bấm, không có chỗ hỏi thêm. Chọn quân khác để sau.
    """
    try:
        move = chess.Move.from_uci(f"{src}{dst}{promotion or ''}")
    except ValueError as error:
        raise ValueError(f"Nước đi không đọc được: {src}{dst}{promotion or ''}") from error
    if move in board.legal_moves:
        return move
    queened = chess.Move.from_uci(f"{src}{dst}q")
    if promotion is None and queened in board.legal_moves:
        return queened
    raise ValueError(f"Nước {move.uci()} không hợp lệ ở thế cờ này")


def _result_text(board: chess.Board) -> str | None:
    """Kết quả ván, hoặc ``None`` nếu còn đang chơi."""
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    reason = TERMINATION_VI.get(outcome.termination.name, outcome.termination.name.lower())
    if outcome.winner is None:
        return f"Hoà — {reason}"
    side = SIDE_VI["white"] if outcome.winner == chess.WHITE else SIDE_VI["black"]
    return f"{side} thắng — {reason}"


def _legal_map(board: chess.Board) -> dict[str, list[str]]:
    """``{ô nguồn: [ô đích, ...]}`` để trình duyệt tô sáng, không tự đoán luật."""
    moves: dict[str, list[str]] = {}
    for move in board.legal_moves:
        moves.setdefault(chess.square_name(move.from_square), []).append(
            chess.square_name(move.to_square)
        )
    return moves


def _facts_payload(facts: PositionFacts) -> dict[str, Any]:
    """Fact đã xác minh, dạng sẵn để hiển thị."""
    return {
        "side_to_move": SIDE_VI[facts.side_to_move],
        "material": facts.material,
        "material_diff": facts.material_diff,
        "in_check": facts.in_check,
        "hanging": [{"square": square, "piece": symbol} for square, symbol in facts.hanging],
        "checks_available": facts.checks_available,
        "captures_available": facts.captures_available,
        "legal_count": len(facts.legal_moves),
        "opening": facts.opening_name,
        "is_checkmate": facts.is_checkmate,
        "is_stalemate": facts.is_stalemate,
    }


def build_state(board: chess.Board, last_move_san: str | None = None) -> dict[str, Any]:
    """Toàn bộ thứ trình duyệt cần để vẽ lại một thế cờ.

    Gom vào một hàm thuần để test được mà không cần dựng server.
    """
    facts = extract_facts(board)
    return {
        "fen": board.fen(),
        "turn": "white" if board.turn == chess.WHITE else "black",
        "turn_vi": SIDE_VI["white" if board.turn == chess.WHITE else "black"],
        "move_number": board.fullmove_number,
        "legal": _legal_map(board),
        "game_over": board.is_game_over(claim_draw=True),
        "result": _result_text(board),
        "last_move": last_move_san,
        "facts": _facts_payload(facts),
        # Đúng khối sẽ nhét vào prompt của chatbot. Hiển thị nguyên văn để
        # thấy model được cho biết những gì — và chỉ những gì đó.
        "facts_block": facts_to_prompt(facts),
    }


class _Bot:
    """Một đối thủ dùng chung, có khoá.

    Engine UCI là một process nói chuyện qua stdin/stdout, gọi song song từ hai
    thread sẽ trộn lẫn lệnh. ``ThreadingHTTPServer`` thì mỗi request một thread,
    nên phải tuần tự hoá ở đây.
    """

    def __init__(self, *, allow_stockfish_fallback: bool) -> None:
        self._opponent = Opponent(allow_stockfish_fallback=allow_stockfish_fallback)
        self._lock = threading.Lock()

    def get_move(self, fen: str, elo: int) -> chess.Move:
        with self._lock:
            return self._opponent.get_move(fen, elo)

    def close(self) -> None:
        with self._lock:
            self._opponent.close()


class _Analyst:
    """Tính fact cho chatbot, kèm eval Stockfish nếu máy có binary.

    Bảng bên phải bàn cờ dùng ``extract_facts`` trần vì nó chạy sau **mỗi**
    nước; chỗ này mới gọi engine, và chỉ khi người dùng hỏi. Eval và nước tốt
    nhất là fact đắt nhất nhưng cũng là fact làm lời giải thích có giá trị.

    Thiếu Stockfish thì vẫn trả fact, chỉ là không có eval — chatbot được dặn
    nói thẳng là không biết thay vì đoán (quy tắc 3 trong SYSTEM_PROMPT).
    """

    def __init__(self) -> None:
        self._engine: StockfishEngine | None = None
        self._unavailable = False
        self._lock = threading.Lock()

    def facts(self, board: chess.Board) -> PositionFacts:
        with self._lock:
            return extract_facts(board, engine=self._get_engine())

    def _get_engine(self) -> StockfishEngine | None:
        if self._unavailable:
            return None
        if self._engine is None:
            binary = Paths().resolve_stockfish()
            if binary is None:
                self._unavailable = True
                logger.warning(
                    "Không có Stockfish — fact sẽ thiếu eval và nước tốt nhất; "
                    "đặt CHESSVI_STOCKFISH_BIN"
                )
                return None
            self._engine = StockfishEngine(binary).__enter__()
        return self._engine

    def close(self) -> None:
        with self._lock:
            if self._engine is not None:
                self._engine.close()
                self._engine = None


def build_llm(
    kind: str,
    *,
    api_base: str | None = None,
    api_model: str = "chessvi",
    api_key: str | None = None,
    gguf: Path | None = None,
    config: ServeConfig | None = None,
) -> LocalLLM | None:
    """Dựng backend chatbot, hoặc ``None`` khi chạy không có chatbot."""
    if kind == "none":
        return None
    if kind == "remote":
        if not api_base:
            raise ValueError("--llm remote cần --api-base, ví dụ https://xxx.trycloudflare.com/v1")
        return RemoteLLM(api_base, api_model, api_key=api_key, config=config)
    if kind == "gguf":
        path = gguf or (config or ServeConfig()).gguf_path
        return LlamaCppLLM(Path(path), config)
    raise ValueError(f"Backend không biết: {kind!r} (có: none, remote, gguf)")


def _make_handler(
    bot: _Bot,
    analyst: _Analyst,
    llm: LocalLLM | None,
    config: ServeConfig,
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "chessvi"

        # -- tiện ích ----------------------------------------------------

        def log_message(self, format: str, *args: Any) -> None:
            """Mặc định của thư viện là in thẳng ra stderr; ta dùng logging."""
            logger.debug("%s %s", self.address_string(), format % args)

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send(status, body, "application/json; charset=utf-8")

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                # Không hút nữa thì phần còn lại sẽ bị đọc thành request sau;
                # đóng kết nối là cách duy nhất thoát sạch.
                self.close_connection = True
                raise ValueError("Request quá lớn")
            raw = self.rfile.read(length) if length else b"{}"
            data = json.loads(raw or b"{}")
            if not isinstance(data, dict):
                raise ValueError("Body phải là một object JSON")
            return data

        def _board_from(self, data: dict[str, Any]) -> chess.Board:
            fen = data.get("fen")
            if fen is None:
                return chess.Board()
            if not isinstance(fen, str):
                raise ValueError("fen phải là chuỗi")
            return chess.Board(fen)  # ném ValueError nếu FEN hỏng

        # -- routing -----------------------------------------------------

        def do_GET(self) -> None:  # noqa: N802 - tên do thư viện chuẩn quy định
            path = "index.html" if self.path in ("/", "") else self.path.lstrip("/")
            target = (STATIC_DIR / path).resolve()
            # Chặn ../: chỉ phục vụ file nằm trong STATIC_DIR.
            if not target.is_file() or STATIC_DIR.resolve() not in target.parents:
                self._send(404, b"Not found", "text/plain; charset=utf-8")
                return
            types = {".html": "text/html", ".css": "text/css", ".js": "text/javascript"}
            content_type = types.get(target.suffix, "application/octet-stream")
            self._send(200, target.read_bytes(), f"{content_type}; charset=utf-8")

        def do_POST(self) -> None:  # noqa: N802 - tên do thư viện chuẩn quy định
            routes = {
                "/api/new": self._new_game,
                "/api/move": self._player_move,
                "/api/ai": self._ai_move,
                "/api/explain": self._explain,
            }
            # Đọc hết body TRƯỚC khi phân nhánh, kể cả khi sắp trả 404. Bỏ dở
            # body trên kết nối keep-alive thì phần thừa bị đọc thành request
            # tiếp theo, và client nhận ConnectionAborted thay vì mã lỗi.
            try:
                data = self._read_json()
            except ValueError as error:
                self._send_json({"error": str(error)}, status=400)
                return

            handler = routes.get(self.path)
            if handler is None:
                self._send_json({"error": "Không có endpoint này"}, status=404)
                return
            try:
                handler(data)
            except ValueError as error:
                # Nước không hợp lệ, FEN hỏng, body sai — lỗi của người gọi.
                self._send_json({"error": str(error)}, status=400)
            except Exception as error:  # noqa: BLE001 - engine ném nhiều kiểu
                logger.exception("Lỗi khi xử lý %s", self.path)
                self._send_json({"error": f"{type(error).__name__}: {error}"}, status=500)

        # -- endpoint ----------------------------------------------------

        def _new_game(self, _data: dict[str, Any]) -> None:
            self._send_json(build_state(chess.Board()))

        def _player_move(self, data: dict[str, Any]) -> None:
            board = self._board_from(data)
            src, dst = data.get("from"), data.get("to")
            if not isinstance(src, str) or not isinstance(dst, str):
                raise ValueError("Thiếu ô nguồn hoặc ô đích")
            move = parse_move(board, src, dst, data.get("promotion"))
            san = board.san(move)  # phải lấy SAN *trước* khi đi
            board.push(move)
            self._send_json(build_state(board, san))

        def _ai_move(self, data: dict[str, Any]) -> None:
            board = self._board_from(data)
            elo = int(data.get("elo") or DEFAULT_ELO)
            if board.is_game_over(claim_draw=True):
                self._send_json(build_state(board))
                return
            move = bot.get_move(board.fen(), elo)
            san = board.san(move)
            board.push(move)
            self._send_json(build_state(board, san))

        def _explain(self, data: dict[str, Any]) -> None:
            if llm is None:
                self._send_json(
                    {
                        "error": "Chưa bật chatbot. Chạy lại server với "
                        "--llm remote --api-base <url>/v1"
                    },
                    status=503,
                )
                return

            board = self._board_from(data)
            question = str(data.get("question") or "").strip() or DEFAULT_QUESTION
            facts = analyst.facts(board)

            def generate(attempt: int) -> str:
                prompt = build_prompt(facts, question)
                if attempt:
                    prompt += RETRY_HINT
                return llm.generate(prompt, temperature=config.temperature + 0.1 * attempt)

            # guarded_generate xác thực mọi nước đi model nhắc tới, sinh lại khi
            # bắt được nước sai, và cuối cùng lùi về câu trả lời chỉ gồm fact.
            # Thà khô khan mà đúng còn hơn trôi chảy mà bịa.
            answer = guarded_generate(
                generate, board, facts, max_regenerations=config.max_regenerations
            )
            # Nói thẳng cho giao diện biết khi model bị chặn hết lượt. Im lặng
            # đưa bản chỉ-fact thì người dùng tưởng trợ lý vốn nói cụt thế.
            self._send_json(
                {
                    "question": question,
                    "answer": answer,
                    "blocked": answer == fallback_answer(facts),
                    "facts_block": facts_to_prompt(facts),
                }
            )

    return Handler


def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    *,
    allow_stockfish_fallback: bool = True,
    llm: LocalLLM | None = None,
    config: ServeConfig | None = None,
) -> None:
    """Chạy server cho tới khi Ctrl+C. Đóng engine và model trong ``finally``."""
    settings = config or ServeConfig()
    bot = _Bot(allow_stockfish_fallback=allow_stockfish_fallback)
    analyst = _Analyst()
    server = ThreadingHTTPServer((host, port), _make_handler(bot, analyst, llm, settings))
    logger.info("Mở http://%s:%d", host, port)
    if llm is None:
        logger.info("Chatbot chưa bật — xem --llm. Bàn cờ và bảng fact vẫn chạy.")
    if allow_stockfish_fallback:
        logger.info(
            "Thiếu binary Maia thì dùng Stockfish yếu thay — đi được nhưng "
            "sai lầm sẽ không giống người thật. Tắt bằng --strict-maia."
        )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Dừng server")
    finally:
        server.server_close()
        bot.close()
        analyst.close()
        if llm is not None:
            llm.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m chessvi.serve.web",
        description="Web demo: bàn cờ, đối thủ theo Elo, fact đã xác minh.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--strict-maia",
        action="store_true",
        help="Không cho Stockfish yếu thay Maia ở mức Elo thấp. Đúng hơn về "
        "độ 'giống người', nhưng máy chưa có binary Maia thì sẽ không đi được.",
    )
    parser.add_argument(
        "--llm",
        default="none",
        choices=["none", "remote", "gguf"],
        help="Backend chatbot. 'remote' gọi một API kiểu OpenAI — dùng khi model "
        "chạy ở Colab (xem notebooks/serve_colab.ipynb). 'gguf' chạy local, chỉ "
        "hợp với bản 4B vì 14B Q4 đã ~9GB.",
    )
    parser.add_argument("--api-base", help="Gốc API kèm /v1, vd https://xxx.trycloudflare.com/v1")
    parser.add_argument("--api-model", default="chessvi", help="Tên model gửi kèm request")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("CHESSVI_API_KEY"),
        help="Mặc định lấy từ CHESSVI_API_KEY. Tunnel Colab thì thường không cần.",
    )
    parser.add_argument("--gguf", type=Path, help="File GGUF cho --llm gguf")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    config = ServeConfig()
    try:
        llm = build_llm(
            args.llm,
            api_base=args.api_base,
            api_model=args.api_model,
            api_key=args.api_key,
            gguf=args.gguf,
            config=config,
        )
    except (ValueError, FileNotFoundError) as error:
        logger.error("%s", error)
        return 2
    serve(
        args.host,
        args.port,
        allow_stockfish_fallback=not args.strict_maia,
        llm=llm,
        config=config,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

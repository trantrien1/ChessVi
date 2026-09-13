"""Web demo: bàn cờ trong trình duyệt, đối thủ theo Elo, fact đã xác minh.

    python -m chessvi.serve.web --port 8000

Không thêm phụ thuộc nào — ``http.server`` của thư viện chuẩn là đủ cho một
demo chạy trên máy mình. Toàn bộ logic cờ nằm ở ``engine/``, lớp này chỉ dịch
HTTP sang lời gọi hàm, nên đổi sang FastAPI sau này là thay mỗi phần routing.

Trạng thái ván **không** giữ ở server: mỗi request mang theo FEN. Nhờ vậy
không cần session, mở nhiều tab không đá nhau, và restart server không mất ván.

Phần LLM chưa nối vào đây. Bảng bên phải hiển thị đúng khối
``[SỰ THẬT ĐÃ XÁC MINH]`` mà chatbot sẽ nhận — xem được nguyên tắc 1 đang làm
gì trước khi có model, và sau này chỉ việc thêm một ô "giải thích".
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import chess

from chessvi.engine.facts import PositionFacts, extract_facts, facts_to_prompt
from chessvi.engine.opponent import Opponent
from chessvi.logging_setup import configure_logging

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_ELO",
    "build_parser",
    "build_state",
    "main",
    "parse_move",
    "serve",
]

STATIC_DIR = Path(__file__).parent / "static"
DEFAULT_ELO = 1500
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


def _make_handler(bot: _Bot) -> type[BaseHTTPRequestHandler]:
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

    return Handler


def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    *,
    allow_stockfish_fallback: bool = True,
) -> None:
    """Chạy server cho tới khi Ctrl+C. Đóng engine trong ``finally``."""
    bot = _Bot(allow_stockfish_fallback=allow_stockfish_fallback)
    server = ThreadingHTTPServer((host, port), _make_handler(bot))
    logger.info("Mở http://%s:%d", host, port)
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
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    serve(args.host, args.port, allow_stockfish_fallback=not args.strict_maia)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

# CLAUDE.md — chess-vi

Trợ lý cờ vua tiếng Việt: một chatbot giải thích nước đi và một đối thủ AI chơi theo mức Elo.

## Nguyên tắc bất di bất dịch

**1. LLM không bao giờ tự suy luận trạng thái bàn cờ.**
Mọi sự kiện kiểm chứng được — nước đi hợp lệ, cán cân lực lượng, quân bị tấn công,
eval, tên khai cuộc — phải tính bằng `python-chess` hoặc Stockfish rồi inject vào
prompt như fact cứng. LLM chỉ diễn giải fact thành tiếng Việt.
Vi phạm nguyên tắc này là nguồn hallucination số một.

**2. Mọi nước đi đi qua parser.**
Bất kỳ chuỗi nào trông giống nước đi (UCI/SAN) xuất hiện ở input hay output đều
phải `board.parse_san()` / `chess.Move.from_uci()` + kiểm tra `in board.legal_moves`.
Không tin chuỗi thô.

**3. Ký hiệu cờ không bao giờ đi qua máy dịch.**
Trước khi dịch phải mask `e2e4`, `Nf3`, `O-O`, FEN thành placeholder. Quên bước
này thì toàn bộ label thành rác.

**4. Diff nhỏ, tập trung.**
Sửa đúng thứ được yêu cầu. Không refactor kèm, không đổi tên biến không liên quan,
không "tiện tay dọn dẹp". Một task một mối quan tâm.

## Phân chia môi trường

| Chạy ở đâu | Cái gì |
|---|---|
| Local (RTX 3050 Ti, 4GB) | engine layer, data pipeline, eval harness, serving, inference Q4 |
| Colab Pro (100 CU/tháng) | SFT, RLVR |
| Kaggle (30h/tuần free) | dịch dữ liệu, eval batch |

**Claude Code không chạy được training.** Với mọi task training, deliverable là
*script chạy được*, và acceptance criteria là **smoke test 50 mẫu trên CPU** hoặc
`--dry-run` hoàn tất không lỗi. Không bao giờ báo "đã train xong".

VRAM local chỉ 4GB thật (không phải 12GB như dxdiag hiển thị — 8GB kia là shared
memory, tràn vào đó sẽ chậm 10–50x). Mọi inference local phải là GGUF Q4 + ctx ≤ 4096.

## Stack

```
python-chess      trạng thái bàn cờ, parse, legal moves
stockfish (UCI)   oracle: eval, best move, PV
maia3             đối thủ giống người theo Elo, cũng là UCI engine
transformers/peft SFT
trl               GRPO
vllm              rollout khi RL (Colab), KHÔNG dùng local
llama-cpp-python  inference local
datasets          data pipeline
pytest            test
```

Pin version trong `pyproject.toml`. Không dùng `latest`.

## Cấu trúc repo

```
src/chessvi/
  engine/    stockfish.py  maia.py  opponent.py  facts.py
  data/      mask.py  translate.py  validate.py  puzzles.py
  train/     sft.py  grpo.py
  serve/     prompt.py  guard.py  chat.py
  eval/      puzzle_acc.py  hallucination.py
tests/
notebooks/   bản Colab của train/*
data/        gitignored
```

## Quy ước code

- Type hints bắt buộc ở mọi public function.
- FEN luôn là `str`, nước đi luôn là `chess.Move` khi ở trong code, chỉ convert
  sang `str` ở biên I/O.
- Engine dùng context manager, không để process rò rỉ.
- Config qua dataclass trong `src/chessvi/config.py`, không hardcode đường dẫn.
- Không print, dùng `logging`.

## Test

- `tests/` chạy được không cần GPU và không cần mạng.
- Fixture Stockfish: skip nếu không tìm thấy binary, đừng fail.
- Mọi hàm trong `data/` phải có test với ít nhất một case nước đi không hợp lệ.

## Bảng thuật ngữ

Ánh xạ này là chuẩn duy nhất, dùng nhất quán toàn dự án (`src/chessvi/data/glossary.py`):

| EN | VI |
|---|---|
| fork | đòn đôi |
| pin | ghim |
| skewer | xiên |
| discovered attack | đòn mở |
| back rank mate | chiếu hết hàng cuối |
| en passant | bắt tốt qua đường |
| castling | nhập thành |
| zugzwang | zugzwang (giữ nguyên) |
| blunder | sai lầm nghiêm trọng |
| hanging piece | quân bỏ ngỏ |

Bổ sung sau khi đọc tay lứa dịch máy đầu tiên (T6). Máy dịch phổ thông dịch
những từ này theo nghĩa đời thường chứ không phải nghĩa cờ vua — quan sát thật:
rook → "cổ tay", pawn → "sát thủ", checkmate → "giao phối", file → "tập tin".

| EN | VI |
|---|---|
| check | chiếu |
| checkmate | chiếu hết |
| stalemate | hoà do hết nước đi |
| passed pawn | tốt thông |
| promotion | phong cấp |
| king | vua |
| queen | hậu |
| rook | xe |
| bishop | tượng |
| knight | mã |
| pawn | tốt |
| file | cột |
| rank | hàng |
| square | ô |
| kingside | cánh vua |
| queenside | cánh hậu |

Bổ sung thì thêm vào bảng này trước, không tự chế trong prompt. Prompt dịch
được **dựng từ chính bảng này** (`build_glossary_block`), nên sửa ở đây là sửa
cả hai nơi cùng lúc.

## Không làm

- Không train lại engine chơi cờ. Stockfish ~3600 Elo đã giải xong bài toán đó.
- Không dùng vLLM local (4GB sẽ OOM).
- Không commit weights, dataset, hay file `.pgn` vào git.
- Không hardcode HF token. Dùng biến môi trường `HF_TOKEN`.

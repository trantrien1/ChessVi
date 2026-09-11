# TASKS.md — backlog cho Claude Code

Mỗi task: một phiên Claude Code, một commit, một PR. Đừng gộp.
Ký hiệu: 🖥️ local · ☁️ Colab · 🟨 Kaggle

---

## T0 — Khởi tạo repo 🖥️

**Phụ thuộc:** không · **Ước tính:** 30 phút

**Deliverable:** `pyproject.toml`, `src/chessvi/__init__.py`, `src/chessvi/config.py`, `.gitignore`, `tests/conftest.py`, CI chạy pytest.

**Nghiệm thu:** `pip install -e .` thành công; `pytest` chạy và pass 0 test không lỗi; `data/` và `*.gguf` nằm trong gitignore.

```
Khởi tạo repo Python cho dự án chess-vi theo cấu trúc trong CLAUDE.md.
Dùng pyproject.toml với hatchling, Python 3.11. Pin version cho python-chess,
transformers, peft, trl, datasets, pytest.
Tạo src/chessvi/config.py với dataclass Paths (stockfish_bin, maia_bin, data_dir,
model_dir) đọc từ biến môi trường có default hợp lý.
Tạo .gitignore chặn data/, *.gguf, *.safetensors, .env.
Thêm GitHub Actions chạy pytest trên push.
Chưa viết logic gì khác.
```

---

## T1 — Wrapper Stockfish 🖥️

**Phụ thuộc:** T0 · **Ước tính:** 1–2 giờ

**Deliverable:** `src/chessvi/engine/stockfish.py`, `tests/test_stockfish.py`

**Nghiệm thu:** từ FEN bất kỳ trả về `EngineAnalysis(best_move, cp, mate, pv)`; xử lý đúng thế chiếu hết và hòa (trả `cp=None`, không crash); engine process đóng sạch sau context manager; test skip gọn nếu thiếu binary.

```
Viết src/chessvi/engine/stockfish.py.

Class StockfishEngine dùng chess.engine.SimpleEngine.popen_uci, hỗ trợ context
manager (__enter__/__exit__).

Method analyse(board: chess.Board, depth: int = 18) -> EngineAnalysis.
EngineAnalysis là dataclass: best_move (chess.Move | None), cp (int | None),
mate (int | None), pv (list[chess.Move]).

Quan trọng: score phải quy về góc nhìn bên đang đi (POV white hay side-to-move —
chọn side-to-move và ghi rõ trong docstring). Thế mate thì cp=None, mate=số nước.
Thế đã kết thúc (checkmate/stalemate) thì best_move=None, không gọi engine.

Thêm method configure_skill(level: int) map sang option "Skill Level" của UCI.

Viết test với FEN cố định: thế khởi đầu, một thế mate-in-1, một thế stalemate.
Test phải skip (không fail) nếu không tìm thấy binary Stockfish.
```

---

## T2 — Wrapper Maia-3 + lớp đối thủ 🖥️

**Phụ thuộc:** T1 · **Ước tính:** 2 giờ

**Deliverable:** `src/chessvi/engine/maia.py`, `src/chessvi/engine/opponent.py`, test

**Nghiệm thu:** `get_move(fen, target_elo)` trả nước hợp lệ ở mọi mức Elo 800–2600; mức ≤2200 route sang Maia, >2200 sang Stockfish; nước trả về luôn nằm trong `board.legal_moves` (assert trong code, không chỉ trong test).

```
Viết src/chessvi/engine/maia.py và opponent.py.

maia.py: MaiaEngine cũng dùng popen_uci (maia3 cài qua pip cung cấp binary UCI).
Cùng interface context manager như StockfishEngine. Method get_move(board, elo).
Nếu Maia trả nước không hợp lệ vì bất kỳ lý do gì thì raise, không im lặng.

opponent.py: class Opponent với API duy nhất get_move(fen: str, target_elo: int).
Routing: elo <= 2200 dùng Maia với Elo condition tương ứng; elo > 2200 dùng
Stockfish với Skill Level và depth scale theo elo.
Lý do routing này có trong CLAUDE.md — Stockfish yếu đi bằng nước ngẫu nhiên kỳ
quặc, Maia sai giống người thật.

Assert nước trả về nằm trong legal_moves trước khi return.
Test: parametrize elo [1000, 1500, 2000, 2500], mỗi mức kiểm tra tính hợp lệ.
```

---

## T3 — Trích xuất fact để grounding 🖥️

**Phụ thuộc:** T1 · **Ước tính:** 3 giờ

**Deliverable:** `src/chessvi/engine/facts.py`, test đầy đủ

**Nghiệm thu:** trả về dataclass `PositionFacts` với ≥6 trường; mọi trường tính thuần bằng python-chess trừ eval; có test cho từng trường với FEN dựng sẵn.

Đây là task quan trọng nhất về mặt chất lượng sản phẩm. Chất lượng grounding quyết định tỷ lệ hallucination ở T10.

```
Viết src/chessvi/engine/facts.py.

Dataclass PositionFacts:
- fen, side_to_move
- legal_moves: list[str]  (SAN)
- material: dict[str, int] + material_diff (quy ước P=1 N=B=3 R=5 Q=9)
- hanging: list[tuple[square_name, piece_symbol]]  quân bị tấn công mà không
  được bảo vệ, dùng board.attackers()
- in_check: bool
- checks_available: list[str]  nước đi gây chiếu
- captures_available: list[str]
- eval_cp / eval_mate  (từ StockfishEngine, optional — cho phép None nếu không
  truyền engine vào)
- opening_name: str | None

Hàm extract_facts(board, engine=None, opening_book=None) -> PositionFacts.

Thêm hàm facts_to_prompt(facts) -> str render thành block tiếng Việt gọn để nhét
vào prompt. Dùng bảng thuật ngữ trong data/glossary.py.

Test: dựng FEN riêng cho từng trường hợp — một thế có quân bỏ ngỏ rõ ràng, một
thế đang bị chiếu, một thế có nước chiếu. Khẳng định chính xác từng trường.
```

---

## T4 — Masking ký hiệu cờ 🖥️

**Phụ thuộc:** T0 · **Ước tính:** 3 giờ

**Deliverable:** `src/chessvi/data/mask.py`, test property-based

**Nghiệm thu:** round-trip `unmask(mask(text)) == text` trên ≥500 mẫu thật từ C1-data; không bắt nhầm từ tiếng Anh thường (`Be`, `Bed`, `a4` trong ngữ cảnh khác).

Đây là task dễ làm ẩu nhất và hỏng ngầm nhất. Đầu tư test tử tế.

```
Viết src/chessvi/data/mask.py.

mask(text: str) -> tuple[str, dict[str, str]]
unmask(text: str, mapping: dict) -> str

Cần nhận diện và thay bằng placeholder:
- UCI: e2e4, a7a8q
- SAN: Nf3, Bxc6+, O-O, O-O-O, exd5, a8=Q#, Rfe1
- FEN đầy đủ (6 trường)
- Số thứ tự nước đi: "1." "23..."

Placeholder dạng <M0> <M1> <F0> <N0> — chọn dạng mà model dịch sẽ giữ nguyên.

Cạm bẫy phải xử lý: SAN trùng từ tiếng Anh. "Be" "Bed" "Ба" không phải nước đi.
Chiến lược: chỉ coi là nước đi khi match regex CHẶT và (nếu có FEN ngữ cảnh) khi
board.parse_san() thành công. Thêm tham số optional context_fen để bật xác thực này.

Test:
1. round-trip identity trên 500 dòng lấy từ C1-data
2. các case âm: câu tiếng Anh chứa "Be careful", "Bed", "a4 paper size"
3. property test với hypothesis nếu tiện
```

---

## T5 — Pipeline dịch 🟨

**Phụ thuộc:** T4 · **Ước tính:** 4 giờ code + thời gian chạy

**Deliverable:** `src/chessvi/data/translate.py`, `src/chessvi/data/glossary.py`

**Nghiệm thu:** `--dry-run` trên 20 mẫu hoàn tất, in ra before/after; glossary được ép nhất quán (kiểm tra bằng test: output không được chứa hai cách dịch khác nhau cho cùng term EN).

```
Viết src/chessvi/data/glossary.py (từ bảng trong CLAUDE.md, dạng dict + hàm
enforce_glossary(text) -> text) và src/chessvi/data/translate.py.

translate.py là CLI:
  python -m chessvi.data.translate --split sft --limit N --dry-run

Luồng: load C1-data từ HF -> mask -> gọi model dịch -> unmask ->
enforce_glossary -> ghi parquet.

Backend dịch pluggable qua tham số --backend, mặc định là một LLM chạy local/Kaggle.
Viết interface Translator abstract, một implementation cụ thể.

Batch, có retry, checkpoint từng 500 mẫu ra đĩa để ngắt giữa chừng chạy tiếp được
(--resume). Colab/Kaggle hay đứt.

--dry-run in ra 20 cặp before/after ra stdout và không ghi file.
```

---

## T6 — Validate dữ liệu đã dịch 🖥️

**Phụ thuộc:** T5 · **Ước tính:** 2 giờ

**Deliverable:** `src/chessvi/data/validate.py`, report

**Nghiệm thu:** in ra tỷ lệ loại bỏ theo từng lý do; ghi ra `data/rejected.jsonl` để soi tay; chạy được như một bước độc lập.

```
Viết src/chessvi/data/validate.py.

Với mỗi mẫu đã dịch, kiểm tra:
1. Mọi nước đi trong output parse được và hợp lệ với FEN của mẫu
2. Nước đi kết luận khớp với label gốc
3. Placeholder không còn sót (<M0> lọt ra ngoài = lỗi unmask)
4. Output không còn chunk tiếng Anh dài (heuristic đơn giản)
5. Glossary nhất quán

CLI in bảng thống kê: tổng, pass, reject theo từng lý do, tỷ lệ %.
Ghi mẫu bị loại ra data/rejected.jsonl kèm trường reason.

Kỳ vọng reject 5–15%. Nếu >25% thì in cảnh báo to rằng pipeline dịch có vấn đề.
```

**Chốt chặn thủ công:** trước khi sang T7, tự đọc tay 200 mẫu ngẫu nhiên đã pass. Không ai thay bạn làm bước này được.

---

## T7 — Chuẩn bị puzzle set 🖥️

**Phụ thuộc:** T0 · **Ước tính:** 2 giờ

**Deliverable:** `src/chessvi/data/puzzles.py`

**Nghiệm thu:** sample 50K train + 2K test cân bằng theo Theme và Rating bucket; không rò rỉ giữa hai tập; phân phối in ra được.

```
Viết src/chessvi/data/puzzles.py.

Load Lichess/chess-puzzles (streaming, dataset lớn).
Sample cân bằng: chia Rating thành bucket 400 điểm, với mỗi (theme, bucket) lấy
tối đa N mẫu. Mục tiêu 50K train / 2K test, không giao nhau.

Lưu ý: Moves trong dataset là chuỗi UCI, nước ĐẦU TIÊN là nước của đối thủ dẫn
vào thế puzzle, không phải lời giải. Phải apply nước đó vào FEN trước rồi mới
lấy phần còn lại làm lời giải. Đây là lỗi kinh điển, đọc kỹ dataset card.

Xuất parquet + in bảng phân phối theme/rating.
```

---

## T8 — Script SFT ☁️

**Phụ thuộc:** T6 · **Ước tính:** 3 giờ

**Deliverable:** `src/chessvi/train/sft.py`, `notebooks/sft_colab.ipynb`

**Nghiệm thu:** smoke test `--limit 50 --max-steps 5` chạy hết trên CPU không lỗi; checkpoint push lên HF Hub với `hub_strategy="checkpoint"`; resume từ checkpoint hoạt động.

```
Viết src/chessvi/train/sft.py — QLoRA fine-tune Qwen3-4B trên dữ liệu đã validate.

Cấu hình: 4-bit nf4, LoRA rank 32 alpha 64, target tất cả linear, seq 2048,
gradient checkpointing, bf16.

TrainingArguments phải có push_to_hub=True, hub_strategy="checkpoint",
save_strategy="steps", save_steps=200, save_total_limit=2, hub_private_repo=True.
Đọc HF_TOKEN từ env.

Thêm cờ --limit và --max-steps để smoke test được trên CPU với model tí hon
(cho phép --base-model để thay bằng Qwen3-0.6B khi test).

notebooks/sft_colab.ipynb: mount, cài đặt, gọi script, có cell resume.
Notebook chỉ gọi script, không copy-paste logic vào cell.

KHÔNG train. Chỉ cần smoke test pass.
```

---

## T9 — Script GRPO ☁️

**Phụ thuộc:** T7, T8 · **Ước tính:** 4 giờ

**Deliverable:** `src/chessvi/train/grpo.py`, `notebooks/grpo_colab.ipynb`

**Nghiệm thu:** hàm reward có unit test riêng (không cần GPU); smoke test 5 step chạy hết; theme-balanced sampler có test phân phối.

```
Viết src/chessvi/train/grpo.py dùng trl GRPOTrainer.

Reward function (tách ra module riêng để test được không cần GPU):
  +1.0  nước đi khớp lời giải puzzle
  +0.1  hợp lệ nhưng sai
  -1.0  không hợp lệ hoặc không parse được
  +0.2  bonus riêng cho đúng format output
Parse nước đi từ output bằng python-chess, không dùng regex trần.

Sampler cân bằng theme (tái dùng logic T7).
Rollout bằng vLLM.

Viết tests/test_reward.py cover cả 4 nhánh reward — đây là phần dễ sai nhất và
rẻ nhất để test.

Smoke test: --max-steps 5 với model nhỏ.
KHÔNG train.
```

---

## T10 — Prompt, guard, chat loop 🖥️

**Phụ thuộc:** T3 · **Ước tính:** 4 giờ

**Deliverable:** `src/chessvi/serve/prompt.py`, `guard.py`, `chat.py`

**Nghiệm thu:** guard bắt được 100% nước đi không hợp lệ trong output (test với output giả có nước sai); chat loop chạy local với GGUF Q4 ctx 4096.

```
Viết lớp serving.

prompt.py: build_prompt(facts, user_question, history) -> str.
Fact block từ T3 đặt ở vị trí cố định, có nhãn rõ ràng, kèm chỉ thị rằng đây là
sự thật đã xác minh và model không được mâu thuẫn với chúng.

guard.py: check_output(text, board) -> GuardResult.
Trích mọi nước đi trong text (tái dùng regex từ mask.py), xác thực từng nước với
board. Trả về danh sách nước vi phạm. Nếu có vi phạm thì caller regenerate,
tối đa 2 lần, sau đó fallback về câu trả lời chỉ gồm fact.

chat.py: vòng lặp CLI, load GGUF Q4 qua llama-cpp-python, n_ctx=4096,
cache-type-k/v = q8_0. Ghép với Opponent từ T2 để chơi và hỏi xen kẽ.
```

---

## T11 — Eval harness 🖥️🟨

**Phụ thuộc:** T7, T10 · **Ước tính:** 4 giờ

**Deliverable:** `src/chessvi/eval/puzzle_acc.py`, `hallucination.py`

**Nghiệm thu:** chạy được trên bất kỳ model endpoint nào qua một interface chung; xuất CSV tách theo Rating bucket và Theme; hallucination check hoàn toàn tự động.

```
Viết eval harness.

puzzle_acc.py: chạy model trên 2K test puzzle, tính accuracy tổng và tách theo
Rating bucket + Theme. Xuất CSV.

hallucination.py: với mỗi câu trả lời, trích các khẳng định kiểm chứng được
(nước đi hợp lệ, quân ở ô nào, ai đang hơn quân, có đang bị chiếu không) và đối
chiếu với facts thật từ T3. Trả về tỷ lệ khẳng định sai.

Cả hai nhận model qua interface chung (abstract class Predictor) để so sánh được
nhiều baseline: model của mình, C1-4B gốc, Qwen3-4B chưa fine-tune.
```

---

## T12 — Human eval + model card 🖥️

**Phụ thuộc:** T11 · **Ước tính:** 3 giờ

**Deliverable:** script export 100 mẫu ra form chấm, model card, README

**Nghiệm thu:** form chấm thang 1–5 hai trục (tự nhiên, đúng thuật ngữ); model card ghi rõ base model, dữ liệu, license, hạn chế đã biết.

```
Viết script export 100 mẫu ngẫu nhiên ra CSV/Google Form để người chấm tay,
hai trục: độ tự nhiên tiếng Việt (1-5), độ đúng thuật ngữ cờ (1-5).
Kèm script gom kết quả và tính agreement giữa các người chấm.

Viết model card cho HF Hub: base model, quy trình dữ liệu (bao gồm việc dịch máy
và tỷ lệ loại bỏ ở T6), kết quả 3 trục eval, hạn chế đã biết, license kế thừa từ
C1-data (apache-2.0) và Lichess (CC0).
```

---

## Thứ tự chạy

```
T0 ──┬── T1 ── T2
     │    └─── T3 ──────────── T10 ── T11 ── T12
     ├── T4 ── T5 ── T6 ── T8 ──┘        │
     └── T7 ─────────────┴── T9 ─────────┘
```

Song song được: nhánh engine (T1–T3) và nhánh dữ liệu (T4–T7) độc lập hoàn toàn.
Nếu làm nhóm thì chia đúng theo hai nhánh này.

## Chốt chặn

Không sang giai đoạn sau khi chưa qua:

- **Sau T6:** đọc tay 200 mẫu. Dịch ẩu thì mọi thứ sau đó vô nghĩa mà bạn không
  phát hiện ra cho tới tận T9.
- **Sau T8:** accuracy trên test set phải đạt 30–38%. Dưới 20% nghĩa là dữ liệu
  có vấn đề — quay lại T5, đừng chạy RL.
- **Sau T9:** RL phải cải thiện ≥4 điểm so với SFT. Không thì cấu hình reward sai.

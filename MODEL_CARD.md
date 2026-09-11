---
language:
  - vi
license: apache-2.0
base_model: Qwen/Qwen3-4B
library_name: peft
tags:
  - chess
  - vietnamese
  - qlora
  - grpo
---

# chessvi-4b — trợ lý cờ vua tiếng Việt

> **Trạng thái: CHƯA TRAIN.** Đây là model card đi kèm mã nguồn. Mọi ô kết quả
> ghi `TBD` phải được điền bằng số thật từ `chessvi.eval.*` trước khi đẩy model
> lên Hub. Không công bố model kèm card còn TBD.

## Tóm tắt

Trợ lý cờ vua tiếng Việt gồm hai phần:

1. **Chatbot giải thích nước đi** — trả lời bằng tiếng Việt, mọi sự kiện về bàn
   cờ đều do `python-chess` và Stockfish tính rồi inject vào prompt. Model chỉ
   diễn giải fact, không tự suy luận trạng thái bàn cờ.
2. **Đối thủ AI theo mức Elo** — Maia-3 cho Elo ≤ 2200 (sai giống người thật),
   Stockfish có giới hạn Skill Level cho Elo > 2200.

Model **không** chơi cờ bằng khả năng của chính nó. Nước đi trong chế độ đối
thủ đến từ engine; điểm mạnh của model là phần *giải thích*.

## Base model

| | |
|---|---|
| Base | `Qwen/Qwen3-4B` |
| Phương pháp | QLoRA (4-bit nf4) → GRPO (RLVR) |
| LoRA | rank 32, alpha 64, dropout 0.05, target `all-linear` |
| Seq length | 2048 |
| Định dạng phát hành | LoRA adapter + bản GGUF Q4_K_M cho inference local |

## Quy trình dữ liệu

### SFT

1. Nguồn: **C1-data** (apache-2.0), dữ liệu suy luận cờ vua tiếng Anh.
2. **Mask ký hiệu cờ** (`chessvi.data.mask`): mọi UCI/SAN/FEN/số thứ tự nước
   được thay bằng placeholder `<M0>`, `<F0>`, `<N0>` **trước khi** đưa qua máy
   dịch. Không mask thì `Nf3` biến thành "Mã f3" và toàn bộ label thành rác.
3. **Dịch máy** sang tiếng Việt (`chessvi.data.translate`). Đây là dữ liệu
   **dịch máy, không phải người Việt viết** — xem phần Hạn chế.
4. **Unmask** rồi **ép bảng thuật ngữ** (`chessvi.data.glossary`) để mỗi thuật
   ngữ tiếng Anh chỉ có đúng một cách dịch trong toàn bộ dữ liệu.
5. **Validate** (`chessvi.data.validate`): loại mẫu có nước đi không hợp lệ với
   FEN, nước kết luận lệch label gốc, còn sót placeholder, còn khúc tiếng Anh
   dài, hoặc thuật ngữ lệch chuẩn.

| Chỉ số | Giá trị |
|---|---|
| Mẫu trước validate | TBD |
| Tỷ lệ loại ở bước validate | TBD (kỳ vọng 5–15%) |
| Mẫu sau validate | TBD |
| Đã đọc tay 200 mẫu ngẫu nhiên | TBD |

### RL (GRPO)

Puzzle từ **Lichess/chess-puzzles** (CC0), sample cân bằng theo `(theme, rating
bucket 400 điểm)`: 50K train / 2K test, hai tập rời nhau.

Reward (`chessvi.train.reward`), kiểm chứng bằng `python-chess`, không dùng LLM
chấm LLM:

| Tình huống | Điểm |
|---|---|
| Nước đi khớp lời giải puzzle | +1.0 |
| Hợp lệ nhưng sai | +0.1 |
| Không hợp lệ hoặc không parse được | −1.0 |
| Bonus riêng cho đúng format output | +0.2 |

## Kết quả eval

Ba trục, mỗi trục một script trong `chessvi.eval`.

### 1. Accuracy giải puzzle (`puzzle_acc.py`, 2K test puzzle)

| Model | Accuracy | Tỷ lệ trả nước hợp lệ |
|---|---|---|
| Qwen3-4B (chưa fine-tune) | TBD | TBD |
| C1-4B (gốc, tiếng Anh) | TBD | TBD |
| chessvi-4b SFT | TBD (mục tiêu 30–38%) | TBD |
| chessvi-4b SFT + GRPO | TBD (mục tiêu ≥ SFT + 4 điểm) | TBD |

Bản CSV đầy đủ tách theo Rating bucket và Theme: `reports/puzzle_acc.csv`.

### 2. Hallucination (`hallucination.py`)

Tỷ lệ khẳng định kiểm chứng được mà **sai**: nước đi không hợp lệ, quân ở sai
ô, sai cán cân lực lượng, sai tình trạng bị chiếu, sai lượt đi.

| Model | Tỷ lệ khẳng định sai | Câu trả lời có ≥1 lỗi |
|---|---|---|
| Qwen3-4B (chưa fine-tune) | TBD | TBD |
| chessvi-4b SFT | TBD | TBD |
| chessvi-4b SFT + GRPO | TBD | TBD |

### 3. Người chấm tay (`human_eval.py`, 100 mẫu, ≥2 người chấm)

| Trục (thang 1–5) | Trung bình | Kappa (trọng số bậc 2) |
|---|---|---|
| Độ tự nhiên tiếng Việt | TBD | TBD |
| Độ đúng thuật ngữ cờ | TBD | TBD |

## Cách dùng

```bash
pip install -e ".[serve]"
export CHESSVI_STOCKFISH_BIN=/duong/dan/stockfish
python -m chessvi.serve.chat --gguf models/chessvi-q4_k_m.gguf --elo 1500
```

Chạy local phải là **GGUF Q4, `n_ctx` ≤ 4096, KV cache q8_0**: card mục tiêu chỉ
có 4GB VRAM thật, tràn sang shared memory sẽ chậm 10–50x.

## Hạn chế đã biết

- **Dữ liệu SFT là dịch máy.** Văn phong có thể còn dấu vết dịch máy dù đã ép
  bảng thuật ngữ và lọc ở bước validate. Trục "độ tự nhiên tiếng Việt" trong
  human eval tồn tại chính vì lý do này.
- **Model không biết chơi cờ.** Mọi đánh giá thế cờ đều đến từ Stockfish. Nếu
  chạy không có Stockfish, khối fact sẽ không có eval và model được dặn phải
  nói là không biết thay vì đoán.
- **Vẫn cần guard.** `chessvi.serve.guard` xác thực lại mọi nước đi trong output
  và sinh lại tối đa 2 lần, sau đó lùi về câu trả lời chỉ gồm fact. Đừng dùng
  model này mà bỏ guard.
- **Bảng thuật ngữ hữu hạn.** Thuật ngữ ngoài `chessvi.data.glossary` không
  được đảm bảo dịch nhất quán.
- **Chế độ đối thủ cần binary engine.** Elo ≤ 2200 cần binary UCI của Maia;
  không có thì lớp đối thủ báo lỗi rõ ràng chứ không im lặng hạ cấp sang
  Stockfish (trừ khi bật `--allow-stockfish-fallback`).
- **Chưa đo an toàn/thiên lệch** ngoài phạm vi cờ vua.

## License

| Thành phần | License |
|---|---|
| Mã nguồn repo này | Apache-2.0 |
| Base model Qwen3-4B | theo license của Qwen |
| Dữ liệu SFT (C1-data và bản dịch) | Apache-2.0 (kế thừa) |
| Puzzle Lichess | CC0 |
| Adapter phát hành | Apache-2.0 |

## Trích dẫn

```bibtex
@software{chessvi,
  title  = {chess-vi: Trợ lý cờ vua tiếng Việt},
  year   = {2026},
  note   = {QLoRA + GRPO trên Qwen3-4B, grounding bằng python-chess và Stockfish}
}
```

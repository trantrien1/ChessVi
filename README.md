# chess-vi

Trợ lý cờ vua tiếng Việt: một chatbot giải thích nước đi và một đối thủ AI chơi
theo mức Elo.

Nguyên tắc gốc: **LLM không bao giờ tự suy luận trạng thái bàn cờ.** Mọi sự kiện
kiểm chứng được — nước đi hợp lệ, cán cân lực lượng, quân bỏ ngỏ, eval, tên khai
cuộc — do `python-chess`/Stockfish tính rồi inject vào prompt như fact cứng. LLM
chỉ diễn giải fact thành tiếng Việt. Chi tiết trong [CLAUDE.md](CLAUDE.md),
backlog trong [TASKS.md](TASKS.md).

## Cài đặt

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows; Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"         # thêm data / train / serve / rl khi cần
pytest
```

Extras:

| Extra | Dùng khi | Ghi chú |
|---|---|---|
| `dev` | chạy test | pytest, hypothesis |
| `data` | pipeline dữ liệu | datasets, pyarrow, pandas |
| `train` | SFT/GRPO | transformers, peft, trl, bitsandbytes |
| `rl` | rollout GRPO | vLLM — **chỉ Colab**, 4GB VRAM local sẽ OOM |
| `serve` | chat local | llama-cpp-python |

## Engine

Cả hai engine là binary UCI bên ngoài, trỏ tới bằng biến môi trường (xem
[.env.example](.env.example)):

```bash
export CHESSVI_STOCKFISH_BIN=/duong/dan/stockfish
export CHESSVI_MAIA_BIN=/duong/dan/maia
```

Không có Stockfish thì test engine **tự skip**, không fail. Không có Maia thì
lớp đối thủ báo lỗi rõ ràng ở mức Elo ≤ 2200 chứ không im lặng hạ cấp.

## Quy trình

```
T0 ──┬── T1 ── T2                              engine: Stockfish, Maia, đối thủ
     │    └─── T3 ──────────── T10 ── T11 ── T12
     ├── T4 ── T5 ── T6 ── T8 ──┘        │     dữ liệu: mask, dịch, validate
     └── T7 ─────────────┴── T9 ─────────┘     puzzle + RL
```

### Dữ liệu

```bash
# chuẩn hoá C1-data về schema pipeline hiểu (tách FEN, rút label)
python -m chessvi.data.c1 --split sft --out data/raw/c1_sft.jsonl

# dịch — luôn thử --dry-run trước và đọc tận mắt 20 cặp before/after.
# backend llm mặc định Qwen/Qwen3-14B qua vLLM (Colab). Máy dịch phổ thông
# (--backend hf) dịch sai nặng từ vựng cờ, xem MODEL_CARD.md.
python -m chessvi.data.translate --input-jsonl data/raw/c1_sft.jsonl     --split sft --backend llm --limit 20 --dry-run
python -m chessvi.data.translate --input-jsonl data/raw/c1_sft.jsonl     --split sft --backend llm --resume

# kiểm tra dữ liệu đã dịch; kỳ vọng loại 5–15%, >25% là pipeline dịch có vấn đề.
# --out-clean ghi tập đã pass ra đĩa — đây là đầu vào của SFT.
python -m chessvi.data.validate --input data/translated/sft \
    --out-clean data/validated/sft

# puzzle set cân bằng theo theme và rating bucket
python -m chessvi.data.puzzles --train-size 50000 --test-size 2000
```

### Huấn luyện

Chạy trên Colab, xem [notebooks/sft_colab.ipynb](notebooks/sft_colab.ipynb) và
[notebooks/grpo_colab.ipynb](notebooks/grpo_colab.ipynb). Notebook chỉ gọi
script, không chứa logic.

```bash
# smoke test trên CPU trước khi tốn CU
python -m chessvi.train.sft --data data/validated/sft --base-model Qwen/Qwen3-0.6B \
    --limit 50 --max-steps 5 --no-push --output-dir outputs/smoke

python -m chessvi.train.grpo --puzzles data/puzzles/train.parquet \
    --base-model Qwen/Qwen3-0.6B --limit 16 --max-steps 5 --no-vllm --no-push \
    --output-dir outputs/grpo-smoke
```

### Phục vụ

```bash
python -m chessvi.serve.chat --gguf models/chessvi-q4_k_m.gguf --elo 1500
```

Mọi câu trả lời đi qua `serve/guard.py`: nước đi không hợp lệ trong output sẽ
làm model sinh lại (tối đa 2 lần), sau đó lùi về câu trả lời chỉ gồm fact.

### Đánh giá

```bash
python -m chessvi.eval.puzzle_acc --puzzles data/puzzles/test.parquet \
    --backend gguf --model-path models/chessvi-q4_k_m.gguf --out reports/acc.csv

python -m chessvi.eval.hallucination --puzzles data/puzzles/test.parquet \
    --backend gguf --model-path models/chessvi-q4_k_m.gguf --out reports/hallu.csv

python -m chessvi.eval.human_eval export --input data/answers.jsonl \
    --out reports/human_eval.csv
python -m chessvi.eval.human_eval collect \
    --ratings reports/rater_a.csv reports/rater_b.csv
```

## Chốt chặn

Không sang giai đoạn sau khi chưa qua:

- **Sau T6** — đọc tay 200 mẫu ngẫu nhiên đã pass. Không ai làm thay được.
- **Sau T8** — accuracy test set 30–38%. Dưới 20% là dữ liệu có vấn đề, quay
  lại T5, **đừng** chạy RL.
- **Sau T9** — RL phải hơn SFT ≥ 4 điểm. Không thì cấu hình reward sai.

## Cấu trúc

```
src/chessvi/
  config.py           dataclass Paths/EngineConfig/ServeConfig, đọc từ env
  engine/             stockfish.py  maia.py  opponent.py  facts.py
  data/               mask.py  glossary.py  translate.py  validate.py  puzzles.py
  train/              dataset.py  sft.py  reward.py  grpo.py
  serve/              prompt.py  guard.py  chat.py
  eval/               predictor.py  puzzle_acc.py  hallucination.py  human_eval.py
tests/                chạy được không GPU, không mạng
notebooks/            bản Colab của train/*
data/                 gitignored
```

## Giấy phép

Mã nguồn Apache-2.0. Dữ liệu và model kế thừa license nguồn — xem
[MODEL_CARD.md](MODEL_CARD.md).

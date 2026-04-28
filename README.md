
# Chess Hub — AI-Powered Chess Training Platform

[![Live Demo](https://img.shields.io/badge/Live%20Demo-HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/Fenil045/chess_hub)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CNN-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

**Chess Hub** is a full-stack chess training ecosystem that integrates custom-built AI engines, Stockfish-powered game analysis, 100,000+ tactical puzzles, and an opponent scouting tool — all in a single, responsive web application.

> 🔴 **[Try it live →](https://huggingface.co/spaces/Fenil045/chess_hub)**

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Technical Deep Dives](#technical-deep-dives)
- [Tech Stack](#tech-stack)
- [Local Setup](#local-setup)
- [Project Structure](#project-structure)
- [Security](#security)

---

## Features

### 🤖 Multi-Engine AI Opponents

Four distinct AI difficulty levels, each powered by a different engine:

| Mode | Engine | Description |
|---|---|---|
| **Easy** | PyTorch CNN | Predicts human-like moves from a trained neural network |
| **Medium** | GhostChessNet | Custom Ghost-module CNN with spatial & channel attention |
| **Hard** | NegaMax Depth 2 | Alpha-Beta search with Transposition Tables & Zobrist Hashing |
| **Expert** | NegaMax Depth 4 | Deeper search with Late Move Reduction (LMR) |

All AI levels consult an **Opening Book** before switching to engine search, giving games a natural, realistic opening phase.

The **Ghost AI** (Medium) uses a hybrid design: a GhostNet-inspired CNN selects a human-like move first, but an NNUE evaluation layer vetoes the choice if it is a blunder — balancing playability with challenge.

---

### 📊 Deep Game Analysis

- Powered by **Stockfish 16.1** running at configurable depth
- Move-by-move classification using an **Expected-Points (Win Probability) model** — the same methodology used by Chess.com's Classification V2:

| Classification | Condition |
|---|---|
| ♟ Brilliant | Best-move piece sacrifice, not in opening, not while winning, no check |
| ⭐ Great | Game-flipping move, or the only good move in a balanced position |
| ✅ Best | Top engine choice, EP loss < 2% |
| 👍 Excellent | Near-best, EP loss < 5% |
| 🟡 Good | EP loss < 5% (non-best) |
| 🟠 Inaccuracy | EP loss 5–10% |
| 🔴 Mistake | EP loss 10–20% |
| ❌ Blunder | EP loss > 20% |
| 💀 Miss | Had a winning position and let it collapse |

- Interactive evaluation bar with move-by-move replay
- Suggested best move for every move

---

### 🧩 Tactical Puzzles

- **100,000+ puzzles** served from a high-performance **DuckDB** database
- Adaptive **Elo rating system** (Elo K=32, calibrated for puzzle difficulty)
- Bonus rating for solving quickly (under 15 seconds)
- Hint system with rating penalty scaling
- **Theme-based filtering**: Mates, Forks, Pins, Skewers, Discovered Attacks, and more
- Streak tracker to encourage consistent solving

---

### 🔍 Opponent Scout

- Fetches game history from **Lichess** and **Chess.com** simultaneously
- Aggregates opening statistics by color and opening name
- Runs a **Random Forest Classifier** to label each opening as `Exploit` or `Avoid`
- Ranking uses a **Wilson Lower Bound** score that accounts for sample size and opponent strength
- Output: top 2 openings to target and top 2 openings to avoid, per color

---

### 📈 Analytics Dashboard (Admin)

- Tracks page views, game starts/ends, puzzle attempts, and scout searches
- Real-time SQLite backend with a **background sync thread** that checkpoints data to Hugging Face Datasets — ensuring zero data loss on ephemeral Spaces storage

---

## Architecture

```
chess_hub/
├── app.py                  # Flask API gateway — all routes, game state, analysis
├── analytics.py            # Thread-safe SQLite analytics + HF Dataset sync
├── opening_book.py         # ECO opening book lookup
│
├── Chess_AI/
│   ├── ChessEngine.py      # Custom board representation, move generation, FEN
│   ├── ChessAI.py          # NegaMax, Alpha-Beta, Transposition Table, LMR
│   ├── predict.py          # ChessPredictor — PyTorch CNN (Easy mode)
│   ├── predict_ghost.py    # GhostPredictor — GhostChessNet (Medium mode)
│   └── RF_model/           # Random Forest for Scout feature
│   └── model/              # CNN model move mapping file
│
├── static/
│   ├── js/game.js          # Frontend game logic, board rendering, move log
│   └── css/                # 3-file CSS split: base / desktop / mobile
│
└── templates/              # Jinja2 HTML templates for each page
```

**Request lifecycle:** Browser → Flask route → `ChessEngine` (move validation) → `ChessAI` / `Stockfish` (AI or analysis) → JSON response → `game.js` (board update).

**State management:** Each session is a UUID-keyed dictionary in server memory with a 3-hour TTL. Position hashes (Zobrist-style) are tracked per game for threefold repetition detection.

---

## Technical Deep Dives

### Custom Chess Engine (`ChessEngine.py`)

Built from scratch in Python with a 2D array board representation. Handles:
- All standard move types: castling, en passant, promotion
- Check, checkmate, and stalemate detection
- FEN generation and parsing
- UCI notation conversion

### NegaMax with Alpha-Beta Pruning

The search algorithm uses:
- **Transposition Table** with Zobrist Hashing — reduced search nodes by ~60% in middlegame testing
- **Late Move Reduction (LMR)** — prunes less promising branches at deeper nodes
- **Quiescence search** — prevents the horizon effect on captures
- Position history passed in to avoid moves that would cause threefold repetition

### GhostChessNet Architecture

```
Input: (batch, 13, 8, 8)
  → Stem Conv (13→64)
  → GhostModule (64→128): primary conv + cheap depthwise conv
  → GhostModule (128→256)
  → ChessAttention: spatial gate (7×7 conv) × channel gate (FC squeeze-excite)
  → AdaptiveAvgPool (4×4) → flatten (4096)
  → Concat phase features (4): material density, pawn density, mobility, major piece ratio
  → Decision Head: 4100 → 512 → 256 → num_classes
Output: (batch, num_classes) logits
```

The 4 phase features are computed analytically from the board tensor before the CNN forward pass — giving the model explicit game-phase context without requiring it to learn this from data alone.


### Expected-Points Move Classification

Each move is evaluated before and after with Stockfish. The centipawn score is converted to a win probability using the logistic curve:

```
WP = 1 / (1 + e^(-0.00368208 × cp))
```

This constant (0.00368208) calibrates the sigmoid so that +100 cp ≈ 73% win probability, matching grandmaster-level baselines — the same model used by Chess.com's classification system.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3.9+, Flask 3.0, Gunicorn |
| **AI / ML** | PyTorch (CNN), Scikit-learn (Random Forest), NNUE |
| **Chess Engine** | Custom Python engine + python-chess + Stockfish 16.1 |
| **Databases** | SQLite (analytics), DuckDB (puzzles) |
| **Frontend** | Vanilla JS, CSS3, HTML5, chessboard.js |
| **Deployment** | Docker, Hugging Face Spaces |
| **Data Sync** | Hugging Face Datasets (HF Hub API) |

---

## Local Setup

```bash
# 1. Clone the repository
# 1. Clone the repository
git clone https://huggingface.co/spaces/Fenil045/chess_hub
cd chess_hub

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup Models & Data
# Large assets are hosted externally. Download them to the correct paths:
mkdir -p Chess_AI/model Chess_AI/RF_model Chess_AI/src
wget -O Chess_AI/model/CHESS_MODEL.pth https://huggingface.co/datasets/Fenil045/chess_models/resolve/main/CHESS_MODEL.pth
wget -O Chess_AI/model/ghost_chess_best.pth https://huggingface.co/datasets/Fenil045/chess_models/resolve/main/ghost_chess_best.pth
wget -O Chess_AI/model/move_to_int https://huggingface.co/datasets/Fenil045/chess_models/resolve/main/move_to_int
wget -O Chess_AI/RF_model/opening_rf.pkl https://huggingface.co/datasets/Fenil045/chess_models/resolve/main/opening_rf.pkl
wget -O Chess_AI/src/puzzles.duckdb https://huggingface.co/datasets/Fenil045/chess_models/resolve/main/puzzles.duckdb

# 4. Run the development server
python app.py
```

### Environment Configuration

The application uses environment variables for security and data persistence. Create a `.env` file in the root directory to configure them:

| Variable | Description |
|---|---|
| `ADMIN_PASSWORD` | The password required to access the `/admin` dashboard. |
| `SECRET_KEY` | A unique string used to encrypt Flask sessions (Puzzle ELO/Game IDs). |
| `HF_TOKEN` | (Optional) HF Write Token to sync analytics to the cloud. |
| `DATASET_ID` | (Optional) The HF Dataset repo ID for cloud storage. |

> [!TIP]
> When deploying to Hugging Face Spaces, set these as **Secrets** in the Settings tab.

Open `http://127.0.0.1:5000` in your browser.

**Docker:**
```bash
   docker build -t chess_hub .
   docker run -p 7860:7860 chess_hub
```

---

## Project Structure

```
chess_hub/
├── app.py
├── analytics.py
├── opening_book.py
├── Dockerfile
├── requirements.txt
├── Chess_AI/
│   ├── ChessEngine.py
│   ├── ChessAI.py
│   ├── predict.py
│   ├── predict_ghost.py
│   ├── RF_model/
│   │   ├── opening_rf.pkl
│   │   ├── opening_rf_meta.pkl
│   │   └── predict.py
│   ├── model/
│   │   ├── CHESS_MODEL.pth
│   │   ├── ghost_chess_best.pth
│   │   └── move_to_int / move_to_int.json
├── static/
│   ├── js/game.js
│   └── css/
│       ├── style.base.css
│       ├── style.desktop.css
│       └── style.mobile.css
└── templates/
    ├── index.html
    ├── game.html
    ├── analysis.html
    ├── puzzle.html
    ├── history.html
    ├── scout.html
    ├── admin.html
    └── admin_login.html
```

---

## Security

- Admin routes are protected by session authentication with `hmac.compare_digest` for timing-safe password comparison
- Login is rate-limited to 5 attempts per IP with a 10-minute lockout
- Session cookies are `SameSite=None; Secure`
- `SECRET_KEY` and `ADMIN_PASSWORD` are loaded from environment variables — never hardcoded
- No sensitive credentials are stored in the repository

---

## License

This project is open-source and available under the [MIT License](LICENSE).

---

*Built by [Fenil](https://huggingface.co/spaces/Fenil045/chess_hub)*
# NOW import the rest of the app
from analytics import analytics
import os
import uuid
import random
import datetime
import chess
import chess.pgn
import chess.engine
from flask import Flask, request, jsonify, render_template, session, send_from_directory, Response
from werkzeug.exceptions import HTTPException
import joblib
import pandas as pd
import math
import duckdb
import threading
import logging
import hmac
import time as _time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from Chess_AI.RF_model.predict import fetch_all_user_games, clean_opening_name
from Chess_AI import ChessEngine
from Chess_AI import ChessAI
from opening_book import OpeningBook

# Use python-dotenv to load env vars locally
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

PUZZLE_DB_PATH = os.path.join(os.path.dirname(__file__), "Chess_AI", "src", "puzzles.duckdb")
_puzzle_local = threading.local()

app = Flask(__name__)

# Use a static fallback key so sessions survive server restarts!
app.secret_key = os.environ.get("SECRET_KEY", "default_dev_secret")
# Inactivity timeout (sliding window): 30 minutes
app.permanent_session_lifetime = datetime.timedelta(minutes=30)
app.config['SESSION_COOKIE_SAMESITE'] = 'None'
app.config['SESSION_COOKIE_SECURE'] = True

# ── Security Check ──────────────────────────────────────────────────────────
if app.secret_key == "default_dev_secret":
    logging.warning("[Security] WARNING: Using default development SECRET_KEY. Not suitable for production.")
else:
    logging.info("[Security] Production SECRET_KEY detected.")


@app.before_request
def log_request_info():
    """Log main page views only, using real IP address and a cooldown to prevent refresh spam."""
    # Only track GET requests to the main UI pages
    if request.method == "GET":
        main_pages = ("/", "/game", "/analysis", "/history", "/puzzles", "/scout")
        if request.path in main_pages:
            # 1. Get real client IP, even through proxies
            # Hugging Face and other proxies use X-Forwarded-For
            forwarded = request.headers.get('X-Forwarded-For')
            if forwarded:
                ip = forwarded.split(',')[0].strip()
            else:
                ip = request.remote_addr or "unknown"

            # 2. Prevent double-logging (e.g. from rapid refreshes)
            # We use a 15-second cooldown per page per session
            now = _time.time()
            log_cache = session.get('_log_cache', {})

            # Clean up old entries from cache to keep it small
            log_cache = {p: t for p, t in log_cache.items() if now - t < 60}

            if now - log_cache.get(request.path, 0) > 30:
                # This is a new, unique visit for this timeframe
                analytics.log_page_view(request.path, ip)
                log_cache[request.path] = now
                session['_log_cache'] = log_cache
                session.permanent = True


ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")
_login_attempts = {}  # ip -> (count, first_attempt_ts) – simple rate limiter
_MAX_ATTEMPTS = 5
_LOCKOUT_SECONDS = 600  # 10 minutes


def _require_admin(f):
    """Decorator: redirect to /admin/login unless session is authenticated."""
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("admin_authenticated"):
            return redirect("/admin/login")
        return f(*args, **kwargs)

    return decorated


def _check_rate_limit(ip: str) -> bool:
    """Returns True if the IP is currently locked out."""
    import time as _t
    now = _t.time()
    entry = _login_attempts.get(ip)
    if entry:
        count, first_ts = entry
        if count >= _MAX_ATTEMPTS and (now - first_ts) < _LOCKOUT_SECONDS:
            return True  # locked out
        if (now - first_ts) >= _LOCKOUT_SECONDS:
            del _login_attempts[ip]  # window expired, reset
    return False


def _record_failed_attempt(ip: str):
    import time as _t
    entry = _login_attempts.get(ip)
    now = _t.time()
    if entry:
        _login_attempts[ip] = (entry[0] + 1, entry[1])
    else:
        _login_attempts[ip] = (1, now)


# ── Admin page routes ─────────────────────────────────────────────────────────
from flask import redirect


@app.route("/admin")
@_require_admin
def admin_dashboard():
    return render_template("admin.html")


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    error = None
    if request.method == "POST":
        ip = request.remote_addr or "unknown"
        if _check_rate_limit(ip):
            error = "Too many failed attempts. Try again in 10 minutes."
        else:
            pw = request.form.get("password", "")
            if not ADMIN_PASSWORD:
                error = "ADMIN_PASSWORD env var is not set on the server."
            elif hmac.compare_digest(pw.encode(), ADMIN_PASSWORD.encode()):
                session["admin_authenticated"] = True
                session.permanent = True
                return redirect("/admin")
            else:
                _record_failed_attempt(ip)
                remaining = _MAX_ATTEMPTS - _login_attempts.get(ip, (0,))[0]
                error = f"Wrong password. {max(0, remaining)} attempt(s) remaining."
    return render_template("admin_login.html", error=error)


@app.route("/admin/logout")
def admin_logout():
    session.pop("admin_authenticated", None)
    return redirect("/admin/login")


# ── Admin API routes ──────────────────────────────────────────────────────────
@app.route("/admin/api/stats")
@_require_admin
def admin_api_stats():
    days = request.args.get("days", 30, type=int)
    days = max(1, min(days, 36500))
    return jsonify(analytics.get_dashboard_stats(days=days))


@app.route("/admin/api/events")
@_require_admin
def admin_api_events():
    return jsonify(analytics.get_recent_events())


@app.route("/admin/api/sync", methods=["POST"])
@_require_admin
def admin_api_sync():
    """Manual sync trigger for debugging."""
    try:
        analytics.sync_to_hub()
        return jsonify({"status": "ok", "msg": "Sync triggered successfully"})
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)}), 500


@app.route("/admin/api/diagnostics")
@_require_admin
def admin_api_diagnostics():
    return jsonify(analytics.get_diagnostics())


@app.route("/admin/api/retention")
@_require_admin
def admin_api_retention():
    days = request.args.get("days", 30, type=int)
    return jsonify(analytics.get_retention(days=days))


@app.route("/logs")
@_require_admin
def view_logs():
    log_file = analytics.get_log_path()
    try:
        if not os.path.exists(log_file):
            return f"Log file not found at: {log_file}", 404
        with open(log_file, 'r') as f:
            lines = f.readlines()
            return "<pre>" + "".join(lines[-500:]) + "</pre>"
    except Exception as e:
        return f"Could not read logs: {e}", 500


# ── In-memory game store with TTL cleanup ─────────────────────────────
games = {}
GAME_TTL_SECONDS = 3 * 60 * 60  # 3 hours – auto-expire idle games


def _prune_old_games():
    """Remove games older than GAME_TTL_SECONDS to prevent memory leaks."""
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=GAME_TTL_SECONDS)
    stale = [gid for gid, g in games.items()
             if g.get("created_at", datetime.datetime.now(datetime.timezone.utc)) < cutoff]
    for gid in stale:
        del games[gid]


# ── Session-level game history (no login required) ────────────────────
# Stored in server-side `games` dict under a "history" key per session.
# Each entry: {w_name, b_name, result, pgn, date, mode}

import shutil

if os.name == 'nt':
    STOCKFISH_PATH = os.path.join(os.path.dirname(__file__), "stockfish", "stockfish.exe")
else:
    # On Linux: prefer local binary, fallback to system-installed stockfish
    _local_sf = os.path.join(os.path.dirname(__file__), "stockfish", "stockfish")
    _system_sf = shutil.which("stockfish")
    STOCKFISH_PATH = _local_sf if os.path.exists(_local_sf) else (_system_sf or _local_sf)


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def get_game():
    gid = session.get("game_id")
    return games.get(gid)


def king_square_uci(gs):
    """Return UCI square of the side-to-move's king if in check, else None."""
    if not gs.inCheck:
        return None
    pos = gs.whiteKingLocation if gs.whiteToMove else gs.blackKingLocation
    return ChessEngine.Move.colsToFiles[pos[1]] + ChessEngine.Move.rowsToRanks[pos[0]]


def check_draw(game):
    """
    Checks all draw conditions after a move has been made.
    Returns (is_draw, draw_reason_string) or (False, None).
    """
    gs = game["gs"]

    # 1. Stalemate (already detected by engine)
    if gs.stalemate:
        return True, "Draw by stalemate"

    # 2. Fifty-move rule
    if gs.half_move_count >= 100:
        return True, "Draw by 50-move rule"

    # 3. Threefold repetition
    current_hash = " ".join(gs.get_fen().split()[:4])
    pos_hist = game.get("position_history", {})
    if pos_hist.get(current_hash, 0) >= 3:
        return True, "Draw by threefold repetition"

    # 4. Insufficient material
    if _is_insufficient_material(gs.board):
        return True, "Draw by insufficient material"

    return False, None


def _is_insufficient_material(board):
    pieces = {"wB": [], "wN": [], "bB": [], "bN": [],
              "wR": 0, "wQ": 0, "wP": 0, "bR": 0, "bQ": 0, "bP": 0}

    for r in range(8):
        for c in range(8):
            p = board[r][c]
            if p == "--" or p in ("wK", "bK"):
                continue
            if p in ("wB", "bB"):
                pieces[p].append((r + c) % 2)
            elif p in ("wN", "bN"):
                pieces[p].append(1)
            elif p in pieces:
                pieces[p] += 1

    if any(pieces[k] > 0 for k in ("wP", "bP", "wR", "bR", "wQ", "bQ")):
        return False

    w_minors = len(pieces["wB"]) + len(pieces["wN"])
    b_minors = len(pieces["bB"]) + len(pieces["bN"])

    if w_minors == 0 and b_minors == 0:
        return True

    if (w_minors == 1 and b_minors == 0) or (b_minors == 1 and w_minors == 0):
        return True

    if (len(pieces["wB"]) == 1 and len(pieces["wN"]) == 0 and
            len(pieces["bB"]) == 1 and len(pieces["bN"]) == 0):
        if pieces["wB"][0] == pieces["bB"][0]:
            return True

    return False


def get_ai_move_obj(game):
    gs = game["gs"]
    history = game["move_history"]
    engine = game.get("engine", "model")

    continuations = game["book"].lookup(history)
    if continuations:
        moves, weights = zip(*continuations)
        chosen_uci = random.choices(moves, weights=weights, k=1)[0]
        move_obj = gs.uci_to_move(chosen_uci)
        if move_obj:
            return move_obj, "book"

    valid_moves = gs.getValidMoves()
    if not valid_moves:
        return None, "engine"

    pos_hist = game.get("position_history", {})

    if engine == "negamax_2":
        move_obj = ChessAI.findBestMoveAtDepth(gs, valid_moves, 2, position_history=pos_hist)
    elif engine == "negamax_5":
        move_obj = ChessAI.findBestMoveAtDepth(gs, valid_moves, 4, position_history=pos_hist)
    elif engine == "ghost_model":
        move_obj = ChessAI.findModelMoveGhost(gs, valid_moves)
    else:
        move_obj = ChessAI.findModelMovePytorch(gs, valid_moves)

    if not move_obj:
        move_obj = ChessAI.findRandomMove(valid_moves)

    return move_obj, "engine"


def build_pgn(game, result="*"):
    gs = game["gs"]
    w_name = game.get("w_name", "White")
    b_name = game.get("b_name", "Black")
    now = datetime.datetime.now()

    headers = [
        '[Event "Ghost AI Game"]',
        '[Site "Local"]',
        f'[Date "{now.strftime("%Y.%m.%d")}"]',
        f'[Time "{now.strftime("%H:%M")}"]',
        f'[White "{w_name}"]',
        f'[Black "{b_name}"]',
        f'[Result "{result}"]',
    ]

    move_parts = []
    for i, move in enumerate(gs.moveLog):
        if i % 2 == 0:
            move_parts.append(f"{i // 2 + 1}.")
        move_parts.append(str(move))

    body = " ".join(move_parts)
    return "\n".join(headers) + "\n\n" + body + f" {result}\n"


def _save_game_to_history(game):
    result = game.get("result", "*")
    pgn_str = build_pgn(game, result)
    entry = {
        "w_name": game.get("w_name", "White"),
        "b_name": game.get("b_name", "Black"),
        "result": result,
        "mode": game.get("mode", "ai"),
        "moves": len(game["gs"].moveLog),
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "pgn": pgn_str,
    }
    analytics.log_event("game_end", {
        "result": result,
        "mode": game.get("mode", "ai"),
        "end_reason": game.get("end_reason", "unknown"),
        "duration_sec": (datetime.datetime.now(datetime.timezone.utc) - game.get("created_at", datetime.datetime.now(
            datetime.timezone.utc))).total_seconds(),
        "total_moves": len(game["gs"].moveLog)
    }, session.get("session_id"))

    return entry


def _cp_to_wp(cp: float) -> float:
    """Convert centipawns → win probability [0, 1] (chess.com Expected-Points model).

    Uses the same logistic curve chess.com's Classification V2 uses.  The
    constant 0.00368208 comes from calibrating the sigmoid so that +100 cp ≈
    73 % win probability, matching grandmaster-strength engine baselines.
    """
    cp = max(-3000.0, min(3000.0, float(cp)))
    return 1.0 / (1.0 + math.exp(-0.00368208 * cp))


def _wilson_lower(positive, total, z=1.96):
    if total == 0: return 0.0
    p = positive / total
    num = p + z * z / (2 * total) - z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
    den = 1 + z * z / total
    return max(0.0, num / den)


def _exploit_score(row, global_avg_opp: float, global_max_games: int) -> float:
    n = int(row['totalgames'])
    losses = int(row['loss'])
    avg_opp = float(row['avg_opp'])
    wilson_loss = _wilson_lower(losses, n)
    opp_factor = avg_opp / max(global_avg_opp, 1)
    sample_factor = math.log1p(n) / math.log1p(max(global_max_games, 1))
    return round(wilson_loss * (0.55 + 0.25 * opp_factor + 0.20 * sample_factor), 6)


def _avoid_score(row, global_avg_opp: float, global_max_games: int) -> float:
    n = int(row['totalgames'])
    wins = int(row['won'])
    avg_opp = float(row['avg_opp'])
    wilson_win = _wilson_lower(wins, n)
    opp_factor = avg_opp / max(global_avg_opp, 1)
    sample_factor = math.log1p(n) / math.log1p(max(global_max_games, 1))
    return round(wilson_win * (0.55 + 0.25 * opp_factor + 0.20 * sample_factor), 6)


# ─────────────────────────────────────────────
#  Page routes
# ─────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/game")
def game_page():
    return render_template("game.html")


@app.route("/analysis")
def analysis_page():
    return render_template("analysis.html")


@app.route("/history")
def history_page():
    return render_template("history.html")


@app.route("/history_data", methods=["GET"])
def history_data():
    history = session.get("game_history", [])
    return jsonify({"history": history})


@app.route("/Chess_AI/images/<path:filename>")
def serve_image(filename):
    images_dir = os.path.join(os.path.dirname(__file__), "Chess_AI", "images")
    return send_from_directory(images_dir, filename)


# ─────────────────────────────────────────────
#  Game logic routes
# ─────────────────────────────────────────────

@app.route("/new_game", methods=["POST"])
def new_game():
    data = request.json or {}
    gid = str(uuid.uuid4())
    session["game_id"] = gid
    gs = ChessEngine.GameState()
    mode = data.get("mode", "ai")
    color = data.get("color", "white")
    engine = data.get("engine", "model")

    engine_names = {
        "model": "Ghost AI  · Easy",
        "ghost_model": "Ghost AI  · Medium",
        "negamax_2": "Ghost AI  · Depth 2",
        "negamax_5": "Ghost AI  · Depth 4",
    }
    ai_display_name = engine_names.get(engine, "Ghost AI")

    player_white = (color == "white")
    player_black = (color == "black")
    w_name = "Human" if color == "white" else ai_display_name
    b_name = "Human" if color == "black" else ai_display_name

    _prune_old_games()

    games[gid] = {
        "gs": gs,
        "book": OpeningBook(data.get("opening", "Sicilian Najdorf")),
        "mode": mode,
        "engine": engine,
        "player_white": player_white,
        "player_black": player_black,
        "human_color": color,
        "w_name": w_name,
        "b_name": b_name,
        "move_history": [],
        "hash_history": [" ".join(gs.get_fen().split()[:4])],
        "result": "*",
        "position_history": {" ".join(gs.get_fen().split()[:4]): 1},
        "created_at": datetime.datetime.now(datetime.timezone.utc),
    }

    analytics.log_event("game_start", {
        "engine": engine,
        "mode": mode,
        "opening": data.get("opening", "Sicilian Najdorf")
    }, session.get("session_id"))

    response = {
        "fen": gs.get_fen(),
        "w_name": w_name,
        "b_name": b_name,
        "ai_san": None,
    }

    if mode == "ai" and not player_white:
        move_obj, _ = get_ai_move_obj(games[gid])
        if move_obj:
            move_obj.validMoves = gs.getValidMoves()
            ai_san = str(move_obj)
            games[gid]["move_history"].append(move_obj.get_uci_notation())
            gs.makeMove(move_obj)
            
            # Update position history for AI move
            new_hash = " ".join(gs.get_fen().split()[:4])
            games[gid]["hash_history"].append(new_hash)
            games[gid]["position_history"][new_hash] = games[gid]["position_history"].get(new_hash, 0) + 1
            
            response["ai_san"] = ai_san
            response["fen"] = gs.get_fen()

    return jsonify(response)


@app.route("/get_valid_moves", methods=["POST"])
def get_valid_moves():
    game = get_game()
    if not game:
        return jsonify({"moves": []})

    gs = game["gs"]
    is_human_turn = ((gs.whiteToMove and game["player_white"]) or (not gs.whiteToMove and game["player_black"]))
    if not is_human_turn:
        return jsonify({"moves": []})

    current_fen = gs.get_fen()
    if game.get("cached_fen") != current_fen:
        game["valid_moves"] = gs.getValidMoves()
        game["cached_fen"] = current_fen

    square = request.json.get("square", "")
    moves_uci = [
        m.get_uci_notation()
        for m in game["valid_moves"]
        if m.getRankFile(m.startRow, m.startCol) == square
    ]
    return jsonify({"moves": moves_uci})


@app.route("/human_move", methods=["POST"])
def human_move():
    game = get_game()
    if not game:
        return jsonify({"error": "No active game"}), 400

    gs = game["gs"]
    uci = request.json.get("move", "")

    is_human_turn = ((gs.whiteToMove and game["player_white"]) or (not gs.whiteToMove and game["player_black"]))
    if not is_human_turn:
        return jsonify({"error": "Not your turn", "fen": gs.get_fen()}), 400

    move_obj = gs.uci_to_move(uci)
    if not move_obj:
        return jsonify({"error": f"Illegal move: {uci}", "fen": gs.get_fen()}), 400

    move_obj.validMoves = gs.getValidMoves()
    human_san = str(move_obj)
    gs.makeMove(move_obj)
    game["move_history"].append(uci)
    
    # Update position history
    new_hash = " ".join(gs.get_fen().split()[:4])
    game["hash_history"].append(new_hash)
    game["position_history"][new_hash] = game["position_history"].get(new_hash, 0) + 1

    status = "ongoing"
    msg = None
    summary = None
    if gs.checkmate:
        status = "over"
        msg = "Checkmate! You win!"
        game["result"] = "1-0" if not gs.whiteToMove else "0-1"
        game["end_reason"] = "checkmate"
        summary = _save_game_to_history(game)
    else:
        is_draw, draw_reason = check_draw(game)
        if is_draw:
            status = "over"
            msg = draw_reason
            game["result"] = "1/2-1/2"
            summary = _save_game_to_history(game)

    return jsonify({
        "human_san": human_san,
        "fen_after_human": gs.get_fen(),
        "in_check_after_human": gs.inCheck,
        "king_square_after_human": king_square_uci(gs),
        "status_after_human": status,
        "msg": msg,
        "is_draw": (status == "over" and game["result"] == "1/2-1/2" and not gs.checkmate),
        "game_summary": summary
    })


@app.route("/ai_move", methods=["POST"])
def ai_move():
    game = get_game()
    if not game:
        return jsonify({"error": "No active game"}), 400

    gs = game["gs"]

    is_ai_turn = ((gs.whiteToMove and not game["player_white"]) or (not gs.whiteToMove and not game["player_black"]))
    if not is_ai_turn:
        return jsonify({"ai_san": None, "fen": gs.get_fen(), "status": "ongoing"})

    move_obj, phase = get_ai_move_obj(game)

    if not move_obj:
        return jsonify({
            "ai_san": None,
            "fen": gs.get_fen(),
            "status": "over",
            "msg": "Stalemate — it's a draw.",
        })

    move_obj.validMoves = gs.getValidMoves()
    ai_san = str(move_obj)
    ai_uci = move_obj.get_uci_notation()
    gs.makeMove(move_obj)
    game["move_history"].append(ai_uci)
    
    # Update position history
    new_hash = " ".join(gs.get_fen().split()[:4])
    game["hash_history"].append(new_hash)
    game["position_history"][new_hash] = game["position_history"].get(new_hash, 0) + 1

    status = "ongoing"
    msg = None
    summary = None
    if gs.checkmate:
        status = "over"
        msg = "Checkmate — Ghost wins."
        game["result"] = "1-0" if not gs.whiteToMove else "0-1"
        game["end_reason"] = "checkmate"
        summary = _save_game_to_history(game)
    else:
        is_draw, draw_reason = check_draw(game)
        if is_draw:
            status = "over"
            msg = draw_reason
            game["result"] = "1/2-1/2"
            game["end_reason"] = "draw"
            summary = _save_game_to_history(game)

    return jsonify({
        "ai_san": ai_san,
        "ai_uci": ai_uci,
        "phase": phase,
        "fen": gs.get_fen(),
        "in_check": gs.inCheck,
        "king_square": king_square_uci(gs),
        "status": status,
        "msg": msg,
        "is_draw": (status == "over" and game["result"] == "1/2-1/2" and not gs.checkmate),
        "game_summary": summary
    })


@app.route("/undo", methods=["POST"])
def undo():
    game = get_game()
    if not game:
        return jsonify({"error": "no game"}), 400
    gs = game["gs"]
    steps = 2 if game["mode"] == "ai" else 1
    for _ in range(steps):
        if gs.moveLog:
            # Get the hash of the position we are leaving
            current_hash = " ".join(gs.get_fen().split()[:4])
            
            gs.undoMove()
            if game["move_history"]:
                game["move_history"].pop()
            
            # Update position history: decrement the count for the state we just left
            if "position_history" in game and current_hash in game["position_history"]:
                game["position_history"][current_hash] -= 1
                if game["position_history"][current_hash] <= 0:
                    del game["position_history"][current_hash]
            
            if "hash_history" in game and game["hash_history"]:
                game["hash_history"].pop()
                
    return jsonify({"fen": gs.get_fen(), "status": "ok"})


@app.route("/resign", methods=["POST"])
def resign():
    game = get_game()
    if not game:
        return jsonify({"error": "No game session"}), 400

    gs = game["gs"]
    human_color = game.get("human_color", "white")
    if human_color == "white":
        msg = game["b_name"] + " wins — you resigned."
        winner = game["b_name"]
        game["result"] = "0-1"
    else:
        msg = game["w_name"] + " wins — you resigned."
        winner = game["w_name"]
        game["result"] = "1-0"

    game["end_reason"] = "resignation"

    summary = _save_game_to_history(game)
    return jsonify({"status": "over", "winner": winner, "msg": msg, "game_summary": summary})


@app.route("/download_pgn", methods=["GET"])
def download_pgn():
    game = get_game()
    if not game:
        return jsonify({"error": "No active game"}), 400

    result = game.get("result", "*")
    pgn_str = build_pgn(game, result)
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"ghost_ai_{now}.pgn"

    return Response(
        pgn_str,
        mimetype="application/x-chess-pgn",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


# ─────────────────────────────────────────────
#  Analysis routes
# ─────────────────────────────────────────────
analysis_progress = {}


@app.route("/analyse_progress", methods=["GET"])
def analyse_progress_route():
    sid = session.get("game_id", "")
    prog = analysis_progress.get(sid, {"done": 0, "total": 0, "status": "idle"})
    return jsonify(prog)


@app.route("/analyse", methods=["POST"])
def analyse():
    data = request.json or {}
    pgn_text = data.get("pgn", "").strip()
    depth = int(data.get("depth", 14))
    sid = session.get("game_id", "")

    if not pgn_text:
        return jsonify({"error": "No PGN provided"}), 400

    thread = threading.Thread(target=_run_analysis, args=(sid, pgn_text, depth))
    thread.daemon = True
    thread.start()

    analytics.log_event("analysis_start", {"depth": depth}, session.get("session_id"))

    return jsonify({"status": "started"})


def _run_analysis(sid, pgn_text, depth):
    piece_values = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330, chess.ROOK: 500, chess.QUEEN: 900,
                    chess.KING: 0}
    analysis_progress[sid] = {"done": 0, "total": 0, "status": "analysing"}

    try:
        import io
        pgn_io = io.StringIO(pgn_text)
        game_obj = chess.pgn.read_game(pgn_io)

        if not game_obj:
            analysis_progress[sid] = {"status": "error", "error": "Could not parse PGN"}
            return

        total_moves = sum(1 for _ in game_obj.mainline())
        analysis_progress[sid] = {"done": 0, "total": total_moves, "status": "analysing"}

        results = []
        if not os.path.exists(STOCKFISH_PATH):
            analysis_progress[sid] = {"status": "error", "error": f"Stockfish not found at {STOCKFISH_PATH}"}
            return

        def get_cp(info, is_white_turn_now):
            # Normalizes perspective purely mathematically based on whose turn it was
            score = info["score"].white()
            if score.is_mate():
                m = score.mate()
                if m == 0:
                    # If mate is exactly 0, the person whose turn it IS was checkmated.
                    return -10000 if is_white_turn_now else 10000
                return 10000 if m > 0 else -10000
            return score.score()

        with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
            engine.configure({"Threads": 2, "Hash": 64})
            board = game_obj.board()
            time_limit = 0.25
            move_num = 0

            for node in game_obj.mainline():
                move = node.move
                fen_before = board.fen()
                move_san = board.san(move)
                move_num += 1

                # Track exactly whose turn it is BEFORE pushing the move
                is_white_mover = board.turn

                analysis_before = engine.analyse(board, chess.engine.Limit(depth=depth, time=time_limit), multipv=2)
                best_move = analysis_before[0]["pv"][0] if len(analysis_before) > 0 else None
                best_san = board.san(best_move) if best_move else None

                cp_before_best = get_cp(analysis_before[0], is_white_mover)
                cp_before_second = get_cp(analysis_before[1], is_white_mover) if len(analysis_before) > 1 else (
                        cp_before_best - 150)

                board.push(move)

                # Now it is the opponent's turn, so pass board.turn for mate calculations
                analysis_after = engine.analyse(board, chess.engine.Limit(depth=depth, time=time_limit))
                score_after_obj = analysis_after["score"].white()
                cp_after = get_cp(analysis_after, board.turn)

                # ── True Mover Perspective ────────────────────────────
                # Calculate the evaluation relative to the player who just moved
                if is_white_mover:
                    eval_before = cp_before_best
                    eval_after = cp_after
                else:
                    eval_before = -cp_before_best
                    eval_after = -cp_after

                delta = eval_after - eval_before

                is_top_choice = (move == best_move) or (delta >= -15)
                is_opening = move_num <= 10

                board.pop()
                target_sq = move.to_square
                captured_piece = board.piece_at(target_sq)
                moving_piece = board.piece_at(move.from_square)
                board.push(move)

                val_lost = piece_values.get(moving_piece.piece_type, 0) if moving_piece else 0
                val_gained = piece_values.get(captured_piece.piece_type, 0) if captured_piece else 0

                # ── Context flags (board already has the move pushed here) ─────
                is_capture = captured_piece is not None
                is_check_after_move = board.is_check()  # we just gave check
                dest_sq = move.to_square
                # Is our freshly-moved piece sitting on a square the opponent can retake?
                # board.turn is now the OPPONENT (they move next), so their colour attacks.
                sq_attacked_by_opp = board.is_attacked_by(board.turn, dest_sq)

                # ── Expected-Points model (mirrors chess.com Classification V2) ─
                # eval_before / eval_after are both from the MOVER's perspective
                # (positive = good for the mover).
                wp_before = _cp_to_wp(eval_before)
                wp_after = _cp_to_wp(eval_after)
                ep_loss = max(0.0, wp_before - wp_after)

                # ── Sacrifice detection ───────────────────────────────────────
                # Pure sacrifice: piece moves to an attacked square WITHOUT capturing.
                is_pure_sacrifice = (
                        not is_capture and
                        val_lost >= 300 and  # at least a minor piece (300 cp ≈ bishop/knight)
                        sq_attacked_by_opp  # piece can be taken on the very next move
                )
                # Exchange sacrifice: capture something but hand over significantly more.
                is_exchange_sacrifice = (
                        is_capture and
                        val_lost >= 300 and
                        val_lost > val_gained + 150  # net material loss ≥ 1.5 pawns
                )
                is_sacrifice = is_pure_sacrifice or is_exchange_sacrifice

                # ── Situational thresholds ───────────────────────────────────
                # Mover is already crushing – can't earn Brilliant by pushing a win
                is_already_crushing = (eval_before > 450)

                # The second-best move is 300+ cp worse: truly the only good path.
                # (old threshold was 200 cp, which fired far too often)
                is_only_good_move_strict = (abs(cp_before_best - cp_before_second) >= 300)

                # Genuine game-flip: was behind, now ahead (not just "from 0 to +1")
                is_game_flipper = (eval_before < -50 and eval_after > 100)

                # ── Classify ─────────────────────────────────────────────────
                classification = "blunder"

                if is_top_choice:
                    # ── BRILLIANT ──────────────────────────────────────────
                    # Chess.com definition: "a good PIECE SACRIFICE that is the
                    # best or near-best move, when you were not already winning."
                    # Additional guards:
                    #   • checks are NOT brilliant – they're obvious to spot
                    #   • must be THE top engine choice (not just near-top)
                    #   • position must be at least slightly good after the sac
                    #   • the sac must be the only real winning path (300 cp gap)
                    if (is_sacrifice and
                            not is_check_after_move and  # checks ≠ brilliant
                            not is_opening and
                            not is_already_crushing and
                            move == best_move and  # strictly the #1 engine pick
                            eval_after > 100 and  # not suicidal
                            is_only_good_move_strict):  # genuinely hard to find
                        classification = "brilliant"

                    # ── GREAT ─────────────────────────────────────────────
                    # Game-critical move: turned the game around, OR the sole
                    # path in a contested position (not while already winning).
                    elif (not is_opening and
                          (is_game_flipper or
                           (is_only_good_move_strict and abs(eval_before) < 300))):
                        classification = "great"

                    # ── BEST ──────────────────────────────────────────────
                    # Top choice with negligible expected-points loss.
                    elif ep_loss < 0.02:
                        classification = "best"

                    # ── EXCELLENT ─────────────────────────────────────────
                    # Very close to best – small EP loss but not quite there.
                    else:
                        classification = "excellent"

                else:
                    # Non-best moves: Expected-Points thresholds
                    # (mirrors the chess.com V2 classification table)
                    if ep_loss < 0.05:
                        classification = "good"
                    elif ep_loss < 0.10:
                        classification = "inaccuracy"
                    elif ep_loss < 0.20:
                        classification = "mistake"
                    else:
                        # Miss: had a winning position but let it slip away
                        if eval_before > 200 and eval_after < 50:
                            classification = "miss"
                        else:
                            classification = "blunder"

                results.append({
                    "move_san": move_san,
                    "fen_before": fen_before,
                    "fen_after": board.fen(),
                    "score_cp": cp_after,
                    "score_mate": score_after_obj.mate() if score_after_obj.is_mate() else None,
                    "best_san": best_san,
                    "classification": classification,
                })

                analysis_progress[sid] = {"done": move_num, "total": total_moves, "status": "analysing"}

        # Aggregated classification counts for analytics
        counts = {}
        for m in results:
            cls = m.get("classification", "unknown")
            counts[cls] = counts.get(cls, 0) + 1
            
        analytics.log_event("analysis_complete", {
            "move_count": total_moves,
            "classifications": counts,
            "depth": depth
        }, sid)

        analysis_progress[sid] = {"done": total_moves, "total": total_moves, "status": "done", "moves": results}

    except Exception as e:
        logging.error(f"Analysis error: {e}")
        analysis_progress[sid] = {"done": 0, "total": 0, "status": "error", "error": str(e)}


# ─────────────────────────────────────────────
#  Scout & Puzzles (Unchanged Core Logic)
# ─────────────────────────────────────────────

@app.route("/get_pgn", methods=["GET"])
def get_pgn():
    game = get_game()
    if not game: return jsonify({"error": "No active game"}), 400
    pgn_str = build_pgn(game, game.get("result", "*"))
    session["last_pgn"] = pgn_str
    return jsonify({"pgn": pgn_str})


@app.route("/session_pgn", methods=["GET"])
def session_pgn():
    return jsonify({"pgn": session.get("last_pgn", "")})


@app.route("/delete_history", methods=["POST"])
def delete_history():
    idx = (request.json or {}).get("index")
    if idx is not None and "game_history" in session:
        history = list(session["game_history"])
        if 0 <= idx < len(history):
            history.pop(idx)
            session["game_history"] = history
            session.modified = True
            return jsonify({"status": "ok"})
    return jsonify({"error": "Invalid index"}), 400


@app.route("/scout")
def scout_page(): return render_template("scout.html")


@app.route("/scout", methods=["POST"])
def scout_api():
    data = request.json or {}
    username = (data.get("username") or "").strip()
    if not username: return jsonify({"error": "Please provide a username."}), 400

    try:
        base_dir = os.path.dirname(__file__)
        meta = joblib.load(os.path.join(base_dir, "Chess_AI", "RF_model", "opening_rf_meta.pkl"))
        rf = joblib.load(os.path.join(base_dir, "Chess_AI", "RF_model", "opening_rf.pkl"))
    except Exception as e:
        return jsonify({"error": f"Model not found: {e}"}), 500

    try:
        raw_data, lichess_count, chesscom_count = fetch_all_user_games(username)
    except Exception as e:
        return jsonify({"error": f"Failed to fetch games: {e}"}), 500

    if not raw_data:
        return jsonify({"error": f"No games found for '{username}'. Check the username and try again."}), 404

    total_games = lichess_count + chesscom_count
    df = pd.DataFrame(raw_data)
    stats = df.groupby(["color", "opening"]).agg(
        totalgames=("win", "count"), won=("win", "sum"), loss=("loss", "sum"),
        avg_rating=("p_rating", "mean"), avg_opp=("opp_rating", "mean"),
    ).reset_index()

    invalid_names = ['Unknown', 'unknown', 'n/a', 'N/A', '?', '']
    stats = stats[stats['opening'].notna() & (~stats['opening'].astype(str).isin(invalid_names)) & (
            stats["totalgames"] >= 5)].copy()

    if stats.empty: return jsonify(
        {"error": f"Not enough named opening data for '{username}' (need ≥5 games per opening)."}), 404

    stats["win_rate"] = stats["won"] / stats["totalgames"]
    stats["loss_rate"] = stats["loss"] / stats["totalgames"]
    stats["draw_rate"] = (stats["totalgames"] - stats["won"] - stats["loss"]) / stats["totalgames"]
    stats["rating_diff"] = stats["avg_rating"] - stats["avg_opp"]

    global_avg_opp = float(stats["avg_opp"].mean())
    global_max_games = int(stats["totalgames"].max())

    stats["exploit_score"] = stats.apply(lambda r: _exploit_score(r, global_avg_opp, global_max_games), axis=1)
    stats["avoid_score"] = stats.apply(lambda r: _avoid_score(r, global_avg_opp, global_max_games), axis=1)

    df_model = pd.get_dummies(stats, columns=["color"], prefix=["color"], dtype=int)
    for col in meta["feature_cols"]:
        if col not in df_model.columns: df_model[col] = 0

    X = df_model[meta["feature_cols"]].fillna(0).astype(float)
    df_model["label"] = rf.predict(X)
    df_model["opening_name"] = df_model["opening"].apply(clean_opening_name)

    for col in ["exploit_score", "avoid_score", "win_rate", "loss_rate", "draw_rate", "rating_diff", "avg_opp",
                "avg_rating"]:
        df_model[col] = stats[col].values

    results = []
    for side in ["white", "black"]:
        side_col = f"color_{side}"
        if side_col not in df_model.columns: continue
        subset = df_model[df_model[side_col] == 1].copy()
        exp = subset[subset["label"] == "exploit"].sort_values("exploit_score", ascending=False).head(2)
        avd = subset[subset["label"] == "avoid"].sort_values("avoid_score", ascending=False).head(2)
        for _, row in exp.iterrows(): results.append(_row_to_dict(row, side, "exploit"))
        for _, row in avd.iterrows(): results.append(_row_to_dict(row, side, "avoid"))

    analytics.log_event("scout_search", {
        "username": username,
        "lichess_games": lichess_count,
        "chesscom_games": chesscom_count,
        "total_games": total_games
    }, session.get("session_id"))

    return jsonify({"username": username, "total_games": total_games, "results": results})


def _row_to_dict(row, color: str, label: str) -> dict:
    return {
        "opening": str(row.get("opening_name", row.get("opening", ""))), "color": color, "label": label,
        "totalgames": int(row["totalgames"]), "won": int(row["won"]), "loss": int(row["loss"]),
        "win_rate": round(float(row["win_rate"]), 3), "loss_rate": round(float(row["loss_rate"]), 3),
        "draw_rate": round(float(row["draw_rate"]), 3), "avg_rating": round(float(row["avg_rating"]), 1),
        "avg_opp": round(float(row["avg_opp"]), 1), "exploit_score": round(float(row.get("exploit_score", 0)), 4),
        "avoid_score": round(float(row.get("avoid_score", 0)), 4),
    }


def _get_puzzle_db():
    if not hasattr(_puzzle_local, "con") or _puzzle_local.con is None:
        _puzzle_local.con = duckdb.connect(PUZZLE_DB_PATH, read_only=True)
    return _puzzle_local.con


@app.route("/puzzles")
def puzzles_page():
    return render_template("puzzle.html", initial_rating=session.get("puzzle_rating", 1200))


def _fetch_puzzles(target, theme, excl_ids, streak, count=1):
    con = _get_puzzle_db()
    quoted_excl = ",".join(f"'{str(x).replace('\"', '')}'" for x in excl_ids[:300]) if excl_ids else ""
    base_band = 100 if streak else 200

    def run_query(use_rating, use_theme, use_exclude, band):
        parts, params = [], []
        if use_rating: parts.append(f"rating BETWEEN {target - band} AND {target + band}")
        if use_theme and theme:
            parts.append("themes ILIKE ?")
            params.append(f"%{theme}%")
        if use_exclude and quoted_excl: parts.append(f"puzzle_id NOT IN ({quoted_excl})")

        where_clause = " AND ".join(parts) if parts else "1=1"
        sql = f"SELECT puzzle_id, fen, moves, rating, themes, opening_tags FROM puzzles WHERE {where_clause} ORDER BY random() LIMIT {int(count)}"
        return con.execute(sql, params).fetchall() if params else con.execute(sql).fetchall()

    rows = run_query(True, True, True, base_band)
    if len(rows) < count: rows = run_query(True, True, True, 400)
    if len(rows) < count: rows = run_query(False, True, True, 0)
    if len(rows) < count: rows = run_query(False, True, False, 0)
    if len(rows) < count: rows = run_query(False, False, False, 0)
    return rows


def _row_to_puzzle(row):
    moves_list = row[2].split()
    return {"puzzle_id": row[0], "fen": row[1], "moves": moves_list, "rating": row[3], "themes": row[4] or "",
            "opening_tags": row[5] or "", "move_count": len(moves_list) - 1}


@app.route("/get_puzzle", methods=["GET"])
def get_puzzle():
    session.permanent = True
    target = request.args.get("rating", session.get("puzzle_rating", 1200), type=int)
    theme = (request.args.get("theme", "") or "").strip()
    excl_ids = [e.strip() for e in request.args.get("exclude", "").split(",") if e.strip()]
    count = min(request.args.get("count", 1, type=int), 5)
    try:
        rows = _fetch_puzzles(target, theme, excl_ids, request.args.get("streak", "0") == "1", count)
        if not rows: return jsonify({"error": "No puzzles found."}), 404
        session["current_puzzle_theme"] = theme if theme else "mixed"
        return jsonify({"puzzles": [_row_to_puzzle(r) for r in rows]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/puzzle_themes", methods=["GET"])
def puzzle_themes():
    try:
        rows = _get_puzzle_db().execute(
            "SELECT DISTINCT themes FROM puzzles WHERE themes IS NOT NULL LIMIT 60000").fetchall()
        theme_set = set()
        for (t,) in rows:
            for tok in t.split(): theme_set.add(tok)
        return jsonify({"themes": sorted(theme_set)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/puzzle_stats", methods=["GET"])
def puzzle_stats():
    try:
        row = _get_puzzle_db().execute("SELECT COUNT(*), MIN(rating), MAX(rating), AVG(rating) FROM puzzles").fetchone()
        return jsonify({"total": row[0], "min_rating": row[1], "max_rating": row[2], "avg_rating": round(row[3], 1)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/puzzle_solved", methods=["POST"])
def puzzle_solved():
    session.permanent = True
    data = request.json or {}
    puzzle_rating = int(data.get("puzzle_rating", 1200))
    solved_flag = bool(data.get("solved", False))
    time_sec = int(data.get("time_seconds", 60))
    hints_used = int(data.get("hints_used", 0))
    rated = bool(data.get("rated", True))

    current = int(data.get("current_rating", session.get("puzzle_rating", 1200)))
    if not rated: return jsonify({"new_rating": current, "delta": 0, "streak": session.get("puzzle_streak", 0)})

    expected = 1 / (1 + 10 ** ((puzzle_rating - current) / 400))
    score = 1.05 if solved_flag and time_sec < 15 else (1.0 if solved_flag else 0.0)
    delta = round(32 * (score - expected))

    if solved_flag:
        if hints_used == 1:
            delta = max(0, int(delta * 0.5))
        elif hints_used >= 2:
            delta = 0

    new_rating = max(400, min(3000, current + delta))
    session["puzzle_rating"] = new_rating
    session["puzzle_streak"] = session.get("puzzle_streak", 0) + 1 if solved_flag and hints_used == 0 else 0

    analytics.log_event("puzzle_attempt", {
        "solved": 1 if solved_flag else 0,
        "rating": puzzle_rating,
        "theme": session.get("current_puzzle_theme") or "mixed"
    }, session.get("session_id"))

    return jsonify({"new_rating": new_rating, "delta": delta, "streak": session.get("puzzle_streak", 0)})


@app.route("/evaluate_fen", methods=["POST"])
def evaluate_fen():
    data = request.json or {}
    if not data.get("fen"): return jsonify({"error": "No FEN provided"}), 400
    if not os.path.exists(STOCKFISH_PATH): return jsonify({"error": "Stockfish not found"}), 500
    try:
        with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
            engine.configure({"Threads": 1, "Hash": 16})
            board_obj = chess.Board(data.get("fen"))
            info = engine.analyse(board_obj, chess.engine.Limit(depth=14, time=0.2))
            score = info["score"].white()
            cp = score.score()
            mate = score.mate() if score.is_mate() else None
            if mate == 0:
                cp = 10000 if board_obj.turn == chess.BLACK else -10000
            elif mate is not None:
                cp = 10000 if mate > 0 else -10000
            return jsonify({"cp": cp, "mate": mate})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.errorhandler(Exception)
def handle_exception(e):
    # Pass through HTTP errors
    if isinstance(e, HTTPException):
        return e
    logging.error(f"Unhandled Exception: {e}", exc_info=True)
    return jsonify({"error": "Internal Server Error", "details": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
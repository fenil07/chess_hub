"""
analytics.py  —  Chess Hub Analytics Engine
--------------------------------------------
Thread-safe SQLite analytics layer.
Drop-in for app.py — import and call log_event() / log_page_view() from any route.

Tables
------
  events      — every discrete action (game_start, game_end, puzzle_solved, etc.)
  page_views  — every page visit with session_id + timestamp
  daily_stats — pre-aggregated daily summaries (refreshed on demand)

Usage
-----
  from analytics import analytics
  analytics.log_event("game_start", {"engine": "ghost_model", "mode": "ai"}, session_id)
  analytics.log_page_view("/game", session_id)
  stats = analytics.get_dashboard_stats(days=30)
"""

import sqlite3
import json
import threading
import os
import datetime
import logging
import time
import atexit

from logging.handlers import TimedRotatingFileHandler

try:
    from huggingface_hub import hf_hub_download, upload_file
    HAS_HF = True
except ImportError:
    HAS_HF = False

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# ── IST Logging Setup ────────────────────────────────────────────────────────
def _setup_logging(data_dir):
    def ist_time(*args):
        # Indian Standard Time (UTC+5:30)
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        now_ist = now_utc + datetime.timedelta(hours=5, minutes=30)
        return now_ist.timetuple()
    
    logging.Formatter.converter = ist_time
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 1. Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)

    # 2. File Handler
    log_path = os.path.join(data_dir, "chess_hub.log")
    file_handler = TimedRotatingFileHandler(log_path, when="D", interval=1, backupCount=7)
    file_handler.setFormatter(log_formatter)
    
    # Configure Root Logger
    root_log = logging.getLogger()
    root_log.setLevel(logging.INFO)
    
    # Clear existing handlers to avoid duplicates/conflicts
    root_log.handlers = []
    root_log.addHandler(console_handler)
    root_log.addHandler(file_handler)
    
    # Silence noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    
    log.debug(f"[Analytics] Logging initialized. IST File: {log_path}")

HF_TOKEN = os.environ.get("HF_TOKEN")
DATASET_ID = os.environ.get("DATASET_ID")

# ── DB path ───────────────────────────────────────────────────────────────────
# Hugging Face Spaces provide persistent storage at /data if enabled.
# We prioritize /data, and create it if we are in a writeable environment.
_DATA_DIR = "/data"
if not os.path.exists(_DATA_DIR):
    try:
        # Fallback to local directory if /data isn't available/writeable
        _DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(_DATA_DIR, exist_ok=True)
    except Exception:
        _DATA_DIR = os.path.dirname(__file__)

def _restore_from_hub(data_dir):
    """Pulls existing DB and Logs from the Hub BEFORE logging is initialized."""
    if not (HAS_HF and HF_TOKEN and DATASET_ID):
        return

    for fname in ["chess_hub_analytics.db", "chess_hub.log"]:
        try:
            hf_hub_download(
                repo_id=DATASET_ID,
                filename=fname,
                repo_type="dataset",
                token=HF_TOKEN,
                local_dir=data_dir,
                local_dir_use_symlinks=False
            )
        except Exception:
            pass # No file on hub yet, that's fine

DB_PATH = os.environ.get("ANALYTICS_DB_PATH", os.path.join(_DATA_DIR, "chess_hub_analytics.db"))
LOG_PATH = os.path.join(_DATA_DIR, "chess_hub.log")

# ── Startup Sequence ──────────────────────────────────────────────────────────
# 1. Restore from Hub BEFORE setting up logging (prevents file descriptor loss)
_restore_from_hub(_DATA_DIR)

# 2. Now setup logging (will append to the restored file)
_setup_logging(_DATA_DIR)
log.debug(f"[Analytics] Initializing. Database path: {DB_PATH}")

_local = threading.local()   # one connection per thread


# ── Connection factory ────────────────────────────────────────────────────────
def _get_conn() -> sqlite3.Connection:
    if not hasattr(_local, "conn") or _local.conn is None:
        try:
            # Ensure the parent directory exists right before connecting
            os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
            _local.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            _local.conn.row_factory = sqlite3.Row
            _local.conn.execute("PRAGMA journal_mode=WAL") # Keep WAL for concurrency
            _local.conn.execute("PRAGMA synchronous=NORMAL")
            _local.conn.execute("PRAGMA busy_timeout=5000") # Wait up to 5s if DB is busy
        except Exception as e:
            log.error(f"[Analytics] Failed to connect to database at {DB_PATH}: {e}")
            raise
    return _local.conn


# ── Schema ────────────────────────────────────────────────────────────────────
SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type  TEXT    NOT NULL,
    data        TEXT    NOT NULL DEFAULT '{}',
    session_id  TEXT,
    ts          TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now'))
);

CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_ts   ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_sess ON events(session_id);

CREATE TABLE IF NOT EXISTS page_views (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    page        TEXT    NOT NULL,
    session_id  TEXT,
    ts          TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now'))
);

CREATE INDEX IF NOT EXISTS idx_pv_page ON page_views(page);
CREATE INDEX IF NOT EXISTS idx_pv_ts   ON page_views(ts);
CREATE INDEX IF NOT EXISTS idx_pv_sess ON page_views(session_id);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    val   TEXT
);
"""


def _init_db():
    # 2. Run schema
    try:
        conn = _get_conn()
        conn.executescript(SCHEMA)
        conn.commit()
        log.info(f"[Analytics] DB ready at {DB_PATH}")
    except Exception as e:
        log.error(f"[Analytics] DB init failed: {e}")


# ── Public API ────────────────────────────────────────────────────────────────
class Analytics:
    def __init__(self):
        _init_db()
        self._start_sync_thread()
        atexit.register(self.sync_to_hub)

    def _start_sync_thread(self):
        if HAS_HF and HF_TOKEN and DATASET_ID:
            t = threading.Thread(target=self._sync_loop, daemon=True)
            t.start()
            log.info("[Analytics] Background sync thread started.")

    def _sync_loop(self):
        # Initial wait to let app boot
        time.sleep(60)
        while True:
            try:
                self.sync_to_hub()
            except Exception as e:
                log.error(f"[Analytics] Loop sync failed: {e}")
            time.sleep(300) # Sync every 5 minutes

    def sync_to_hub(self):
        if not HAS_HF or not HF_TOKEN or not DATASET_ID:
            log.error("[Analytics] Cannot sync: HF_TOKEN or DATASET_ID missing.")
            return
        try:
            # 1. Checkpoint WAL -> DB (Flushes all data into the main .db file)
            try:
                conn = _get_conn()
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                log.debug("[Analytics] WAL Checkpoint complete.")
            except Exception as e:
                log.warning(f"[Analytics] Checkpoint failed (continuing anyway): {e}")

            # 2. Sync DB
            if os.path.exists(DB_PATH):
                log.debug(f"[Analytics] Uploading DB to {DATASET_ID}...")
                upload_file(
                    path_or_fileobj=DB_PATH,
                    path_in_repo="chess_hub_analytics.db",
                    repo_id=DATASET_ID,
                    repo_type="dataset",
                    token=HF_TOKEN
                )
            
            # Sync Log
            log_path = os.path.join(_DATA_DIR, "chess_hub.log")
            if os.path.exists(log_path):
                upload_file(
                    path_or_fileobj=log_path,
                    path_in_repo="chess_hub.log",
                    repo_id=DATASET_ID,
                    repo_type="dataset",
                    token=HF_TOKEN
                )
            log.debug("[Analytics] DB & Logs successfully pushed to Hub.")
        except Exception as e:
            log.error(f"[Analytics] Hub upload failed: {e}")

    def get_diagnostics(self):
        """Check status of sync variables."""
        import os
        conn = _get_conn()
        pv_count = conn.execute("SELECT COUNT(*) FROM page_views").fetchone()[0]
        ev_count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        return {
            "has_hf_lib": HAS_HF,
            "has_token": bool(HF_TOKEN),
            "has_dataset_id": bool(DATASET_ID),
            "dataset_id": DATASET_ID,
            "db_path": DB_PATH,
            "db_exists": os.path.exists(DB_PATH),
            "db_size": os.path.getsize(DB_PATH) if os.path.exists(DB_PATH) else 0,
            "row_counts": {"page_views": pv_count, "events": ev_count},
            "data_dir": _DATA_DIR,
            "data_dir_writable": os.access(_DATA_DIR, os.W_OK) if os.path.exists(_DATA_DIR) else False
        }

    # ── Write helpers ──────────────────────────────────────────────────────
    def log_event(self, event_type: str, data: dict = None, session_id: str = None):
        """Fire-and-forget event logger. Never blocks the Flask response."""
        try:
            conn = _get_conn()
            conn.execute(
                "INSERT INTO events (event_type, data, session_id) VALUES (?, ?, ?)",
                (event_type, json.dumps(data or {}), session_id)
            )
            conn.commit()
            # If it's a major event, trigger a sync soon
            if event_type in ("game_end", "analysis_complete"):
                pass # Could trigger sync, but background loop is safer
        except Exception as e:
            log.warning(f"[Analytics] log_event failed: {e}")

    def log_page_view(self, page: str, session_id: str = None):
        try:
            conn = _get_conn()
            conn.execute(
                "INSERT INTO page_views (page, session_id) VALUES (?, ?)",
                (page, session_id)
            )
            conn.commit()
        except Exception as e:
            log.warning(f"[Analytics] log_page_view failed: {e}")

    # ── Dashboard stats ────────────────────────────────────────────────────
    def get_dashboard_stats(self, days: int = 30) -> dict:
        """Return all stats needed by the admin dashboard in one call.
        When days >= 36500 (the 'All Time' sentinel), the date filter is
        skipped entirely so every row in the DB is included.
        """
        conn = _get_conn()
        ALL_TIME = days >= 36500
        since = "1970-01-01T00:00:00" if ALL_TIME else (
            datetime.datetime.utcnow() - datetime.timedelta(days=days)
        ).strftime("%Y-%m-%dT%H:%M:%S")

        # ── Overview totals ────────────────────────────────────────────────
        total_visits = conn.execute(
            "SELECT COUNT(*) FROM page_views WHERE ts >= ?", (since,)
        ).fetchone()[0]

        unique_sessions = conn.execute(
            "SELECT COUNT(DISTINCT session_id) FROM page_views WHERE ts >= ? AND session_id IS NOT NULL",
            (since,)
        ).fetchone()[0]

        games_played = conn.execute(
            "SELECT COUNT(*) FROM events WHERE event_type='game_end' AND ts >= ?", (since,)
        ).fetchone()[0]

        puzzles_attempted = conn.execute(
            "SELECT COUNT(*) FROM events WHERE event_type='puzzle_attempt' AND ts >= ?", (since,)
        ).fetchone()[0]

        scout_searches = conn.execute(
            "SELECT COUNT(*) FROM events WHERE event_type='scout_search' AND ts >= ?", (since,)
        ).fetchone()[0]

        analyses_run = conn.execute(
            "SELECT COUNT(*) FROM events WHERE event_type='analysis_start' AND ts >= ?", (since,)
        ).fetchone()[0]

        # ── All-time totals ────────────────────────────────────────────────
        all_time_visits  = conn.execute("SELECT COUNT(*) FROM page_views").fetchone()[0]
        all_time_games   = conn.execute("SELECT COUNT(*) FROM events WHERE event_type='game_end'").fetchone()[0]
        all_time_puzzles = conn.execute("SELECT COUNT(*) FROM events WHERE event_type='puzzle_attempt'").fetchone()[0]
        all_time_scout   = conn.execute("SELECT COUNT(*) FROM events WHERE event_type='scout_search'").fetchone()[0]

        # ── Daily visits (for sparkline chart) ────────────────────────────
        daily_rows = conn.execute("""
            SELECT substr(ts,1,10) AS day, COUNT(*) AS cnt
            FROM page_views WHERE ts >= ?
            GROUP BY day ORDER BY day
        """, (since,)).fetchall()
        daily_visits = [{"date": r["day"], "visits": r["cnt"]} for r in daily_rows]

        # ── Daily unique sessions ──────────────────────────────────────────
        daily_users_rows = conn.execute("""
            SELECT substr(ts,1,10) AS day, COUNT(DISTINCT session_id) AS cnt
            FROM page_views WHERE ts >= ? AND session_id IS NOT NULL
            GROUP BY day ORDER BY day
        """, (since,)).fetchall()
        daily_users = [{"date": r["day"], "users": r["cnt"]} for r in daily_users_rows]

        # ── Daily games ────────────────────────────────────────────────────
        daily_games_rows = conn.execute("""
            SELECT substr(ts,1,10) AS day, COUNT(*) AS cnt
            FROM events WHERE event_type='game_end' AND ts >= ?
            GROUP BY day ORDER BY day
        """, (since,)).fetchall()
        daily_games = [{"date": r["day"], "games": r["cnt"]} for r in daily_games_rows]

        # ── Engine popularity ──────────────────────────────────────────────
        engine_rows = conn.execute("""
            SELECT json_extract(data,'$.engine') AS engine, COUNT(*) AS cnt
            FROM events WHERE event_type='game_start' AND ts >= ?
            GROUP BY engine ORDER BY cnt DESC
        """, (since,)).fetchall()
        engine_stats = [{"engine": r["engine"] or "unknown", "count": r["cnt"]} for r in engine_rows]

        # ── AI win/loss/draw ───────────────────────────────────────────────
        wl_rows = conn.execute("""
            SELECT json_extract(data,'$.result') AS result, COUNT(*) AS cnt
            FROM events WHERE event_type='game_end' AND ts >= ?
            AND json_extract(data,'$.mode') = 'ai'
            GROUP BY result
        """, (since,)).fetchall()
        ai_wl = {r["result"]: r["cnt"] for r in wl_rows}

        # ── End reason breakdown ───────────────────────────────────────────
        end_reason_rows = conn.execute("""
            SELECT json_extract(data,'$.end_reason') AS reason, COUNT(*) AS cnt
            FROM events WHERE event_type='game_end' AND ts >= ?
            GROUP BY reason ORDER BY cnt DESC
        """, (since,)).fetchall()
        end_reasons = [{"reason": r["reason"] or "unknown", "count": r["cnt"]} for r in end_reason_rows]

        # ── Game mode split ────────────────────────────────────────────────
        mode_rows = conn.execute("""
            SELECT json_extract(data,'$.mode') AS mode, COUNT(*) AS cnt
            FROM events WHERE event_type='game_start' AND ts >= ?
            GROUP BY mode
        """, (since,)).fetchall()
        game_modes = {r["mode"]: r["cnt"] for r in mode_rows}

        # ── Puzzle stats ───────────────────────────────────────────────────
        puzzle_solved = conn.execute("""
            SELECT COUNT(*) FROM events WHERE event_type='puzzle_attempt'
            AND json_extract(data,'$.solved') = 1 AND ts >= ?
        """, (since,)).fetchone()[0]

        puzzle_themes_rows = conn.execute("""
            SELECT json_extract(data,'$.theme') AS theme, COUNT(*) AS cnt
            FROM events WHERE event_type='puzzle_attempt' AND ts >= ?
            AND theme IS NOT NULL AND theme != ''
            GROUP BY theme ORDER BY cnt DESC LIMIT 10
        """, (since,)).fetchall()
        puzzle_themes = [{"theme": r["theme"], "count": r["cnt"]} for r in puzzle_themes_rows]

        daily_puzzles_rows = conn.execute("""
            SELECT substr(ts,1,10) AS day, COUNT(*) AS cnt
            FROM events WHERE event_type='puzzle_attempt' AND ts >= ?
            GROUP BY day ORDER BY day
        """, (since,)).fetchall()
        daily_puzzles = [{"date": r["day"], "puzzles": r["cnt"]} for r in daily_puzzles_rows]

        # ── Page popularity ────────────────────────────────────────────────
        page_rows = conn.execute("""
            SELECT page, COUNT(*) AS cnt
            FROM page_views WHERE ts >= ?
            GROUP BY page ORDER BY cnt DESC
        """, (since,)).fetchall()
        page_stats = [{"page": r["page"], "views": r["cnt"]} for r in page_rows]

        # ── Hourly activity heatmap (hour-of-day × day-of-week) ───────────
        heatmap_rows = conn.execute("""
            SELECT
              CAST(strftime('%H', ts, '+5 hours', '+30 minutes') AS INTEGER) AS hour,
              CAST(strftime('%w', ts, '+5 hours', '+30 minutes') AS INTEGER) AS dow,
              COUNT(*) AS cnt
            FROM page_views WHERE ts >= ?
            GROUP BY hour, dow
        """, (since,)).fetchall()
        heatmap = [{"hour": r["hour"], "dow": r["dow"], "count": r["cnt"]} for r in heatmap_rows]

        # ── Average game duration ──────────────────────────────────────────
        avg_duration_row = conn.execute("""
            SELECT AVG(CAST(json_extract(data,'$.duration_sec') AS REAL)) AS avg_dur
            FROM events WHERE event_type='game_end' AND ts >= ?
            AND json_extract(data,'$.duration_sec') IS NOT NULL
        """, (since,)).fetchone()
        avg_game_duration = round(avg_duration_row["avg_dur"] or 0, 1)

        # ── Average moves per game ─────────────────────────────────────────
        avg_moves_row = conn.execute("""
            SELECT AVG(CAST(json_extract(data,'$.total_moves') AS REAL)) AS avg_moves
            FROM events WHERE event_type='game_end' AND ts >= ?
        """, (since,)).fetchone()
        avg_moves = round(avg_moves_row["avg_moves"] or 0, 1)

        # ── Scout platform breakdown ───────────────────────────────────────
        scout_lc = conn.execute("""
            SELECT SUM(CAST(json_extract(data,'$.lichess_games') AS INTEGER))
            FROM events WHERE event_type='scout_search' AND ts >= ?
        """, (since,)).fetchone()[0] or 0

        scout_cc = conn.execute("""
            SELECT SUM(CAST(json_extract(data,'$.chesscom_games') AS INTEGER))
            FROM events WHERE event_type='scout_search' AND ts >= ?
        """, (since,)).fetchone()[0] or 0

        # ── Top searched usernames ─────────────────────────────────────────
        scout_users_rows = conn.execute("""
            SELECT json_extract(data,'$.username') AS username, COUNT(*) AS cnt
            FROM events WHERE event_type='scout_search' AND ts >= ?
            AND username IS NOT NULL AND username != ''
            GROUP BY username ORDER BY cnt DESC LIMIT 10
        """, (since,)).fetchall()
        top_scouted = [{"username": r["username"], "count": r["cnt"]} for r in scout_users_rows]

        # ── Analysis move classifications ──────────────────────────────────
        analysis_rows = conn.execute("""
            SELECT json_extract(data,'$.classifications') AS cls_json
            FROM events WHERE event_type='analysis_complete' AND ts >= ?
        """, (since,)).fetchall()
        move_cls_totals = {}
        for r in analysis_rows:
            try:
                counts = json.loads(r["cls_json"])
                for k, v in counts.items():
                    move_cls_totals[k] = move_cls_totals.get(k, 0) + v
            except: continue
        cls_order = ["brilliant", "great", "best", "excellent", "good", "inaccuracy", "mistake", "miss", "blunder"]
        move_classification_stats = [{"label": c, "count": move_cls_totals.get(c, 0)} for c in cls_order]

        # ── Opening popularity ─────────────────────────────────────────────
        opening_rows = conn.execute("""
            SELECT json_extract(data,'$.opening') AS opening, COUNT(*) AS cnt
            FROM events WHERE event_type='game_start' AND ts >= ?
            AND opening IS NOT NULL AND opening != ''
            GROUP BY opening ORDER BY cnt DESC LIMIT 10
        """, (since,)).fetchall()
        top_openings = [{"opening": r["opening"], "count": r["cnt"]} for r in opening_rows]

        # ── Recent events (live log) ───────────────────────────────────────
        recent_events_rows = conn.execute("""
            SELECT event_type, data, session_id, ts
            FROM events ORDER BY id DESC LIMIT 50
        """).fetchall()
        recent_events = [
            {
                "type": r["event_type"],
                "data": json.loads(r["data"]),
                "session": (r["session_id"] or "")[:8],
                "ts": r["ts"]
            }
            for r in recent_events_rows
        ]

        # ── First event date (Persistent Launch Date) ───────────────────────
        launch_row = conn.execute("SELECT val FROM settings WHERE key='launch_date'").fetchone()
        if launch_row:
            first_event_date = launch_row[0]
        else:
            first_row = conn.execute("SELECT MIN(ts) FROM events").fetchone()[0]
            first_event_date = (first_row or datetime.datetime.utcnow().strftime("%Y-%m-%d"))[:10]
            # Save it so it never changes again
            if first_event_date:
                try:
                    conn.execute("INSERT OR IGNORE INTO settings (key, val) VALUES ('launch_date', ?)", (first_event_date,))
                    conn.commit()
                except: pass

        return {
            "period_days": "All Time" if ALL_TIME else days,
            "overview": {
                "total_visits": total_visits,
                "unique_sessions": unique_sessions,
                "games_played": games_played,
                "puzzles_attempted": puzzles_attempted,
                "scout_searches": scout_searches,
                "analyses_run": analyses_run,
            },
            "all_time": {
                "visits": all_time_visits,
                "games": all_time_games,
                "puzzles": all_time_puzzles,
                "scout": all_time_scout,
                "first_event_date": first_event_date,
            },
            "charts": {
                "daily_visits": daily_visits,
                "daily_users": daily_users,
                "daily_games": daily_games,
                "daily_puzzles": daily_puzzles,
                "page_stats": page_stats,
                "engine_stats": engine_stats,
                "game_modes": game_modes,
                "end_reasons": end_reasons,
                "top_openings": top_openings,
                "puzzle_themes": puzzle_themes,
                "heatmap": heatmap,
                "top_scouted": top_scouted,
                "move_classifications": move_classification_stats
            },
            "game_meta": {
                "ai_wl": ai_wl,
                "avg_game_duration_sec": avg_game_duration,
                "avg_moves_per_game": avg_moves,
                "puzzle_solve_rate": round(puzzle_solved / max(puzzles_attempted, 1) * 100, 1),
                "puzzle_solved": puzzle_solved,
                "scout_lichess_games": int(scout_lc),
                "scout_chesscom_games": int(scout_cc),
            },
            "recent_events": recent_events,
        }

    def get_retention(self, days: int = 30) -> list:
        """DAU over last N days for retention chart."""
        conn = _get_conn()
        since = (datetime.datetime.utcnow() - datetime.timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")
        rows = conn.execute("""
            SELECT substr(ts,1,10) AS day,
                   COUNT(DISTINCT session_id) AS dau
            FROM page_views WHERE ts >= ? AND session_id IS NOT NULL
            GROUP BY day ORDER BY day
        """, (since,)).fetchall()
        return [{"date": r["day"], "dau": r["dau"]} for r in rows]

    def get_raw_events(self, page: int = 1, per_page: int = 100,
                       event_type: str = None) -> dict:
        """Paginated raw event log for the admin event feed."""
        conn = _get_conn()
        offset = (page - 1) * per_page
        if event_type:
            rows = conn.execute(
                "SELECT * FROM events WHERE event_type=? ORDER BY id DESC LIMIT ? OFFSET ?",
                (event_type, per_page, offset)
            ).fetchall()
            total = conn.execute(
                "SELECT COUNT(*) FROM events WHERE event_type=?", (event_type,)
            ).fetchone()[0]
        else:
            rows = conn.execute(
                "SELECT * FROM events ORDER BY id DESC LIMIT ? OFFSET ?",
                (per_page, offset)
            ).fetchall()
            total = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]

        return {
            "events": [
                {
                    "id": r["id"], "type": r["event_type"],
                    "data": json.loads(r["data"]),
                    "session": (r["session_id"] or "")[:8],
                    "ts": r["ts"]
                }
                for r in rows
            ],
            "total": total,
            "page": page,
            "per_page": per_page,
            "pages": max(1, (total + per_page - 1) // per_page),
        }

    def get_log_path(self):
        return LOG_PATH


# ── Singleton ─────────────────────────────────────────────────────────────────
analytics = Analytics()
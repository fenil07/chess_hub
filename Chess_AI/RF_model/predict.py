import json
import logging
import re
import joblib
import requests
import pandas as pd

# --- CONFIGURATION ---
MODEL_PATH = "opening_rf.pkl"
META_PATH = "opening_rf_meta.pkl"
CHESSCOM_ARC = "https://api.chess.com/pub/player/{username}/games/archives"
LICHESS_API = "https://lichess.org/api/games/user/{username}"

# Mappings
CHESSCOM_TC_MAP = {"daily": "classical", "rapid": "rapid", "blitz": "blitz"}
CHESSCOM_RESULT_MAP = {
    "win": 1, "checkmated": 0, "timeout": 0, "resigned": 0, "lose": 0,
    "insufficient": 0.5, "stalemate": 0.5, "repetition": 0.5,
    "agreed": 0.5, "50move": 0.5, "timevsinsufficient": 0.5,
}

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)


# --- HELPERS ---
def clean_opening_name(name):
    if not isinstance(name, str):
        return name
    if "chess.com/openings/" in name:
        name = re.sub(r'https?://www.chess.com/openings/', '', name)
        return name.replace('-', ' ').replace('...', '').strip()
    return name


def fetch_all_user_games(username):
    rows = []

    lichess_count = 0
    chesscom_count = 0

    # =========================
    # LICHESS (PER TIME CONTROL)
    # =========================
    try:
        log.info("Fetching Lichess games...")

        for tc in ["blitz", "rapid", "classical"]:
            params = {
                "opening": "true",
                "perfType": tc
            }

            if tc in ["blitz", "rapid"]:
                params["max"] = 2000

            r = requests.get(
                LICHESS_API.format(username=username),
                params=params,
                headers={"Accept": "application/x-ndjson"},
                stream=True,
                timeout=30
            )

            if r.status_code == 404:
                continue

            count = 0

            for line in r.iter_lines():
                if not line:
                    continue

                g = json.loads(line)

                try:
                    color = "white" if g['players']['white']['user']['name'].lower() == username.lower() else "black"
                except:
                    continue

                winner = g.get("winner")

                rows.append({
                    "time_control": tc,
                    "color": color,
                    "opening": g.get("opening", {}).get("name", "Unknown"),
                    "win": 1 if winner == color else 0,
                    "loss": 1 if winner and winner != color else 0,
                    "p_rating": g['players'][color].get("rating", 1500),
                    "opp_rating": g['players']["black" if color == "white" else "white"].get("rating", 1500)
                })

                count += 1
                lichess_count += 1

                if tc in ["blitz", "rapid"] and count >= 2000:
                    break

            log.info(f"Lichess {tc}: {count} games")

    except Exception as e:
        log.error(f"Lichess fetch error: {e}")

    # =========================
    # CHESS.COM (CAP 3000 LATEST)
    # =========================
    try:
        log.info("Fetching Chess.com archives...")

        r = requests.get(
            CHESSCOM_ARC.format(username=username),
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15
        )

        if r.status_code in (404, 301, 410):
            log.warning("Chess.com user not found")
            return rows, lichess_count, chesscom_count

        if r.status_code != 200:
            log.error(f"Chess.com failed: {r.status_code}")
            return rows, lichess_count, chesscom_count

        archives = r.json().get("archives", [])
        log.info(f"Total archives: {len(archives)}")

        for url in reversed(archives):  # newest first
            if chesscom_count >= 3000:
                break

            resp = requests.get(
                url,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=30
            )

            if resp.status_code != 200:
                continue

            games = resp.json().get("games", [])

            for g in games:
                if chesscom_count >= 3000:
                    break

                tc = CHESSCOM_TC_MAP.get(g.get("time_class"))
                if not tc:
                    continue

                w, b = g['white'], g['black']

                if w['username'].lower() == username.lower():
                    color, res, pr, opr = "white", w['result'], w['rating'], b['rating']
                else:
                    color, res, pr, opr = "black", b['result'], b['rating'], w['rating']

                out = CHESSCOM_RESULT_MAP.get(res, 0.5)

                eco_url = g.get("eco", "")
                opening_name = clean_opening_name(eco_url) if eco_url else "Unknown"

                rows.append({
                    "time_control": tc,
                    "color": color,
                    "opening": opening_name,
                    "win": 1 if out == 1 else 0,
                    "loss": 1 if out == 0 else 0,
                    "p_rating": pr,
                    "opp_rating": opr
                })

                chesscom_count += 1

    except Exception as e:
        log.error(f"Chess.com fetch error: {e}")

    return rows, lichess_count, chesscom_count


# --- MAIN REPORT ---
def run_scouting_report(username):
    meta = joblib.load(META_PATH)
    rf = joblib.load(MODEL_PATH)

    raw_data, lichess_count, chesscom_count = fetch_all_user_games(username)

    if not raw_data:
        logging.info("No data found.")
        return

    logging.info(f"\nTotal Lichess games fetched: {lichess_count}")
    logging.info(f"Total Chess.com games fetched: {chesscom_count}")
    logging.info(f"Total combined games: {len(raw_data)}")

    df = pd.DataFrame(raw_data)

    stats = df.groupby(["color", "opening"]).agg(
        totalgames=("win", "count"),
        won=("win", "sum"),
        loss=("loss", "sum"),
        avg_rating=("p_rating", "mean"),
        avg_opp=("opp_rating", "mean")
    ).reset_index()

    invalid_names = ['Unknown', 'unknown', 'n/a', 'N/A', '?', '']

    stats = stats[
        stats['opening'].notna() &
        (~stats['opening'].astype(str).isin(invalid_names)) &
        (stats["totalgames"] >= 5)
    ].copy()

    logging.info(f"Unique openings with >=5 games: {len(stats)}")

    if stats.empty:
        logging.info(f"No valid named openings for {username}")
        return

    stats["win_rate"] = stats["won"] / stats["totalgames"]
    stats["loss_rate"] = stats["loss"] / stats["totalgames"]
    stats["draw_rate"] = (stats["totalgames"] - stats["won"] - stats["loss"]) / stats["totalgames"]
    stats["rating_diff"] = stats["avg_rating"] - stats["avg_opp"]

    df_model = pd.get_dummies(stats, columns=['color'], prefix=['color'], dtype=int)

    for col in meta['feature_cols']:
        if col not in df_model.columns:
            df_model[col] = 0

    X = df_model[meta['feature_cols']].fillna(0).astype(float)

    df_model['label'] = rf.predict(X)
    df_model['opening_name'] = df_model['opening'].apply(clean_opening_name)

    final_results = []

    for side in ['white', 'black']:
        side_col = f'color_{side}'
        if side_col not in df_model.columns:
            continue

        subset = df_model[df_model[side_col] == 1]

        exp = subset[subset['label'] == 'exploit'].sort_values('loss_rate', ascending=False).head(1)
        avd = subset[subset['label'] == 'avoid'].sort_values('win_rate', ascending=False).head(1)

        if not exp.empty:
            final_results.append(exp)
        if not avd.empty:
            final_results.append(avd)

    if not final_results:
        logging.info("No strong exploits/avoids found.")
        return

    output_df = pd.concat(final_results)

    output_df['color'] = output_df.apply(
        lambda x: 'white' if x.get('color_white', 0) == 1 else 'black', axis=1
    )

    logging.info("\n" + "=" * 95)
    logging.info(f"CLEAN SCOUTING REPORT: {username}")
    logging.info("=" * 95)

    logging.info(f"{'LABEL':<10} | {'PLAYERS COLOR':<15} | {'OPENING':<40} | {'TOTAL':<7} | {'WON':<5} | {'LOSS':<5}")
    logging.info(f"{'-' * 10}-|-{'-' * 15}-|-{'-' * 40}-|-{'-' * 7}-|-{'-' * 5}-|-{'-' * 5}")

    for _, row in output_df.iterrows():
        logging.info(f"{row['label'].upper():<10} | "
              f"{row['color']:<15} | "
              f"{row['opening_name'][:40]:<40} | "
              f"{int(row['totalgames']):<7} | "
              f"{int(row['won']):<5} | "
              f"{int(row['loss']):<5}")

    logging.info("=" * 95)


# --- RUN ---
if __name__ == "__main__":
    user = "mayur_c"
    run_scouting_report(user)
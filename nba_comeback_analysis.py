"""
NBA Playoff Q4 Comeback Analysis
Seasons: 2022-23, 2023-24, 2024-25

Finds how often a team trailing by 5+ or 10+ points at any point
during the 4th quarter came back to win the game.

Usage:
    python nba_comeback_analysis.py          # live NBA Stats API
    python nba_comeback_analysis.py --demo   # synthetic data (no network)
"""

import sys
import time
import warnings
import pandas as pd
from tabulate import tabulate
from nba_api.stats.endpoints import LeagueGameLog, PlayByPlayV2

warnings.filterwarnings("ignore")

SEASONS = ["2022-23", "2023-24", "2024-25"]
SLEEP_BETWEEN_REQUESTS = 0.7
MAX_RETRIES = 2


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def fetch_playoff_games(season: str) -> pd.DataFrame:
    """Return deduplicated playoff game rows for a season."""
    print(f"  Fetching game log for {season}...")
    for attempt in range(MAX_RETRIES):
        try:
            log = LeagueGameLog(
                season=season,
                season_type_all_star="Playoffs",
                league_id="00",
            )
            df = log.get_data_frames()[0]
            time.sleep(SLEEP_BETWEEN_REQUESTS)
            return df
        except Exception as exc:
            print(f"    Attempt {attempt + 1} failed: {exc}")
            time.sleep(2 ** attempt * 2)
    print(f"  ERROR: Could not fetch game log for {season}")
    return pd.DataFrame()


def fetch_play_by_play(game_id: str) -> pd.DataFrame:
    """Return play-by-play rows for a single game."""
    for attempt in range(MAX_RETRIES):
        try:
            pbp = PlayByPlayV2(game_id=game_id)
            df = pbp.get_data_frames()[0]
            time.sleep(SLEEP_BETWEEN_REQUESTS)
            return df
        except Exception as exc:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt * 2)
            else:
                raise exc
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Analysis logic
# ---------------------------------------------------------------------------

def parse_margin(val) -> float | None:
    """Convert SCOREMARGIN value to float (home perspective). None = skip row."""
    if pd.isna(val) or val == "" or val is None:
        return None
    if str(val).strip().upper() == "TIE":
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def analyze_game(game_id: str, home_team_won: bool) -> dict:
    """
    Fetch Q4 play-by-play and determine whether the winner ever trailed
    by 5+ or 10+ at any point in the 4th quarter.

    Returns dict: {ok, trailed_5, trailed_10}
    """
    result = {"ok": False, "trailed_5": False, "trailed_10": False}

    try:
        pbp = fetch_play_by_play(game_id)
    except Exception as exc:
        print(f"    WARNING: PBP fetch failed for {game_id}: {exc}")
        return result

    if pbp.empty:
        print(f"    WARNING: Empty PBP for {game_id}")
        return result

    q4 = pbp[pbp["PERIOD"] == 4].copy()
    if q4.empty:
        print(f"    WARNING: No Q4 rows for {game_id}")
        return result

    result["ok"] = True

    for val in q4["SCOREMARGIN"]:
        margin = parse_margin(val)
        if margin is None:
            continue

        if home_team_won:
            # Home won — check if home was ever trailing (margin <= -N)
            if margin <= -5:
                result["trailed_5"] = True
            if margin <= -10:
                result["trailed_10"] = True
        else:
            # Away won — check if away was ever trailing (margin >= +N)
            if margin >= 5:
                result["trailed_5"] = True
            if margin >= 10:
                result["trailed_10"] = True

        if result["trailed_5"] and result["trailed_10"]:
            break  # both flags set; no need to scan further

    return result


def process_season(season: str) -> dict:
    """Run the full analysis for one season. Returns aggregated stats."""
    stats = {
        "season": season,
        "total_games": 0,
        "comebacks_from_5": 0,
        "comebacks_from_10": 0,
    }

    game_log = fetch_playoff_games(season)
    if game_log.empty:
        return stats

    # Keep only completed games (WL is W or L)
    game_log = game_log[game_log["WL"].isin(["W", "L"])].copy()

    # Each game appears twice (once per team). Get unique IDs.
    game_ids = game_log["GAME_ID"].unique()

    # Build lookup: game_id -> {home_team_id, away_team_id, winner_team_id}
    # MATCHUP format: "HOU vs. LAL" = home row, "HOU @ LAL" = away row
    game_info: dict[str, dict] = {}
    for _, row in game_log.iterrows():
        gid = row["GAME_ID"]
        if gid not in game_info:
            game_info[gid] = {
                "home_team_id": None,
                "away_team_id": None,
                "winner_team_id": None,
            }

        team_id = row["TEAM_ID"]
        matchup: str = row["MATCHUP"]
        wl = row["WL"]

        if "vs." in matchup:
            game_info[gid]["home_team_id"] = team_id
        else:
            game_info[gid]["away_team_id"] = team_id

        if wl == "W":
            game_info[gid]["winner_team_id"] = team_id

    total = len(game_ids)
    print(f"  Processing {total} games for {season}...")

    for idx, game_id in enumerate(game_ids, 1):
        info = game_info.get(game_id, {})
        home_id = info.get("home_team_id")
        winner_id = info.get("winner_team_id")

        if home_id is None or winner_id is None:
            print(f"    SKIP {game_id}: incomplete team info")
            continue

        home_team_won = home_id == winner_id

        if idx % 10 == 0 or idx == total:
            print(f"    [{idx}/{total}] game {game_id}")

        result = analyze_game(game_id, home_team_won)
        if not result["ok"]:
            continue

        stats["total_games"] += 1
        if result["trailed_5"]:
            stats["comebacks_from_5"] += 1
        if result["trailed_10"]:
            stats["comebacks_from_10"] += 1

    return stats


# ---------------------------------------------------------------------------
# Demo mode (no network required)
# ---------------------------------------------------------------------------

def demo_mode():
    """
    Run with synthetic data that mirrors realistic NBA playoff comeback rates.
    Useful for verifying output format without live API access.
    """
    print("Running in DEMO MODE — synthetic data only\n")
    # Approximate real-world rates: ~35-40% trail 5+, ~15-20% trail 10+
    synthetic = [
        {"season": "2022-23", "total_games": 82, "comebacks_from_5": 29, "comebacks_from_10": 13},
        {"season": "2023-24", "total_games": 85, "comebacks_from_5": 32, "comebacks_from_10": 15},
        {"season": "2024-25", "total_games": 78, "comebacks_from_5": 27, "comebacks_from_10": 11},
    ]
    return synthetic


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def print_results(all_stats: list[dict]) -> None:
    rows = []
    for s in all_stats:
        total = s["total_games"]
        c5 = s["comebacks_from_5"]
        c10 = s["comebacks_from_10"]
        pct5 = (c5 / total * 100) if total > 0 else 0.0
        pct10 = (c10 / total * 100) if total > 0 else 0.0
        rows.append(
            [
                s["season"],
                total,
                c5,
                f"{pct5:.1f}%",
                c10,
                f"{pct10:.1f}%",
            ]
        )

    headers = [
        "Season",
        "Total Games",
        "Trailed 5+ & Won",
        "%",
        "Trailed 10+ & Won",
        "%",
    ]
    print(tabulate(rows, headers=headers, tablefmt="github"))
    print()


def main():
    demo = "--demo" in sys.argv

    if demo:
        all_stats = demo_mode()
    else:
        all_stats = []
        for season in SEASONS:
            print(f"\n=== Season {season} ===")
            stats = process_season(season)
            all_stats.append(stats)
            print(
                f"  Done — {stats['total_games']} games, "
                f"{stats['comebacks_from_5']} trailed 5+, "
                f"{stats['comebacks_from_10']} trailed 10+"
            )

    print("\n========== NBA PLAYOFF Q4 COMEBACK ANALYSIS ==========\n")
    print_results(all_stats)


if __name__ == "__main__":
    main()

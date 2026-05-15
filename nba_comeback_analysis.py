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


def analyze_game(game_id: str) -> dict:
    """
    Fetch Q4 play-by-play and determine, independently for each team,
    whether they trailed by 5+ or 10+ at any point in the 4th quarter.

    Margin is always from the home team's perspective (positive = home leading).

    Returns dict:
        ok                 – play-by-play was valid
        home_trailed_5     – home team was ever down 5+ in Q4
        home_trailed_10    – home team was ever down 10+ in Q4
        away_trailed_5     – away team was ever down 5+ in Q4
        away_trailed_10    – away team was ever down 10+ in Q4
    """
    result = {
        "ok": False,
        "home_trailed_5": False,
        "home_trailed_10": False,
        "away_trailed_5": False,
        "away_trailed_10": False,
    }

    try:
        pbp = fetch_play_by_play(game_id)
    except Exception as exc:
        print(f"    WARNING: PBP fetch failed for {game_id}: {exc}")
        return result

    if pbp.empty:
        print(f"    WARNING: Empty PBP for {game_id}")
        return result

    # PERIOD may come back as int or str depending on API version
    q4 = pbp[pbp["PERIOD"].astype(str) == "4"].copy()
    if q4.empty:
        print(f"    WARNING: No Q4 rows for {game_id} (periods seen: {pbp['PERIOD'].unique().tolist()})")
        return result

    result["ok"] = True

    for val in q4["SCOREMARGIN"]:
        margin = parse_margin(val)
        if margin is None:
            continue

        # margin <= -N  →  home is trailing by N+
        if margin <= -5:
            result["home_trailed_5"] = True
        if margin <= -10:
            result["home_trailed_10"] = True

        # margin >= +N  →  away is trailing by N+
        if margin >= 5:
            result["away_trailed_5"] = True
        if margin >= 10:
            result["away_trailed_10"] = True

    return result


def process_season(season: str) -> dict:
    """Run the full analysis for one season. Returns aggregated stats."""
    stats = {
        "season": season,
        "total_games": 0,
        "games_trailed_5": 0,   # games where any team trailed 5+ in Q4
        "comeback_wins_5": 0,   # of those, games won by the trailing team
        "games_trailed_10": 0,
        "comeback_wins_10": 0,
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
    skipped_info = 0
    failed_pbp = 0
    print(f"  Processing {total} games for {season}...")

    for idx, game_id in enumerate(game_ids, 1):
        info = game_info.get(game_id, {})
        home_id = info.get("home_team_id")
        winner_id = info.get("winner_team_id")

        if home_id is None or winner_id is None:
            skipped_info += 1
            if skipped_info <= 3:  # avoid flooding console
                print(f"    SKIP {game_id}: incomplete team info")
            continue

        home_team_won = home_id == winner_id

        if idx % 10 == 0 or idx == total:
            print(f"    [{idx}/{total}] game {game_id}")

        result = analyze_game(game_id)
        if not result["ok"]:
            failed_pbp += 1
            continue

        stats["total_games"] += 1

        any_trailed_5 = result["home_trailed_5"] or result["away_trailed_5"]
        any_trailed_10 = result["home_trailed_10"] or result["away_trailed_10"]

        # Did the team that trailed eventually win?
        winner_trailed_5 = (
            (home_team_won and result["home_trailed_5"]) or
            (not home_team_won and result["away_trailed_5"])
        )
        winner_trailed_10 = (
            (home_team_won and result["home_trailed_10"]) or
            (not home_team_won and result["away_trailed_10"])
        )

        if any_trailed_5:
            stats["games_trailed_5"] += 1
        if winner_trailed_5:
            stats["comeback_wins_5"] += 1

        if any_trailed_10:
            stats["games_trailed_10"] += 1
        if winner_trailed_10:
            stats["comeback_wins_10"] += 1

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
    # Approximate real-world rates for illustration
    synthetic = [
        {"season": "2022-23", "total_games": 82, "games_trailed_5": 61, "comeback_wins_5": 29, "games_trailed_10": 34, "comeback_wins_10": 13},
        {"season": "2023-24", "total_games": 85, "games_trailed_5": 64, "comeback_wins_5": 32, "games_trailed_10": 37, "comeback_wins_10": 15},
        {"season": "2024-25", "total_games": 78, "games_trailed_5": 58, "comeback_wins_5": 27, "games_trailed_10": 31, "comeback_wins_10": 11},
    ]
    return synthetic


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def print_results(all_stats: list[dict]) -> None:
    rows = []
    for s in all_stats:
        total = s["total_games"]
        t5  = s["games_trailed_5"]
        w5  = s["comeback_wins_5"]
        t10 = s["games_trailed_10"]
        w10 = s["comeback_wins_10"]
        pct5  = (w5  / t5  * 100) if t5  > 0 else 0.0
        pct10 = (w10 / t10 * 100) if t10 > 0 else 0.0
        rows.append([
            s["season"],
            total,
            t5,
            w5,
            f"{pct5:.1f}%",
            t10,
            w10,
            f"{pct10:.1f}%",
        ])

    headers = [
        "Season",
        "Total Games",
        "Games w/ 5+ trail in Q4",
        "Comeback Wins (5+)",
        "Win %",
        "Games w/ 10+ trail in Q4",
        "Comeback Wins (10+)",
        "Win %",
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
            s = stats
            print(
                f"  Done — {s['total_games']} games analysed "
                f"(skipped {skipped_info} missing team info, {failed_pbp} PBP failures) | "
                f"trailed 5+: {s['games_trailed_5']} games, {s['comeback_wins_5']} wins | "
                f"trailed 10+: {s['games_trailed_10']} games, {s['comeback_wins_10']} wins"
            )

    print("\n========== NBA PLAYOFF Q4 COMEBACK ANALYSIS ==========\n")
    print_results(all_stats)


if __name__ == "__main__":
    main()

"""
nba_fetcher.py
--------------
Récupère les données de matchs NBA via l'API officielle (gratuit, sans clé).
Fournit : pace, offensive/defensive rating, forme récente des équipes.
"""

import pandas as pd
import time
from nba_api.stats.endpoints import (
    leaguegamefinder,
    teamdashboardbygeneralsplits,
    leaguedashteamstats,
)
from nba_api.stats.static import teams as nba_teams_static

# ── Constantes ──────────────────────────────────────────────────────────────
CURRENT_SEASON = "2024-25"
RECENT_GAMES   = 10          # nb de matchs pour calculer la "forme" d'une équipe
API_DELAY      = 0.6         # secondes entre chaque appel (évite le rate-limit NBA)


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_all_teams() -> dict:
    """Retourne un dict {nom_equipe: team_id} pour toutes les équipes NBA."""
    teams = nba_teams_static.get_teams()
    return {t["full_name"]: t["id"] for t in teams}


def get_team_id(team_name: str) -> int | None:
    """Trouve l'ID d'une équipe depuis son nom (partiel ou complet)."""
    all_teams = get_all_teams()
    for name, tid in all_teams.items():
        if team_name.lower() in name.lower():
            return tid
    return None


# ── Statistiques avancées de la ligue ────────────────────────────────────────

def get_league_advanced_stats(season: str = CURRENT_SEASON) -> pd.DataFrame:
    """
    Récupère les stats avancées de toutes les équipes NBA pour la saison.
    Colonnes clés : TEAM_NAME, PACE, OFF_RATING, DEF_RATING, NET_RATING, W_PCT
    """
    print(f"[NBA] Chargement des stats avancées ({season})...")
    time.sleep(API_DELAY)

    stats = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        measure_type_detailed_defense="Advanced",
    )
    df = stats.get_data_frames()[0]

    # Colonnes utiles pour notre modèle
    cols = [
        "TEAM_ID", "TEAM_NAME",
        "W", "L", "W_PCT",
        "PACE", "OFF_RATING", "DEF_RATING", "NET_RATING",
        "EFG_PCT", "TS_PCT",
    ]
    # Garde seulement les colonnes qui existent
    available = [c for c in cols if c in df.columns]
    df = df[available].copy()

    print(f"[NBA] {len(df)} équipes chargées.")
    return df


# ── Forme récente d'une équipe ────────────────────────────────────────────────

def get_team_recent_form(team_id: int, n_games: int = RECENT_GAMES) -> dict:
    """
    Calcule la forme récente d'une équipe sur ses n derniers matchs.
    Retourne : pts_for_avg, pts_against_avg, win_rate_recent, total_avg
    """
    time.sleep(API_DELAY)

    finder = leaguegamefinder.LeagueGameFinder(
        team_id_nullable=team_id,
        season_nullable=CURRENT_SEASON,
        season_type_nullable="Regular Season",
    )
    df = finder.get_data_frames()[0]

    if df.empty:
        return {}

    # Trier par date décroissante, prendre les n derniers
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE", ascending=False).head(n_games)

    # Identifier les matchs à domicile/extérieur
    df["IS_HOME"] = ~df["MATCHUP"].str.contains("@")
    df["WON"]     = df["WL"] == "W"

    return {
        "team_id":          team_id,
        "games_analyzed":   len(df),
        "pts_for_avg":      round(df["PTS"].mean(), 1),
        "pts_against_avg":  round((df["PTS"] - df["PLUS_MINUS"]).mean(), 1),  # PTS adversaire
        "plus_minus_avg":   round(df["PLUS_MINUS"].mean(), 1),
        "win_rate_recent":  round(df["WON"].mean(), 3),
        "total_avg":        round(df["PTS"].mean() + (df["PTS"] - df["PLUS_MINUS"]).mean(), 1),
        "home_pts_avg":     round(df[df["IS_HOME"]]["PTS"].mean(), 1) if df["IS_HOME"].any() else None,
        "away_pts_avg":     round(df[~df["IS_HOME"]]["PTS"].mean(), 1) if (~df["IS_HOME"]).any() else None,
    }


# ── Prédiction du total d'un match ───────────────────────────────────────────

def predict_match_total(home_team_name: str, away_team_name: str,
                         league_df: pd.DataFrame | None = None) -> dict:
    """
    Prédit le total de points attendu pour un match (home vs away).

    Méthode : moyenne pondérée entre :
      - OFF_RATING domicile + DEF_RATING extérieur (stats de saison complète)
      - Forme récente des deux équipes (n derniers matchs)

    Retourne un dict avec le total prédit et les détails du calcul.
    """
    # Charger les stats de ligue si non fournies
    if league_df is None:
        league_df = get_league_advanced_stats()

    # Récupérer les IDs
    home_id = get_team_id(home_team_name)
    away_id = get_team_id(away_team_name)

    if not home_id or not away_id:
        raise ValueError(f"Équipe introuvable : {home_team_name} ou {away_team_name}")

    # Stats de saison des deux équipes
    home_stats = league_df[league_df["TEAM_ID"] == home_id].iloc[0] if home_id in league_df["TEAM_ID"].values else None
    away_stats = league_df[league_df["TEAM_ID"] == away_id].iloc[0] if away_id in league_df["TEAM_ID"].values else None

    # Forme récente
    print(f"[NBA] Forme récente : {home_team_name}...")
    home_form = get_team_recent_form(home_id)
    print(f"[NBA] Forme récente : {away_team_name}...")
    away_form = get_team_recent_form(away_id)

    # ── Calcul du total prédit ────────────────────────────────────────────────
    # Méthode 1 : OFF_RATING + DEF_RATING (sur toute la saison, normalisé en points)
    # OFF_RATING = points pour 100 possessions → on normalise avec le PACE
    total_season = None
    if home_stats is not None and away_stats is not None:
        pace         = (home_stats.get("PACE", 98) + away_stats.get("PACE", 98)) / 2
        home_off_pts = (home_stats.get("OFF_RATING", 110) / 100) * (pace / 2)
        away_off_pts = (away_stats.get("OFF_RATING", 110) / 100) * (pace / 2)
        home_def_pts = (home_stats.get("DEF_RATING", 110) / 100) * (pace / 2)
        away_def_pts = (away_stats.get("DEF_RATING", 110) / 100) * (pace / 2)

        home_pts_pred = (home_off_pts + away_def_pts) / 2
        away_pts_pred = (away_off_pts + home_def_pts) / 2
        total_season  = round(home_pts_pred + away_pts_pred, 1)

    # Méthode 2 : forme récente brute
    total_recent = None
    if home_form and away_form:
        total_recent = round(
            (home_form["pts_for_avg"] + away_form["pts_for_avg"] +
             home_form["pts_against_avg"] + away_form["pts_against_avg"]) / 2,
            1
        )

    # Pondération finale : 60% saison, 40% forme récente
    if total_season and total_recent:
        predicted_total = round(0.6 * total_season + 0.4 * total_recent, 1)
    elif total_season:
        predicted_total = total_season
    elif total_recent:
        predicted_total = total_recent
    else:
        predicted_total = None

    return {
        "home_team":       home_team_name,
        "away_team":       away_team_name,
        "predicted_total": predicted_total,
        "total_season":    total_season,
        "total_recent":    total_recent,
        "home_form":       home_form,
        "away_form":       away_form,
    }


# ── Test rapide ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  NBA FETCHER — Test")
    print("=" * 55)

    # Stats avancées de la ligue
    df = get_league_advanced_stats()
    print("\nTop 5 équipes par NET_RATING :")
    print(df.nlargest(5, "NET_RATING")[["TEAM_NAME", "NET_RATING", "PACE", "OFF_RATING", "DEF_RATING"]].to_string(index=False))

    # Prédiction d'un match exemple
    print("\n" + "=" * 55)
    result = predict_match_total("Boston Celtics", "Golden State Warriors", df)
    print(f"\nMatch : {result['home_team']} vs {result['away_team']}")
    print(f"  Total prédit (saison)  : {result['total_season']}")
    print(f"  Total prédit (forme)   : {result['total_recent']}")
    print(f"  Total prédit (final)   : {result['predicted_total']}")
    print(f"  Forme Boston           : {result['home_form'].get('pts_for_avg')} pts/match (récent)")
    print(f"  Forme Golden State     : {result['away_form'].get('pts_for_avg')} pts/match (récent)")

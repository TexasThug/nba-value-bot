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
B2B_PENALTY    = 2.5         # points retirés du total prédit par équipe en back-to-back


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
        "last_game_date":   df.iloc[0]["GAME_DATE"].strftime("%Y-%m-%d"),
    }


# ── Prédiction du total d'un match ───────────────────────────────────────────

def _is_back_to_back(last_game_date: str | None, game_date: str | None) -> bool:
    """True si l'équipe a joué la veille du match ciblé."""
    if not last_game_date or not game_date:
        return False
    from datetime import datetime
    try:
        delta = datetime.strptime(game_date, "%Y-%m-%d") - datetime.strptime(last_game_date, "%Y-%m-%d")
        return delta.days == 1
    except ValueError:
        return False


def predict_match_total(home_team_name: str, away_team_name: str,
                         league_df: pd.DataFrame | None = None,
                         game_date: str | None = None) -> dict:
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

    # ── Méthode 1 : stats saison — formule multiplicative ────────────────────
    #
    # Ancienne formule (additive) : (home_OFF + away_DEF) / 2
    #   → régresse vers la moyenne, std dev prédictions = 4.7 pts (actuals = 20 pts)
    #
    # Nouvelle formule (multiplicative) :
    #   E[home_score] = home_OFF × (away_DEF / league_avg_DEF) × pace / 100
    #   Standard en NBA analytics. Préserve les interactions extrêmes :
    #   bonne attaque × mauvaise défense → score élevé (pas juste leur moyenne).
    #
    # Pace : moyenne harmonique plutôt qu'arithmétique.
    #   Une équipe lente contraint plus le tempo qu'une équipe rapide ne peut
    #   l'accélérer. La moyenne harmonique est plus proche du pace réel observé.
    #
    total_season  = None
    spread_season = None
    if home_stats is not None and away_stats is not None:
        h_pace = home_stats.get("PACE", 98)
        a_pace = away_stats.get("PACE", 98)
        # Moyenne harmonique du pace (favorise le tempo lent)
        pace = 2 * h_pace * a_pace / (h_pace + a_pace) if (h_pace + a_pace) > 0 else 98

        h_off = home_stats.get("OFF_RATING", 110)
        a_off = away_stats.get("OFF_RATING", 110)
        h_def = home_stats.get("DEF_RATING", 110)
        a_def = away_stats.get("DEF_RATING", 110)

        # Moyenne ligue : base de normalisation pour l'ajustement défensif
        league_avg_def = float(league_df["DEF_RATING"].mean()) if "DEF_RATING" in league_df.columns else 115.0

        # Score prédit : OFF de l'attaquant × ajustement défensif de l'adversaire
        home_pts_pred = h_off * (a_def / league_avg_def) * pace / 100
        away_pts_pred = a_off * (h_def / league_avg_def) * pace / 100
        total_season  = round(home_pts_pred + away_pts_pred, 1)
        spread_season = round(home_pts_pred - away_pts_pred, 1)

    # ── Méthode 2 : forme récente ─────────────────────────────────────────────
    # Total  : (pts_for + pts_against) pour chaque équipe, moyennés
    # Spread : (plus_minus_home - plus_minus_away) / 2
    #   → formule équivalente à (E[home_score] - E[away_score]) dans ce matchup
    total_recent  = None
    spread_recent = None
    if home_form and away_form:
        total_via_home = home_form["pts_for_avg"] + home_form["pts_against_avg"]
        total_via_away = away_form["pts_for_avg"] + away_form["pts_against_avg"]
        total_recent   = round((total_via_home + total_via_away) / 2, 1)
        spread_recent  = round(
            (home_form["plus_minus_avg"] - away_form["plus_minus_avg"]) / 2, 1
        )

    # ── Pondération finale : 60% saison, 40% forme récente ───────────────────
    if total_season and total_recent:
        predicted_total  = round(0.6 * total_season  + 0.4 * total_recent,  1)
        predicted_spread = round(0.6 * spread_season + 0.4 * spread_recent, 1)
    elif total_season:
        predicted_total  = total_season
        predicted_spread = spread_season
    elif total_recent:
        predicted_total  = total_recent
        predicted_spread = spread_recent
    else:
        predicted_total  = None
        predicted_spread = None

    # ── Ajustement back-to-back ────────────────────────────────────────────
    b2b_home = _is_back_to_back(home_form.get("last_game_date"), game_date)
    b2b_away = _is_back_to_back(away_form.get("last_game_date"), game_date)
    b2b_adjustment = -B2B_PENALTY * (int(b2b_home) + int(b2b_away))

    # Ajustement total : les deux équipes marquent moins en B2B
    # Ajustement spread : home B2B → spread baisse ; away B2B → spread monte
    b2b_spread_adj = -B2B_PENALTY * int(b2b_home) + B2B_PENALTY * int(b2b_away)

    if b2b_adjustment != 0:
        b2b_who = (("domicile" if b2b_home else "") +
                   ("+" if b2b_home and b2b_away else "") +
                   ("extérieur" if b2b_away else ""))
        print(f"[NBA] Ajustement B2B ({b2b_who}) : total {b2b_adjustment:+.1f} pts | "
              f"spread {b2b_spread_adj:+.1f} pts")
        if predicted_total  is not None:
            predicted_total  = round(predicted_total  + b2b_adjustment,   1)
        if predicted_spread is not None:
            predicted_spread = round(predicted_spread + b2b_spread_adj,   1)

    return {
        "home_team":        home_team_name,
        "away_team":        away_team_name,
        "predicted_total":  predicted_total,
        "predicted_spread": predicted_spread,   # positif = home favori (ex: +7.2)
        "total_season":     total_season,
        "spread_season":    spread_season,
        "total_recent":     total_recent,
        "spread_recent":    spread_recent,
        "b2b_home":         b2b_home,
        "b2b_away":         b2b_away,
        "b2b_adjustment":   b2b_adjustment,
        "b2b_spread_adj":   b2b_spread_adj,
        "home_form":        home_form,
        "away_form":        away_form,
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

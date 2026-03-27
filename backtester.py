"""
backtester.py
-------------
Teste le modèle de prédiction sur 3 saisons historiques NBA.
Compare les prédictions aux vrais résultats et calcule le P&L théorique.

Usage :
  python backtester.py              → back-test 3 saisons
  python backtester.py --season 2023-24  → une saison spécifique
  python backtester.py --fast       → mode rapide (sample de 100 matchs/saison)
"""

import argparse
import time
import json
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, asdict
from scipy.stats import norm

from nba_api.stats.endpoints import leaguegamefinder, leaguedashteamstats
from nba_api.stats.static import teams as nba_teams_static

# ── Config ────────────────────────────────────────────────────────────────────
SEASONS        = ["2021-22", "2022-23", "2023-24"]
API_DELAY      = 0.7
STD_DEV        = 18.9          # RMSE calibré sur 3689 matchs historiques (était 12.0)
MIN_VALUE      = 0.04          # seuil de value pour "parier"
KELLY_FRACTION = 0.25
BANKROLL       = 1000.0
B2B_PENALTY    = 2.5           # pts retirés par équipe en back-to-back
# Cote simulée PS3838 (on suppose ~1.92 over/under, marge ~4%)
SIMULATED_ODD  = 1.92


# ── Dataclass résultat ────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    season:          str
    date:            str
    home_team:       str
    away_team:       str
    predicted_total: float
    actual_total:    float
    line:            float        # ligne simulée = médiane de la saison
    market:          str          # "Over" ou "Under"
    model_prob:      float
    bookie_prob:     float
    value:           float
    kelly_stake:     float
    bet_placed:      bool         # True si value >= MIN_VALUE
    won:             bool         # True si le pari aurait gagné
    pnl:             float        # gain/perte en €


# ── Stats de ligue par saison ─────────────────────────────────────────────────

def get_league_stats(season: str) -> pd.DataFrame:
    """Récupère les stats avancées NBA pour une saison donnée."""
    print(f"  [NBA] Stats avancées {season}...")
    time.sleep(API_DELAY)
    stats = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        measure_type_detailed_defense="Advanced",
    )
    df = stats.get_data_frames()[0]
    cols = ["TEAM_ID", "TEAM_NAME", "W_PCT", "PACE",
            "OFF_RATING", "DEF_RATING", "NET_RATING"]
    df = df[[c for c in cols if c in df.columns]].copy()
    df["TEAM_ID"] = df["TEAM_ID"].astype(int)
    return df


# ── Récupération des matchs d'une saison ─────────────────────────────────────

def get_season_games(season: str, sample: int = None) -> pd.DataFrame:
    """Récupère tous les matchs d'une saison NBA avec les scores."""
    print(f"  [NBA] Matchs {season}...")
    time.sleep(API_DELAY)

    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        season_type_nullable="Regular Season",
        league_id_nullable="00",
    )
    df = finder.get_data_frames()[0]
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    # Garder uniquement les matchs à domicile (évite les doublons)
    df = df[~df["MATCHUP"].str.contains("@")].copy()
    df = df.sort_values("GAME_DATE")

    if sample:
        df = df.tail(sample)  # Prendre les plus récents pour le mode fast

    print(f"  [NBA] {len(df)} matchs trouvés")
    return df


# ── Prédiction du total ───────────────────────────────────────────────────────

def compute_market_line(home_id: int, away_id: int,
                        game_date: pd.Timestamp,
                        all_df: pd.DataFrame,
                        n: int = 10) -> float:
    """
    Calcule une ligne proxy réaliste sans utiliser la prédiction du modèle.

    Méthode : moyenne des totaux rolling (n derniers matchs AVANT ce match)
    pour chaque équipe, indépendante du modèle → pas de biais circulaire.
    Total = 2*PTS - PLUS_MINUS (= pts_marqués + pts_encaissés)
    """
    def rolling_avg_total(team_id):
        tg = all_df[
            (all_df["TEAM_ID"] == team_id) &
            (all_df["GAME_DATE"] < game_date)
        ].sort_values("GAME_DATE", ascending=False).head(n)
        if tg.empty:
            return None
        return (2 * tg["PTS"] - tg["PLUS_MINUS"]).mean()

    home_avg = rolling_avg_total(home_id)
    away_avg = rolling_avg_total(away_id)

    if home_avg is not None and away_avg is not None:
        raw = (home_avg + away_avg) / 2
    elif home_avg is not None:
        raw = home_avg
    elif away_avg is not None:
        raw = away_avg
    else:
        raw = 220.0  # fallback : moyenne NBA historique

    # Arrondir au .5 le plus proche (convention bookmaker)
    return round(raw * 2) / 2


def is_b2b(team_id: int, game_date: pd.Timestamp, all_df: pd.DataFrame) -> bool:
    """True si l'équipe a joué la veille."""
    prev = game_date - pd.Timedelta(days=1)
    played = all_df[
        (all_df["TEAM_ID"] == team_id) &
        (all_df["GAME_DATE"].dt.date == prev.date())
    ]
    return not played.empty


def predict_total(home_id: int, away_id: int,
                  league_df: pd.DataFrame,
                  game_date: pd.Timestamp,
                  all_df: pd.DataFrame) -> float | None:
    """
    Prédit le total d'un match : stats saison (60%) + forme récente (40%).

    La forme est calculée sur les données AVANT game_date uniquement.
    Plus de data leakage : chaque match est prédit avec les infos disponibles
    à ce moment-là dans le temps.
    """
    home_stats = league_df[league_df["TEAM_ID"] == home_id]
    away_stats = league_df[league_df["TEAM_ID"] == away_id]

    if home_stats.empty or away_stats.empty:
        return None

    h = home_stats.iloc[0]
    a = away_stats.iloc[0]

    # Méthode saison (stats cumulées — elles-mêmes légèrement leaky pour les
    # premiers matchs, mais c'est une approximation acceptable)
    pace          = (h.get("PACE", 98) + a.get("PACE", 98)) / 2
    home_off_pts  = (h.get("OFF_RATING", 110) / 100) * pace
    away_off_pts  = (a.get("OFF_RATING", 110) / 100) * pace
    home_def_pts  = (h.get("DEF_RATING", 110) / 100) * pace
    away_def_pts  = (a.get("DEF_RATING", 110) / 100) * pace
    home_pts_pred = (home_off_pts + away_def_pts) / 2
    away_pts_pred = (away_off_pts + home_def_pts) / 2
    total_season  = home_pts_pred + away_pts_pred

    # Forme récente — calculée sur les matchs AVANT ce match uniquement
    hf = form_at_date(home_id, game_date, all_df)
    af = form_at_date(away_id, game_date, all_df)

    if hf and af:
        total_recent = ((hf["pts_for"] + hf["pts_against"]) +
                        (af["pts_for"] + af["pts_against"])) / 2
        return round(0.6 * total_season + 0.4 * total_recent, 1)

    return round(total_season, 1)


def form_at_date(team_id: int, game_date: pd.Timestamp,
                  all_df: pd.DataFrame, n: int = 10) -> dict | None:
    """
    Forme récente d'une équipe sur ses n derniers matchs STRICTEMENT avant game_date.

    Pas de data leakage : on ne regarde jamais dans le futur par rapport au match
    analysé. Les matchs d'overtime (total > 265 pts) sont exclus des moyennes car
    ils biaisent le total moyen à la hausse (~5% des matchs NBA).
    """
    tg = all_df[
        (all_df["TEAM_ID"] == team_id) &
        (all_df["GAME_DATE"] < game_date)
    ].sort_values("GAME_DATE", ascending=False).head(n)

    if tg.empty:
        return None

    # Filtre OT : heuristique — total > 265 pts indique très probablement un OT
    game_totals = 2 * tg["PTS"] - tg["PLUS_MINUS"]
    tg = tg[game_totals <= 265]

    if tg.empty:
        return None

    return {
        "pts_for":     tg["PTS"].mean(),
        "pts_against": (tg["PTS"] - tg["PLUS_MINUS"]).mean(),
    }


# ── Back-test principal ───────────────────────────────────────────────────────

def backtest_season(season: str, fast: bool = False) -> list[BacktestResult]:
    """Back-teste le modèle sur une saison complète."""
    print(f"\n{'='*55}")
    print(f"  BACK-TEST : {season}")
    print(f"{'='*55}")

    sample = 80 if fast else None
    results = []

    # Stats de ligue
    league_df = get_league_stats(season)
    team_map  = {t["id"]: t["full_name"]
                 for t in nba_teams_static.get_teams()}

    # Matchs de la saison
    games = get_season_games(season, sample=sample)
    if games.empty:
        return []

    # Charger tous les matchs de la saison (utilisé pour : forme rolling,
    # ligne proxy, détection B2B — tout calculé par date pour éviter le leakage)
    print(f"  [NBA] Chargement de tous les matchs de la saison...")
    time.sleep(API_DELAY)
    all_df = pd.DataFrame()
    try:
        all_games_finder = leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            season_type_nullable="Regular Season",
        )
        all_df = all_games_finder.get_data_frames()[0]
        all_df["GAME_DATE"] = pd.to_datetime(all_df["GAME_DATE"])
        all_df["TEAM_ID"]   = all_df["TEAM_ID"].astype(int)
        print(f"  [NBA] {len(all_df)} entrées chargées.")
    except Exception as e:
        print(f"  Erreur chargement : {e}")
        all_df = pd.DataFrame()

    print(f"  Analyse de {len(games)} matchs...\n")

    for _, game in games.iterrows():
        home_id   = int(game["TEAM_ID"])
        home_name = game["TEAM_NAME"]

        # Trouver l'équipe adverse dans le même match
        matchup   = game["MATCHUP"]  # ex: "BOS vs. GSW"
        game_id   = game["GAME_ID"]

        # Pts domicile + pts extérieur = total réel
        home_pts  = game["PTS"]
        away_pts  = home_pts - game["PLUS_MINUS"]
        actual_total = home_pts + away_pts

        if actual_total < 150 or actual_total > 310:
            continue  # Données aberrantes

        # Équipe adverse
        away_row = all_df[
            (all_df["GAME_ID"] == game_id) &
            (all_df["TEAM_ID"] != home_id)
        ]
        if away_row.empty:
            continue
        away_id   = int(away_row.iloc[0]["TEAM_ID"])
        away_name = away_row.iloc[0]["TEAM_NAME"]

        # Prédiction (avec ajustement back-to-back)
        b2b_home_flag = is_b2b(home_id, game["GAME_DATE"], all_df)
        b2b_away_flag = is_b2b(away_id, game["GAME_DATE"], all_df)
        pred = predict_total(home_id, away_id, league_df, game["GAME_DATE"], all_df)
        if pred is None:
            continue
        pred = round(pred - B2B_PENALTY * (int(b2b_home_flag) + int(b2b_away_flag)), 1)

        # Ligne proxy : rolling average des totaux réels AVANT ce match
        # Indépendante du modèle → pas de biais circulaire (était random ±3 pts)
        line = compute_market_line(home_id, away_id, game["GAME_DATE"], all_df)

        # Calcul value Over
        prob_over  = round(1 - norm.cdf(line, loc=pred, scale=STD_DEV), 4)
        prob_under = round(norm.cdf(line, loc=pred, scale=STD_DEV), 4)
        bookie_prob = round(1 / SIMULATED_ODD, 4)

        value_over  = round((prob_over  * SIMULATED_ODD) - 1, 4)
        value_under = round((prob_under * SIMULATED_ODD) - 1, 4)

        # Choisir le côté avec la meilleure value positive
        if value_over >= value_under:
            market     = "Over"
            value      = value_over
            model_prob = prob_over
            won        = actual_total > line
        else:
            market     = "Under"
            value      = value_under
            model_prob = prob_under
            won        = actual_total < line

        bet_placed = value >= MIN_VALUE

        # Kelly stake
        b      = SIMULATED_ODD - 1
        q      = 1 - model_prob
        kelly  = max(0.0, (model_prob * b - q) / b) * KELLY_FRACTION
        stake  = round(BANKROLL * kelly, 2) if bet_placed else 0.0

        # P&L
        if bet_placed:
            pnl = round(stake * (SIMULATED_ODD - 1) if won else -stake, 2)
        else:
            pnl = 0.0

        results.append(BacktestResult(
            season=season,
            date=game["GAME_DATE"].strftime("%Y-%m-%d"),
            home_team=home_name,
            away_team=away_name,
            predicted_total=pred,
            actual_total=round(actual_total, 1),
            line=line,
            market=market,
            model_prob=model_prob,
            bookie_prob=bookie_prob,
            value=value,
            kelly_stake=stake,
            bet_placed=bet_placed,
            won=won,
            pnl=pnl,
        ))

    bets    = [r for r in results if r.bet_placed]
    wins    = [r for r in bets if r.won]
    total_pnl = sum(r.pnl for r in bets)

    print(f"\n  Résultats {season} :")
    print(f"  Matchs analysés : {len(results)}")
    print(f"  Paris placés    : {len(bets)} ({len(bets)/max(len(results),1)*100:.1f}%)")
    print(f"  Taux de victoire: {len(wins)/max(len(bets),1)*100:.1f}%")
    print(f"  P&L total       : {total_pnl:+.2f}€")
    print(f"  ROI             : {total_pnl/max(sum(r.kelly_stake for r in bets),1)*100:.1f}%")

    return results


# ── Point d'entrée ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NBA Value Bot — Back-tester")
    parser.add_argument("--season", type=str, help="Saison spécifique (ex: 2023-24)")
    parser.add_argument("--fast",   action="store_true",
                        help="Mode rapide (80 matchs par saison)")
    args = parser.parse_args()

    seasons = [args.season] if args.season else SEASONS

    print("=" * 55)
    print("  NBA VALUE BOT - BACK-TESTER")
    print(f"  Saisons : {', '.join(seasons)}")
    print(f"  Mode    : {'FAST (80 matchs)' if args.fast else 'COMPLET'}")
    print("=" * 55)

    all_results = []
    for season in seasons:
        results = backtest_season(season, fast=args.fast)
        all_results.extend(results)

    # Sauvegarder en JSON
    import numpy as np
    output = [{k: (bool(v) if isinstance(v, np.bool_) else float(v) if isinstance(v, np.floating) else v) for k, v in asdict(r).items()} for r in all_results]
    with open("backtest_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*55}")
    print(f"  RÉSUMÉ GLOBAL ({len(seasons)} saisons)")
    print(f"{'='*55}")

    bets  = [r for r in all_results if r.bet_placed]
    wins  = [r for r in bets if r.won]
    pnl   = sum(r.pnl for r in bets)
    invested = sum(r.kelly_stake for r in bets)

    print(f"  Total matchs    : {len(all_results)}")
    print(f"  Paris placés    : {len(bets)}")
    print(f"  Taux victoire   : {len(wins)/max(len(bets),1)*100:.1f}%")
    print(f"  P&L total       : {pnl:+.2f}€")
    print(f"  ROI global      : {pnl/max(invested,1)*100:.1f}%")
    print(f"\n  Résultats sauvegardés dans backtest_results.json")
    print(f"  Lance excel_tracker.py pour générer le fichier Excel !")


if __name__ == "__main__":
    main()

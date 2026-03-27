"""
odds_fetcher.py
---------------
Récupère les cotes NBA en temps réel via la librairie ps3838api.
Authentification : username + password Asian Connect via variables d'env.

pip install ps3838api
"""

import os
from dotenv import load_dotenv
load_dotenv()

from ps3838api.api.client import PinnacleClient
from datetime import datetime

client = PinnacleClient()

SPORT_ID   = 4    # Basketball
LEAGUE_NBA = 487  # NBA


def odd_to_prob(odd: float) -> float:
    """Probabilité implicite brute (inclut le vig). Ne pas utiliser pour comparer au modèle."""
    return round(1 / odd, 4) if odd > 1 else 0.0

def prob_to_odd(prob: float) -> float:
    return round(1 / prob, 2) if prob > 0 else 0.0

def odd_to_fair_prob(over_odd: float, under_odd: float) -> tuple[float, float]:
    """
    Retire le vig pour obtenir la probabilité fair de chaque côté.

    Avec 1.92/1.92 : brut = 52.08%/52.08% (somme = 104.2%) → fair = 50%/50%
    Avec 1.93/1.91 : fair_over = 49.7%, fair_under = 50.3%

    C'est la vraie probabilité que le bookmaker assigne, sans sa marge.
    Utile pour affichage et pour savoir si le modèle a vraiment un edge.
    Note : la formule value = model_prob × odd - 1 reste inchangée.
    """
    p_over  = 1 / over_odd  if over_odd  > 1 else 0.0
    p_under = 1 / under_odd if under_odd > 1 else 0.0
    total   = p_over + p_under
    if total <= 0:
        return 0.5, 0.5
    return round(p_over / total, 4), round(p_under / total, 4)


def get_nba_fixtures() -> list[dict]:
    print("[PS3838] Récupération des fixtures NBA...")
    try:
        resp = client.get_fixtures(sport_id=SPORT_ID, league_ids=[LEAGUE_NBA])
    except Exception as e:
        print(f"[PS3838] Erreur fixtures : {e}")
        return []

    fixtures = []
    for league in resp.get("league", []):
        for event in league.get("events", []):
            if event.get("liveStatus", 0) == 1:
                continue
            if event.get("status", "O") != "O":
                continue
            starts = event.get("starts", "")
            try:
                dt       = datetime.fromisoformat(starts.replace("Z", "+00:00"))
                date_str = dt.strftime("%d/%m %H:%M")
            except Exception:
                date_str = starts
            fixtures.append({
                "event_id":  event["id"],
                "home_team": event.get("home", ""),
                "away_team": event.get("away", ""),
                "starts":    starts,
                "date":      date_str,
            })

    print(f"[PS3838] {len(fixtures)} matchs NBA trouvés")
    return fixtures


def get_nba_odds_raw() -> dict:
    print("[PS3838] Récupération des cotes NBA...")
    try:
        return client.get_odds(sport_id=SPORT_ID, league_ids=[LEAGUE_NBA])
    except Exception as e:
        print(f"[PS3838] Erreur cotes : {e}")
        return {}


def parse_totals(fixtures: list[dict], odds_data: dict) -> list[dict]:
    fixture_map = {f["event_id"]: f for f in fixtures}
    results     = []

    for league in odds_data.get("leagues", []):
        for event in league.get("events", []):
            event_id = event["id"]
            fixture  = fixture_map.get(event_id)
            if not fixture:
                continue
            for period in event.get("periods", []):
                if period.get("number") != 0:
                    continue
                if period.get("status") != 1:
                    continue
                totals = period.get("totals", [])
                if not totals:
                    continue
                main_total = next(
                    (t for t in totals if not t.get("altLineId")),
                    totals[0]
                )
                line      = main_total.get("points", 0)
                over_odd  = main_total.get("over", 0)
                under_odd = main_total.get("under", 0)
                if not line or not over_odd or not under_odd:
                    continue
                results.append({
                    "event_id":     event_id,
                    "home_team":    fixture["home_team"],
                    "away_team":    fixture["away_team"],
                    "date":         fixture["date"],
                    "total_line":   line,
                    "best_over":    {"bookmaker": "PS3838", "point": line, "price": over_odd},
                    "best_under":   {"bookmaker": "PS3838", "point": line, "price": under_odd},
                    "n_bookmakers": 1,
                    "raw_totals":   totals,
                })

    print(f"[PS3838] {len(results)} matchs avec cotes totals")
    return results


def parse_spreads(fixtures: list[dict], odds_data: dict) -> list[dict]:
    """
    Parse le marché spreads (handicap asiatique) depuis les données PS3838.

    Convention : spread_line = handicap appliqué à l'équipe domicile.
      spread_line = -7.5  →  home donne 7.5 pts (favori)
      spread_line = +4.5  →  home reçoit 4.5 pts (outsider)

    Pour couvrir en tant que home sur -7.5 : home doit gagner de 8+ pts.
    """
    fixture_map = {f["event_id"]: f for f in fixtures}
    results = []

    for league in odds_data.get("leagues", []):
        for event in league.get("events", []):
            event_id = event["id"]
            fixture  = fixture_map.get(event_id)
            if not fixture:
                continue
            for period in event.get("periods", []):
                if period.get("number") != 0:
                    continue
                if period.get("status") != 1:
                    continue
                spreads = period.get("spreads", [])
                if not spreads:
                    continue
                # Ligne principale (pas de ligne alternative)
                main = next(
                    (s for s in spreads if not s.get("altLineId")),
                    spreads[0]
                )
                hdp      = main.get("hdp")   # handicap home (négatif = favori)
                home_odd = main.get("home")  # cote home covering
                away_odd = main.get("away")  # cote away covering
                if hdp is None or not home_odd or not away_odd:
                    continue
                results.append({
                    "event_id":    event_id,
                    "home_team":   fixture["home_team"],
                    "away_team":   fixture["away_team"],
                    "date":        fixture["date"],
                    "starts":      fixture["starts"],
                    "spread_line": hdp,
                    "best_home":   {"bookmaker": "PS3838", "point": hdp,  "price": home_odd},
                    "best_away":   {"bookmaker": "PS3838", "point": -hdp, "price": away_odd},
                    "n_bookmakers": 1,
                    "raw_spreads": spreads,
                })

    print(f"[PS3838] {len(results)} matchs avec cotes spreads")
    return results


def get_nba_odds_parsed() -> list[dict]:
    fixtures  = get_nba_fixtures()
    if not fixtures:
        return []
    odds_data = get_nba_odds_raw()
    if not odds_data:
        return []
    return parse_totals(fixtures, odds_data)


def get_nba_odds_and_spreads() -> tuple[list[dict], list[dict]]:
    """
    Retourne (totals, spreads) en un seul appel API.
    Utiliser cette fonction dans value_bot.py pour éviter les double-appels.
    """
    fixtures = get_nba_fixtures()
    if not fixtures:
        return [], []
    odds_data = get_nba_odds_raw()
    if not odds_data:
        return [], []
    totals  = parse_totals(fixtures, odds_data)
    spreads = parse_spreads(fixtures, odds_data)
    return totals, spreads


if __name__ == "__main__":
    print("=" * 55)
    print("  PS3838 ODDS FETCHER — Test")
    print("=" * 55)
    try:
        balance = client.get_client_balance()
        print(f"\n Connecté ! Solde : {balance.get('availableBalance', '?')} {balance.get('currency', '')}")
    except Exception as e:
        print(f"\n Erreur de connexion : {e}")
        print("  Vérifie PS3838_USERNAME et PS3838_PASSWORD dans ton .env")
        exit(1)

    totals = get_nba_odds_parsed()
    if not totals:
        print("\nAucun match NBA disponible pour le moment.")
    else:
        print(f"\n{len(totals)} matchs disponibles :\n")
        for m in totals[:5]:
            print(f"  {m['date']} | {m['away_team']} @ {m['home_team']}")
            print(f"    Ligne  : {m['total_line']} pts")
            print(f"    Over   : {m['best_over']['price']}")
            print(f"    Under  : {m['best_under']['price']}")
            print()

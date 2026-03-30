"""
value_bot.py
------------
Cerveau du bot : compare les prédictions NBA avec les cotes bookmakers,
détecte les value bets, et envoie des alertes Telegram.

Usage :
  python value_bot.py             → scan complet
  python value_bot.py --dry-run   → affiche les value bets sans envoyer
"""

import os
import sys
import json
import time
import argparse
import requests
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict

from nba_fetcher import get_league_advanced_stats, predict_match_total
from odds_fetcher import get_nba_odds_and_spreads, odd_to_prob, odd_to_fair_prob


# ── Config ───────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN",  "REMPLACE_PAR_TON_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID","REMPLACE_PAR_TON_CHAT_ID")

# Seuils de détection
# MIN_VALUE = 0.04 était trop bas : avec std_dev=18.9, 2 pts suffisaient à
# déclencher 4% de value — le modèle n'a pas ce niveau de précision.
# À 8%, il faut ~5 pts d'écart, ce qui est plus exigeant et filtre mieux.
# Recalibrer après 500+ vrais résultats.
MIN_VALUE        = 0.08   # valeur minimum pour alerter (8%)
MIN_BOOKMAKERS   = 1      # ignorer les matchs couverts par peu de bookies
KELLY_FRACTION   = 0.25   # Kelly fractionné — conservé pour référence uniquement
BANKROLL         = 1000   # Bankroll fictive de test
FIXED_STAKE      = 10.0   # Mise fixe par bet (phase de test 2 semaines)

# Modèle de distribution des totaux NBA
# RMSE calibré sur 3689 matchs historiques (2021-24) : 18.87 pts
STD_DEV = 18.9

# Modèle de distribution des spreads NBA
# RMSE estimé ~12 pts (à recalibrer après 200+ bets réels)
# Var(spread_error) ≈ Var(home_error) + Var(away_error) - 2·Cov → ~12 pts
STD_DEV_SPREAD = 13.6   # calibré sur backtest 3 saisons (RMSE spreads = 13.57)

# Mapping noms d'équipes The Odds API → nba_api (quand ils diffèrent)
TEAM_NAME_MAP = {
    "LA Clippers":         "Los Angeles Clippers",
    "LA Lakers":           "Los Angeles Lakers",
    "GS Warriors":         "Golden State Warriors",
    "NY Knicks":           "New York Knicks",
    "NJ Nets":             "Brooklyn Nets",
    "NO Pelicans":         "New Orleans Pelicans",
    "SA Spurs":            "San Antonio Spurs",
    "OKC Thunder":         "Oklahoma City Thunder",
}

def normalize_team(name: str) -> str:
    return TEAM_NAME_MAP.get(name, name)


# ── Dataclass ValueBet ────────────────────────────────────────────────────────

@dataclass
class SpreadBet:
    home_team:        str
    away_team:        str
    date:             str
    side:             str    # "Home -7.5" ou "Away +7.5"
    bookmaker:        str
    bookie_odd:       float
    bookie_prob:      float  # probabilité fair (vig retiré)
    model_prob:       float  # notre P(couvre le spread)
    value:            float  # (model_prob × odd) - 1
    kelly_stake:      float
    predicted_spread: float  # notre prédiction de marge (positif = home favori)
    spread_line:      float  # ligne bookmaker (ex: -7.5 pour home favori)
    b2b_home:         bool = False
    b2b_away:         bool = False


@dataclass
class ValueBet:
    home_team:       str
    away_team:       str
    date:            str
    market:          str          # "Over 224.5" ou "Under 224.5"
    bookmaker:       str
    bookie_odd:      float        # cote proposée par le bookmaker
    bookie_prob:     float        # probabilité implicite du bookie
    model_prob:      float        # notre estimation de probabilité
    value:           float        # value = (model_prob * odd) - 1
    kelly_stake:     float        # mise conseillée (Kelly fractionné)
    predicted_total: float        # total prédit par notre modèle
    total_line:      float        # ligne de total du bookmaker
    b2b_home:        bool = False # équipe domicile en back-to-back
    b2b_away:        bool = False # équipe extérieure en back-to-back


# ── Calcul de probabilité modèle ─────────────────────────────────────────────

def model_probability_cover(predicted_spread: float, spread_line: float,
                             std_dev: float = STD_DEV_SPREAD) -> float:
    """
    P(home team covers the spread).

    predicted_spread : marge prédite côté home (positif = home favori)
    spread_line      : handicap home (ex: -7.5 = home donne 7.5 pts)

    P(home covers) = P(actual_spread > -spread_line)
      Exemple : spread_line=-7.5 → P(home gagne de 8+ pts)
                spread_line=+4.5 → P(home perd de 4 ou moins, ou gagne)
    """
    from scipy.stats import norm
    cutoff = -spread_line  # pour -7.5 : cutoff = 7.5 (home doit gagner de 7.5+)
    prob = 1 - norm.cdf(cutoff, loc=predicted_spread, scale=std_dev)
    return round(float(prob), 4)


def model_probability_over(predicted_total: float, line: float,
                            std_dev: float = STD_DEV) -> float:
    """
    Estime la probabilité que le match dépasse la ligne (Over).

    On modélise les totaux NBA comme une distribution normale :
    - Moyenne = notre total prédit
    - Écart-type = STD_DEV calibré sur 3689 matchs historiques (RMSE = 18.87 pts)
    """
    from scipy.stats import norm
    prob_over = 1 - norm.cdf(line, loc=predicted_total, scale=std_dev)
    return round(float(prob_over), 4)


def fractional_kelly(prob: float, odd: float, fraction: float = KELLY_FRACTION) -> float:
    """
    Calcule la mise optimale selon le critère de Kelly fractionné.
    Kelly pur = (p*b - q) / b  où b = odd - 1
    """
    b = odd - 1
    q = 1 - prob
    kelly = (prob * b - q) / b
    return max(0.0, round(kelly * fraction, 4))


# ── Détection de value ────────────────────────────────────────────────────────

def detect_spread_bets(spreads: list[dict], league_df) -> list[SpreadBet]:
    """
    Pour chaque match avec cotes spreads, prédit la marge et cherche des value bets.
    """
    spread_bets = []
    n = len(spreads)

    for i, match in enumerate(spreads, 1):
        home = normalize_team(match["home_team"])
        away = normalize_team(match["away_team"])

        print(f"\n[{i}/{n}] Spread : {away} @ {home}")

        game_date = None
        starts_raw = match.get("starts", "")
        if starts_raw:
            try:
                dt_utc = datetime.fromisoformat(starts_raw.replace("Z", "+00:00"))
                dt_et  = dt_utc.astimezone(timezone(timedelta(hours=-5)))
                game_date = dt_et.strftime("%Y-%m-%d")
            except Exception:
                pass

        try:
            prediction   = predict_match_total(home, away, league_df, game_date=game_date)
            pred_spread  = prediction.get("predicted_spread")
        except Exception as e:
            print(f"  Erreur prédiction : {e}")
            continue

        if pred_spread is None:
            print(f"  Pas de prédiction disponible")
            continue

        line = match["spread_line"]
        print(f"  Spread prédit : {pred_spread:+.1f} pts | Ligne : {line:+g} pts")

        prob_home = model_probability_cover(pred_spread, line)
        prob_away = round(1 - prob_home, 4)
        print(f"  P(Home couvre)={prob_home:.1%}  P(Away couvre)={prob_away:.1%}")

        b2b_home = prediction.get("b2b_home", False)
        b2b_away = prediction.get("b2b_away", False)

        best_home = match["best_home"]
        best_away = match["best_away"]
        bookie_prob_home, bookie_prob_away = odd_to_fair_prob(
            best_home["price"], best_away["price"]
        )

        # Value Home cover
        value_home = round((prob_home * best_home["price"]) - 1, 4)
        if value_home >= MIN_VALUE:
            side  = f"Home {line:+g}"
            print(f"  [VALUE BET] {side} : value={value_home:.1%} | cote={best_home['price']} | mise: {FIXED_STAKE}€")
            spread_bets.append(SpreadBet(
                home_team=home, away_team=away, date=match["date"],
                side=side, bookmaker=best_home["bookmaker"],
                bookie_odd=best_home["price"], bookie_prob=bookie_prob_home,
                model_prob=prob_home, value=value_home, kelly_stake=FIXED_STAKE,
                predicted_spread=pred_spread, spread_line=line,
                b2b_home=b2b_home, b2b_away=b2b_away,
            ))

        # Value Away cover
        value_away = round((prob_away * best_away["price"]) - 1, 4)
        if value_away >= MIN_VALUE:
            side  = f"Away {-line:+g}"
            print(f"  [VALUE BET] {side} : value={value_away:.1%} | cote={best_away['price']} | mise: {FIXED_STAKE}€")
            spread_bets.append(SpreadBet(
                home_team=home, away_team=away, date=match["date"],
                side=side, bookmaker=best_away["bookmaker"],
                bookie_odd=best_away["price"], bookie_prob=bookie_prob_away,
                model_prob=prob_away, value=value_away, kelly_stake=FIXED_STAKE,
                predicted_spread=pred_spread, spread_line=line,
                b2b_home=b2b_home, b2b_away=b2b_away,
            ))

        time.sleep(0.3)

    return spread_bets


def detect_value_bets(totals: list[dict], league_df) -> list[ValueBet]:
    """
    Pour chaque match avec cotes, prédit le total NBA et cherche des value bets.
    """
    value_bets = []
    n = len(totals)

    for i, match in enumerate(totals, 1):
        home = normalize_team(match["home_team"])
        away = normalize_team(match["away_team"])

        print(f"\n[{i}/{n}] Analyse : {away} @ {home}")

        # Ignorer les matchs avec peu de bookmakers (faible liquidité)
        if match["n_bookmakers"] < MIN_BOOKMAKERS:
            print(f"  >>  Ignoré ({match['n_bookmakers']} bookmakers seulement)")
            continue

        # Convertir la date UTC Pinnacle en date locale US (ET = UTC-5)
        game_date = None
        starts_raw = match.get("starts", "")
        if starts_raw:
            try:
                dt_utc = datetime.fromisoformat(starts_raw.replace("Z", "+00:00"))
                dt_et  = dt_utc.astimezone(timezone(timedelta(hours=-5)))
                game_date = dt_et.strftime("%Y-%m-%d")
            except Exception:
                pass

        # Prédire le total via notre modèle NBA
        try:
            prediction = predict_match_total(home, away, league_df, game_date=game_date)
            pred_total = prediction.get("predicted_total")
        except Exception as e:
            print(f"  Erreur prédiction : {e}")
            continue

        if pred_total is None:
            print(f"  Pas de prédiction disponible")
            continue

        line = match["total_line"]
        print(f"  Total prédit : {pred_total} pts | Ligne bookie : {line} pts")

        # Probabilité Over selon notre modèle
        prob_over  = model_probability_over(pred_total, line)
        prob_under = round(1 - prob_over, 4)

        print(f"  P(Over)={prob_over:.1%}  P(Under)={prob_under:.1%}")

        b2b_home = prediction.get("b2b_home", False)
        b2b_away = prediction.get("b2b_away", False)

        # Probabilités fair (vig retiré) — affichage uniquement.
        # La formule value = model_prob × odd - 1 est correcte sans correction.
        best_over  = match["best_over"]
        best_under = match["best_under"]
        bookie_prob_over, bookie_prob_under = odd_to_fair_prob(
            best_over["price"], best_under["price"]
        )

        value_over = round((prob_over * best_over["price"]) - 1, 4)

        if value_over >= MIN_VALUE:
            print(f"  [VALUE BET OVER]  : value={value_over:.1%} | cote={best_over['price']} @ {best_over['bookmaker']} | mise : {FIXED_STAKE}€")
            value_bets.append(ValueBet(
                home_team=home, away_team=away, date=match["date"],
                market=f"Over {line}", bookmaker=best_over["bookmaker"],
                bookie_odd=best_over["price"], bookie_prob=bookie_prob_over,
                model_prob=prob_over, value=value_over, kelly_stake=FIXED_STAKE,
                predicted_total=pred_total, total_line=line,
                b2b_home=b2b_home, b2b_away=b2b_away,
            ))

        value_under = round((prob_under * best_under["price"]) - 1, 4)

        if value_under >= MIN_VALUE:
            print(f"  [VALUE BET UNDER] : value={value_under:.1%} | cote={best_under['price']} @ {best_under['bookmaker']} | mise : {FIXED_STAKE}€")
            value_bets.append(ValueBet(
                home_team=home, away_team=away, date=match["date"],
                market=f"Under {line}", bookmaker=best_under["bookmaker"],
                bookie_odd=best_under["price"], bookie_prob=bookie_prob_under,
                model_prob=prob_under, value=value_under, kelly_stake=FIXED_STAKE,
                predicted_total=pred_total, total_line=line,
                b2b_home=b2b_home, b2b_away=b2b_away,
            ))

        # Petit délai pour ne pas spammer l'API NBA
        time.sleep(0.3)

    return value_bets


# ── Alertes Telegram ──────────────────────────────────────────────────────────

def send_telegram(message: str) -> bool:
    """Envoie un message Telegram."""
    if TELEGRAM_TOKEN == "REMPLACE_PAR_TON_TOKEN":
        print("[TELEGRAM] Token non configuré, message non envoyé.")
        print(f"[TELEGRAM] Message :\n{message}")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id":    TELEGRAM_CHAT_ID,
        "text":       message,
        "parse_mode": "HTML",
    }
    resp = requests.post(url, json=payload, timeout=10)
    return resp.status_code == 200


def format_spread_bet_message(sb: SpreadBet) -> str:
    """Formate un spread bet en message Telegram lisible."""
    value_pct  = f"{sb.value * 100:.1f}%"
    edge_emoji = "[!!]" if sb.value >= 0.08 else "[OK]"

    b2b_flags = []
    if sb.b2b_home:
        b2b_flags.append(f"{sb.home_team} B2B")
    if sb.b2b_away:
        b2b_flags.append(f"{sb.away_team} B2B")
    b2b_line = f"\n  ⚠️ Back-to-back : {', '.join(b2b_flags)}" if b2b_flags else ""

    # Explication humaine du pari
    if sb.side.startswith("Home"):
        cover_desc = f"{sb.home_team} gagne de {abs(sb.spread_line):.1f}+ pts"
    else:
        margin = abs(sb.spread_line)
        cover_desc = (f"{sb.away_team} gagne, ou perd de moins de {margin:.1f} pts")

    return (
        f"{edge_emoji} <b>VALUE BET NBA — SPREAD</b>\n"
        f"\n"
        f" <b>{sb.away_team} @ {sb.home_team}</b>\n"
        f" {sb.date}{b2b_line}\n"
        f"\n"
        f" <b>Pari : {sb.side}</b>\n"
        f"  → {cover_desc}\n"
        f"\n"
        f" <b>Analyse :</b>\n"
        f"  Marge prédite  : <b>{sb.predicted_spread:+.1f} pts</b> (home)\n"
        f"  Ligne bookmaker: {sb.spread_line:+g} pts\n"
        f"  Notre P(couvre): {sb.model_prob:.1%}\n"
        f"  Prob. fair bookie: {sb.bookie_prob:.1%}\n"
        f"\n"
        f" <b>Value : {value_pct}</b>\n"
        f"  Cote : {sb.bookie_odd} @ {sb.bookmaker}\n"
        f"  Mise fixe : <b>{sb.kelly_stake}€</b> (bankroll {BANKROLL}€)"
    )


def format_value_bet_message(vb: ValueBet) -> str:
    """Formate une value bet en message Telegram lisible."""
    value_pct   = f"{vb.value * 100:.1f}%"
    edge_emoji  = "[!!]" if vb.value >= 0.08 else "[OK]"

    b2b_flags = []
    if vb.b2b_home:
        b2b_flags.append(f"{vb.home_team} B2B")
    if vb.b2b_away:
        b2b_flags.append(f"{vb.away_team} B2B")
    b2b_line = f"\n  ⚠️ Back-to-back : {', '.join(b2b_flags)}" if b2b_flags else ""

    return (
        f"{edge_emoji} <b>VALUE BET NBA — {vb.market.upper()}</b>\n"
        f"\n"
        f" <b>{vb.away_team} @ {vb.home_team}</b>\n"
        f" {vb.date}{b2b_line}\n"
        f"\n"
        f" <b>Analyse :</b>\n"
        f"  Total prédit   : <b>{vb.predicted_total} pts</b>\n"
        f"  Ligne bookmaker: {vb.total_line} pts\n"
        f"  Notre P(gagner): {vb.model_prob:.1%}\n"
        f"  Prob. implicite: {vb.bookie_prob:.1%}\n"
        f"\n"
        f" <b>Value : {value_pct}</b>\n"
        f"  Cote : {vb.bookie_odd} @ {vb.bookmaker}\n"
        f"  Mise fixe : <b>{vb.kelly_stake}€</b> (bankroll {BANKROLL}€)"
    )


def send_summary(value_bets: list[ValueBet], spread_bets: list[SpreadBet],
                  dry_run: bool = False) -> None:
    """Envoie un résumé + chaque value bet (totaux + spreads) en Telegram."""
    now   = datetime.now().strftime("%d/%m/%Y %H:%M")
    total = len(value_bets) + len(spread_bets)

    if total == 0:
        msg = (
            f" <b>NBA Value Bot — {now}</b>\n\n"
            f"Aucune value bet détectée aujourd'hui.\n"
            f"Patience, la value viendra."
        )
        if not dry_run:
            send_telegram(msg)
        else:
            print(f"\n[DRY RUN]\n{msg}")
        return

    summary = (
        f" <b>NBA Value Bot — {now}</b>\n\n"
        f"<b>{total} value bet(s) détectée(s)</b>\n"
        f"  Spreads : {len(spread_bets)} | Totaux : {len(value_bets)}\n"
        f"Seuil minimum : {int(MIN_VALUE*100)}%\n"
        f"Bankroll : {BANKROLL}€"
    )

    if not dry_run:
        send_telegram(summary)
        for sb in spread_bets:
            time.sleep(0.5)
            send_telegram(format_spread_bet_message(sb))
        for vb in value_bets:
            time.sleep(0.5)
            send_telegram(format_value_bet_message(vb))
    else:
        print(f"\n{'='*55}")
        print("[DRY RUN] Messages qui seraient envoyés :")
        print(f"{'='*55}")
        print(summary)
        for sb in spread_bets:
            print(f"\n{format_spread_bet_message(sb)}")
        for vb in value_bets:
            print(f"\n{format_value_bet_message(vb)}")


# ── Point d'entrée ────────────────────────────────────────────────────────────

def main():
    global MIN_VALUE

    parser = argparse.ArgumentParser(description="NBA Value Bet Bot")
    parser.add_argument("--dry-run", action="store_true",
                        help="Affiche les value bets sans envoyer de Telegram")
    parser.add_argument("--min-value", type=float, default=MIN_VALUE,
                        help=f"Seuil de value minimum (défaut: {MIN_VALUE})")
    args = parser.parse_args()

    MIN_VALUE = args.min_value

    print("=" * 55)
    print("  NBA VALUE BOT")
    print(f"  {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    print(f"  Seuil value : {int(MIN_VALUE*100)}%")
    print(f"  Mode : {'DRY RUN' if args.dry_run else 'LIVE'}")
    print("=" * 55)

    # 1. Charger les stats NBA
    print("\n[1/3] Chargement stats NBA...")
    league_df = get_league_advanced_stats()

    # 2. Récupérer les cotes (spreads + totaux en un seul appel API)
    print("\n[2/3] Récupération des cotes PS3838...")
    totals, spreads = get_nba_odds_and_spreads()

    print(f"  {len(spreads)} matchs avec cotes spreads")
    print(f"  {len(totals)} matchs avec cotes totaux")

    if not spreads and not totals:
        print("\n[!] Aucun match trouvé. Vérifie tes identifiants PS3838.")
        sys.exit(0)

    # 3. Détecter les value bets
    spread_bets = []
    value_bets  = []

    if spreads:
        print(f"\n[3a/3] Analyse spreads ({len(spreads)} matchs)...")
        spread_bets = detect_spread_bets(spreads, league_df)

    if totals:
        print(f"\n[3b/3] Analyse totaux ({len(totals)} matchs)...")
        value_bets = detect_value_bets(totals, league_df)

    # 4. Envoyer les alertes
    total_bets = len(spread_bets) + len(value_bets)
    print(f"\n{'='*55}")
    print(f"  Résultat : {total_bets} value bet(s)")
    print(f"  Spreads : {len(spread_bets)} | Totaux : {len(value_bets)}")
    print(f"{'='*55}")

    send_summary(value_bets, spread_bets, dry_run=args.dry_run)

    # Sauvegarder dans un fichier log
    all_bets = spread_bets + value_bets
    if all_bets:
        log = {
            "timestamp":    datetime.now().isoformat(),
            "spread_bets":  [asdict(sb) for sb in spread_bets],
            "value_bets":   [asdict(vb) for vb in value_bets],
        }
        with open("value_bets_log.json", "a") as f:
            f.write(json.dumps(log, ensure_ascii=False) + "\n")
        print(f"\n[>] Bets sauvegardées dans value_bets_log.json")


if __name__ == "__main__":
    main()

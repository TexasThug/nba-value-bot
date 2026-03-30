"""
check_results.py
----------------
Le lendemain matin, va chercher les vrais scores NBA et marque chaque
value bet Won / Lost dans le log.

Usage :
  python check_results.py              -> verifie toutes les bets non resolues
  python check_results.py --date 2026-03-17  -> verifie une date specifique
"""

import argparse
import json
import time
from datetime import datetime, timedelta

from nba_api.stats.endpoints import leaguegamefinder

LOG_FILE          = "value_bets_log.json"
RESULTS_FILE      = "value_bets_results.json"   # combined (source de vérité)
RESULTS_TOTAUX    = "results_totaux.json"
RESULTS_SPREADS   = "results_spreads.json"
API_DELAY         = 0.7


# ── Charger le log des bets ───────────────────────────────────────────────────

def load_bets() -> list[dict]:
    """
    Charge toutes les bets (spreads + totaux) du log.
    Chaque bet reçoit un champ 'bet_type' : 'spread' ou 'total'.
    """
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        bets = []
        for line in lines:
            entry = json.loads(line)
            ts = entry["timestamp"]
            for sb in entry.get("spread_bets", []):
                sb["logged_at"] = ts
                sb["bet_type"]  = "spread"
                bets.append(sb)
            for vb in entry.get("value_bets", []):
                vb["logged_at"] = ts
                vb["bet_type"]  = "total"
                bets.append(vb)
        return bets
    except FileNotFoundError:
        print(f"[!] {LOG_FILE} introuvable. Lance d'abord value_bot.py.")
        return []


def load_results() -> dict:
    """Charge les resultats deja connus (pour ne pas re-checker)."""
    try:
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_results(results: dict):
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


# ── Recuperer les vrais scores pour une date ──────────────────────────────────

def fetch_scores_for_date(date_str: str) -> list[dict]:
    """
    Recupere tous les scores NBA pour une date donnee (format YYYY-MM-DD).
    Retourne une liste de dicts : home_team, away_team, home_pts, away_pts, total.
    """
    print(f"  [NBA] Scores du {date_str}...")
    time.sleep(API_DELAY)

    # L'API prend des dates au format MM/DD/YYYY
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    date_from = dt.strftime("%m/%d/%Y")
    date_to   = (dt + timedelta(days=1)).strftime("%m/%d/%Y")

    finder = leaguegamefinder.LeagueGameFinder(
        date_from_nullable=date_from,
        date_to_nullable=date_to,
        season_type_nullable="Regular Season",
        league_id_nullable="00",
    )
    df = finder.get_data_frames()[0]

    if df.empty:
        print(f"  [!] Aucun match trouve pour le {date_str} (matchs pas encore joues ?)")
        return []

    # Garder uniquement les matchs a domicile pour eviter les doublons
    home_games = df[~df["MATCHUP"].str.contains("@")].copy()

    scores = []
    for _, row in home_games.iterrows():
        game_id   = row["GAME_ID"]
        home_team = row["TEAM_NAME"]
        home_pts  = int(row["PTS"])
        away_pts  = int(home_pts - row["PLUS_MINUS"])
        total     = home_pts + away_pts

        # Trouver l'equipe adverse dans le meme match
        away_row  = df[(df["GAME_ID"] == game_id) & (df["TEAM_ID"] != row["TEAM_ID"])]
        away_team = away_row.iloc[0]["TEAM_NAME"] if not away_row.empty else "Unknown"

        scores.append({
            "home_team": home_team,
            "away_team": away_team,
            "home_pts":  home_pts,
            "away_pts":  away_pts,
            "total":     total,
        })

    print(f"  [NBA] {len(scores)} matchs trouves")
    return scores


# ── Matcher une bet avec un score ─────────────────────────────────────────────

def find_score(bet: dict, scores: list[dict]) -> dict | None:
    """Cherche le score correspondant a une value bet (matching partiel des noms)."""
    home = bet["home_team"].lower()
    away = bet["away_team"].lower()

    for s in scores:
        s_home = s["home_team"].lower()
        s_away = s["away_team"].lower()
        # Matching flexible : on cherche si un mot cle est present
        home_match = any(w in s_home for w in home.split() if len(w) > 3)
        away_match = any(w in s_away for w in away.split() if len(w) > 3)
        if home_match and away_match:
            return s
    return None


# ── Calculer Won / Lost ───────────────────────────────────────────────────────

def resolve_bet(bet: dict, score: dict) -> dict:
    """Determine si la bet est gagnee ou perdue (supporte spreads et totaux)."""
    actual_total  = score["total"]
    actual_spread = score["home_pts"] - score["away_pts"]  # marge home
    stake = bet["kelly_stake"]
    odd   = bet["bookie_odd"]

    if bet.get("bet_type") == "spread":
        # Spread bet : "Home -7.5" ou "Away +7.5"
        side        = bet["side"]   # ex: "Home -7.5"
        spread_line = bet["spread_line"]

        if side.startswith("Home"):
            # Home couvre si actual_spread > -spread_line
            won = actual_spread > -spread_line
        else:
            # Away couvre si actual_spread < -spread_line
            won = actual_spread < -spread_line

        pnl = round(stake * (odd - 1) if won else -stake, 2)
        return {
            **bet,
            "actual_spread": actual_spread,
            "actual_total":  actual_total,
            "home_pts":      score["home_pts"],
            "away_pts":      score["away_pts"],
            "won":           won,
            "pnl":           pnl,
            "resolved_at":   datetime.now().isoformat(),
        }
    else:
        # Total bet : "Over 224.5" ou "Under 224.5"
        line   = bet["total_line"]
        market = bet["market"]
        won    = actual_total > line if market.startswith("Over") else actual_total < line
        pnl    = round(stake * (odd - 1) if won else -stake, 2)
        return {
            **bet,
            "actual_total": actual_total,
            "home_pts":     score["home_pts"],
            "away_pts":     score["away_pts"],
            "won":          won,
            "pnl":          pnl,
            "resolved_at":  datetime.now().isoformat(),
        }


# ── Affichage des resultats ───────────────────────────────────────────────────

def print_summary(resolved: list[dict]):
    if not resolved:
        return

    print(f"\n{'='*55}")
    print(f"  RESULTATS DES VALUE BETS")
    print(f"{'='*55}")

    wins   = [r for r in resolved if r["won"]]
    losses = [r for r in resolved if not r["won"]]
    pnl    = sum(r["pnl"] for r in resolved)
    invest = sum(r["kelly_stake"] for r in resolved)

    for r in resolved:
        status = "WON  " if r["won"] else "LOST "
        print(f"\n  [{status}] {r['away_team']} @ {r['home_team']}")
        if r.get("bet_type") == "spread":
            margin = r.get("actual_spread", "?")
            print(f"    {r['side']} | Marge reelle: {margin:+g} pts ({r['home_pts']}-{r['away_pts']})")
        else:
            print(f"    {r['market']} | Ligne: {r['total_line']} | Total: {r['actual_total']} ({r['home_pts']}-{r['away_pts']})")
        print(f"    Mise: {r['kelly_stake']}e | P&L: {r['pnl']:+.2f}e")

    print(f"\n{'-'*55}")
    print(f"  Bets resolues   : {len(resolved)}")
    print(f"  Victoires       : {len(wins)} ({len(wins)/max(len(resolved),1)*100:.0f}%)")
    print(f"  P&L total       : {pnl:+.2f}e")
    print(f"  ROI             : {pnl/max(invest,1)*100:.1f}%")
    print(f"{'='*55}")


# ── Point d'entree ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NBA Value Bot - Check Results")
    parser.add_argument("--date", type=str, help="Date a verifier (YYYY-MM-DD)")
    args = parser.parse_args()

    print("=" * 55)
    print("  NBA VALUE BOT - CHECK RESULTS")
    print(f"  {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    print("=" * 55)

    # Charger les bets loggees
    bets = load_bets()
    if not bets:
        return

    print(f"\n  {len(bets)} bet(s) dans le log")

    # Charger les resultats deja connus
    known_results = load_results()

    # Filtrer par date si demande
    if args.date:
        bets = [b for b in bets if b.get("date", "")[:10] == args.date]
        print(f"  Filtre sur {args.date} : {len(bets)} bet(s)")

    # Trouver les dates uniques a verifier
    # Le format du log peut etre "17/03 23:10" (DD/MM HH:MM) ou "2026-03-17"
    def normalize_date(raw: str) -> str:
        raw = raw.strip()
        if len(raw) >= 10 and raw[4] == "-":
            return raw[:10]  # deja au format YYYY-MM-DD
        # Format DD/MM HH:MM — les matchs avant 10h00 heure locale sont en fait
        # des matchs du soir precedent cote US (decalage horaire ~6h)
        part = raw[:5]   # "17/03" ou "18/03"
        time_part = raw[6:] if len(raw) > 5 else "20:00"  # "02:10"
        day, month = part.split("/")
        hour = int(time_part.split(":")[0]) if ":" in time_part else 20
        year = datetime.now().year
        dt = datetime(year, int(month), int(day))
        if hour < 10:
            dt = dt - timedelta(days=1)  # heure US = veille
        return dt.strftime("%Y-%m-%d")

    dates = sorted(set(normalize_date(b["date"]) for b in bets))
    scores_by_date = {}
    for date in dates:
        scores_by_date[date] = fetch_scores_for_date(date)

    # Resoudre chaque bet
    newly_resolved = []
    still_pending  = []

    for bet in bets:
        # Cle unique pour eviter les doublons
        key = f"{bet['date'][:10]}|{bet['home_team']}|{bet.get('market') or bet.get('side', '')}"

        if key in known_results:
            continue  # deja resolue

        date   = normalize_date(bet["date"])
        scores = scores_by_date.get(date, [])

        if not scores:
            still_pending.append(bet)
            continue

        score = find_score(bet, scores)
        if score is None:
            print(f"  [!] Match introuvable pour {bet['away_team']} @ {bet['home_team']} ({date})")
            still_pending.append(bet)
            continue

        result = resolve_bet(bet, score)
        known_results[key] = result
        newly_resolved.append(result)

    # Sauvegarder
    save_results(known_results)

    # Afficher
    if newly_resolved:
        print_summary(newly_resolved)
        print(f"\n  Resultats sauvegardes dans {RESULTS_FILE}")
    else:
        print("\n  Aucune nouvelle bet a resoudre.")

    if still_pending:
        print(f"  {len(still_pending)} bet(s) en attente (matchs pas encore joues ?)")

    # Sauvegarder les résultats séparés par type
    all_resolved = list(known_results.values())
    totaux_resolved  = [r for r in all_resolved if r.get("bet_type") == "total"]
    spreads_resolved = [r for r in all_resolved if r.get("bet_type") == "spread"]
    with open(RESULTS_TOTAUX,  "w", encoding="utf-8") as f:
        json.dump(totaux_resolved,  f, ensure_ascii=False, indent=2)
    with open(RESULTS_SPREADS, "w", encoding="utf-8") as f:
        json.dump(spreads_resolved, f, ensure_ascii=False, indent=2)

    # Afficher le cumul global par type
    def _summary_line(label, results):
        if not results:
            return f"  {label:8s}: 0 bets"
        wins   = sum(1 for r in results if r["won"])
        pnl    = sum(r["pnl"] for r in results)
        invest = sum(r["kelly_stake"] for r in results)
        roi    = pnl / max(invest, 1) * 100
        return (f"  {label:8s}: {len(results)} bets | "
                f"Win {wins}/{len(results)} ({wins/len(results)*100:.0f}%) | "
                f"P&L {pnl:+.2f}e | ROI {roi:.1f}%")

    if all_resolved:
        print(f"\n  {'='*53}")
        print(f"  CUMUL GLOBAL ({len(all_resolved)} bets resolues)")
        print(f"  {'-'*53}")
        print(_summary_line("TOTAUX",  totaux_resolved))
        print(_summary_line("SPREADS", spreads_resolved))
        print(f"  {'-'*53}")
        print(_summary_line("TOTAL",   all_resolved))
        print(f"  {'='*53}")


if __name__ == "__main__":
    main()

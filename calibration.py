"""
calibration.py
--------------
Analyse la qualité et la fiabilité du modèle NBA value bot.

Métriques calculées :
  1. Précision des prédictions (RMSE, MAE, biais)
  2. Calibration des probabilités (prob. prédite vs fréquence réelle)
  3. Brier score (qualité globale des probabilités)
  4. Significativité statistique de l'edge
  5. P&L et ROI par saison

Usage :
  python calibration.py                   → rapport complet sur backtest
  python calibration.py --live            → inclut les vrais résultats live
  python calibration.py --min-bets 5      → seuil min par bucket de calibration
"""

import argparse
import json
import math
import sys
from collections import defaultdict

# Force UTF-8 output (nécessaire sur Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


BACKTEST_FILE = "backtest_results.json"
RESULTS_FILE  = "value_bets_results.json"


# ── Chargement des données ────────────────────────────────────────────────────

def load_backtest() -> list[dict]:
    try:
        with open(BACKTEST_FILE, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[!] {BACKTEST_FILE} introuvable. Lance d'abord backtester.py.")
        return []


def load_live_results() -> list[dict]:
    try:
        with open(RESULTS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return list(data.values())
    except FileNotFoundError:
        return []


# ── Métriques de prédiction ───────────────────────────────────────────────────

def prediction_accuracy(records: list[dict]) -> dict:
    """RMSE, MAE, biais sur les totaux prédits vs réels."""
    errors = [r["predicted_total"] - r["actual_total"] for r in records]
    n = len(errors)
    if n == 0:
        return {}
    mean_err = sum(errors) / n
    rmse = math.sqrt(sum(e ** 2 for e in errors) / n)
    mae  = sum(abs(e) for e in errors) / n
    std  = math.sqrt(sum((e - mean_err) ** 2 for e in errors) / n)

    # Erreur < 5 pts, < 10 pts, < 15 pts
    within_5  = sum(1 for e in errors if abs(e) < 5)
    within_10 = sum(1 for e in errors if abs(e) < 10)
    within_15 = sum(1 for e in errors if abs(e) < 15)

    return {
        "n":           n,
        "bias":        round(mean_err, 2),
        "rmse":        round(rmse, 2),
        "mae":         round(mae, 2),
        "std":         round(std, 2),
        "within_5":    round(within_5 / n * 100, 1),
        "within_10":   round(within_10 / n * 100, 1),
        "within_15":   round(within_15 / n * 100, 1),
    }


# ── Calibration des probabilités ──────────────────────────────────────────────

def calibration_table(bets: list[dict], n_buckets: int = 10,
                       min_bets: int = 5) -> list[dict]:
    """
    Divise les paris par tranches de probabilité prédite.
    Compare P(modèle) vs fréquence réelle de victoire.
    """
    buckets = defaultdict(list)
    for b in bets:
        prob = b["model_prob"]
        bucket = int(prob * n_buckets) / n_buckets  # arrondit au bucket
        buckets[bucket].append(b["won"])

    table = []
    for low in sorted(buckets):
        wins = buckets[low]
        n    = len(wins)
        if n < min_bets:
            continue
        win_rate = sum(wins) / n
        table.append({
            "prob_range": f"{low:.0%}–{low + 1/n_buckets:.0%}",
            "predicted":  round(low + 0.5 / n_buckets, 3),
            "actual":     round(win_rate, 3),
            "n":          n,
            "gap":        round(win_rate - (low + 0.5 / n_buckets), 3),
        })
    return table


# ── Brier Score ───────────────────────────────────────────────────────────────

def brier_score(bets: list[dict]) -> float:
    """Brier score = MSE des probabilités. Parfait = 0, pire = 1. Baseline (50%) = 0.25."""
    if not bets:
        return float("nan")
    return round(sum((b["model_prob"] - int(b["won"])) ** 2 for b in bets) / len(bets), 4)


# ── Significativité statistique ───────────────────────────────────────────────

def edge_significance(bets: list[dict]) -> dict:
    """
    Test statistique de l'edge.
    H0 : win rate = 52.1% (break-even à cote 1.92)
    H1 : win rate > 52.1%
    Retourne z-score, p-value approx, et IC à 95%.
    """
    n = len(bets)
    if n == 0:
        return {}

    wins     = sum(1 for b in bets if b["won"])
    win_rate = wins / n
    # Break-even à cote 1.92 = 1/1.92 = 52.08%
    p0       = 1 / 1.92

    # Z-score (test unilatéral)
    se = math.sqrt(p0 * (1 - p0) / n)
    z  = (win_rate - p0) / se if se > 0 else 0

    # IC 95% (Wilson interval approximation)
    se_obs   = math.sqrt(win_rate * (1 - win_rate) / n) if n > 0 else 0
    ci_low   = max(0, win_rate - 1.96 * se_obs)
    ci_high  = min(1, win_rate + 1.96 * se_obs)

    # Combien de bets pour atteindre p<0.05 avec ce win rate ?
    if win_rate > p0:
        n_needed = math.ceil((1.645 * math.sqrt(p0 * (1 - p0)) /
                              (win_rate - p0)) ** 2)
    else:
        n_needed = None

    return {
        "n":          n,
        "wins":       wins,
        "win_rate":   round(win_rate, 4),
        "breakeven":  round(p0, 4),
        "z_score":    round(z, 2),
        "ci_95_low":  round(ci_low, 4),
        "ci_95_high": round(ci_high, 4),
        "significant": z >= 1.645,   # p < 0.05 unilatéral
        "n_needed":   n_needed,
    }


# ── P&L par saison ────────────────────────────────────────────────────────────

def pnl_by_season(bets: list[dict]) -> list[dict]:
    seasons = defaultdict(list)
    for b in bets:
        seasons[b.get("season", "live")].append(b)

    rows = []
    for season in sorted(seasons):
        sb    = seasons[season]
        wins  = sum(1 for b in sb if b["won"])
        pnl   = sum(b["pnl"] for b in sb)
        inv   = sum(b["kelly_stake"] for b in sb)
        rows.append({
            "season":    season,
            "n_bets":    len(sb),
            "wins":      wins,
            "win_rate":  round(wins / len(sb) * 100, 1) if sb else 0,
            "pnl":       round(pnl, 2),
            "invested":  round(inv, 2),
            "roi":       round(pnl / inv * 100, 1) if inv > 0 else 0,
        })
    return rows


# ── Affichage ─────────────────────────────────────────────────────────────────

def print_report(records: list[dict], bets: list[dict], min_bets: int = 5,
                 source_label: str = "backtest"):
    sep = "=" * 58

    # 1. Précision des prédictions
    print(f"\n{sep}")
    print(f"  1. PRÉCISION DES PRÉDICTIONS ({source_label})")
    print(sep)
    acc = prediction_accuracy(records)
    if acc:
        bias_arrow = "^" if acc["bias"] > 0 else "v" if acc["bias"] < 0 else "-"
        print(f"  Matchs analysés  : {acc['n']}")
        print(f"  Biais            : {acc['bias']:+.2f} pts {bias_arrow}  "
              f"({'sur-prédit' if acc['bias'] > 0 else 'sous-prédit' if acc['bias'] < 0 else 'neutre'})")
        print(f"  RMSE             : {acc['rmse']:.2f} pts  (std_dev à utiliser dans le modèle)")
        print(f"  MAE              : {acc['mae']:.2f} pts")
        print(f"  Précision ±5 pts : {acc['within_5']:.1f}%")
        print(f"  Précision ±10 pts: {acc['within_10']:.1f}%")
        print(f"  Précision ±15 pts: {acc['within_15']:.1f}%")

    # 2. Paris placés
    print(f"\n{sep}")
    print(f"  2. PARIS PLACES ({len(bets)} bets, seuil value >= 4%)")
    print(sep)

    if not bets:
        print("  Aucun pari placé dans les données.")
    else:
        # Brier score
        bs = brier_score(bets)
        bs_ref = 0.25  # baseline 50%
        bs_skill = round((1 - bs / bs_ref) * 100, 1)
        print(f"  Brier Score      : {bs:.4f}  (ref naive=0.25, skill={bs_skill:+.1f}%)")

        # 3. Calibration
        print(f"\n{sep}")
        print(f"  3. CALIBRATION DES PROBABILITÉS")
        print(sep)
        table = calibration_table(bets, min_bets=min_bets)
        if table:
            print(f"  {'Tranche':<12} {'P(modèle)':>10} {'P(réelle)':>10} {'Écart':>7} {'N':>6}")
            print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*7} {'-'*6}")
            for row in table:
                gap_marker = " OK" if abs(row["gap"]) < 0.03 else (" ^" if row["gap"] > 0 else " v")
                print(f"  {row['prob_range']:<12} {row['predicted']:>9.1%} {row['actual']:>9.1%} "
                      f"  {row['gap']:>+.1%}{gap_marker}   {row['n']:>5}")
        else:
            print(f"  Pas assez de données par bucket (min_bets={min_bets}).")

        # 4. Significativité
        print(f"\n{sep}")
        print(f"  4. SIGNIFICATIVITÉ STATISTIQUE DE L'EDGE")
        print(sep)
        sig = edge_significance(bets)
        if sig:
            sig_label = "[OK] SIGNIFICATIF (p < 0.05)" if sig["significant"] else "[--] PAS encore significatif"
            print(f"  Paris placés     : {sig['n']}")
            print(f"  Win rate observé : {sig['win_rate']:.1%}  (break-even : {sig['breakeven']:.1%})")
            print(f"  IC 95%           : [{sig['ci_95_low']:.1%} – {sig['ci_95_high']:.1%}]")
            print(f"  Z-score          : {sig['z_score']:.2f}  → {sig_label}")
            if not sig["significant"] and sig["n_needed"]:
                print(f"  Bets nécessaires : ~{sig['n_needed']} pour p<0.05 avec ce win rate")

        # 5. P&L par saison
        print(f"\n{sep}")
        print(f"  5. P&L PAR SAISON (bankroll simulée 1000€)")
        print(sep)
        seasons = pnl_by_season(bets)
        print(f"  {'Saison':<12} {'Bets':>6} {'W%':>7} {'P&L':>10} {'Investi':>10} {'ROI':>7}")
        print(f"  {'-'*12} {'-'*6} {'-'*7} {'-'*10} {'-'*10} {'-'*7}")
        for s in seasons:
            roi_marker = " +" if s["roi"] > 0 else " -"
            print(f"  {s['season']:<12} {s['n_bets']:>6} {s['win_rate']:>6.1f}% "
                  f"{s['pnl']:>+9.2f}€ {s['invested']:>9.2f}€ {s['roi']:>+6.1f}%{roi_marker}")
        total_pnl  = sum(s["pnl"] for s in seasons)
        total_inv  = sum(s["invested"] for s in seasons)
        total_roi  = total_pnl / total_inv * 100 if total_inv > 0 else 0
        total_bets = sum(s["n_bets"] for s in seasons)
        print(f"  {'─'*12} {'─'*6} {'─'*7} {'─'*10} {'─'*10} {'─'*7}")
        roi_marker = " +" if total_roi > 0 else " -"
        print(f"  {'TOTAL':<12} {total_bets:>6}        "
              f"{total_pnl:>+9.2f}€ {total_inv:>9.2f}€ {total_roi:>+6.1f}%{roi_marker}")

    print(f"\n{sep}\n")


# ── Point d'entrée ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NBA Value Bot — Calibration")
    parser.add_argument("--live",      action="store_true",
                        help="Inclure les résultats live (value_bets_results.json)")
    parser.add_argument("--min-bets",  type=int, default=5,
                        help="Nombre min de bets par bucket de calibration (défaut: 5)")
    args = parser.parse_args()

    print("=" * 58)
    print("  NBA VALUE BOT — RAPPORT DE CALIBRATION")
    print("=" * 58)

    backtest = load_backtest()
    if not backtest:
        return

    bets_backtest = [r for r in backtest if r.get("bet_placed")]

    if args.live:
        live = load_live_results()
        if live:
            print(f"\n  Mode : backtest ({len(bets_backtest)} bets) + live ({len(live)} bets)")
            # Normaliser les champs live pour correspondre au format backtest
            for r in live:
                r.setdefault("season", "live")
                r.setdefault("bet_placed", True)
            all_bets = bets_backtest + live
            print_report(backtest, all_bets, min_bets=args.min_bets,
                         source_label=f"backtest + {len(live)} live")
        else:
            print("\n  Aucun résultat live trouvé, rapport backtest seul.")
            print_report(backtest, bets_backtest, min_bets=args.min_bets)
    else:
        print_report(backtest, bets_backtest, min_bets=args.min_bets)


if __name__ == "__main__":
    main()

"""
=============================================================
 ÉVALUATION DES STRATÉGIES DE PROMPTING - PRÉDICTION VENTES
 LLM  : Gemini (API Google AI Studio, offre gratuite)
 Données : SAPiBench (ventes quotidiennes de 100 articles, 2019-2025)
 Stratégies : Zero-Shot | Few-Shot | Chain-of-Thought (CoT)
 Métriques  : RMSE | MAPE
=============================================================
"""

import pandas as pd
import numpy as np
import urllib.request
import holidays as pyholidays
import re
import os
import time
from google import genai
from google.genai import types

# ─── CONFIGURATION GEMINI ─────────────────────────────────
# Clé gratuite : https://aistudio.google.com/apikey
# À définir dans l'environnement : export GEMINI_API_KEY="..."
MODEL = "gemini-2.5-flash"
SLEEP_BETWEEN_CALLS = 6        # secondes (offre gratuite : ~10 requêtes/min)
MAX_RETRIES = 4

API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    raise SystemExit("❌ Définissez la variable d'environnement GEMINI_API_KEY "
                     "(clé gratuite : https://aistudio.google.com/apikey)")
client = genai.Client(api_key=API_KEY)

CUTOFF_DATE = pd.Timestamp("2025-11-23")   # date de référence
TEST_START  = pd.Timestamp("2025-11-24")
TEST_END    = pd.Timestamp("2025-11-30")   # 7 jours à prédire (inclut Thanksgiving)
HISTORY_DAYS = 14                          # jours d'historique donnés au LLM
PERIOD_LABEL = f"{TEST_START.day}-{TEST_END.day} novembre 2025"

# Articles représentatifs (catégories variées)
ITEMS_TO_TEST = ["MAT001", "MAT011", "MAT021", "MAT026",
                 "MAT031", "MAT036", "MAT041", "MAT046"]

# ─── CHARGEMENT DES DONNÉES ───────────────────────────────
print("=" * 65)
print("  CHARGEMENT DES DONNÉES")
print("=" * 65)
DATA_DIR = "data"
BASE_URL = ("https://raw.githubusercontent.com/clcamar74/"
            "SAPiBench-inventory-simulation-dataset-2026/main/")
os.makedirs(DATA_DIR, exist_ok=True)
for fname in ("synthetic_7year_demand.csv", "material_master.csv"):
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        print(f"  Téléchargement de {fname} ...")
        urllib.request.urlretrieve(BASE_URL + fname, path)

# On renomme les colonnes pour garder le reste du code identique
sales = (pd.read_csv(f"{DATA_DIR}/synthetic_7year_demand.csv", parse_dates=["Date"])
           .rename(columns={"Date": "date", "Material_ID": "item_nbr",
                            "Final_Demand": "unit_sales"})
           [["date", "item_nbr", "unit_sales"]])
master = pd.read_csv(f"{DATA_DIR}/material_master.csv")
items = master.rename(columns={"Material_ID": "item_nbr", "Category": "family"})
items["perishable"] = (items["family"] == "Perishable").astype(int)

# Jours fériés US générés avec la librairie `holidays` (pas de fichier externe)
us_hol = pyholidays.US(years=range(2019, 2026))
holidays = pd.DataFrame({"date": pd.to_datetime(list(us_hol.keys()))})
holidays["is_Holiday"] = True
holidays["is_likely_closed"] = holidays["date"].dt.strftime("%m-%d") == "12-25"

print(f"  SalesData : {len(sales):,} lignes | "
      f"{sales['date'].min().date()} → {sales['date'].max().date()}")
print(f"  Items     : {len(items)} produits | "
      f"{items['family'].nunique()} familles")
print(f"  Période test : {TEST_START.date()} → {TEST_END.date()} (7 jours)\n")

# ─── CONSTRUCTION DU CONTEXTE POUR UN ARTICLE ─────────────
def get_context(item_id):
    item_data = items[items["item_nbr"] == item_id]
    family     = item_data["family"].iloc[0]          
    perishable = "périssable" if item_data["perishable"].iloc[0] == 1 else "non-périssable"  

    hist = (sales[(sales["item_nbr"] == item_id) &
                  (sales["date"] > CUTOFF_DATE - pd.Timedelta(days=HISTORY_DAYS)) &
                  (sales["date"] <= CUTOFF_DATE)]
            .sort_values("date"))
    avg14 = hist["unit_sales"].mean() if len(hist) else 0.0

    history_str = "\n".join(
        f"  {r['date'].strftime('%Y-%m-%d')} ({r['date'].strftime('%A')[:3]}): "
        f"{r['unit_sales']:.0f} unités"
        for _, r in hist.iterrows()
    )

    future_dates = pd.date_range(TEST_START, TEST_END)

    future_rows = []
    for d in future_dates:
        h = holidays[holidays["date"] == d]
        is_hol    = bool(h["is_Holiday"].values[0])       if len(h) else False   
        is_closed = bool(h["is_likely_closed"].values[0]) if len(h) else False   
        flags = []
        if is_hol:    flags.append("FERIÉ")
        if is_closed: flags.append("FERMÉ")
        future_rows.append(
            f"  {d.strftime('%Y-%m-%d')} ({d.strftime('%A')[:3]}): "
            + (", ".join(flags) if flags else "normal")
        )

    return dict(item_id=item_id, family=family, perishable=perishable,
                avg14=avg14, history_str=history_str,
                future_str="\n".join(future_rows))

def get_actuals(item_id):
    a = (sales[(sales["item_nbr"] == item_id) &
               (sales["date"] >= TEST_START) &
               (sales["date"] <= TEST_END)]
         .sort_values("date")["unit_sales"].tolist())
    while len(a) < 7:
        a.append(0.0)
    return a[:7]

# ─── MÉTRIQUES ────────────────────────────────────────────
def rmse(act, pred):
    return float(np.sqrt(np.mean((np.array(act) - np.array(pred))**2)))

def mape(act, pred, eps=0.5):
    act, pred = np.array(act, float), np.array(pred, float)
    mask = act >= eps
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((act[mask]-pred[mask])/act[mask]))*100)

# ─── EXTRACTION DES NOMBRES DEPUIS LA RÉPONSE LLM ─────────
def parse_numbers(text, n=7):
    # cherche d'abord un bloc "PRÉDICTION FINALE: [...]" (CoT)
    bloc = re.search(r"PRÉDICTION FINALE\s*:\s*\[([^\]]+)\]", text, re.I)
    if bloc:
        raw = bloc.group(1)
    else:
        raw = text
    nums = [float(x) for x in re.findall(r"\b(\d+(?:\.\d+)?)\b", raw)
            if 0 <= float(x) <= 5000]
    if len(nums) < n and len(nums) > 0:
        nums += [np.median(nums)] * (n - len(nums))
    if len(nums) == 0:
        nums = [0.0] * n
    return nums[:n]

# ─── APPEL API GEMINI ──────────────────────────────────────
def call_llm(prompt):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    # Pas de "thinking" interne : on compare uniquement les prompts
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            return (resp.text or "").strip()
        except Exception as e:
            wait = 15 * attempt
            print(f"\n[⚠️ ERREUR GEMINI] tentative {attempt}/{MAX_RETRIES} : {e}"
                  f"\n   nouvelle tentative dans {wait}s ...")
            time.sleep(wait)
    print("[❌] Échec de l'appel Gemini, prédiction remplacée par des zéros.")
    return ""

# ═══════════════════════════════════════════════════════════
#  STRATÉGIE 1 — ZERO-SHOT
# ═══════════════════════════════════════════════════════════
def prompt_zero_shot(c):
    return f"""Tu es un expert en prévision des ventes de supermarché.

ARTICLE #{c['item_id']} | Famille: {c['family']} | {c['perishable']}

HISTORIQUE DES 14 DERNIERS JOURS:
{c['history_str']}

SEMAINE À PRÉVOIR ({PERIOD_LABEL}):
{c['future_str']}

Donne uniquement 7 nombres entiers séparés par des virgules
représentant les ventes prévues du {TEST_START.day} au {TEST_END.day} novembre 2025.
Format: n1, n2, n3, n4, n5, n6, n7
"""

# ═══════════════════════════════════════════════════════════
#  STRATÉGIE 2 — FEW-SHOT
# ═══════════════════════════════════════════════════════════
def prompt_few_shot(c):
    return f"""Tu es un expert en prévision des ventes de supermarché.
Voici 3 exemples de prévisions réussies pour guider ta réponse:

EXEMPLE 1 — BEVERAGES (non-périssable), moyenne=15/j
Historique: Lun 12, Mar 14, Mer 10, Jeu 15, Ven 18, Sam 22, Dim 20
Semaine suivante: pas de promo, pas de férié
→ PRÉVISION: 13, 14, 11, 15, 19, 23, 21
Logique: weekend +40%, tendance stable → ajustement saisonnier

EXEMPLE 2 — BREAD/BAKERY (périssable), moyenne=8/j
Historique: Lun 7, Mar 8, Mer 6, Jeu 8, Ven 9, Sam 11, Dim 0
Semaine: vendredi FERIÉ (-15%), dimanche FERMÉ
→ PRÉVISION: 7, 8, 6, 8, 8, 11, 0
Logique: périssable = 0 si fermé; baisse légère les jours fériés

EXEMPLE 3 — DAIRY (périssable), moyenne=20/j
Historique: Lun 18, Mar 20, Mer 22, Jeu 19, Ven 21, Sam 25, Dim 23
Semaine: pas de promo, pas de férié
→ PRÉVISION: 19, 20, 21, 19, 21, 26, 23
Logique: tendance stable, léger pic le samedi

---
MAINTENANT POUR CET ARTICLE:
Article #{c['item_id']} | Famille: {c['family']} | {c['perishable']}
Moyenne 14j: {c['avg14']:.1f} unités/j

HISTORIQUE:
{c['history_str']}

SEMAINE À PRÉVOIR:
{c['future_str']}

Donne uniquement 7 nombres entiers séparés par des virgules ({TEST_START.day} au {TEST_END.day} novembre).
Format: n1, n2, n3, n4, n5, n6, n7
"""

# ═══════════════════════════════════════════════════════════
#  STRATÉGIE 3 — CHAIN-OF-THOUGHT (CoT)
# ═══════════════════════════════════════════════════════════
def prompt_cot(c):
    return f"""Tu es un expert en prévision des ventes de supermarché.

Article #{c['item_id']} | Famille: {c['family']} | {c['perishable']}
Moyenne 14j: {c['avg14']:.1f} unités/j

HISTORIQUE:
{c['history_str']}

SEMAINE À PRÉVOIR ({PERIOD_LABEL}):
{c['future_str']}

Raisonne étape par étape:
ÉTAPE 1 - BASE: Quelle est la vente moyenne récente? Quelle tendance?
ÉTAPE 2 - SAISONNALITÉ: Pattern jour de semaine visible (weekend vs semaine)?
ÉTAPE 3 - ÉVÉNEMENTS: Ajustements pour fériés/fermetures?
ÉTAPE 4 - PÉRISABILITÉ: Impact sur les jours de fermeture?
ÉTAPE 5 - SYNTHÈSE: Combine tout pour 7 valeurs finales.

Termine ta réponse par exactement cette ligne:
PRÉDICTION FINALE: [val1, val2, val3, val4, val5, val6, val7]
"""

# ═══════════════════════════════════════════════════════════
#  BOUCLE PRINCIPALE
# ═══════════════════════════════════════════════════════════
strategies = {
    "Zero-Shot": prompt_zero_shot,
    "Few-Shot" : prompt_few_shot,
    "CoT"      : prompt_cot,
}

all_results  = {s: [] for s in strategies}
raw_log      = []

print("=" * 65)
print(f"  LANCEMENT DE L'ÉVALUATION ({MODEL} via API Gemini)")
print(f"  Articles : {ITEMS_TO_TEST}")
print(f"  Stratégies : {list(strategies.keys())}")
print("=" * 65)

for item_id in ITEMS_TO_TEST:
    ctx     = get_context(item_id)
    actuals = get_actuals(item_id)
    fam     = ctx["family"]
    print(f"\n── Article {item_id} ({fam}, {ctx['perishable']}) ──")
    print(f"   Réels  ({TEST_START:%d/%m}→{TEST_END:%d/%m}): {[round(a) for a in actuals]}")
    print(f"   Moy. 14j: {ctx['avg14']:.1f}")

    for sname, pfn in strategies.items():
        prompt = pfn(ctx)
        raw    = call_llm(prompt)
        preds  = parse_numbers(raw)
        r      = rmse(actuals, preds)
        m      = mape(actuals, preds)

        all_results[sname].append(dict(
            item_id=item_id, family=fam,
            perishable=ctx["perishable"], avg14=ctx["avg14"],
            actuals=actuals, preds=preds, rmse=r, mape=m
        ))
        raw_log.append(dict(strategy=sname, item_id=item_id,
                            raw=raw, preds=preds,
                            actuals=actuals, rmse=r, mape=m))

        print(f"   [{sname:<10}] Préd: {[round(p) for p in preds]}"
              f"  RMSE={r:.2f}  MAPE={m:.1f}%")
        
        time.sleep(SLEEP_BETWEEN_CALLS)   # respecte le quota gratuit de l'API

# ═══════════════════════════════════════════════════════════
#  RÉSULTATS AGRÉGÉS
# ═══════════════════════════════════════════════════════════
print("\n\n" + "=" * 65)
print("  TABLEAU RÉCAPITULATIF — RÉSULTATS PAR STRATÉGIE")
print("=" * 65)
print(f"{'Stratégie':<12} {'RMSE moyen':>11} {'RMSE médian':>12} {'MAPE moyen':>11} {'MAPE médian':>12}")
print("-" * 60)

summary = {}
for sname, results in all_results.items():
    rl = [r["rmse"] for r in results if not np.isnan(r["rmse"])]
    ml = [r["mape"] for r in results if not np.isnan(r["mape"])]
    summary[sname] = dict(
        rmse_mean=np.mean(rl), rmse_med=np.median(rl), rmse_std=np.std(rl),
        mape_mean=np.mean(ml), mape_med=np.median(ml), mape_std=np.std(ml),
    )
    print(f"{sname:<12} {np.mean(rl):>10.3f}  {np.median(rl):>11.3f}"
          f"  {np.mean(ml):>9.1f}%  {np.median(ml):>10.1f}%")

ranked = sorted(summary.items(), key=lambda x: x[1]["rmse_mean"])
print("\n  CLASSEMENT (RMSE croissant — meilleur = plus bas):")
for i, (nm, s) in enumerate(ranked, 1):
    delta = ((s["rmse_mean"] - ranked[0][1]["rmse_mean"]) / 
             ranked[0][1]["rmse_mean"] * 100) if i > 1 else 0
    sign  = f"+{delta:.0f}% vs meilleur" if i > 1 else "← MEILLEUR"
    print(f"  {i}. {nm:<12} RMSE={s['rmse_mean']:.3f}±{s['rmse_std']:.2f}"
          f"  MAPE={s['mape_mean']:.1f}%±{s['mape_std']:.1f}  {sign}")

# ═══════════════════════════════════════════════════════════
#  TABLEAU DÉTAILLÉ PAR ARTICLE
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  DÉTAIL PAR ARTICLE")
print("=" * 65)
hdr = f"{'Article':<10} {'Famille':<16} {'Moy14j':>6}"
for s in strategies:
    hdr += f"  {s:<10} RMSE   MAPE"
print(hdr)
print("-" * (len(hdr)+4))

for item_id in ITEMS_TO_TEST:
    fam = items[items["item_nbr"]==item_id]["family"].iloc[0] 
    ctx = get_context(item_id)
    row = f"{item_id:<10} {fam:<16} {ctx['avg14']:>6.1f}"
    for sname, results in all_results.items():
        r = next(x for x in results if x["item_id"]==item_id)
        row += f"  {sname:<10} {r['rmse']:>5.2f}  {r['mape']:>5.1f}%"
    print(row)

# ═══════════════════════════════════════════════════════════
#  ANALYSE QUALITATIVE
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  ANALYSE QUALITATIVE : SUCCÈS ET ÉCHECS")
print("=" * 65)

if raw_log:
    best = min(raw_log, key=lambda x: x["rmse"])
    print(f"\n  ★ MEILLEUR CAS — {best['strategy']} | Article {best['item_id']}")
    print(f"    Réels  : {[round(a) for a in best['actuals']]}")
    print(f"    Prédit : {[round(p) for p in best['preds']]}")
    print(f"    RMSE={best['rmse']:.3f} | MAPE={best['mape']:.1f}%")
    print(f"    Réponse LLM:\n    {best['raw'][:500].replace(chr(10), chr(10)+'    ')}")

    worst = max(raw_log, key=lambda x: x["rmse"])
    print(f"\n  ✗ PIRE CAS — {worst['strategy']} | Article {worst['item_id']}")
    print(f"    Réels  : {[round(a) for a in worst['actuals']]}")
    print(f"    Prédit : {[round(p) for p in worst['preds']]}")
    print(f"    RMSE={worst['rmse']:.3f} | MAPE={worst['mape']:.1f}%")
    print(f"    Réponse LLM:\n    {worst['raw'][:500].replace(chr(10), chr(10)+'    ')}")

    cot_log = sorted([x for x in raw_log if x["strategy"]=="CoT"], key=lambda x: x["rmse"])
    if cot_log:
        ex = cot_log[len(cot_log)//2]
        print(f"\n  ◆ EXEMPLE CoT MÉDIAN — Article {ex['item_id']}")
        print(f"    Réels  : {[round(a) for a in ex['actuals']]}")
        print(f"    Prédit : {[round(p) for p in ex['preds']]}")
        print(f"    RMSE={ex['rmse']:.3f} | MAPE={ex['mape']:.1f}%")
        print(f"    Raisonnement LLM:\n    {ex['raw'][:700].replace(chr(10), chr(10)+'    ')}")

# ═══════════════════════════════════════════════════════════
#  ANALYSE DES ERREURS SYSTÉMATIQUES
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  ANALYSE DES ERREURS SYSTÉMATIQUES PAR STRATÉGIE")
print("=" * 65)

for sname, results in all_results.items():
    if not results: continue
    errors = []
    for r in results:
        errors.extend((np.array(r["preds"]) - np.array(r["actuals"])).tolist())
    errors = np.array(errors)
    bias   = np.mean(errors)
    print(f"\n  {sname}:")
    print(f"    Biais moyen (prédit - réel): {bias:+.2f}  "
          f"→ {'SURESTIMATION' if bias>0 else 'SOUS-ESTIMATION'}")
    print(f"    Écart-type des erreurs     : {np.std(errors):.2f}")
    print(f"    Erreur absolue max         : {np.max(np.abs(errors)):.1f} unités")
    print(f"    % préd. à < 2 unités d'erreur : "
          f"{(np.abs(errors)<2).mean()*100:.0f}%")
    print(f"    % préd. à < 5 unités d'erreur : "
          f"{(np.abs(errors)<5).mean()*100:.0f}%")

# ═══════════════════════════════════════════════════════════
#  CONCLUSION
# ═══════════════════════════════════════════════════════════
if ranked:
    print("\n" + "=" * 65)
    print("  CONCLUSION — RÉSUMÉ FINAL")
    print("=" * 65)
    best_name = ranked[0][0] 
    for sname, s in summary.items():
        delta = ((s["rmse_mean"] - summary[best_name]["rmse_mean"]) /
                 summary[best_name]["rmse_mean"] * 100)
        tag = " ← MEILLEURE STRATÉGIE" if sname == best_name else f" (+{delta:.0f}%)"
        print(f"  {sname:<12}: RMSE={s['rmse_mean']:.3f} | MAPE={s['mape_mean']:.1f}%{tag}")

    print(f"\n  → Stratégie recommandée : {best_name}")
    print(f"  → La {ranked[-1][0]} est " 
          f"{(ranked[-1][1]['rmse_mean']/ranked[0][1]['rmse_mean']-1)*100:.0f}%"
          f" moins précise que {best_name}")
    
print("\n  FIN DE L'ÉVALUATION")
print("=" * 65)
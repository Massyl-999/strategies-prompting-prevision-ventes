# Évaluation de stratégies de prompting pour la prévision de ventes

Comparaison de trois stratégies de prompting — **Zero-Shot**, **Few-Shot** et **Chain-of-Thought (CoT)** — pour prédire les ventes quotidiennes d'articles de supermarché sur 7 jours, avec un LLM **Gemini** (API Google AI Studio, offre gratuite).

Métriques : **RMSE** et **MAPE**, plus une analyse qualitative (meilleur/pire cas) et une analyse des biais.

## Données

[SAPiBench](https://github.com/clcamar74/SAPiBench-inventory-simulation-dataset-2026) (licence MIT) : ventes quotidiennes de 100 articles sur 2019-2025.
Les fichiers sont téléchargés automatiquement dans `data/` au premier lancement.

> ⚠️ Ce dataset est **synthétique** (demande de base × facteur saisonnier × facteur hebdomadaire × bruit). Les résultats mesurent donc la capacité du LLM à retrouver ces motifs, pas à prédire des ventes réelles.

Les jours fériés (US) sont générés avec la librairie [`holidays`](https://pypi.org/project/holidays/).

## Installation

```bash
git clone https://github.com/<votre-compte>/<votre-repo>.git
cd <votre-repo>
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration

Créez une clé gratuite sur [Google AI Studio](https://aistudio.google.com/apikey), puis :

```bash
export GEMINI_API_KEY="votre_cle"        # Windows PowerShell : $env:GEMINI_API_KEY="votre_cle"
```

## Lancement

```bash
python Code_Prompting.py
```

8 articles × 3 stratégies = 24 appels API (~3 minutes avec la pause de 6 s qui respecte le quota gratuit).

## Paramètres modifiables (début de `Code_Prompting.py`)

| Paramètre | Rôle |
|---|---|
| `MODEL` | Modèle Gemini (défaut : `gemini-2.5-flash`) |
| `TEST_START` / `TEST_END` / `CUTOFF_DATE` | Semaine à prédire (défaut : 24-30 nov. 2025, avec Thanksgiving) |
| `HISTORY_DAYS` | Jours d'historique fournis au LLM |
| `ITEMS_TO_TEST` | Articles évalués |
| `SLEEP_BETWEEN_CALLS` | Pause entre appels (quota gratuit) |

## Licence

MIT — voir [LICENSE](LICENSE). Le dataset SAPiBench est sous licence MIT (auteur : clcamar74).

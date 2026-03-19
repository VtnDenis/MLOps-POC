# Pipeline de Réconciliation Automatisée (PoC)

## 1) Vue d'ensemble

Ce dépôt implémente un PoC de réconciliation Front-Office / Back-Office pour des transactions financières, avec :

- un moteur de matching hybride (règles métier + probabilité ML) ;
- une interface Streamlit pour exécuter les runs et analyser les résultats ;
- un module d'explication IA (DeepSeek) pour les cas ambigus ;
- une persistance PostgreSQL optionnelle des runs et résultats.

## 2) État actuel du projet (mars 2026)

### ✅ Fonctionnalités implémentées

- Upload de 2 fichiers CSV (FO/BO) via l'interface.
- Chargement d'un dataset local de démonstration (`dataset/`).
- Réconciliation FO→BO avec filtrage des candidats BO par ISIN identique.
- Scoring règles pondéré : ISIN, quantité, prix, contrepartie (fuzzy), date.
- Entraînement automatique d'un classifieur `LogisticRegression` (à chaque run) sur des paires heuristiquement labellisées.
- Score global hybride :

$$
Score_{global} = (1 - w_{ml}) \times Score_{règles} + w_{ml} \times Probabilité_{ML}
$$

- KPIs affichés dans Streamlit : auto-match, suggestions, unmatched, confiance ML moyenne.
- Affichage des métriques d'entraînement ML (taille dataset, taux de positifs, accuracy, ROC-AUC si disponible).
- Explication IA DeepSeek pour :
  - une suggestion sélectionnée,
  - ou toutes les suggestions.
- Persistance PostgreSQL optionnelle : insertion d'un run + des lignes de résultats.

### ⚠️ Points importants sur le comportement actuel

- Le statut `MATCH` est déterminé par `Rule_Score == 1.0` (et non par `Global_Score`).
- Le statut `SUGGESTION` est attribué si `Rule_Score < 1.0` et `Global_Score >= threshold`.
- Le statut `UNMATCHED` couvre le reste des cas.
- Sans clé API DeepSeek valide, l'application fonctionne, mais les explications IA retournent un message d'erreur.
- Sans `DATABASE_URL`, la persistance est désactivée automatiquement.

## 3) Structure du projet

```text
.
├── app.py                  # Interface Streamlit
├── engine.py               # Moteur de réconciliation + module DeepSeek
├── db.py                   # Persistance PostgreSQL (optionnelle)
├── data_generator.py       # Génération de données synthétiques FO/BO
├── dataset/
│   ├── data_front_office.csv
│   └── data_back_office.csv
├── requirements.txt        # Dépendances Python
├── dockerfile              # Image Docker Streamlit
└── README.md
```

## 4) Schéma des données attendu

### Front-Office (`data_front_office.csv`)

- `Trade_ID`
- `ISIN`
- `Counterparty`
- `Date`
- `Quantity`
- `Price`
- `Currency`

### Back-Office (`data_back_office.csv`)

- `ISIN`
- `Counterparty`
- `Date`
- `Quantity`
- `Price`
- `Currency`

> Le flux BO n'inclut pas `Trade_ID` dans ce PoC.

## 5) Logique de matching

Le score règles est pondéré comme suit :

- `ISIN`: 40%
- `Quantity`: 30%
- `Price`: 15%
- `Counterparty` (TheFuzz): 10%
- `Date`: 5%

La prédiction ML utilise notamment ces features :

- `isin_match`, `qty_exact`, `qty_diff_ratio`
- `price_diff_abs`, `price_diff_ratio`
- `counterparty_fuzzy`, `date_diff_days`
- `currency_match`, `rule_score`

Le poids ML `w_ml` est paramétrable dans l'UI (`0.00` à `0.70`, défaut `0.35`).

## 6) Prérequis

- Python 3.10+
- `pip`
- Docker (optionnel)

## 7) Installation et exécution locale

### 7.1 Environnement virtuel (Windows / PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 7.2 Variables d'environnement (`.env`)

Créer/compléter le fichier `.env` à la racine :

```env
DEEPSEEK_API_KEY=votre_cle_deepseek
DATABASE_URL=postgresql://user:password@host:5432/dbname
```

- `DEEPSEEK_API_KEY` : requis uniquement pour les explications IA.
- `DATABASE_URL` : requis uniquement pour la persistance PostgreSQL.

### 7.3 Lancement

```powershell
streamlit run .\app.py
```

Puis ouvrir `http://localhost:8501`.

## 8) Exécution Docker

```powershell
docker build -t mlops-app -f dockerfile .
docker run -p 8501:8501 --env-file .env mlops-app
```

## 9) Génération de jeux de données synthétiques

Le script `data_generator.py` produit :

- un flux FO « propre » ;
- un flux BO bruité (typos, décalages de date, micro-écarts de prix, lignes manquantes).

```powershell
python .\data_generator.py
```

Les fichiers sont écrits dans `dataset/`.

## 10) Dépendances principales

- `pandas`
- `numpy`
- `thefuzz`
- `scikit-learn`
- `python-dotenv`
- `streamlit`
- `psycopg2-binary`

## 11) Persistance PostgreSQL

Si `DATABASE_URL` est défini, l'application :

1. crée le schéma si nécessaire (`reconciliation_runs`, `reconciliation_results`, index),
2. sauvegarde chaque run,
3. sauvegarde le détail des résultats du run.

La persistance reste optionnelle (mode dégradé sans base de données).
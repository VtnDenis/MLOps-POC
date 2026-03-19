import pandas as pd
import numpy as np
import random
from datetime import timedelta

def mock_data_generator(n_rows=1000):
    np.random.seed(42)
    
    # 1. Création de la base "Front-Office" (La référence propre)
    data_fo = {
        'Trade_ID': [f"TRD_{i:04d}" for i in range(n_rows)],
        'ISIN': [f"FR{np.random.randint(100000, 999999)}00" for _ in range(n_rows)],
        'Counterparty': np.random.choice(['BNP Paribas', 'Societe Generale', 'JP Morgan', 'Goldman Sachs', 'HSBC'], n_rows),
        'Date': [pd.Timestamp('2026-03-01') + timedelta(days=np.random.randint(0, 15)) for _ in range(n_rows)],
        'Quantity': np.random.randint(10, 5000, n_rows),
        'Price': np.round(np.random.uniform(50, 1500, n_rows), 2),
        'Currency': 'EUR'
    }
    df_fo = pd.DataFrame(data_fo)

    # 2. Création de la base "Back-Office" (La base avec erreurs à réconcilier)
    df_bo = df_fo.copy()

    # --- Simulation des anomalies métier (Réconciliation IA) ---
    
    # A. Erreurs de libellé (Fuzzy Matching)
    typos = {'BNP Paribas': 'BNP PARIBAS SA', 'Societe Generale': 'SOCGEN', 'HSBC': 'HSBC LTD'}
    idx_typo = df_bo.sample(frac=0.1).index
    df_bo.loc[idx_typo, 'Counterparty'] = df_bo.loc[idx_typo, 'Counterparty'].map(lambda x: typos.get(x, x))

    # B. Décalages de date (T+1 / T+2)
    idx_date = df_bo.sample(frac=0.15).index
    df_bo.loc[idx_date, 'Date'] = df_bo.loc[idx_date, 'Date'] + timedelta(days=1)

    # C. Écarts de montant (Frais de courtage / Arrondis)
    idx_price = df_bo.sample(frac=0.1).index
    df_bo.loc[idx_price, 'Price'] = df_bo.loc[idx_price, 'Price'] + np.random.uniform(0.01, 0.05, len(idx_price))

    # D. Ruptures de flux (Lignes manquantes ou en trop)
    df_bo = df_bo.drop(df_bo.sample(frac=0.05).index) # Lignes perdues
    
    # E. Supprimer les IDs pour forcer l'IA à matcher sur les autres colonnes
    df_bo = df_bo.drop(columns=['Trade_ID'])
    
    return df_fo, df_bo

# Génération et sauvegarde
df_front, df_back = mock_data_generator()
df_front.to_csv('data_front_office.csv', index=False)
df_back.to_csv('data_back_office.csv', index=False)

print(f"Datasets générés : {len(df_front)} lignes FO, {len(df_back)} lignes BO.")
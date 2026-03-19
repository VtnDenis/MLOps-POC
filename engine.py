import pandas as pd
import numpy as np
from thefuzz import fuzz
import json
import urllib.request
import time
from dotenv import load_dotenv
import os

load_dotenv()

class ReconciliationEngine:
    """
    Matching des transactions avec du scoring
    """
    
    def __init__(self, threshold=0.8):
        self.threshold = threshold
        # On pondère les colonnes car chaque colonne a sa propre importance
        self.weights = {
            'ISIN': 0.40,
            'Quantity': 0.30,
            'Price': 0.15,
            'Counterparty': 0.10,
            'Date': 0.05
        }

    def _score_isin(self, val1, val2):
        """Match exact sur l'ISIN"""
        return 1.0 if str(val1).strip() == str(val2).strip() else 0.0

    def _score_counterparty(self, val1, val2):
        """Fuzzy matching sur le nom de la contrepartie (0 à 1)"""
        return fuzz.token_sort_ratio(str(val1), str(val2)) / 100.0

    def _score_price(self, val1, val2, tolerance=0.10):
        """Score basé sur l'écart relatif de prix"""
        diff = abs(val1 - val2)
        if diff == 0: return 1.0
        if diff <= tolerance: return 0.8
        return max(0, 1 - (diff / val1))

    def _score_date(self, date1, date2):
        """Score dégressif selon l'écart de jours"""
        delta = abs((date1 - date2).days)
        if delta == 0: return 1.0
        if delta == 1: return 0.7
        if delta == 2: return 0.3
        return 0.0

    def compute_row_score(self, row_fo, row_bo):
        """Calcule un score global pondéré entre deux lignes."""
        scores = {
            'ISIN': self._score_isin(row_fo['ISIN'], row_bo['ISIN']),
            'Quantity': 1.0 if row_fo['Quantity'] == row_bo['Quantity'] else 0.0,
            'Price': self._score_price(row_fo['Price'], row_bo['Price']),
            'Counterparty': self._score_counterparty(row_fo['Counterparty'], row_bo['Counterparty']),
            'Date': self._score_date(row_fo['Date'], row_bo['Date'])
        }
        
        # Somme pondérée
        total_score = sum(scores[col] * self.weights[col] for col in self.weights)
        return total_score, scores

    def reconcile(self, df_fo, df_bo):
        """
        Tente de trouver le meilleur match dans le dataset du back-office pour chaque ligne du front-office.
        """
        results = []
        
        for _, row_fo in df_fo.iterrows():
            best_match_idx = -1
            max_score = -1
            best_detailed_scores = {}

            # On ne compare que les transactions avec le même ISIN
            potential_matches = df_bo[df_bo['ISIN'] == row_fo['ISIN']]
            
            if not potential_matches.empty:
                for idx_bo, row_bo in potential_matches.iterrows():
                    score, detailed = self.compute_row_score(row_fo, row_bo)
                    if score > max_score:
                        max_score = score
                        best_match_idx = idx_bo
                        best_detailed_scores = detailed
            else:
                # Cas où il y a des transactions manquantes
                max_score = 0

            # Statut de la réconciliation
            status = "MATCH" if max_score >= self.threshold else "UNMATCHED"
            if max_score < 1.0 and max_score >= self.threshold:
                status = "SUGGESTION"

            results.append({
                'FO_Trade_ID': row_fo['Trade_ID'],
                'BO_Index': best_match_idx,
                'Global_Score': round(max_score, 4),
                'Status': status,
                'Break_Reason': self._get_break_reason(best_detailed_scores) if status != "MATCH" else ""
            })
            
        return pd.DataFrame(results)

    def _get_break_reason(self, scores):
        """Identifie la cause principale de l'écart."""
        if not scores:
            return "ISIN introuvable dans le Back-Office"
            
        lowest_score_col = min(scores, key=scores.get)
        if scores[lowest_score_col] < 1.0:
            return f"Écart sur {lowest_score_col}"
        return "Inconnu"
    
class AIReconciliationExplainer:
    """
    Module utilisant l'IA Générative (DeepSeek) pour expliquer les écarts de réconciliation
    et suggérer des actions correctives aux opérateurs back-office.
    """
    
    def __init__(self):
        # L'environnement fournit la clé API automatiquement
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.model_name = "deepseek-chat"
        self.url = "https://api.deepseek.com/chat/completions"

    def _call_api_with_retry(self, prompt, retries=5):
        """Appel à l'API DeepSeek avec exponential backoff."""
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system", 
                    "content": "Tu es un expert en réconciliation bancaire chez Amundi. Ton rôle est d'analyser les écarts entre le Front-Office et le Back-Office et de proposer une explication métier concise et professionnelle en français."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "stream": False
        }
        
        data = json.dumps(payload).encode("utf-8")
        
        for i in range(retries):
            try:
                req = urllib.request.Request(
                    self.url,
                    data=data,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}"
                    }
                )
                with urllib.request.urlopen(req) as response:
                    result = json.loads(response.read().decode("utf-8"))
                    return result['choices'][0]['message']['content']
            except Exception:
                if i < retries - 1:
                    time.sleep(2 ** i) # Exponential backoff: 1s, 2s, 4s, 8s, 16s
                else:
                    return "Erreur d'analyse IA : Impossible de générer l'explication."

    def generate_explanation(self, row_fo, row_bo, reason, score):
        """Génère une explication métier basée sur les données des deux flux."""
        
        prompt = f"""
        Analyse l'écart suivant :
        - Statut : {reason}
        - Score de confiance : {score:.2f}
        
        Données Front-Office :
        {row_fo.to_dict()}
        
        Données Back-Office :
        {row_bo.to_dict() if row_bo is not None else 'Aucune correspondance trouvée'}
        
        Instructions : 
        1. Explique pourquoi le score n'est pas de 1.0 (ex: écart de prix, nom de contrepartie différent).
        2. Propose une cause métier probable (frais de courtage, erreur de saisie, décalage de date de valeur).
        3. Sois très bref (2 phrases maximum).
        """
        
        return self._call_api_with_retry(prompt)


# df_front = pd.read_csv('./dataset/data_front_office.csv', parse_dates=['Date'])
# df_back = pd.read_csv('./dataset/data_back_office.csv', parse_dates=['Date'])

# engine = ReconciliationEngine(threshold=0.85)
# recon_results = engine.reconcile(df_front, df_back)

# explainer = AIReconciliationExplainer()
# fo_by_trade = df_front.set_index('Trade_ID')

# recon_results['AI_Explanation'] = ""
# for idx, result_row in recon_results.iterrows():
#     if result_row['Status'] == "MATCH":
#         continue

#     row_fo = fo_by_trade.loc[result_row['FO_Trade_ID']]
#     if isinstance(row_fo, pd.DataFrame):
#         row_fo = row_fo.iloc[0]

#     row_bo = None
#     if result_row['BO_Index'] != -1:
#         row_bo = df_back.loc[int(result_row['BO_Index'])]

#     recon_results.at[idx, 'AI_Explanation'] = explainer.generate_explanation(
#         row_fo,
#         row_bo,
#         result_row['Break_Reason'],
#         result_row['Global_Score']
#     )


import streamlit as st
import pandas as pd
from engine import ReconciliationEngine, AIReconciliationExplainer

# Configuration de la page
st.set_page_config(page_title="Amundi IA Reconciliation", layout="wide")

st.title("🛡️ Plateforme de Réconciliation Automatisée (PoC)")
st.markdown("""
Cette application utilise un moteur de matching probabiliste et l'IA Générative pour identifier et expliquer les écarts de réconciliation entre le Front-Office et le Back-Office.
""")

# Initialisation des modules
engine = ReconciliationEngine(threshold=0.85)
explainer = AIReconciliationExplainer()

# --- Sidebar : Import des données ---
st.sidebar.header("📁 Chargement des Flux")
fo_file = st.sidebar.file_uploader("Fichier Front-Office (CSV)", type="csv")
bo_file = st.sidebar.file_uploader("Fichier Back-Office (CSV)", type="csv")

if fo_file and bo_file:
    df_fo = pd.read_csv(fo_file, parse_dates=['Date'])
    df_bo = pd.read_csv(bo_file, parse_dates=['Date'])

    # Réinitialise l'état si les fichiers changent
    current_signature = (fo_file.name, fo_file.size, bo_file.name, bo_file.size)
    if st.session_state.get("file_signature") != current_signature:
        st.session_state.pop("recon_results", None)
        st.session_state.pop("df_fo", None)
        st.session_state.pop("df_bo", None)
        st.session_state["file_signature"] = current_signature
    
    if st.sidebar.button("Lancer la Réconciliation"):
        with st.spinner("Analyse des flux en cours..."):
            # 1. Exécution du moteur de matching
            st.session_state["recon_results"] = engine.reconcile(df_fo, df_bo)
            st.session_state["df_fo"] = df_fo
            st.session_state["df_bo"] = df_bo

    if "recon_results" in st.session_state:
        results = st.session_state["recon_results"]
        df_fo = st.session_state["df_fo"]
        df_bo = st.session_state["df_bo"]

        # 2. Calcul des KPIs
        total = len(results)
        matches = len(results[results['Status'] == 'MATCH'])
        suggestions = len(results[results['Status'] == 'SUGGESTION'])
        unmatched = len(results[results['Status'] == 'UNMATCHED'])

        # --- Affichage des KPIs ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Taux d'Auto-Match", f"{(matches/total):.1%}", delta=None)
        col2.metric("Suggestions IA", suggestions)
        col3.metric("Alertes (Unmatched)", unmatched)

        # --- Table des résultats ---
        st.subheader("📋 Détail des opérations")

        # On ne garde que les lignes intéressantes pour la démo (Anomalies & Suggestions)
        view_filter = st.selectbox("Filtrer la vue :", ["Toutes", "Suggestions IA uniquement", "Écarts critiques"])

        if view_filter == "Suggestions IA uniquement":
            display_df = results[results['Status'] == 'SUGGESTION']
        elif view_filter == "Écarts critiques":
            display_df = results[results['Status'] == 'UNMATCHED']
        else:
            display_df = results

        # Affichage interactif
        st.dataframe(display_df, width='stretch')

        # --- Focus sur une suggestion (Analyse IA) ---
        if suggestions > 0:
            st.subheader("🤖 Analyse approfondie par l'IA")
            st.info("L'IA analyse ici les lignes en statut 'SUGGESTION' pour proposer une correction.")

            suggestions_df = results[results['Status'] == 'SUGGESTION'].copy()

            mode = st.radio(
                "Mode d'analyse :",
                ["Transaction individuelle", "Toutes les suggestions"],
                horizontal=True
            )

            def _get_rows_from_result(result_row):
                fo_candidates = df_fo[df_fo['Trade_ID'] == result_row['FO_Trade_ID']]
                row_fo_local = fo_candidates.iloc[0] if not fo_candidates.empty else None

                row_bo_local = None
                bo_idx = int(result_row['BO_Index']) if pd.notna(result_row['BO_Index']) else -1
                if bo_idx >= 0 and bo_idx in df_bo.index:
                    row_bo_local = df_bo.loc[bo_idx]

                return row_fo_local, row_bo_local

            if mode == "Transaction individuelle":
                selected_idx = st.selectbox(
                    "Choisissez une suggestion à analyser :",
                    suggestions_df.index,
                    format_func=lambda idx: (
                        f"Trade {suggestions_df.loc[idx, 'FO_Trade_ID']} | "
                        f"Score {suggestions_df.loc[idx, 'Global_Score']:.2f} | "
                        f"{suggestions_df.loc[idx, 'Break_Reason']}"
                    )
                )

                row_res = suggestions_df.loc[selected_idx]
                row_fo, row_bo = _get_rows_from_result(row_res)

                col_a, col_b = st.columns(2)
                with col_a:
                    st.write("**Données Front :**")
                    st.json(row_fo.to_dict() if row_fo is not None else {"erreur": "Trade FO introuvable"})
                with col_b:
                    st.write("**Données Back (Match Probable) :**")
                    st.json(row_bo.to_dict() if row_bo is not None else {"info": "Aucune correspondance BO valide"})

                if st.button("Générer l'explication métier", key="generate_one"):
                    if row_fo is None:
                        st.error("Impossible de générer l'explication : transaction Front-Office introuvable.")
                    else:
                        explanation = explainer.generate_explanation(
                            row_fo, row_bo, row_res['Break_Reason'], row_res['Global_Score']
                        )
                        st.success(f"**Analyse Amundi IA :** {explanation}")

            else:
                st.caption(f"{len(suggestions_df)} suggestion(s) seront analysées en une fois.")

                if st.button("Générer les explications pour toutes les suggestions", key="generate_all"):
                    explanations = []
                    with st.spinner("Génération des explications en cours..."):
                        for _, row_res in suggestions_df.iterrows():
                            row_fo, row_bo = _get_rows_from_result(row_res)
                            if row_fo is None:
                                explanation = "Transaction Front-Office introuvable, explication impossible."
                            else:
                                explanation = explainer.generate_explanation(
                                    row_fo, row_bo, row_res['Break_Reason'], row_res['Global_Score']
                                )

                            explanations.append({
                                "FO_Trade_ID": row_res['FO_Trade_ID'],
                                "Global_Score": row_res['Global_Score'],
                                "Break_Reason": row_res['Break_Reason'],
                                "AI_Explanation": explanation
                            })

                    st.success("Explications générées pour toutes les suggestions.")
                    st.dataframe(pd.DataFrame(explanations), width='stretch')
    else:
        st.info("Cliquez sur 'Lancer la Réconciliation' pour afficher les résultats.")

else:
    st.info("Veuillez charger les deux fichiers CSV dans la barre latérale pour commencer.")
import os
import uuid
import importlib
from datetime import datetime, timezone

import pandas as pd


class PostgresPersistence:
    """Persistance PostgreSQL pour les runs de réconciliation et leurs résultats."""

    def __init__(self):
        self.database_url = os.getenv("DATABASE_URL")

    @property
    def enabled(self):
        return bool(self.database_url)

    def _connect(self):
        try:
            psycopg2 = importlib.import_module("psycopg2")
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "Le package psycopg2-binary est requis pour la persistance PostgreSQL. "
                "Installez les dépendances du projet."
            ) from e

        return psycopg2.connect(self.database_url)

    def ensure_schema(self):
        ddl_runs = """
        CREATE TABLE IF NOT EXISTS reconciliation_runs (
            run_id UUID PRIMARY KEY,
            created_at TIMESTAMPTZ NOT NULL,
            source_type TEXT,
            source_ref TEXT,
            threshold NUMERIC(5, 4) NOT NULL,
            ml_weight NUMERIC(5, 4) NOT NULL,
            total_count INTEGER NOT NULL,
            match_count INTEGER NOT NULL,
            suggestion_count INTEGER NOT NULL,
            unmatched_count INTEGER NOT NULL,
            avg_ml_probability NUMERIC(6, 4),
            ml_samples INTEGER,
            ml_positive_rate NUMERIC(6, 4),
            ml_accuracy NUMERIC(6, 4),
            ml_roc_auc NUMERIC(6, 4)
        );
        """

        ddl_results = """
        CREATE TABLE IF NOT EXISTS reconciliation_results (
            id BIGSERIAL PRIMARY KEY,
            run_id UUID NOT NULL REFERENCES reconciliation_runs(run_id) ON DELETE CASCADE,
            fo_trade_id TEXT,
            bo_index INTEGER,
            rule_score NUMERIC(6, 4),
            ml_probability NUMERIC(6, 4),
            global_score NUMERIC(6, 4),
            status TEXT NOT NULL,
            break_reason TEXT
        );
        """

        ddl_indexes = """
        CREATE INDEX IF NOT EXISTS idx_recon_results_run_id ON reconciliation_results(run_id);
        CREATE INDEX IF NOT EXISTS idx_recon_results_status ON reconciliation_results(status);
        CREATE INDEX IF NOT EXISTS idx_recon_results_fo_trade_id ON reconciliation_results(fo_trade_id);
        """

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(ddl_runs)
                cur.execute(ddl_results)
                cur.execute(ddl_indexes)
            conn.commit()

    @staticmethod
    def _nan_to_none(value):
        if pd.isna(value):
            return None
        return value

    def save_run_and_results(
        self,
        results_df,
        threshold,
        ml_weight,
        source_type,
        source_ref,
        ml_metrics=None,
    ):
        if ml_metrics is None:
            ml_metrics = {}

        run_id = uuid.uuid4()
        created_at = datetime.now(timezone.utc)

        total_count = int(len(results_df))
        match_count = int((results_df["Status"] == "MATCH").sum())
        suggestion_count = int((results_df["Status"] == "SUGGESTION").sum())
        unmatched_count = int((results_df["Status"] == "UNMATCHED").sum())

        avg_ml_probability = None
        if "ML_Probability" in results_df.columns:
            avg_val = results_df["ML_Probability"].dropna().mean()
            if pd.notna(avg_val):
                avg_ml_probability = float(avg_val)

        run_values = (
            str(run_id),
            created_at,
            source_type,
            source_ref,
            float(threshold),
            float(ml_weight),
            total_count,
            match_count,
            suggestion_count,
            unmatched_count,
            avg_ml_probability,
            ml_metrics.get("samples"),
            ml_metrics.get("positive_rate"),
            ml_metrics.get("accuracy"),
            ml_metrics.get("roc_auc"),
        )

        result_rows = []
        for _, row in results_df.iterrows():
            bo_index = self._nan_to_none(row.get("BO_Index"))
            result_rows.append(
                (
                    str(run_id),
                    self._nan_to_none(row.get("FO_Trade_ID")),
                    int(bo_index) if bo_index is not None else None,
                    self._nan_to_none(row.get("Rule_Score")),
                    self._nan_to_none(row.get("ML_Probability")),
                    self._nan_to_none(row.get("Global_Score")),
                    self._nan_to_none(row.get("Status")),
                    self._nan_to_none(row.get("Break_Reason")),
                )
            )

        with self._connect() as conn:
            try:
                execute_values = importlib.import_module("psycopg2.extras").execute_values
            except ModuleNotFoundError as e:
                raise RuntimeError(
                    "Le package psycopg2-binary est requis pour la persistance PostgreSQL."
                ) from e

            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO reconciliation_runs (
                        run_id, created_at, source_type, source_ref, threshold, ml_weight,
                        total_count, match_count, suggestion_count, unmatched_count,
                        avg_ml_probability, ml_samples, ml_positive_rate, ml_accuracy, ml_roc_auc
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    run_values,
                )

                execute_values(
                    cur,
                    """
                    INSERT INTO reconciliation_results (
                        run_id, fo_trade_id, bo_index, rule_score, ml_probability, global_score, status, break_reason
                    ) VALUES %s
                    """,
                    result_rows,
                )
            conn.commit()

        return str(run_id)
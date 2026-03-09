"""
Output Writer Module
--------------------
Saves labeled scenario windows to BigQuery (production) or local CSV (dev/testing).

BigQuery schema is auto-created on first write.
"""

import os
import json
import logging
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

BIGQUERY_AVAILABLE = False
try:
    from google.cloud import bigquery
    BIGQUERY_AVAILABLE = True
except ImportError:
    logger.warning("google-cloud-bigquery not installed. BigQuery writes disabled.")


BQ_SCHEMA = [
    {"name": "scenario_type",        "type": "STRING"},
    {"name": "ego_id",               "type": "INTEGER"},
    {"name": "window_start_frame",   "type": "INTEGER"},
    {"name": "window_end_frame",     "type": "INTEGER"},
    {"name": "ego_trajectory",       "type": "JSON"},
    {"name": "surrounding_vehicles", "type": "JSON"},
    {"name": "num_surrounding",      "type": "INTEGER"},
    {"name": "leader_id",            "type": "INTEGER"},
    {"name": "avg_gap_m",            "type": "FLOAT"},
    {"name": "speed_corr",           "type": "FLOAT"},
    {"name": "duration_frames",      "type": "INTEGER"},
    {"name": "from_lane",            "type": "INTEGER"},
    {"name": "to_lane",              "type": "INTEGER"},
    {"name": "lateral_disp_m",       "type": "FLOAT"},
    {"name": "cutter_id",            "type": "INTEGER"},
    {"name": "speed_drop_ms",        "type": "FLOAT"},
    {"name": "gap_after_m",          "type": "FLOAT"},
    {"name": "cutin_frame",          "type": "INTEGER"},
    {"name": "ingested_at",          "type": "TIMESTAMP"},
]


def _add_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ingested_at"] = datetime.utcnow().isoformat()
    return df


def write_to_bigquery(
    samples: pd.DataFrame,
    project_id: str,
    dataset_id: str,
    table_id: str = "scenario_windows",
) -> bool:
    """
    Write samples to BigQuery. Creates table if it doesn't exist.
    Returns True on success.
    """
    if not BIGQUERY_AVAILABLE:
        logger.error("BigQuery client not available. Install google-cloud-bigquery.")
        return False

    client    = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset_id}.{table_id}"

    samples = _add_metadata(samples)

    # Convert JSON cols to strings if not already
    for col in ["ego_trajectory", "surrounding_vehicles"]:
        if col in samples.columns and samples[col].dtype != object:
            samples[col] = samples[col].apply(json.dumps)

    try:
        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
            autodetect=True,
        )
        job = client.load_table_from_dataframe(samples, table_ref, job_config=job_config)
        job.result()  # wait for job to finish
        logger.info(f"Wrote {len(samples)} rows to {table_ref}")
        return True
    except Exception as e:
        logger.error(f"BigQuery write failed: {e}")
        return False


def write_to_csv(
    samples: pd.DataFrame,
    output_dir: str = "output",
    filename: str = None,
) -> str:
    """
    Save samples to a local CSV file. Used as fallback when BQ is unavailable
    or during local development / testing.
    Returns path to the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    if filename is None:
        ts       = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"scenario_windows_{ts}.csv"

    samples = _add_metadata(samples)
    path    = os.path.join(output_dir, filename)
    samples.to_csv(path, index=False)
    logger.info(f"Wrote {len(samples)} rows to {path}")
    return path


def write_output(
    samples: pd.DataFrame,
    use_bigquery: bool = False,
    project_id: str = None,
    dataset_id: str = "ngsim_scenarios",
    output_dir: str = "output",
) -> str:
    """
    Smart output writer: use BigQuery if configured, else CSV.
    """
    if use_bigquery and project_id:
        success = write_to_bigquery(samples, project_id, dataset_id)
        if success:
            return f"bigquery://{project_id}.{dataset_id}.scenario_windows"

    # Fall back to CSV
    return write_to_csv(samples, output_dir=output_dir)
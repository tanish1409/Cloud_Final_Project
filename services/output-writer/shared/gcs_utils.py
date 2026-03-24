"""
GCS utility functions shared across all microservices.
"""
import os, logging, pandas as pd
from google.cloud import storage
logger = logging.getLogger(__name__)
BUCKET_NAME = os.environ.get("GCS_BUCKET", "ngsim-raw-data-ngsim-scenarios-proj")

def _get_bucket():
    return storage.Client().bucket(BUCKET_NAME)

def download_file(gcs_path, local_path):
    _get_bucket().blob(gcs_path).download_to_filename(local_path)
    logger.info(f"Downloaded gs://{BUCKET_NAME}/{gcs_path} -> {local_path}")

def upload_file(local_path, gcs_path):
    _get_bucket().blob(gcs_path).upload_from_filename(local_path)
    logger.info(f"Uploaded {local_path} -> gs://{BUCKET_NAME}/{gcs_path}")

def read_parquet(gcs_path):
    local_tmp = f"/tmp/{os.path.basename(gcs_path)}"
    download_file(gcs_path, local_tmp)
    df = pd.read_parquet(local_tmp); os.remove(local_tmp)
    logger.info(f"Read {len(df)} rows from {gcs_path}"); return df

def write_parquet(df, gcs_path):
    local_tmp = f"/tmp/{os.path.basename(gcs_path)}"
    df.to_parquet(local_tmp, index=False, engine="pyarrow")
    upload_file(local_tmp, gcs_path); os.remove(local_tmp)
    logger.info(f"Wrote {len(df)} rows to {gcs_path}")

def write_marker(gcs_path):
    _get_bucket().blob(gcs_path).upload_from_string("")
    logger.info(f"Wrote marker: {gcs_path}")

def marker_exists(gcs_path):
    return _get_bucket().blob(gcs_path).exists()

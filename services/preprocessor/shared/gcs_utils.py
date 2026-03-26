"""
GCS utility functions shared across all microservices.
Handles reading/writing Parquet files and marker files to Google Cloud Storage.
"""
import os
import uuid
import logging
import pandas as pd
from google.cloud import storage

logger = logging.getLogger(__name__)

BUCKET_NAME = os.environ.get("GCS_BUCKET", "ngsim-raw-data-ngsim-scenarios-proj")


def _get_bucket():
    client = storage.Client()
    return client.bucket(BUCKET_NAME)


def _unique_tmp(basename: str) -> str:
    """Generate a unique temp path to avoid collisions between concurrent requests."""
    name, ext = os.path.splitext(basename)
    return f"/tmp/{name}_{uuid.uuid4().hex[:8]}{ext}"


def download_file(gcs_path: str, local_path: str):
    bucket = _get_bucket()
    blob = bucket.blob(gcs_path)
    blob.download_to_filename(local_path)
    logger.info(f"Downloaded gs://{BUCKET_NAME}/{gcs_path} -> {local_path}")


def upload_file(local_path: str, gcs_path: str):
    bucket = _get_bucket()
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    logger.info(f"Uploaded {local_path} -> gs://{BUCKET_NAME}/{gcs_path}")


def read_parquet(gcs_path: str) -> pd.DataFrame:
    local_tmp = _unique_tmp(os.path.basename(gcs_path))
    download_file(gcs_path, local_tmp)
    df = pd.read_parquet(local_tmp)
    os.remove(local_tmp)
    logger.info(f"Read {len(df)} rows from {gcs_path}")
    return df


def write_parquet(df: pd.DataFrame, gcs_path: str):
    local_tmp = _unique_tmp(os.path.basename(gcs_path))
    df.to_parquet(local_tmp, index=False, engine="pyarrow")
    upload_file(local_tmp, gcs_path)
    os.remove(local_tmp)
    logger.info(f"Wrote {len(df)} rows to {gcs_path}")


def write_marker(gcs_path: str):
    bucket = _get_bucket()
    blob = bucket.blob(gcs_path)
    blob.upload_from_string("")
    logger.info(f"Wrote marker: {gcs_path}")


def marker_exists(gcs_path: str) -> bool:
    bucket = _get_bucket()
    blob = bucket.blob(gcs_path)
    return blob.exists()

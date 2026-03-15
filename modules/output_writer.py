import os
import json
import logging
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

GCS_AVAILABLE = False
try:
    from google.cloud import storage as gcs
    GCS_AVAILABLE = True
except ImportError:
    logger.warning("google-cloud-storage not installed. GCS writes disabled.")


def _add_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ingested_at"] = datetime.utcnow().isoformat()
    return df


def _ensure_json_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure JSON columns are strings for Parquet compatibility."""
    for col in ["ego_trajectory", "surrounding_vehicles"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda v: json.dumps(v) if not isinstance(v, str) else v
            )
    return df


def write_to_gcs_parquet(
    samples: pd.DataFrame,
    bucket_name: str,
    prefix: str = "output",
) -> str:
    """
    Write samples to GCS as a Parquet file.
    Returns the gs:// URI of the written file.
    """
    if not GCS_AVAILABLE:
        logger.error("google-cloud-storage not available.")
        return None

    samples = _add_metadata(samples)
    samples = _ensure_json_strings(samples)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"scenario_windows_{ts}.parquet"
    blob_path = f"{prefix}/{filename}"
    gcs_uri = f"gs://{bucket_name}/{blob_path}"

    # Write parquet to a temp local file first, then upload
    local_tmp = f"/tmp/{filename}"
    samples.to_parquet(local_tmp, index=False, engine="pyarrow")

    client = gcs.Client()
    bucket = client.bucket(bucket_name)
    bucket.blob(blob_path).upload_from_filename(local_tmp)
    os.remove(local_tmp)

    logger.info(f"Wrote {len(samples)} rows to {gcs_uri}")
    return gcs_uri


def write_to_local_parquet(
    samples: pd.DataFrame,
    output_dir: str = "output",
    filename: str = None,
) -> str:
    """Write samples to local Parquet file. Returns path."""
    os.makedirs(output_dir, exist_ok=True)
    if filename is None:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"scenario_windows_{ts}.parquet"

    samples = _add_metadata(samples)
    samples = _ensure_json_strings(samples)
    path = os.path.join(output_dir, filename)
    samples.to_parquet(path, index=False, engine="pyarrow")
    logger.info(f"Wrote {len(samples)} rows to {path}")
    return path


def write_output(
    samples: pd.DataFrame,
    use_gcs: bool = False,
    bucket_name: str = None,
    gcs_prefix: str = "output",
    output_dir: str = "output",
) -> str:
    """
    Smart output writer:
      - GCS Parquet if use_gcs=True and bucket_name provided
      - Local Parquet otherwise
    Returns path or gs:// URI.
    """
    if use_gcs and bucket_name:
        uri = write_to_gcs_parquet(samples, bucket_name, prefix=gcs_prefix)
        if uri:
            return uri
        logger.warning("GCS write failed, falling back to local Parquet.")

    return write_to_local_parquet(samples, output_dir=output_dir)

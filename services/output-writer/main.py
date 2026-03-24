"""
Output writer microservice.
Subscribes to windows-ready, writes final Parquet to GCS.
"""
import os
import json
import logging
from datetime import datetime
from flask import Flask, request, jsonify

from shared.gcs_utils import read_parquet, write_parquet
from shared.pubsub_utils import parse_pubsub_message

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("output-writer-service")

app = Flask(__name__)


def _add_metadata(df):
    import pandas as pd
    df = df.copy()
    df["ingested_at"] = datetime.utcnow().isoformat()
    return df


def _ensure_json_strings(df):
    for col in ["ego_trajectory", "surrounding_vehicles"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda v: json.dumps(v) if not isinstance(v, str) else v
            )
    return df


@app.route("/", methods=["POST"])
def handle():
    try:
        payload = parse_pubsub_message(request)
    except Exception as e:
        logger.error(f"Bad request: {e}")
        return "Bad request", 400

    run_id = payload["run_id"]
    windows_path = payload["windows_path"]

    logger.info(f"Output writer starting: run_id={run_id}")

    # Read windows from GCS
    samples = read_parquet(windows_path)

    # Add metadata and ensure JSON strings
    samples = _add_metadata(samples)
    samples = _ensure_json_strings(samples)

    # Write final output
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_path = f"pipeline/{run_id}/final/scenario_windows_{ts}.parquet"
    write_parquet(samples, output_path)

    logger.info(f"Final output: {len(samples)} windows written to {output_path}")

    return jsonify({
        "status": "success",
        "windows_written": len(samples),
        "output_path": output_path,
    }), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

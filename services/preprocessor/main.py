"""
Preprocessor microservice.
Triggered manually (or via Eventarc). Downloads raw CSV from GCS,
runs the preprocessing pipeline, writes clean.parquet to GCS,
and publishes to the clean-data-ready Pub/Sub topic.
"""
import os
import logging
from datetime import datetime
from flask import Flask, request, jsonify

from shared.gcs_utils import download_file, write_parquet
from shared.pubsub_utils import publish
from preprocessor import preprocess

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("preprocessor-service")

app = Flask(__name__)

BUCKET = os.environ.get("GCS_BUCKET", "ngsim-raw-data-ngsim-scenarios-proj")
INPUT_GCS_PATH = os.environ.get("INPUT_GCS_PATH", "i80/trajectories-0400-0415.csv")


@app.route("/", methods=["POST"])
def handle():
    """
    Accepts a POST request to start preprocessing.
    Can be triggered manually or via Pub/Sub/Eventarc.
    """
    # Generate a unique run ID
    run_id = datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")

    # Check if the request body has a custom run_id or input path
    body = request.get_json(silent=True) or {}
    run_id = body.get("run_id", run_id)
    input_path = body.get("input_gcs_path", INPUT_GCS_PATH)

    logger.info(f"Starting preprocessing: run_id={run_id}, input={input_path}")

    # Step 1: Download raw CSV from GCS
    local_csv = "/tmp/raw_input.csv"
    download_file(input_path, local_csv)

    # Step 2: Run preprocessing (existing function, unchanged)
    df = preprocess(local_csv)
    os.remove(local_csv)

    if df.empty:
        logger.error("Preprocessing returned empty DataFrame")
        return jsonify({"error": "Empty result"}), 500

    logger.info(f"Clean data: {len(df)} rows, {df['Vehicle_ID'].nunique()} vehicles")

    # Step 3: Write clean.parquet to GCS
    gcs_output_path = f"pipeline/{run_id}/clean.parquet"
    write_parquet(df, gcs_output_path)

    # Step 4: Publish to Pub/Sub
    publish("clean-data-ready", run_id, clean_path=gcs_output_path)

    return jsonify({
        "status": "success",
        "run_id": run_id,
        "clean_rows": len(df),
        "unique_vehicles": int(df["Vehicle_ID"].nunique()),
        "output": gcs_output_path,
    }), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

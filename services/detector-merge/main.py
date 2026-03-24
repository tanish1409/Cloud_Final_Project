"""
On-ramp merge detector microservice.
Subscribes to clean-data-ready, runs merge detection,
writes events to GCS, publishes to events-ready.
"""
import os
import logging
from flask import Flask, request, jsonify

from shared.gcs_utils import read_parquet, write_parquet, write_marker
from shared.pubsub_utils import publish, parse_pubsub_message
from detector import detect_onramp_merge

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("detector-merge")

app = Flask(__name__)


@app.route("/", methods=["POST"])
def handle():
    try:
        payload = parse_pubsub_message(request)
    except Exception as e:
        logger.error(f"Bad request: {e}")
        return "Bad request", 400

    run_id = payload["run_id"]
    clean_path = payload["clean_path"]

    logger.info(f"On-ramp merge detection starting: run_id={run_id}")

    df = read_parquet(clean_path)
    events = detect_onramp_merge(df)
    logger.info(f"On-ramp merge events detected: {len(events)}")

    output_path = f"pipeline/{run_id}/events/onramp_merge.parquet"
    write_parquet(events, output_path)
    write_marker(f"pipeline/{run_id}/events/.done_onramp_merge")
    publish("events-ready", run_id, detector="onramp_merge", event_count=len(events))

    return jsonify({"status": "success", "events": len(events)}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

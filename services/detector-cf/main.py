"""
Car-following detector microservice.
Subscribes to clean-data-ready, runs car-following detection,
writes events to GCS, publishes to events-ready.
"""
import os
import logging
from flask import Flask, request, jsonify

from shared.gcs_utils import read_parquet, write_parquet, write_marker
from shared.pubsub_utils import publish, parse_pubsub_message
from detector import detect_car_following

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("detector-cf")

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

    logger.info(f"Car-following detection starting: run_id={run_id}")

    # Read clean data from GCS
    df = read_parquet(clean_path)

    # Run detection
    events = detect_car_following(df)
    logger.info(f"Car-following events detected: {len(events)}")

    # Write results to GCS
    output_path = f"pipeline/{run_id}/events/car_following.parquet"
    if not events.empty:
        write_parquet(events, output_path)
    else:
        # Write empty parquet so downstream can still read it
        write_parquet(events, output_path)

    # Write completion marker
    write_marker(f"pipeline/{run_id}/events/.done_car_following")

    # Publish to events-ready
    publish("events-ready", run_id, detector="car_following", event_count=len(events))

    return jsonify({"status": "success", "events": len(events)}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

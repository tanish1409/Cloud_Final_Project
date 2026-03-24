"""
Windower microservice.
Subscribes to events-ready. On each trigger, checks if all 3 detectors
have completed (via marker files). If yes, merges events, runs windowing,
writes windows.parquet, and publishes to windows-ready.
"""
import os
import logging
import pandas as pd
from flask import Flask, request, jsonify

from shared.gcs_utils import read_parquet, write_parquet, marker_exists
from shared.pubsub_utils import publish, parse_pubsub_message
from windower import segment_windows

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("windower-service")

app = Flask(__name__)

DETECTOR_MARKERS = [".done_car_following", ".done_onramp_merge", ".done_lane_cutin"]
EVENT_FILES = ["car_following.parquet", "onramp_merge.parquet", "lane_cutin.parquet"]


@app.route("/", methods=["POST"])
def handle():
    try:
        payload = parse_pubsub_message(request)
    except Exception as e:
        logger.error(f"Bad request: {e}")
        return "Bad request", 400

    run_id = payload["run_id"]
    logger.info(f"Windower triggered: run_id={run_id}, from detector={payload.get('detector')}")

    # ── Merge barrier: check if all 3 detectors are done ──────────────────
    events_dir = f"pipeline/{run_id}/events"
    all_done = all(marker_exists(f"{events_dir}/{m}") for m in DETECTOR_MARKERS)

    if not all_done:
        logger.info("Not all detectors done yet, acknowledging and waiting.")
        return "Waiting for other detectors", 200

    logger.info("All 3 detectors complete — proceeding with windowing.")

    # ── Read and merge all event files ────────────────────────────────────
    all_events = []
    for fname in EVENT_FILES:
        try:
            events_df = read_parquet(f"{events_dir}/{fname}")
            if not events_df.empty:
                all_events.append(events_df)
                logger.info(f"  {fname}: {len(events_df)} events")
        except Exception as e:
            logger.warning(f"  {fname}: failed to read ({e})")

    if not all_events:
        logger.warning("No events found from any detector.")
        return jsonify({"status": "no_events"}), 200

    merged_events = pd.concat(all_events, ignore_index=True)
    merged_events = merged_events.sort_values(["ego_id", "start_frame"]).reset_index(drop=True)
    logger.info(f"Total merged events: {len(merged_events)}")

    # ── Read clean data (needed for surrounding vehicles) ─────────────────
    clean_path = f"pipeline/{run_id}/clean.parquet"
    df = read_parquet(clean_path)

    # ── Run windowing (existing function, unchanged) ──────────────────────
    samples = segment_windows(df, merged_events)

    if samples.empty:
        logger.warning("No windows produced.")
        return jsonify({"status": "no_windows"}), 200

    # ── Write windows.parquet to GCS ──────────────────────────────────────
    output_path = f"pipeline/{run_id}/windows.parquet"
    write_parquet(samples, output_path)

    # ── Publish to windows-ready ──────────────────────────────────────────
    publish("windows-ready", run_id, windows_path=output_path, window_count=len(samples))

    return jsonify({
        "status": "success",
        "total_events": len(merged_events),
        "windows_created": len(samples),
    }), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

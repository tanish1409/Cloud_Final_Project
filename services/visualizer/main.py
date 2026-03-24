"""
Visualizer microservice.
Subscribes to windows-ready, generates diagnostic plots,
uploads PNGs to GCS.
"""
import os
import logging
from flask import Flask, request, jsonify

from shared.gcs_utils import read_parquet, upload_file
from shared.pubsub_utils import parse_pubsub_message
from visualizer import visualize_all, plot_summary_dashboard

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("visualizer-service")

app = Flask(__name__)


@app.route("/", methods=["POST"])
def handle():
    try:
        payload = parse_pubsub_message(request)
    except Exception as e:
        logger.error(f"Bad request: {e}")
        return "Bad request", 400

    run_id = payload["run_id"]
    windows_path = payload["windows_path"]

    logger.info(f"Visualizer starting: run_id={run_id}")

    # Read data from GCS
    samples = read_parquet(windows_path)
    clean_path = f"pipeline/{run_id}/clean.parquet"
    df = read_parquet(clean_path)

    # Generate plots to local /tmp
    local_plot_dir = "/tmp/plots"
    os.makedirs(local_plot_dir, exist_ok=True)

    plot_paths = visualize_all(df, samples, output_dir=local_plot_dir)
    summary_path = plot_summary_dashboard(samples, os.path.join(local_plot_dir, "summary.png"))
    plot_paths.append(summary_path)

    logger.info(f"Generated {len(plot_paths)} plots")

    # Upload plots to GCS
    uploaded = 0
    for local_path in plot_paths:
        if local_path and os.path.exists(local_path):
            fname = os.path.basename(local_path)
            gcs_path = f"pipeline/{run_id}/plots/{fname}"
            upload_file(local_path, gcs_path)
            os.remove(local_path)
            uploaded += 1

    logger.info(f"Uploaded {uploaded} plots to GCS")

    return jsonify({"status": "success", "plots_uploaded": uploaded}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

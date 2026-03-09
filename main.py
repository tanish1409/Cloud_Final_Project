"""
main.py  –  NGSIM Scenario Extraction Pipeline (Phase 1 Monolithic)
-------------------------------------------------------------------
Entry point for the Cloud Run service.
Can also be run locally:

    python main.py --input data/i80_trajectories.csv --output output/ --visualize

Environment variables (for Cloud Run / GCP deployment):
    INPUT_GCS_PATH    gs://your-bucket/ngsim/i80_trajectories.csv
    USE_BIGQUERY      true | false  (default: false → local CSV)
    GCP_PROJECT_ID    your-project-id
    BQ_DATASET_ID     ngsim_scenarios
    VISUALIZE         true | false  (default: true)
"""

import os
import sys
import argparse
import logging
import traceback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger("main")

from modules.preprocessor     import preprocess
from modules.scenario_detector import detect_all_scenarios
from modules.windower          import segment_windows
from modules.output_writer     import write_output
from modules.visualizer        import visualize_all, plot_summary_dashboard


def run_pipeline(
    input_path: str,
    output_dir: str = "output",
    use_bigquery: bool = False,
    project_id: str = None,
    dataset_id: str = "ngsim_scenarios",
    visualize: bool = True,
) -> dict:
    """
    Full end-to-end pipeline.
    Returns a summary dict with counts and output paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

    # ── Step 1: Preprocess ────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1 – Preprocessing")
    logger.info("=" * 60)
    df = preprocess(input_path)

    if df.empty:
        logger.error("Preprocessing returned empty DataFrame. Check your input file.")
        sys.exit(1)

    # ── Step 2: Detect Scenarios ──────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2 – Scenario Detection")
    logger.info("=" * 60)
    events = detect_all_scenarios(df)

    if events.empty:
        logger.warning("No scenarios detected. Check thresholds or input data.")
        return {"status": "no_events", "clean_rows": len(df)}

    logger.info(f"\nEvent breakdown:\n{events['scenario_type'].value_counts().to_string()}")

    # ── Step 3: Window Segmentation ───────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3 – Window Segmentation")
    logger.info("=" * 60)
    samples = segment_windows(df, events)

    # ── Step 4: Visualize ─────────────────────────────────────────────────────
    plot_paths = []
    if visualize and not samples.empty:
        logger.info("=" * 60)
        logger.info("STEP 4 – Visualization")
        logger.info("=" * 60)
        plot_paths = visualize_all(df, samples, output_dir=os.path.join(output_dir, "plots"))
        summary_plot = plot_summary_dashboard(samples, os.path.join(output_dir, "plots", "summary.png"))
        plot_paths.append(summary_plot)
        logger.info(f"Generated {len(plot_paths)} plot(s)")

    # ── Step 5: Write Output ──────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5 – Writing Output")
    logger.info("=" * 60)
    output_path = write_output(
        samples,
        use_bigquery=use_bigquery,
        project_id=project_id,
        dataset_id=dataset_id,
        output_dir=output_dir,
    )

    summary = {
        "status":          "success",
        "clean_rows":      len(df),
        "unique_vehicles": int(df["Vehicle_ID"].nunique()),
        "events_detected": len(events),
        "windows_created": len(samples),
        "scenario_counts": events["scenario_type"].value_counts().to_dict(),
        "output_path":     output_path,
        "plots":           plot_paths,
    }

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"  Clean rows:       {summary['clean_rows']}")
    logger.info(f"  Unique vehicles:  {summary['unique_vehicles']}")
    logger.info(f"  Events detected:  {summary['events_detected']}")
    logger.info(f"  Windows created:  {summary['windows_created']}")
    logger.info(f"  Output:           {summary['output_path']}")
    logger.info("=" * 60)

    return summary


def _parse_args():
    parser = argparse.ArgumentParser(description="NGSIM Scenario Extraction Pipeline")
    parser.add_argument("--input",      required=True, help="Path to raw NGSIM CSV or gs:// URI")
    parser.add_argument("--output",     default="output", help="Output directory")
    parser.add_argument("--bigquery",   action="store_true", help="Write output to BigQuery")
    parser.add_argument("--project",    default=None, help="GCP Project ID")
    parser.add_argument("--dataset",    default="ngsim_scenarios", help="BigQuery dataset")
    parser.add_argument("--visualize",  action="store_true", default=True, help="Generate plots")
    parser.add_argument("--no-visualize", dest="visualize", action="store_false")
    return parser.parse_args()


if __name__ == "__main__":
    # Support both CLI args and environment variables (for Cloud Run)
    if len(sys.argv) > 1:
        args = _parse_args()
        input_path   = args.input
        output_dir   = args.output
        use_bigquery = args.bigquery
        project_id   = args.project
        dataset_id   = args.dataset
        visualize    = args.visualize
    else:
        # Read from environment variables (Cloud Run deployment mode)
        input_path   = os.environ.get("INPUT_GCS_PATH") or os.environ.get("INPUT_PATH", "")
        output_dir   = os.environ.get("OUTPUT_DIR", "output")
        use_bigquery = os.environ.get("USE_BIGQUERY", "false").lower() == "true"
        project_id   = os.environ.get("GCP_PROJECT_ID")
        dataset_id   = os.environ.get("BQ_DATASET_ID", "ngsim_scenarios")
        visualize    = os.environ.get("VISUALIZE", "true").lower() == "true"

        if not input_path:
            logger.error("No input path provided. Set INPUT_GCS_PATH or INPUT_PATH env var, or use CLI args.")
            sys.exit(1)

    # GCS download if needed
    if input_path.startswith("gs://"):
        logger.info(f"Downloading from GCS: {input_path}")
        from google.cloud import storage
        local_path = "/tmp/ngsim_raw.csv"
        bucket_name = input_path.replace("gs://", "").split("/")[0]
        blob_path   = "/".join(input_path.replace("gs://", "").split("/")[1:])
        gcs_client  = storage.Client()
        bucket      = gcs_client.bucket(bucket_name)
        bucket.blob(blob_path).download_to_filename(local_path)
        logger.info(f"Downloaded to {local_path}")
        input_path = local_path

    try:
        summary = run_pipeline(
            input_path=input_path,
            output_dir=output_dir,
            use_bigquery=use_bigquery,
            project_id=project_id,
            dataset_id=dataset_id,
            visualize=visualize,
        )
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        traceback.print_exc()
        sys.exit(1)
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

from modules.visualizer import visualize_all, plot_summary_dashboard
from modules.output_writer import write_output
from modules.windower import segment_windows
from modules.scenario_detector import detect_all_scenarios
from modules.preprocessor import preprocess
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


def run_pipeline(
    input_path: str,
    output_dir: str = "output",
    use_gcs: bool = False,
    bucket_name: str = None,
    gcs_prefix: str = "output",
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
        logger.error(
            "Preprocessing returned empty DataFrame. Check your input file.")
        sys.exit(1)

    # ── Step 2: Detect Scenarios ──────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2 – Scenario Detection")
    logger.info("=" * 60)
    events = detect_all_scenarios(df)

    if events.empty:
        logger.warning(
            "No scenarios detected. Check thresholds or input data.")
        return {"status": "no_events", "clean_rows": len(df)}

    logger.info(
        f"\nEvent breakdown:\n{events['scenario_type'].value_counts().to_string()}")

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
        plot_paths = visualize_all(
            df, samples, output_dir=os.path.join(output_dir, "plots"))
        summary_plot = plot_summary_dashboard(
            samples, os.path.join(output_dir, "plots", "summary.png"))
        plot_paths.append(summary_plot)
        logger.info(f"Generated {len(plot_paths)} plot(s)")

    # ── Step 5: Write Output ──────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5 – Writing Output")
    logger.info("=" * 60)
    output_path = write_output(
        samples,
        use_gcs=use_gcs,
        bucket_name=bucket_name,
        gcs_prefix=gcs_prefix,
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
    parser = argparse.ArgumentParser(
        description="NGSIM Scenario Extraction Pipeline")
    parser.add_argument("--input",      required=True,
                        help="Path to raw NGSIM CSV or gs:// URI")
    parser.add_argument("--output",     default="output",
                        help="Output directory")
    parser.add_argument("--gcs",        action="store_true",
                        help="Write output to GCS as Parquet")
    parser.add_argument("--bucket",     default=None, help="GCS bucket name")
    parser.add_argument("--gcs-prefix", default="output",
                        help="GCS path prefix")
    parser.add_argument("--visualize",  action="store_true",
                        default=True, help="Generate plots")
    parser.add_argument("--no-visualize", dest="visualize",
                        action="store_false")
    return parser.parse_args()


if __name__ == "__main__":
    # Support both CLI args and environment variables (for Cloud Run)
    if len(sys.argv) > 1:
        args = _parse_args()
        input_path = args.input
        output_dir = args.output
        use_gcs = args.gcs
        bucket_name = args.bucket
        gcs_prefix = args.gcs_prefix
        visualize = args.visualize
    else:
        input_path = os.environ.get(
            "INPUT_GCS_PATH") or os.environ.get("INPUT_PATH", "")
        output_dir = os.environ.get("OUTPUT_DIR", "output")
        use_gcs = os.environ.get("USE_GCS", "false").lower() == "true"
        bucket_name = os.environ.get("GCS_BUCKET_NAME")
        gcs_prefix = os.environ.get("GCS_PREFIX", "output")
        visualize = os.environ.get("VISUALIZE", "true").lower() == "true"

        if not input_path:
            logger.error(
                "No input path provided. Set INPUT_GCS_PATH or INPUT_PATH env var, or use CLI args.")
            sys.exit(1)

    # GCS download if needed
    if input_path.startswith("gs://"):
        logger.info(f"Downloading from GCS: {input_path}")
        local_path = "/tmp/ngsim_raw.csv"
        try:
            from google.cloud import storage as gcs
            # parse bucket and blob from gs://bucket/path/to/file.csv
            # strip gs://
            path_no_scheme = input_path[5:]
            bucket_part = path_no_scheme.split("/")[0]
            blob_part = "/".join(path_no_scheme.split("/")[1:])
            gcs_client = gcs.Client()
            bucket_obj = gcs_client.bucket(bucket_part)
            blob_obj = bucket_obj.blob(blob_part)
            blob_obj.download_to_filename(local_path)
            logger.info(
                f"Downloaded to {local_path} ({os.path.getsize(local_path):,} bytes)")
        except Exception as e:
            logger.error(f"GCS download failed: {e}")
            traceback.print_exc()
            sys.exit(1)
        input_path = local_path

    try:
        summary = run_pipeline(
            input_path=input_path,
            output_dir=output_dir,
            use_gcs=use_gcs,
            bucket_name=bucket_name,
            gcs_prefix=gcs_prefix,
            visualize=visualize,
        )

        # Upload plots to GCS if running in cloud mode
        if use_gcs and bucket_name and summary.get("plots"):
            logger.info("Uploading plots to GCS...")
            try:
                from google.cloud import storage as gcs
                gcs_client = gcs.Client()
                bucket_obj = gcs_client.bucket(bucket_name)
                uploaded = 0
                for local_plot in summary["plots"]:
                    if os.path.exists(local_plot):
                        fname = os.path.basename(local_plot)
                        gcs_path = f"{gcs_prefix}/plots/{fname}"
                        blob_obj = bucket_obj.blob(gcs_path)
                        blob_obj.upload_from_filename(local_plot)
                        uploaded += 1
                logger.info(
                    f"Uploaded {uploaded} plot(s) to gs://{bucket_name}/{gcs_prefix}/plots/")
            except Exception as e:
                logger.warning(f"Plot upload failed (non-fatal): {e}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        traceback.print_exc()
        sys.exit(1)

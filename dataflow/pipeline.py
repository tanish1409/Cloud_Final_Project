"""
NGSIM Scenario Extraction — Apache Beam / Dataflow Pipeline
============================================================
Single DAG that replaces the 7 Cloud Run + Pub/Sub microservices.

Stages:
  1. Read raw CSV from GCS → Preprocess
  2. Fan-out to 3 parallel detectors (car-following, merge, cut-in)
  3. Flatten/merge all events
  4. Window segmentation (gap-aware)
  5. Fan-out to Visualizer + Output Writer (parallel)
  6. Write results to GCS

Usage (Dataflow):
  python pipeline.py \
    --runner DataflowRunner \
    --project ngsim-scenarios-proj \
    --region us-central1 \
    --temp_location gs://ngsim-raw-data-ngsim-scenarios-proj/dataflow/tmp \
    --staging_location gs://ngsim-raw-data-ngsim-scenarios-proj/dataflow/staging \
    --input_csv gs://ngsim-raw-data-ngsim-scenarios-proj/i80/trajectories-0400-0415.csv \
    --output_bucket ngsim-raw-data-ngsim-scenarios-proj \
    --machine_type n1-standard-2 \
    --max_num_workers 1 \
    --disk_size_gb 30 \
    --no_use_public_ips \
    --setup_file ./setup.py
"""
import argparse
import json
import logging
import os
import tempfile
from datetime import datetime

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions

from modules.preprocessor import preprocess
from modules.detector_cf import detect_car_following
from modules.detector_merge import detect_onramp_merge
from modules.detector_cutin import detect_lane_cutin
from modules.windower import segment_windows
from modules.visualizer import visualize_all, plot_summary_dashboard

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("ngsim-pipeline")


# ── Helper: upload a local file to GCS ────────────────────────────────────────

def _upload_to_gcs(local_path, bucket_name, gcs_path):
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    logger.info(f"Uploaded {local_path} -> gs://{bucket_name}/{gcs_path}")


def _write_parquet_to_gcs(df, bucket_name, gcs_path):
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        tmp_path = f.name
    df.to_parquet(tmp_path, index=False, engine="pyarrow")
    _upload_to_gcs(tmp_path, bucket_name, gcs_path)
    os.remove(tmp_path)


# ══════════════════════════════════════════════════════════════════════════════
# DoFn definitions — each wraps an existing module function
# ══════════════════════════════════════════════════════════════════════════════

class PreprocessFn(beam.DoFn):
    """Download CSV from GCS, preprocess, emit clean DataFrame."""

    def process(self, csv_gcs_path):
        from google.cloud import storage
        import tempfile, os

        logger.info(f"Preprocessing: {csv_gcs_path}")

        # Download CSV to local temp
        bucket_name = csv_gcs_path.replace("gs://", "").split("/")[0]
        blob_path = "/".join(csv_gcs_path.replace("gs://", "").split("/")[1:])

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)

        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        blob.download_to_filename(tmp.name)
        tmp.close()

        df = preprocess(tmp.name)
        os.remove(tmp.name)

        logger.info(f"Preprocessed: {len(df)} rows, {df['Vehicle_ID'].nunique()} vehicles")
        yield df


class DetectCarFollowingFn(beam.DoFn):
    """Run car-following detection on the clean DataFrame."""

    def process(self, df):
        events = detect_car_following(df)
        logger.info(f"Car-following: {len(events)} events")
        yield events


class DetectOnrampMergeFn(beam.DoFn):
    """Run on-ramp merge detection on the clean DataFrame."""

    def process(self, df):
        events = detect_onramp_merge(df)
        logger.info(f"On-ramp merge: {len(events)} events")
        yield events


class DetectLaneCutinFn(beam.DoFn):
    """Run lane cut-in detection on the clean DataFrame."""

    def process(self, df):
        events = detect_lane_cutin(df)
        logger.info(f"Lane cut-in: {len(events)} events")
        yield events


class MergeEventsFn(beam.DoFn):
    """Merge all event DataFrames into one."""

    def process(self, event_dfs):
        non_empty = [df for df in event_dfs if not df.empty]
        if not non_empty:
            logger.warning("No events from any detector.")
            yield pd.DataFrame()
            return
        merged = pd.concat(non_empty, ignore_index=True)
        merged = merged.sort_values(["ego_id", "start_frame"]).reset_index(drop=True)
        logger.info(f"Merged events: {len(merged)}")
        yield merged


class WindowFn(beam.DoFn):
    """Segment events into 5-second windows using clean data as side input."""

    def process(self, events_df, clean_df):
        if events_df.empty:
            logger.warning("No events to window.")
            yield pd.DataFrame()
            return
        samples = segment_windows(clean_df, events_df)
        logger.info(f"Windows created: {len(samples)}")
        yield samples


class VisualizeFn(beam.DoFn):
    """Generate plots and upload to GCS."""

    def __init__(self, bucket_name, run_id):
        self.bucket_name = bucket_name
        self.run_id = run_id

    def process(self, element):
        samples, clean_df = element

        if samples.empty:
            logger.warning("No samples to visualize.")
            yield "no_plots"
            return

        # Generate plots locally
        plot_dir = tempfile.mkdtemp(prefix="plots_")
        plot_paths = visualize_all(clean_df, samples, output_dir=plot_dir)
        summary_path = plot_summary_dashboard(
            samples, os.path.join(plot_dir, "summary.png")
        )
        plot_paths.append(summary_path)

        # Upload to GCS
        uploaded = 0
        for local_path in plot_paths:
            if local_path and os.path.exists(local_path):
                fname = os.path.basename(local_path)
                gcs_path = f"pipeline/{self.run_id}/plots/{fname}"
                _upload_to_gcs(local_path, self.bucket_name, gcs_path)
                os.remove(local_path)
                uploaded += 1

        logger.info(f"Uploaded {uploaded} plots")
        yield f"uploaded_{uploaded}_plots"


class OutputWriterFn(beam.DoFn):
    """Write final labeled Parquet to GCS."""

    def __init__(self, bucket_name, run_id):
        self.bucket_name = bucket_name
        self.run_id = run_id

    def process(self, element):
        samples, _ = element

        if samples.empty:
            logger.warning("No samples to write.")
            yield "no_output"
            return

        # Add metadata
        samples = samples.copy()
        samples["ingested_at"] = datetime.utcnow().isoformat()

        # Ensure JSON strings
        for col in ["ego_trajectory", "surrounding_vehicles"]:
            if col in samples.columns:
                samples[col] = samples[col].apply(
                    lambda v: json.dumps(v) if not isinstance(v, str) else v
                )

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        gcs_path = f"pipeline/{self.run_id}/final/scenario_windows_{ts}.parquet"
        _write_parquet_to_gcs(samples, self.bucket_name, gcs_path)

        logger.info(f"Final output: {len(samples)} windows -> gs://{self.bucket_name}/{gcs_path}")
        yield f"wrote_{len(samples)}_windows"


class WriteCleanParquetFn(beam.DoFn):
    """Write clean.parquet to GCS and pass the DataFrame through."""

    def __init__(self, bucket_name, run_id):
        self.bucket_name = bucket_name
        self.run_id = run_id

    def process(self, df):
        gcs_path = f"pipeline/{self.run_id}/clean.parquet"
        _write_parquet_to_gcs(df, self.bucket_name, gcs_path)
        logger.info(f"Clean parquet written: gs://{self.bucket_name}/{gcs_path}")
        yield df


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline definition
# ══════════════════════════════════════════════════════════════════════════════

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True,
                        help="GCS path to raw NGSIM CSV (gs://bucket/path)")
    parser.add_argument("--output_bucket", required=True,
                        help="GCS bucket name for outputs")
    parser.add_argument("--run_id", default=None,
                        help="Pipeline run ID (auto-generated if not set)")
    known_args, pipeline_args = parser.parse_known_args()

    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = True

    run_id = known_args.run_id or datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")
    bucket = known_args.output_bucket

    logger.info(f"Starting pipeline: run_id={run_id}")
    logger.info(f"Input: {known_args.input_csv}")
    logger.info(f"Output bucket: {bucket}")

    with beam.Pipeline(options=pipeline_options) as p:

        # ── Step 1: Preprocess ────────────────────────────────────────────
        clean_df = (
            p
            | "CreateInput" >> beam.Create([known_args.input_csv])
            | "Preprocess" >> beam.ParDo(PreprocessFn())
            | "WriteClean" >> beam.ParDo(WriteCleanParquetFn(bucket, run_id))
        )

        # ── Step 2: Fan-out to 3 parallel detectors ──────────────────────
        cf_events = clean_df | "DetectCF" >> beam.ParDo(DetectCarFollowingFn())
        merge_events = clean_df | "DetectMerge" >> beam.ParDo(DetectOnrampMergeFn())
        cutin_events = clean_df | "DetectCutin" >> beam.ParDo(DetectLaneCutinFn())

        # ── Step 3: Merge all events ──────────────────────────────────────
        all_events = (
            (cf_events, merge_events, cutin_events)
            | "FlattenEvents" >> beam.Flatten()
            | "CollectEvents" >> beam.combiners.ToList()
            | "MergeEvents" >> beam.ParDo(MergeEventsFn())
        )

        # ── Step 4: Window segmentation (clean_df as side input) ──────────
        # We need the clean DataFrame as a singleton side input
        clean_side = beam.pvalue.AsSingleton(clean_df)

        windows = (
            all_events
            | "Segment" >> beam.ParDo(WindowFn(), clean_side)
        )

        # ── Step 5: Combine windows + clean_df for downstream stages ─────
        # Create (samples_df, clean_df) tuples for viz and output writer
        combined = (
            windows
            | "PairWithClean" >> beam.Map(lambda samples, cdf: (samples, cdf), clean_side)
        )

        # ── Step 6a: Visualizer (parallel branch) ────────────────────────
        _ = combined | "Visualize" >> beam.ParDo(VisualizeFn(bucket, run_id))

        # ── Step 6b: Output Writer (parallel branch) ─────────────────────
        _ = combined | "WriteOutput" >> beam.ParDo(OutputWriterFn(bucket, run_id))

    logger.info(f"Pipeline complete: run_id={run_id}")


if __name__ == "__main__":
    run()

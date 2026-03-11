"""
Scenario Validator
------------------
Verifies that detected scenarios actually satisfy their detection criteria
by re-checking each sample against the clean preprocessed data.

Produces:
  - Per-scenario pass/fail with specific failure reasons
  - Overall confidence report
  - output/validation_report.txt  (human readable)
  - output/validation_results.csv (machine readable)

Run:
    python validate.py --data trajectories-0400-0415.csv \
                       --samples output/scenario_windows_<timestamp>.parquet
"""

from modules.preprocessor import preprocess
import argparse
import json
import logging
import os
import sys
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("validator")

sys.path.insert(0, os.path.dirname(__file__))

# ── Same thresholds as detector ───────────────────────────────────────────────
CF_MAX_GAP_M = 80.0
CF_SPEED_CORR_MIN = 0.0
CF_SAME_LANE_RATIO = 0.8
ONRAMP_LANE_ID = 6
MAINLINE_LANES = {1, 2, 3, 4, 5}
MERGE_MIN_LATERAL_M = 1.5
CUTIN_SPEED_DROP_MS = 1.0


def _parse(val):
    if isinstance(val, str):
        return json.loads(val)
    return val or []


# ── Per-scenario validators ───────────────────────────────────────────────────

def validate_car_following(row, vdict):
    checks = {}
    ego_id = int(row["ego_id"])
    leader_id = row.get("leader_id")
    issues = []

    if pd.isna(leader_id):
        return False, ["No leader_id recorded"]
    leader_id = int(leader_id)

    if ego_id not in vdict:
        return False, ["Ego vehicle not found in clean data"]
    if leader_id not in vdict:
        return False, ["Leader vehicle not found in clean data"]

    ego_df = vdict[ego_id]
    leader_df = vdict[leader_id]

    w_start = int(row["window_start_frame"])
    w_end = int(row["window_end_frame"])

    ego_w = ego_df[ego_df["Frame_ID"].between(w_start, w_end)]
    ldr_w = leader_df[leader_df["Frame_ID"].between(w_start, w_end)]

    if ego_w.empty or ldr_w.empty:
        return False, ["No data in window for ego or leader"]

    merged = ego_w.merge(ldr_w[["Frame_ID", "Lane_ID", "speed_ms", "y_m"]],
                         on="Frame_ID", suffixes=("_ego", "_ldr"))
    if merged.empty:
        return False, ["No overlapping frames between ego and leader"]

    # Check 1: Lane match
    lane_match = (merged["Lane_ID_ego"] == merged["Lane_ID_ldr"]).mean()
    checks["lane_match_ratio"] = round(lane_match, 3)
    if lane_match < CF_SAME_LANE_RATIO:
        issues.append(
            f"Lane match {lane_match:.2f} < {CF_SAME_LANE_RATIO} threshold")

    # Check 2: Gap
    gap = (merged["y_m_ldr"] - merged["y_m_ego"]).abs()
    avg_gap = gap.mean()
    checks["avg_gap_m"] = round(avg_gap, 2)
    if avg_gap > CF_MAX_GAP_M:
        issues.append(f"Avg gap {avg_gap:.1f}m > {CF_MAX_GAP_M}m threshold")

    # Check 3: Speed correlation
    if merged["speed_ms_ego"].std() < 0.01:
        corr = 1.0
    else:
        corr = merged["speed_ms_ego"].corr(merged["speed_ms_ldr"])
    checks["speed_corr"] = round(corr, 3) if not pd.isna(corr) else None
    avg_speed = merged["speed_ms_ego"].mean()
    if avg_speed >= 3.0 and (pd.isna(corr) or corr < CF_SPEED_CORR_MIN):
        issues.append(
            f"Speed correlation {corr:.2f} < {CF_SPEED_CORR_MIN} threshold")

    # Check 4: Leader is actually ahead of ego
    ahead = (merged["y_m_ldr"] > merged["y_m_ego"]).mean()
    checks["leader_ahead_ratio"] = round(ahead, 3)
    if ahead < 0.7:
        issues.append(
            f"Leader ahead only {ahead:.0%} of frames (should be leading)")

    passed = len(issues) == 0
    return passed, issues, checks


def validate_onramp_merge(row, vdict):
    checks = {}
    ego_id = int(row["ego_id"])
    issues = []

    if ego_id not in vdict:
        return False, ["Ego vehicle not found in clean data"]

    ego_df = vdict[ego_id]
    w_start = int(row["window_start_frame"])
    w_end = int(row["window_end_frame"])
    ego_w = ego_df[ego_df["Frame_ID"].between(
        w_start, w_end)].sort_values("Frame_ID")

    if ego_w.empty:
        return False, ["No ego data in window"]

    lanes = ego_w["Lane_ID"].values

    # Check 1: Vehicle starts on ramp
    starts_on_ramp = lanes[0] == ONRAMP_LANE_ID
    checks["starts_on_ramp"] = bool(starts_on_ramp)
    if not starts_on_ramp:
        issues.append(
            f"Vehicle starts in lane {lanes[0]}, not ramp lane {ONRAMP_LANE_ID}")

    # Check 2: Vehicle ends in mainline
    ends_in_mainline = lanes[-1] in MAINLINE_LANES
    checks["ends_in_mainline"] = bool(ends_in_mainline)
    if not ends_in_mainline:
        issues.append(f"Vehicle ends in lane {lanes[-1]}, not a mainline lane")

    # Check 3: Lane transition actually happens
    unique_lanes = set(lanes)
    has_transition = bool(unique_lanes & {ONRAMP_LANE_ID}) and bool(
        unique_lanes & MAINLINE_LANES)
    checks["lane_transition_present"] = has_transition
    if not has_transition:
        issues.append(
            f"No ramp→mainline transition found. Lanes seen: {sorted(unique_lanes)}")

    # Check 4: Lateral displacement
    lateral = ego_w["lateral_disp"].sum() * 0.3048
    checks["lateral_disp_m"] = round(lateral, 3)
    if lateral < MERGE_MIN_LATERAL_M:
        issues.append(
            f"Lateral displacement {lateral:.2f}m < {MERGE_MIN_LATERAL_M}m threshold")

    passed = len(issues) == 0
    return passed, issues, checks


def validate_lane_cutin(row, vdict):
    checks = {}
    ego_id = int(row["ego_id"])
    cutter_id = row.get("cutter_id")
    issues = []

    if pd.isna(cutter_id):
        return False, ["No cutter_id recorded"]
    cutter_id = int(cutter_id)

    if ego_id not in vdict:
        return False, ["Ego vehicle not found in clean data"]
    if cutter_id not in vdict:
        return False, ["Cutter vehicle not found in clean data"]

    ego_df = vdict[ego_id]
    cutter_df = vdict[cutter_id]

    cutin_frame = row.get("cutin_frame")
    if pd.isna(cutin_frame):
        return False, ["No cutin_frame recorded"]
    cutin_frame = int(cutin_frame)

    # Check 1: Cutter changed lane around cutin frame
    c_window = cutter_df[cutter_df["Frame_ID"].between(
        cutin_frame - 15, cutin_frame + 15)]
    cutter_changed_lane = c_window["Lane_ID"].nunique(
    ) > 1 if not c_window.empty else False
    checks["cutter_changed_lane"] = bool(cutter_changed_lane)
    if not cutter_changed_lane:
        issues.append("Cutter did not change lane near the cut-in frame")

    # Check 2: Ego decelerates after cutin
    before = ego_df[ego_df["Frame_ID"].between(cutin_frame - 20, cutin_frame)]
    after = ego_df[ego_df["Frame_ID"].between(cutin_frame, cutin_frame + 30)]

    if before.empty or after.empty:
        issues.append("Not enough ego data before/after cutin frame")
        speed_drop = None
    else:
        speed_drop = before["speed_ms"].mean() - after["speed_ms"].mean()
        checks["speed_drop_ms"] = round(speed_drop, 3)
        if speed_drop < CUTIN_SPEED_DROP_MS:
            issues.append(
                f"Ego speed drop {speed_drop:.2f} m/s < {CUTIN_SPEED_DROP_MS} m/s threshold")

    # Check 3: Cutter ends up ahead of ego after cutin
    ego_after = ego_df[ego_df["Frame_ID"].between(cutin_frame, cutin_frame+20)]
    cutter_after = cutter_df[cutter_df["Frame_ID"].between(
        cutin_frame, cutin_frame+20)]
    if not ego_after.empty and not cutter_after.empty:
        cutter_ahead = cutter_after["y_m"].mean() > ego_after["y_m"].mean()
        checks["cutter_ends_ahead"] = bool(cutter_ahead)
        if not cutter_ahead:
            issues.append("Cutter is not ahead of ego after cut-in")

    passed = len(issues) == 0
    return passed, issues, checks


# ── Main validation loop ──────────────────────────────────────────────────────

def validate(clean_df: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    vdict = {vid: grp.reset_index(drop=True)
             for vid, grp in clean_df.groupby("Vehicle_ID")}

    validators = {
        "car_following": validate_car_following,
        "onramp_merge":  validate_onramp_merge,
        "lane_cutin":    validate_lane_cutin,
    }

    results = []
    for _, row in samples.iterrows():
        sc_type = row["scenario_type"]
        fn = validators.get(sc_type)
        if fn is None:
            continue

        try:
            result = fn(row, vdict)
            if len(result) == 3:
                passed, issues, checks = result
            else:
                passed, issues = result
                checks = {}
        except Exception as e:
            passed, issues, checks = False, [f"Validator error: {e}"], {}

        results.append({
            "scenario_type":  sc_type,
            "ego_id":         row["ego_id"],
            "window_start":   row["window_start_frame"],
            "window_end":     row["window_end_frame"],
            "passed":         passed,
            "issues":         "; ".join(issues) if issues else "None",
            **{f"check_{k}": v for k, v in checks.items()},
        })

    return pd.DataFrame(results)


def print_report(results: pd.DataFrame):
    total = len(results)
    passed = results["passed"].sum()

    lines = []
    lines.append("=" * 65)
    lines.append("  SCENARIO VALIDATION REPORT")
    lines.append("=" * 65)
    lines.append(f"  Total windows validated : {total}")
    lines.append(
        f"  Passed                  : {passed}  ({passed/total*100:.1f}%)")
    lines.append(
        f"  Failed                  : {total - passed}  ({(total-passed)/total*100:.1f}%)")
    lines.append("")

    for sc_type in ["car_following", "onramp_merge", "lane_cutin"]:
        sub = results[results["scenario_type"] == sc_type]
        if sub.empty:
            continue
        p = sub["passed"].sum()
        n = len(sub)
        lines.append(f"  {sc_type.upper().replace('_', ' ')}")
        lines.append(f"    Passed : {p}/{n}  ({p/n*100:.1f}%)")

        # Show most common failure reasons
        failed = sub[~sub["passed"]]
        if not failed.empty:
            all_issues = []
            for iss in failed["issues"]:
                all_issues.extend([i.strip() for i in iss.split(";")])
            from collections import Counter
            common = Counter(all_issues).most_common(3)
            lines.append("    Top failure reasons:")
            for reason, count in common:
                lines.append(f"      [{count}x] {reason}")

        # Show key metric distributions for passed ones
        passed_sub = sub[sub["passed"]]
        if sc_type == "car_following" and "check_avg_gap_m" in passed_sub:
            g = passed_sub["check_avg_gap_m"].dropna()
            if not g.empty:
                lines.append(f"    Avg gap (passed):  mean={g.mean():.1f}m  "
                             f"min={g.min():.1f}m  max={g.max():.1f}m")
            c = passed_sub.get("check_speed_corr", pd.Series()).dropna()
            if not c.empty:
                lines.append(f"    Speed corr (passed): mean={c.mean():.2f}  "
                             f"min={c.min():.2f}  max={c.max():.2f}")

        elif sc_type == "lane_cutin" and "check_speed_drop_ms" in passed_sub:
            d = passed_sub["check_speed_drop_ms"].dropna()
            if not d.empty:
                lines.append(f"    Speed drop (passed): mean={d.mean():.2f} m/s  "
                             f"min={d.min():.2f}  max={d.max():.2f}")
        lines.append("")

    lines.append("=" * 65)
    report = "\n".join(lines)
    print(report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate detected scenarios")
    parser.add_argument("--data",    required=True,
                        help="Path to raw NGSIM CSV")
    parser.add_argument("--samples", required=True,
                        help="Path to scenario_windows parquet")
    parser.add_argument("--output",  default="output",
                        help="Output directory for report")
    args = parser.parse_args()

    logger.info("Preprocessing data for validation...")
    clean_df = preprocess(args.data)

    logger.info("Loading detected samples...")
    samples = pd.read_parquet(args.samples)
    logger.info(f"Validating {len(samples)} windows...")

    results = validate(clean_df, samples)

    os.makedirs(args.output, exist_ok=True)
    report_str = print_report(results)

    # Save outputs
    report_path = os.path.join(args.output, "validation_report.txt")
    csv_path = os.path.join(args.output, "validation_results.csv")
    with open(report_path, "w") as f:
        f.write(report_str)
    results.to_csv(csv_path, index=False)

    logger.info(f"Report saved to {report_path}")
    logger.info(f"Full results saved to {csv_path}")

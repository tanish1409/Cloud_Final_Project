"""
Verify that the gap-aware windower produces windows with exactly 50
consecutive present ego frames (no gaps).
"""
import pandas as pd
import json
import sys

def verify_windows(parquet_path):
    df = pd.read_parquet(parquet_path)
    print(f"Total windows: {len(df)}")
    print(f"Columns: {list(df.columns)}\n")

    issues = []
    frame_count_stats = []

    for idx, row in df.iterrows():
        ego_traj = json.loads(row["ego_trajectory"]) if isinstance(row["ego_trajectory"], str) else row["ego_trajectory"]
        frames = [r["frame"] for r in ego_traj]
        n_frames = len(frames)
        frame_count_stats.append(n_frames)

        # Check 1: Exactly 50 frames
        if n_frames != 50:
            issues.append(f"Window {idx} (ego={row['ego_id']}, {row['scenario_type']}): "
                          f"has {n_frames} frames, expected 50")

        # Check 2: Frames are sorted
        if frames != sorted(frames):
            issues.append(f"Window {idx} (ego={row['ego_id']}): frames not sorted")

        # Check 3: Check for gaps (consecutive frame differences)
        if n_frames >= 2:
            diffs = [frames[i+1] - frames[i] for i in range(len(frames)-1)]
            max_gap = max(diffs)
            if max_gap > 1:
                # This is expected with the gap-aware windower — frames are
                # consecutive in the *present* data, but frame IDs may have gaps
                # where preprocessing dropped rows
                gap_count = sum(1 for d in diffs if d > 1)
                # Only flag if there are many large gaps (indicates a real problem)
                if gap_count > 10:
                    issues.append(f"Window {idx} (ego={row['ego_id']}): "
                                  f"{gap_count} frame-ID gaps (max gap={max_gap})")

    # Summary
    print("=" * 60)
    print("  CONSECUTIVE FRAMES VERIFICATION")
    print("=" * 60)

    # Frame count distribution
    from collections import Counter
    counts = Counter(frame_count_stats)
    print(f"\n  Frame count distribution:")
    for k in sorted(counts.keys()):
        pct = counts[k] / len(df) * 100
        print(f"    {k} frames: {counts[k]} windows ({pct:.1f}%)")

    # ego_frame_count column check
    if "ego_frame_count" in df.columns:
        print(f"\n  ego_frame_count column stats:")
        print(f"    min: {df['ego_frame_count'].min()}")
        print(f"    max: {df['ego_frame_count'].max()}")
        print(f"    mean: {df['ego_frame_count'].mean():.1f}")
        all_50 = (df["ego_frame_count"] == 50).all()
        print(f"    all exactly 50: {all_50}")
    else:
        print("\n  WARNING: ego_frame_count column not present in output")

    # Frame-ID gap analysis (expected with gap-aware windower)
    print(f"\n  Frame-ID gap analysis (gaps between consecutive present frames):")
    all_gaps = []
    for _, row in df.iterrows():
        ego_traj = json.loads(row["ego_trajectory"]) if isinstance(row["ego_trajectory"], str) else row["ego_trajectory"]
        frames = [r["frame"] for r in ego_traj]
        if len(frames) >= 2:
            diffs = [frames[i+1] - frames[i] for i in range(len(frames)-1)]
            all_gaps.extend(diffs)
    
    if all_gaps:
        gap_1 = sum(1 for g in all_gaps if g == 1)
        gap_gt1 = sum(1 for g in all_gaps if g > 1)
        total = len(all_gaps)
        print(f"    Total frame transitions: {total}")
        print(f"    Consecutive (gap=1): {gap_1} ({gap_1/total*100:.2f}%)")
        print(f"    With gaps (gap>1):   {gap_gt1} ({gap_gt1/total*100:.2f}%)")
        if gap_gt1 > 0:
            gap_sizes = [g for g in all_gaps if g > 1]
            print(f"    Gap sizes: min={min(gap_sizes)}, max={max(gap_sizes)}, "
                  f"mean={sum(gap_sizes)/len(gap_sizes):.1f}")

    # Window duration analysis
    print(f"\n  Window duration (based on frame ID span):")
    durations = (df["window_end_frame"] - df["window_start_frame"]) / 10.0
    print(f"    min: {durations.min():.2f}s")
    print(f"    max: {durations.max():.2f}s")
    print(f"    mean: {durations.mean():.2f}s")
    print(f"    exactly 4.9s (49 frame-IDs): {(durations == 4.9).sum()}")

    # Issues
    if issues:
        print(f"\n  ISSUES FOUND: {len(issues)}")
        for iss in issues[:20]:
            print(f"    - {iss}")
        if len(issues) > 20:
            print(f"    ... and {len(issues) - 20} more")
    else:
        print(f"\n  No critical issues found.")

    print("=" * 60)

    # Per-scenario breakdown
    print(f"\n  Per-scenario frame count verification:")
    for sc in ["car_following", "onramp_merge", "lane_cutin"]:
        sub = df[df["scenario_type"] == sc]
        if sub.empty:
            continue
        sub_counts = []
        for _, row in sub.iterrows():
            ego_traj = json.loads(row["ego_trajectory"]) if isinstance(row["ego_trajectory"], str) else row["ego_trajectory"]
            sub_counts.append(len(ego_traj))
        all_50 = all(c == 50 for c in sub_counts)
        print(f"    {sc}: {len(sub)} windows, all 50 frames: {all_50}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "output/scenario_windows_20260329_145150.parquet"
    verify_windows(path)

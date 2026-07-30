#!/usr/bin/env python3
"""
evaluate_baseline.py
====================
Evaluates the classical optical-flow golf strike detector against GolfDB
ground-truth annotations.

Inputs
------
  --results   : path to results.txt  (detector console output)
  --golfdb    : path to shortlisted_golfDB.json
  --fps       : frames per second (default 29.97)
  --tiou_thr  : tIoU threshold for TP (default 0.5)

Outputs
-------
  per_clip_results.csv   – per-clip tIoU, classification, error
  summary.json           – aggregate metrics
  Console summary table
"""

import re
import json
import csv
import argparse
import statistics
import os
from pathlib import Path

FPS      = 29.97
GT_PRE   = 60   # frames before impact  (= 2.0 s at ~30 fps)
GT_POST  = 45   # frames after  impact  (= 1.5 s at ~30 fps)


# ──────────────────────────────────────────────
# 1.  Parse detector output from results.txt
# ──────────────────────────────────────────────
def parse_results(results_path: str) -> dict:
    """
    Returns a dict keyed by youtube_id:
      {
        'pred_impact_frame': int,
        'pred_start_sec':    float,
        'pred_end_sec':      float,
        'trimmed_duration':  float,   # duration of the trimmed clip (s)
      }
    """
    detections = {}
    current_id = None

    # Patterns
    p_vid    = re.compile(r"Processing:\s+(.+?)\.mp4")
    p_impact = re.compile(r"\[Step 4\] Impact: frame\s+(\d+)")
    p_output = re.compile(r"\[Step 5\] Output:\s+([\d.]+)s\s*->\s*([\d.]+)s")
    p_fps    = re.compile(r"FPS=[\d.]+,\s*Frames=\d+,\s*Duration=([\d.]+)s")

    trimmed_dur = None

    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            m = p_vid.search(line)
            if m:
                current_id   = m.group(1).strip()
                trimmed_dur  = None
                continue

            if current_id is None:
                continue

            m = p_fps.search(line)
            if m:
                trimmed_dur = float(m.group(1))
                continue

            m = p_impact.search(line)
            if m:
                frame = int(m.group(1))
                detections.setdefault(current_id, {})["pred_impact_frame"] = frame
                if trimmed_dur is not None:
                    detections[current_id]["trimmed_duration"] = trimmed_dur
                continue

            m = p_output.search(line)
            if m:
                detections.setdefault(current_id, {})["pred_start_sec"] = float(m.group(1))
                detections[current_id]["pred_end_sec"]                   = float(m.group(2))

    return detections


# ──────────────────────────────────────────────
# 2.  Load ground-truth from GolfDB JSON
# ──────────────────────────────────────────────
def load_golfdb(golfdb_path: str) -> list:
    with open(golfdb_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────────
# 3.  tIoU helper
# ──────────────────────────────────────────────
def compute_tiou(gt_start, gt_end, pred_start, pred_end) -> float:
    intersection = max(0, min(gt_end, pred_end) - max(gt_start, pred_start))
    union        = max(gt_end, pred_end) - min(gt_start, pred_start)
    if union == 0:
        return 0.0
    return intersection / union


# ──────────────────────────────────────────────
# 4.  Main evaluation
# ──────────────────────────────────────────────
def evaluate(results_path, golfdb_path, fps=FPS, tiou_thr=0.5, out_dir="."):
    detections = parse_results(results_path)
    gt_entries = load_golfdb(golfdb_path)

    print(f"\nLoaded {len(detections)} detection(s) and {len(gt_entries)} GT entry/entries.\n")

    # ------------------------------------------------------------------
    # Handle videos that appear more than once in the JSON (e.g. two
    # swings in the same recording).  The detector ran once per video
    # and found a single impact frame; match it to the nearest GT swing.
    # ------------------------------------------------------------------
    # Group GT entries by youtube_id
    from collections import defaultdict
    gt_by_vid = defaultdict(list)
    for entry in gt_entries:
        gt_by_vid[entry["youtube_id"]].append(entry)

    per_clip_rows = []
    total_duration_s = 0.0

    for yt_id, det in detections.items():
        if yt_id not in gt_by_vid:
            print(f"  [SKIP] {yt_id} — not in shortlisted GolfDB, skipping.")
            continue

        pred_impact = det.get("pred_impact_frame")
        pred_start_sec = det.get("pred_start_sec")
        pred_end_sec   = det.get("pred_end_sec")

        if pred_impact is None or pred_start_sec is None or pred_end_sec is None:
            print(f"  [WARN] {yt_id} — incomplete detection record, skipping.")
            continue

        # Convert prediction window to frames
        pred_start = int(round(pred_start_sec * fps))
        pred_end   = int(round(pred_end_sec   * fps))

        # Pick the GT entry whose impact frame is closest to pred_impact
        entries_for_vid = gt_by_vid[yt_id]
        best_entry = min(entries_for_vid,
                         key=lambda e: abs(e["events"][5] - pred_impact))

        gt_impact  = best_entry["events"][5]
        gt_start   = gt_impact - GT_PRE
        gt_end     = gt_impact + GT_POST
        view       = best_entry.get("view", "unknown")
        sex        = best_entry.get("sex", "?")
        club       = best_entry.get("club", "?")
        player     = best_entry.get("player", "?")

        tiou = compute_tiou(gt_start, gt_end, pred_start, pred_end)
        classification = "TP" if tiou >= tiou_thr else "FP/FN"
        impact_error   = abs(gt_impact - pred_impact)

        # Accumulate total processed duration for FPPM
        total_duration_s += det.get("trimmed_duration", 0.0)

        per_clip_rows.append({
            "youtube_id":          yt_id,
            "player":              player,
            "sex":                 sex,
            "club":                club,
            "view":                view,
            "gt_impact":           gt_impact,
            "pred_impact":         pred_impact,
            "impact_error_frames": impact_error,
            "impact_error_sec":    round(impact_error / fps, 3),
            "gt_start":            gt_start,
            "gt_end":              gt_end,
            "pred_start":          pred_start,
            "pred_end":            pred_end,
            "tIoU":                round(tiou, 4),
            "classification":      classification,
        })

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------
    n_total = len(per_clip_rows)
    tp_rows = [r for r in per_clip_rows if r["classification"] == "TP"]
    fp_rows = [r for r in per_clip_rows if r["classification"] == "FP/FN"]

    TP = len(tp_rows)
    FP = len(fp_rows)
    FN = FP   # symmetric in single-swing-per-clip setup

    precision = TP / n_total if n_total > 0 else 0.0
    recall    = TP / n_total if n_total > 0 else 0.0
    f1        = precision    # precision == recall == f1 in this setup

    total_minutes  = total_duration_s / 60.0
    fppm           = FP / total_minutes if total_minutes > 0 else float("nan")

    errors = [r["impact_error_frames"] for r in per_clip_rows]
    mae    = statistics.mean(errors)   if errors else float("nan")
    med    = statistics.median(errors) if errors else float("nan")
    pct_5  = sum(1 for e in errors if e <= 5)  / n_total * 100 if n_total else 0
    pct_10 = sum(1 for e in errors if e <= 10) / n_total * 100 if n_total else 0

    # TP-only error (more meaningful for reporting)
    tp_errors = [r["impact_error_frames"] for r in tp_rows]
    tp_mae    = statistics.mean(tp_errors)   if tp_errors else float("nan")
    tp_med    = statistics.median(tp_errors) if tp_errors else float("nan")

    # ------------------------------------------------------------------
    # Per-angle breakdown
    # ------------------------------------------------------------------
    views = sorted(set(r["view"] for r in per_clip_rows))
    view_stats = {}
    for v in views:
        vrows = [r for r in per_clip_rows if r["view"] == v]
        vtp   = [r for r in vrows if r["classification"] == "TP"]
        v_err = [r["impact_error_frames"] for r in vrows]
        view_stats[v] = {
            "total":   len(vrows),
            "tp":      len(vtp),
            "recall":  len(vtp) / len(vrows) if vrows else 0.0,
            "mae_frames": statistics.mean(v_err) if v_err else float("nan"),
        }

    # ------------------------------------------------------------------
    # Write per-clip CSV
    # ------------------------------------------------------------------
    csv_path = os.path.join(out_dir, "per_clip_results.csv")
    fieldnames = [
        "youtube_id", "player", "sex", "club", "view",
        "gt_impact", "pred_impact", "impact_error_frames", "impact_error_sec",
        "gt_start", "gt_end", "pred_start", "pred_end", "tIoU", "classification",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(per_clip_rows, key=lambda r: r["youtube_id"]):
            writer.writerow(row)
    print(f"  Saved: {csv_path}")

    # ------------------------------------------------------------------
    # Write summary JSON
    # ------------------------------------------------------------------
    summary = {
        "total_clips":            n_total,
        "tp":                     TP,
        "fp":                     FP,
        "fn":                     FN,
        "precision":              round(precision, 4),
        "recall":                 round(recall, 4),
        "f1":                     round(f1, 4),
        "tiou_threshold":         tiou_thr,
        "total_processed_min":    round(total_minutes, 2),
        "fppm":                   round(fppm, 4) if not isinstance(fppm, float) or not fppm != fppm else "nan",
        "mae_frames_all_clips":   round(mae, 2),
        "median_error_frames_all": round(med, 2),
        "mae_frames_tp_only":     round(tp_mae, 2) if tp_errors else "nan",
        "median_error_frames_tp": round(tp_med, 2) if tp_errors else "nan",
        "pct_within_5_frames":    round(pct_5, 1),
        "pct_within_10_frames":   round(pct_10, 1),
        "per_view":               {
            v: {
                "total":      s["total"],
                "tp":         s["tp"],
                "recall":     round(s["recall"], 4),
                "mae_frames": round(s["mae_frames"], 2),
            }
            for v, s in view_stats.items()
        },
    }

    json_path = os.path.join(out_dir, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {json_path}\n")

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    BOLD  = "\033[1m"
    GREEN = "\033[92m"
    RED   = "\033[91m"
    CYAN  = "\033[96m"
    RST   = "\033[0m"

    sep = "─" * 62
    print(f"\n{BOLD}{'═'*62}{RST}")
    print(f"{BOLD}  EVALUATION SUMMARY  (tIoU ≥ {tiou_thr}){RST}")
    print(f"{BOLD}{'═'*62}{RST}")
    print(f"  Total clips evaluated : {n_total}")
    print(f"  True  Positives  (TP) : {GREEN}{TP}{RST}")
    print(f"  False Positives  (FP) : {RED}{FP}{RST}")
    print(f"  False Negatives  (FN) : {RED}{FN}{RST}")
    print(sep)
    print(f"  Precision              : {CYAN}{precision:.4f}  ({precision*100:.1f}%){RST}")
    print(f"  Recall                 : {CYAN}{recall:.4f}  ({recall*100:.1f}%){RST}")
    print(f"  F1 Score               : {CYAN}{f1:.4f}  ({f1*100:.1f}%){RST}")
    print(f"  FPPM                   : {fppm:.3f} false positives/min")
    print(f"  Total processed        : {total_minutes:.2f} min")
    print(sep)
    print(f"  Impact Error (all clips) :")
    print(f"    MAE              : {mae:.2f} frames  ({mae/fps:.3f} s)")
    print(f"    Median AE        : {med:.2f} frames  ({med/fps:.3f} s)")
    print(f"    ≤ 5  frames      : {pct_5:.1f}%")
    print(f"    ≤ 10 frames      : {pct_10:.1f}%")
    if tp_errors:
        print(f"  Impact Error (TP clips only) :")
        print(f"    MAE              : {tp_mae:.2f} frames  ({tp_mae/fps:.3f} s)")
        print(f"    Median AE        : {tp_med:.2f} frames  ({tp_med/fps:.3f} s)")
    print(sep)

    # Per-clip table
    print(f"\n{BOLD}  PER-CLIP RESULTS{RST}")
    print(f"  {'ID':<18} {'GT':>5} {'Pred':>5} {'Err':>4} {'tIoU':>6}  {'Class':<8}  View")
    print(f"  {'─'*18} {'─'*5} {'─'*5} {'─'*4} {'─'*6}  {'─'*8}  {'─'*14}")
    for row in sorted(per_clip_rows, key=lambda r: r["youtube_id"]):
        tag   = GREEN + "TP    " + RST if row["classification"] == "TP" else RED + "FP/FN " + RST
        print(f"  {row['youtube_id']:<18} {row['gt_impact']:>5} {row['pred_impact']:>5} "
              f"{row['impact_error_frames']:>4} {row['tIoU']:>6.3f}  {tag}  {row['view']}")

    # Per-angle table
    print(f"\n{BOLD}  PER-ANGLE BREAKDOWN{RST}")
    print(f"  {'View':<16} {'N':>4} {'TP':>4} {'Recall':>8} {'MAE (frames)':>14}")
    print(f"  {'─'*16} {'─'*4} {'─'*4} {'─'*8} {'─'*14}")
    for v, s in sorted(view_stats.items()):
        print(f"  {v:<16} {s['total']:>4} {s['tp']:>4} {s['recall']*100:>7.1f}%  {s['mae_frames']:>12.2f}")
    # All row
    all_recall = TP / n_total * 100 if n_total else 0
    print(f"  {'All':<16} {n_total:>4} {TP:>4} {all_recall:>7.1f}%  {mae:>12.2f}")
    print(f"\n{'═'*62}\n")

    return summary, per_clip_rows


# ──────────────────────────────────────────────
# Entry-point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate classical golf strike detector against GolfDB ground truth."
    )
    parser.add_argument(
        "--results", default="results.txt",
        help="Path to detector console output (results.txt)"
    )
    parser.add_argument(
        "--golfdb", default="shortlisted_golfDB.json",
        help="Path to shortlisted_golfDB.json"
    )
    parser.add_argument(
        "--fps", type=float, default=FPS,
        help=f"Frames per second (default: {FPS})"
    )
    parser.add_argument(
        "--tiou_thr", type=float, default=0.5,
        help="tIoU threshold for TP classification (default: 0.5)"
    )
    parser.add_argument(
        "--out_dir", default=".",
        help="Directory to write output files (default: current dir)"
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    evaluate(args.results, args.golfdb, args.fps, args.tiou_thr, args.out_dir)

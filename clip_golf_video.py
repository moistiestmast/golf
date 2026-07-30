import os
import sys
import argparse
import subprocess
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


# ==============================================================================
# Shi-Tomasi Corner Attention Map
# ==============================================================================

def compute_attention_map(
    gray_frame,
    max_corners=300,
    quality=0.01,
    min_dist=5,
    block_size=5,
    blur_sigma=20,
):
    """
    Returns (attention, num_corners) where attention is a (H, W) float32
    map in [0, 1] centred on Shi-Tomasi corner features, and num_corners is
    the raw detection count (stored once, reused for diagnostics).
    """
    corners = cv2.goodFeaturesToTrack(
        gray_frame,
        maxCorners=max_corners,
        qualityLevel=quality,
        minDistance=min_dist,
        blockSize=block_size,
        useHarrisDetector=False,
    )

    h, w = gray_frame.shape
    num_corners = len(corners) if corners is not None else 0
    # Fallback: too few corners (blank / very dark frame) — no masking.
    if corners is None or num_corners < 10:
        return np.ones((h, w), dtype=np.float32), num_corners

    attention = np.zeros((h, w), dtype=np.float32)
    for c in corners:
        x, y = int(round(c[0][0])), int(round(c[0][1]))
        if 0 <= x < w and 0 <= y < h:
            attention[y, x] = 1.0

    attention = cv2.GaussianBlur(attention, (0, 0), sigmaX=blur_sigma)
    attention = attention / (attention.max() + 1e-8)
    return attention, num_corners

try:
    import imageio_ffmpeg
    DEFAULT_FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    DEFAULT_FFMPEG_PATH = "ffmpeg"


def get_ffmpeg_cmd(custom_ffmpeg=None):
    if custom_ffmpeg and os.path.exists(custom_ffmpeg):
        return custom_ffmpeg
    return DEFAULT_FFMPEG_PATH


# ==============================================================================
# STAGE 1: Audio Silence Detection — removes slow-mo section
# Input:  videos/full/*.mp4
# Output: videos/trimmed/*_trimmed.mp4
# ==============================================================================

def detect_slomo_cutoff(
    video_path,
    ffmpeg_path,
    win_sec=0.05,
    silence_threshold=5.0,
    min_active_sec=0.3,
    min_silence_sec=1.0,
):
    """Finds the timestamp where normal audio ends and mute slow-mo begins."""
    cmd = [
        ffmpeg_path, "-y", "-i", video_path,
        "-f", "s16le", "-ac", "1", "-ar", "16000", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    raw_audio, _ = proc.communicate()

    if not raw_audio:
        return None

    audio = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32)
    sr = 16000
    win_len = int(sr * win_sec)
    num_wins = len(audio) // win_len

    min_active_wins = int(min_active_sec / win_sec)
    min_silence_wins = int(min_silence_sec / win_sec)

    active_count = 0
    has_started = False
    silence_start_win = None
    silence_count = 0

    for i in range(num_wins):
        chunk = audio[i * win_len : (i + 1) * win_len]
        rms = np.sqrt(np.mean(chunk ** 2))

        if not has_started:
            if rms > silence_threshold:
                active_count += 1
                if active_count >= min_active_wins:
                    has_started = True
            else:
                active_count = 0
        else:
            if rms <= silence_threshold:
                if silence_count == 0:
                    silence_start_win = i
                silence_count += 1
                if silence_count >= min_silence_wins:
                    return silence_start_win * win_sec
            else:
                silence_count = 0
                silence_start_win = None

    return None


def trim_slomo(video_path, output_path, ffmpeg_path, fast_mode=False):
    """Cuts the video at the slow-mo transition point."""
    cut_time = detect_slomo_cutoff(video_path, ffmpeg_path)
    if cut_time is None:
        print(f"  [Stage 1] No slow-mo transition found in '{os.path.basename(video_path)}'.")
        return False

    print(f"  [Stage 1] Slow-mo transition at: {cut_time:.2f}s")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if fast_mode:
        cmd = [
            ffmpeg_path, "-y", "-i", video_path,
            "-ss", "0", "-to", str(cut_time),
            "-c", "copy", output_path,
        ]
    else:
        cmd = [
            ffmpeg_path, "-y", "-i", video_path,
            "-ss", "0", "-to", str(cut_time),
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "aac", "-b:a", "192k",
            output_path,
        ]

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode == 0 and os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  [Stage 1] Saved trimmed clip: {output_path} ({size_mb:.2f} MB)")
        return True
    return False


# ==============================================================================
# STAGE 2: Dense Optical Flow Strike Detection (Farneback)
# Input:  videos/trimmed/*_trimmed.mp4
# Output: videos/strikes/*_strike.mp4
# ==============================================================================

def detect_audio_onsets(video_path, ffmpeg_path, sr=16000,
                        bandpass_low=2000, bandpass_high=7900,
                        win_sec=0.025, hop_sec=0.01,
                        thresh_mult=3.0, min_dist_sec=0.5):
    """
    Extract audio, compute spectral-flux onset strength in the 2-8 kHz band,
    and return a list of (timestamp_sec, peak_amplitude) for detected onsets.
    """
    from scipy.ndimage import maximum_filter1d
    from scipy.signal import butter, sosfilt

    cmd = [ffmpeg_path, "-y", "-i", video_path,
           "-f", "s16le", "-ac", "1", "-ar", str(sr), "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    raw, _ = proc.communicate()
    if not raw:
        return []

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    # Band-pass 2–8 kHz to isolate club-ball "thwack" from wind/cart hum
    sos = butter(4, [bandpass_low, bandpass_high], btype='band', fs=sr, output='sos')
    audio_filt = sosfilt(sos, audio)

    win_len = int(sr * win_sec)
    hop_len = int(sr * hop_sec)
    n_fft = 2 ** int(np.ceil(np.log2(win_len)))

    onset_env, prev_mag = [], None
    for start in range(0, len(audio_filt) - win_len, hop_len):
        frame = audio_filt[start:start + win_len]
        spec = np.abs(np.fft.rfft(frame * np.hanning(win_len), n=n_fft))
        if prev_mag is not None:
            onset_env.append(float(np.sum(np.maximum(spec - prev_mag, 0))))
        prev_mag = spec

    onset_env = np.array(onset_env)
    if len(onset_env) == 0:
        return []

    med = np.median(onset_env)
    thresh = med + thresh_mult * np.median(np.abs(onset_env - med))
    min_dist_frames = max(1, int(min_dist_sec / hop_sec))
    local_max = (onset_env == maximum_filter1d(onset_env, size=min_dist_frames * 2 + 1))
    peaks_idx = np.where(local_max & (onset_env >= thresh))[0]

    peaks_sec = peaks_idx * hop_sec + win_sec / 2.0
    return list(zip(peaks_sec.tolist(), onset_env[peaks_idx].tolist()))


def extract_golf_strike(video_path, output_path, ffmpeg_path, fast_mode=False,
                        use_attention_mask=True, use_audio_gate=False):
    """
    Detects the golf strike impact frame using Dense Optical Flow (Farneback).

    Per-frame motion score = 99th-percentile of the optical flow magnitude map,
    computed only when BOTH the previous and current frames are bright enough
    (mean pixel intensity > 30). This dual brightness gate eliminates:
      - Blackout frames at clip start/end
      - Fade-in / fade-out transition artifacts
      - Any frame where the camera is transitioning

    The impact frame is the GLOBAL ARGMAX of the Gaussian-smoothed signal.
    By physics, the golf clubhead at impact is the fastest object in any clip —
    walking (~2-5 px/frame) is an order of magnitude below impact (~15-30 px/frame).
    No heuristics, run segmentation, or duration filtering needed.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [Error] Cannot open: {video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    filename = os.path.basename(video_path)
    print(f"\n--- Extracting Golf Strike from: {filename} ---")
    print(f"  FPS={fps:.2f}, Frames={total_frames}, Duration={total_frames/fps:.2f}s")

    # Step 1: Read all frames into 320x180 grayscale arrays
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(cv2.resize(gray, (320, 180)))
    cap.release()

    if len(frames) < 2:
        print("  [Error] Not enough frames.")
        return False

    # Step 1b: Compute Shi-Tomasi attention maps (one per frame).
    # Corner counts are stored here and reused in Step 2b — no second pass.
    if use_attention_mask:
        attention_maps = []
        corner_counts = []
        for f in frames:
            att, n = compute_attention_map(f)
            attention_maps.append(att)
            corner_counts.append(n)
    else:
        attention_maps = None
        corner_counts = []

    # Raw per-frame mean brightness — used for all four scene-stability gates.
    # (The attention map is only applied inside the optical flow computation.)
    mean_brightness = [float(np.mean(f)) for f in frames]

    # Step 2: Compute per-frame 99th-percentile optical flow magnitude.
    # Four gates to suppress fade-in / flash / transition / intro artifacts:
    #   Gate A — Absolute brightness: both frames must be > 30 mean intensity.
    #   Gate B — Delta brightness: scene mean cannot change > 2.0 px between
    #             consecutive frames (fade-in ramp signature).
    #   Gate C — Stability window: require 15 consecutive stable frames before
    #             accepting a score (prevents early-ramp peaks).
    #   Gate D — Modal brightness proximity: the video's dominant scene has a
    #             characteristic brightness (histogram mode). Frames deviating
    #             more than 20 units from that mode are intro/outro regions
    #             (different lighting) and are rejected.
    BRIGHTNESS_GATE   = 30.0   # absolute minimum scene brightness
    DELTA_BRIGHT_GATE = 2.0    # max allowed frame-to-frame mean-brightness change
    STABILITY_WINDOW  = 15     # frames of brightness stability required (~0.5s)
    MODAL_TOLERANCE   = 20.0   # max deviation from dominant scene brightness

    # mean_brightness is raw frame brightness (used for all gates).
    mb_arr = np.array(mean_brightness)

    # Gate D: find the modal (most common) brightness bin using a 10-unit histogram
    valid_mb = mb_arr[mb_arr >= BRIGHTNESS_GATE]
    if len(valid_mb) > 0:
        hist, bin_edges = np.histogram(valid_mb, bins=np.arange(0, 256, 10))
        modal_bin_idx = int(np.argmax(hist))
        modal_brightness = float(bin_edges[modal_bin_idx] + 5)  # bin centre
    else:
        modal_brightness = float(np.median(mb_arr))
    MODAL_LOW  = modal_brightness - MODAL_TOLERANCE
    MODAL_HIGH = modal_brightness + MODAL_TOLERANCE

    # Build per-frame "stable" flag: True if the last STABILITY_WINDOW frames
    # are all bright, not fading, and within the modal brightness band.
    stable = [False] * len(frames)
    consec_stable = 0
    for i in range(1, len(frames)):
        delta = abs(mean_brightness[i] - mean_brightness[i - 1])
        if (mean_brightness[i] >= BRIGHTNESS_GATE
                and MODAL_LOW <= mean_brightness[i] <= MODAL_HIGH
                and delta < DELTA_BRIGHT_GATE):
            consec_stable += 1
        else:
            consec_stable = 0
        stable[i] = (consec_stable >= STABILITY_WINDOW)

    motion_raw = [0.0]
    p99_raw    = [0.0]  # cached raw p99 per frame — reused in Step 4, no recompute
    gated_count = 0
    max_dx = max_dy = 0.0  # ego-motion diagnostic accumulators
    for i in range(1, len(frames)):
        prev_gray = frames[i - 1]
        curr_gray = frames[i]

        prev_bright = mean_brightness[i - 1]
        curr_bright = mean_brightness[i]
        delta_bright = abs(curr_bright - prev_bright)

        # Apply all four gates
        if (prev_bright < BRIGHTNESS_GATE
                or curr_bright < BRIGHTNESS_GATE
                or not (MODAL_LOW <= prev_bright <= MODAL_HIGH)
                or not (MODAL_LOW <= curr_bright <= MODAL_HIGH)
                or delta_bright >= DELTA_BRIGHT_GATE
                or not stable[i]):
            motion_raw.append(0.0)
            p99_raw.append(0.0)
            gated_count += 1
            continue

        if use_attention_mask:
            # Combine attention maps: conservative minimum — both frames must
            # agree a region is corner-rich before it contributes to flow.
            att_combined = np.minimum(attention_maps[i - 1], attention_maps[i])
            masked_prev = (prev_gray.astype(np.float32) * att_combined).astype(np.uint8)
            masked_curr = (curr_gray.astype(np.float32) * att_combined).astype(np.uint8)
            # Ego-motion estimate from cheap raw (unmasked) flow — masked background
            # biases the median so we use the original frames at reduced quality.
            flow_raw = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=2, winsize=15,
                iterations=1, poly_n=5, poly_sigma=1.1, flags=0
            )
            dx_median = np.median(flow_raw[..., 0])
            dy_median = np.median(flow_raw[..., 1])
            flow = cv2.calcOpticalFlowFarneback(
                masked_prev, masked_curr, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            # Cancel global camera translation
            flow[..., 0] -= dx_median
            flow[..., 1] -= dy_median
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mag_masked = mag * att_combined
            mag_nonzero = mag_masked[mag_masked > 0]
            if len(mag_nonzero) < 100:
                motion_raw.append(0.0)
                p99_raw.append(0.0)
                continue
            p95 = float(np.percentile(mag_nonzero, 95))
            p99 = float(np.percentile(mag_nonzero, 99))
        else:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            # Cancel global camera translation
            dx_median = np.median(flow[..., 0])
            dy_median = np.median(flow[..., 1])
            flow[..., 0] -= dx_median
            flow[..., 1] -= dy_median
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            p95 = float(np.percentile(mag, 95))
            p99 = float(np.percentile(mag, 99))
        # Track max ego-motion for diagnostic
        max_dx = max(max_dx, abs(dx_median))
        max_dy = max(max_dy, abs(dy_median))
        # Score = p99 * (p99/p95): amplifies localized fast motion (clubhead) vs
        # diffuse slow motion (walking). Strike: p99>>p95 => high score.
        # Walking: p99~p95 => score barely above p99.
        motion_raw.append(p99 * (p99 / (p95 + 1e-4)))
        p99_raw.append(p99)

    print(f"  [Step 2] Modal scene brightness: {modal_brightness:.1f} "
          f"(accepted range {MODAL_LOW:.1f}–{MODAL_HIGH:.1f})")
    if gated_count > 0:
        print(f"  [Step 2] Gated {gated_count} frame pair(s) "
              f"(blackout / fade / unstable / off-modal).")
    print(f"  [Ego-Motion] Max median flow: dx={max_dx:.2f}, dy={max_dy:.2f} px/frame")

    # Step 2b: Shi-Tomasi diagnostic — reads corner_counts stored in Step 1b (no recompute)
    if use_attention_mask:
        avg_corners = float(np.mean(corner_counts))
        print(f"  [Shi-Tomasi] Avg corners/frame: {avg_corners:.1f}")
        if avg_corners < 20:
            print("  [Shi-Tomasi Warning] Low corner count — attention mask may be weak. "
                  "Consider lowering qualityLevel.")

    # Step 3: Gaussian smoothing (sigma=2, lighter than MOG2 — flow is cleaner)
    motion = np.array(motion_raw)
    smoothed = gaussian_filter1d(motion, sigma=2.0)
    max_flow = float(np.max(smoothed))

    print(f"  [Step 3] Smoothed flow range: 0.00 – {max_flow:.2f} px/frame")

    # Step 3b: Audio onset detection (if gate enabled)
    audio_peaks = []
    if use_audio_gate:
        audio_peaks = detect_audio_onsets(video_path, ffmpeg_path)
        if audio_peaks:
            print(f"  [Audio] Detected {len(audio_peaks)} onset(s)")
        else:
            print(f"  [Audio] No onsets found — audio gate skipped")

    # Step 4: Impact frame localized via the last intense swing peak.
    # Find all local maxima (peaks) in the smoothed motion signal.
    # A peak is considered a candidate swing if:
    # 1. Its smoothed score is at least 70% of the global max score.
    # 2. The raw p99 flow magnitude at that peak frame is at least 3.2 px/frame.
    # This filters out slow continuous movements and selects the last practice/real swing.
    peaks, _ = find_peaks(smoothed, distance=int(1.5 * fps))

    swing_candidates = []
    for p in peaks:
        if p < 1 or p >= len(p99_raw):
            continue
        if smoothed[p] >= max_flow * 0.70 and p99_raw[p] >= 3.2:
            swing_candidates.append(p)

    # Diagnostic: all qualifying peaks
    print(f"  [Step 4] Found {len(swing_candidates)} qualifying peak(s):")
    for p in swing_candidates:
        print(f"           frame {p:4d} ({p/fps:.2f}s)  "
              f"smoothed={smoothed[p]:.2f}  raw_p99={p99_raw[p]:.2f} px/frame")

    # Step 4b: Audio-assisted tie-breaking
    AUDIO_WINDOW_SEC = 0.5

    def audio_score_for_frame(frame_idx):
        t = frame_idx / fps
        best_amp, best_dist = 0.0, AUDIO_WINDOW_SEC + 1.0
        for t_a, amp in audio_peaks:
            d = abs(t_a - t)
            if d <= AUDIO_WINDOW_SEC and d < best_dist:
                best_dist, best_amp = d, amp
        return best_amp, best_dist

    if len(swing_candidates) == 0:
        if use_audio_gate and audio_peaks:
            strongest = max(audio_peaks, key=lambda x: x[1])
            impact_frame = int(strongest[0] * fps)
            print(f"  [Step 4] No flow peaks — using strongest audio onset at "
                  f"{strongest[0]:.2f}s (amp={strongest[1]:.1f})")
        else:
            impact_frame = int(np.argmax(smoothed))
            print(f"  [Step 4] No flow peaks — global argmax fallback frame {impact_frame}")

    elif len(swing_candidates) == 1:
        impact_frame = swing_candidates[0]
        if use_audio_gate and audio_peaks:
            amp, dist = audio_score_for_frame(impact_frame)
            tag = f"confirmed by audio (dist={dist:.2f}s)" if amp > 0 else f"no audio match within {AUDIO_WINDOW_SEC}s"
            print(f"  [Step 4] Single candidate — {tag}")

    else:
        print(f"  [Step 4] Multiple candidates ({len(swing_candidates)}) — audio tie-breaker")
        best_frame, best_combined = None, -1.0
        for p in swing_candidates:
            amp, dist = audio_score_for_frame(p)
            if amp > 0:
                combined = smoothed[p] * amp / (1.0 + dist * 10)
                print(f"           frame {p:4d} ({p/fps:.2f}s)  flow={smoothed[p]:.2f}  "
                      f"audio_amp={amp:.1f}  dist={dist:.2f}s  combined={combined:.2f}")
                if combined > best_combined:
                    best_combined, best_frame = combined, p
        if best_frame is not None:
            impact_frame = best_frame
            print(f"  [Step 4] Selected frame {impact_frame} by audio tie-breaker")
        else:
            # No audio match — fall back to highest raw p99
            best_idx = int(np.argmax([p99_raw[p] for p in swing_candidates]))
            impact_frame = swing_candidates[best_idx]
            print(f"  [Step 4] No audio match — using highest raw p99 frame {impact_frame}")

    impact_sec = impact_frame / fps
    print(f"  [Step 4] Impact: frame {impact_frame} ({impact_sec:.2f}s), "
          f"peak flow = {smoothed[impact_frame]:.2f} px/frame")

    # Step 5: Expand clip window: -2.0s before, +1.5s after impact
    t_start_sec = max(0.0, impact_sec - 2.0)
    t_end_sec = min(total_frames / fps, impact_sec + 1.5)

    print(f"  [Step 5] Output: {t_start_sec:.2f}s -> {t_end_sec:.2f}s "
          f"({t_end_sec - t_start_sec:.2f}s)")

    # Step 6: Export with FFmpeg
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if fast_mode:
        cmd = [
            ffmpeg_path, "-y",
            "-i", video_path,
            "-ss", f"{t_start_sec:.3f}",
            "-to", f"{t_end_sec:.3f}",
            "-c", "copy",
            output_path,
        ]
    else:
        cmd = [
            ffmpeg_path, "-y",
            "-i", video_path,
            "-ss", f"{t_start_sec:.3f}",
            "-to", f"{t_end_sec:.3f}",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "aac", "-b:a", "192k",
            output_path,
        ]

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode == 0 and os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  [Done] Strike clip saved: {output_path} ({size_mb:.2f} MB)")
        return True
    else:
        print(f"  [Error] FFmpeg export failed for '{filename}'.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end Golf Strike Clip Processor.\n"
            "Stage 1: Audio silence detection removes slow-mo tail.\n"
            "Stage 2: Farneback dense optical flow detects the impact frame."
        )
    )
    parser.add_argument(
        "--full-dir", default=os.path.join("videos", "full"),
        help="Folder of raw full MP4 videos (default: videos/full).",
    )
    parser.add_argument(
        "--trimmed-dir", default=os.path.join("videos", "trimmed"),
        help="Folder for normal-speed clips (default: videos/trimmed).",
    )
    parser.add_argument(
        "--strikes-dir", default=os.path.join("videos", "strikes"),
        help="Folder for final 3-4s strike clips (default: videos/strikes).",
    )
    parser.add_argument("-f", "--fast", action="store_true", help="Stream copy, no re-encode.")
    parser.add_argument("--ffmpeg-path", help="Path to custom ffmpeg binary.")
    parser.add_argument("--audio", action="store_true",
                        help="Enable audio onset gate for tie-breaking between flow peaks.")

    args = parser.parse_args()
    ffmpeg_path = get_ffmpeg_cmd(args.ffmpeg_path)

    full_dir = os.path.abspath(args.full_dir)
    trimmed_dir = os.path.abspath(args.trimmed_dir)
    strikes_dir = os.path.abspath(args.strikes_dir)

    print("=========================================================")
    print(" GOLF STRIKE EXTRACTION PIPELINE")
    print(" Stage 1: Audio Silence -> Slow-Mo Removal")
    print(" Stage 2: Farneback Optical Flow -> Impact Detection")
    print(f" Full Videos Dir:    {full_dir}")
    print(f" Trimmed Clips Dir:  {trimmed_dir}")
    print(f" Strikes Output Dir: {strikes_dir}")
    print("=========================================================\n")

    if not os.path.exists(full_dir):
        print(f"Directory '{full_dir}' does not exist.")
        sys.exit(1)

    raw_files = sorted(f for f in os.listdir(full_dir) if f.lower().endswith(".mp4"))
    if not raw_files:
        print(f"No MP4 files found in '{full_dir}'.")
        sys.exit(0)

    print(f"Found {len(raw_files)} video(s) to process.\n")
    ok_count = 0

    for fname in raw_files:
        name, ext = os.path.splitext(fname)
        full_path = os.path.join(full_dir, fname)
        trimmed_path = os.path.join(trimmed_dir, f"{name}_trimmed{ext}")
        strike_path = os.path.join(strikes_dir, f"{name}_strike{ext}")

        print(f"Processing: {fname}")
        ok_s1 = trim_slomo(full_path, trimmed_path, ffmpeg_path, fast_mode=args.fast)
        if not ok_s1:
            print(f"  Skipping Stage 2 for '{fname}'.")
            continue

        trim_src = trimmed_path if os.path.exists(trimmed_path) else full_path
        ok_s2 = extract_golf_strike(trim_src, strike_path, ffmpeg_path,
                                    fast_mode=args.fast, use_audio_gate=args.audio)
        if ok_s2:
            ok_count += 1

    print(f"\nPipeline Complete! Extracted {ok_count}/{len(raw_files)} strike clip(s).")


if __name__ == "__main__":
    main()

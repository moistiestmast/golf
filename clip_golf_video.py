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
                pyr_scale=0.5, levels=4, winsize=9,
                iterations=5, poly_n=5, poly_sigma=1.1, flags=0
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
            p95 = max(float(np.percentile(mag_nonzero, 95)), 0.5)
            p99 = float(np.percentile(mag_nonzero, 99))
        else:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=4, winsize=9,
                iterations=5, poly_n=5, poly_sigma=1.1, flags=0
            )
            # Cancel global camera translation
            dx_median = np.median(flow[..., 0])
            dy_median = np.median(flow[..., 1])
            flow[..., 0] -= dx_median
            flow[..., 1] -= dy_median
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            p95 = max(float(np.percentile(mag, 95)), 0.5)
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

    # Temporal bandpass: subtract sustained low-frequency motion (walking)
    # so that brief swing bursts stand out.  Walking raises the baseline
    # slowly; the swing is a sharp 2-4 s spike.  Removing ~70 % of the
    # 3-second low-pass component kills walking while preserving the swing.
    # Temporal bandpass: compute a very-smoothed version (10-second low-pass)
    # that represents the "background motion level" from sustained walking.
    # Subtract it to zero out walking, then clip negative values.
    # The swing peak, being brief (2-4s), is barely affected by a 10s low-pass.
    background_motion = gaussian_filter1d(smoothed, sigma=10.0 * fps)
    smoothed = smoothed - background_motion * 0.5
    smoothed = np.maximum(smoothed, 0.0)

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

    # Step 4: Score every peak by swing-likeness physics and pick the winner.
    peaks, peak_props = find_peaks(smoothed, distance=int(1.5 * fps), prominence=1.0)

    # -- helper: measure how "swing-like" a peak's temporal shape is ----------
    def swing_score(peak_frame):
        """Return a float score; higher means more swing-like."""
        half_w = int(1.5 * fps)                        # ±1.5 s analysis window
        lo, hi = max(0, peak_frame - half_w), min(len(smoothed) - 1, peak_frame + half_w)
        local = smoothed[lo:hi + 1]
        pi = peak_frame - lo

        if len(local) < 10:
            return 0.0

        # 1) Crest factor — how tall is the peak relative to its surroundings?
        crest = smoothed[peak_frame] / (np.mean(local) + 1e-6)

        # 2) Decay speed — sharp drop after impact? (swing = abrupt stop)
        n_decay = min(5, len(local) - pi - 1)
        post = local[pi + 1:pi + 1 + n_decay] if n_decay > 0 else local[pi:pi + 1]
        decay_rate = max(0.0, (smoothed[peak_frame] - np.min(post)) / n_decay) if n_decay > 0 else 0.0

        # 3) Asymmetry — rise area vs fall area (swing: fast rise, faster fall)
        rise_area = np.trapezoid(local[:pi + 1]) if pi > 0 else 0.0
        fall_area = np.trapezoid(local[pi:])
        asymmetry = rise_area / (fall_area + 1e-6) if rise_area > 0 else 1.0

        # 4) Isolation — how much taller than the second-biggest peak in the window?
        sorted_vals = np.sort(local)[::-1]
        isolation = sorted_vals[0] / (sorted_vals[1] + 1e-6) if len(sorted_vals) >= 2 else 1.0

        # 5) Raw p99 speed at this frame (absolute motion)
        raw_p99 = p99_raw[peak_frame] if peak_frame < len(p99_raw) else 0.0

        # 6) Spatial concentration — did the *raw* motion score come from a few
        #    pixels (clubhead) or many pixels (walking)?  The motion_raw array
        #    already encodes p99*(p99/p95) which is high for localized motion.
        #    We read the pre-smoothed value at this frame.
        raw_motion = motion_raw[peak_frame] if peak_frame < len(motion_raw) else 0.0
        concentration = raw_motion / (smoothed[peak_frame] + 1e-6)
        # concentration > 1.0 means the peak was sharper before smoothing
        # (localized), < 1.0 means smoothing amplified it (diffuse).

        return (crest * 1.0
                + decay_rate * 2.0
                + asymmetry * 0.5
                + isolation * 1.0
                + raw_p99 * 0.3
                + concentration * 3.0)

    # -- score every peak and keep them sorted best-first --------------------
    candidates = []
    for p in peaks:
        if p < 1 or p >= len(p99_raw):
            continue
        if p99_raw[p] < 2.0:          # hard floor: must have at least *some* motion
            continue
        sc = swing_score(p)
        candidates.append((p, sc))

    print(f"  [Step 4] {len(candidates)} candidate peak(s) scored:")
    for p, sc in sorted(candidates, key=lambda x: x[1], reverse=True):
        print(f"           frame {p:4d} ({p/fps:.2f}s)  "
              f"smoothed={smoothed[p]:.2f}  raw_p99={p99_raw[p]:.2f}  "
              f"raw_motion={motion_raw[p]:.2f}  swing_score={sc:.2f}")

    # -- audio booster (if enabled) -----------------------------------------
    if use_audio_gate and audio_peaks:
        def audio_boost(frame_idx, base_score):
            t = frame_idx / fps
            best_amp, best_dist = 0.0, 0.5 + 1.0
            for t_a, amp in audio_peaks:
                d = abs(t_a - t)
                if d <= 0.5 and d < best_dist:
                    best_dist, best_amp = d, amp
            if best_amp > 0:
                # Audio amplitude: typical strike = 50-200+, background = 2-15
                amp_factor = best_amp / 15.0
                # Gaussian distance penalty: σ = 0.25s
                # at 0.0s → 1.0, at 0.25s → 0.37, at 0.5s → 0.02
                dist_penalty = np.exp(-0.5 * (best_dist / 0.25) ** 2)
                boost = 1.0 + amp_factor * dist_penalty
                return base_score * boost, best_dist, best_amp
            return base_score, 999.0, 0.0

        boosted = []
        for p, sc in candidates:
            new_sc, dist, amp = audio_boost(p, sc)
            boosted.append((p, new_sc))
            if amp > 0:
                print(f"           frame {p:4d} audio boost: {sc:.2f} -> {new_sc:.2f} "
                      f"(dist={dist:.2f}s, amp={amp:.1f})")
        candidates = sorted(boosted, key=lambda x: x[1], reverse=True)

    # Audio-first fallback: if we have strong audio onsets but no flow peak
    # within 0.5s of them, create synthetic candidates from audio alone.
    if use_audio_gate and audio_peaks:
        for t_a, amp in audio_peaks:
            if amp < 30:  # only strong onsets (real strikes are 50+)
                continue
            frame_a = int(t_a * fps)
            # Check if any existing candidate is within 0.5s of this onset
            nearby = any(abs(p - frame_a) < int(0.5 * fps) for p, _ in candidates)
            if not nearby and 0 <= frame_a < len(smoothed):
                # Create synthetic candidate: score based primarily on audio
                synthetic_score = amp * 0.5  # audio-dominant score
                candidates.append((frame_a, synthetic_score))
                print(f"           frame {frame_a:4d} ({t_a:.2f}s)  "
                      f"audio-only candidate  amp={amp:.1f}  score={synthetic_score:.2f}")
        # Sort candidates again to include any new synthetic candidates
        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)

    # -- pick the winner ----------------------------------------------------
    if len(candidates) == 0:
        if use_audio_gate and audio_peaks:
            strongest = max(audio_peaks, key=lambda x: x[1])
            impact_frame = int(round(strongest[0] * fps))
            print(f"  [Step 4] No flow peaks — strongest audio onset at "
                  f"{strongest[0]:.2f}s (amp={strongest[1]:.1f})")
        else:
            impact_frame = int(np.argmax(smoothed))
            print(f"  [Step 4] No flow peaks — global argmax fallback frame {impact_frame}")
    else:
        impact_frame = candidates[0][0]
        print(f"  [Step 4] Selected frame {impact_frame} ({impact_frame/fps:.2f}s) "
              f"with swing_score={candidates[0][1]:.2f}")

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

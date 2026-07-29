# Automated Golf Strike Extraction Pipeline

An automated, high-precision computer vision and audio analysis pipeline designed to extract 3.5-second golf strike clips from raw video footage (typically ~20s stationary camera clips containing intro setup, practice swings, impact, and slow-motion replays).

The pipeline replaces traditional global motion heuristics with **Farneback Dense Optical Flow**, **Shi-Tomasi Corner Attention Masking**, **skewness motion scoring**, and a **4-gate temporal filtering engine**.

---

## 📐 System Architecture & Flow

```
                 +-----------------------------------+
                 |        Raw Video Footage          |
                 |       (videos/full/*.mp4)         |
                 +-----------------------------------+
                                   |
                                   v
+---------------------------------------------------------------------+
| STAGE 1: Audio Silence Detection (Slow-Mo Replay Cutoff)            |
| - Extract 16kHz Mono PCM Audio Stream via FFmpeg                    |
| - Compute 50ms sliding window RMS amplitude                         |
| - Trim video at transition point where audio becomes silent         |
+---------------------------------------------------------------------+
                                   |
                                   v
                 +-----------------------------------+
                 |      Normal-Speed Clip            |
                 |     (videos/trimmed/*.mp4)        |
                 +-----------------------------------+
                                   |
                                   v
+---------------------------------------------------------------------+
| STAGE 2: Dense Optical Flow & Attention Strike Extraction          |
| 1. Downsample frames to standard 320x180 grayscale                  |
| 2. Compute Shi-Tomasi Corner Attention Maps (Gaussian Blurred σ=20) |
| 3. Apply 4-Gate Artifact Suppression:                               |
|    - Gate A: Absolute Brightness (> 30.0)                           |
|    - Gate B: Delta Brightness (< 2.0 px change)                     |
|    - Gate C: Stability Window (15 consecutive stable frames)       |
|    - Gate D: Modal Brightness Proximity (Within ±20 of mode)        |
| 4. Compute Masked Farneback Dense Optical Flow                      |
| 5. Calculate Skewness Motion Score: p99 * (p99 / (p95 + 1e-4))      |
| 6. Smooth motion signal with 1D Gaussian Filter (σ=2.0)             |
| 7. Localize impact frame via Peak Finding & Peak Velocity Criteria  |
| 8. Slice clip window [-2.0s, +1.5s] relative to impact              |
+---------------------------------------------------------------------+
                                   |
                                   v
                 +-----------------------------------+
                 |        Final Strike Clip          |
                 |     (videos/strikes/*.mp4)        |
                 +-----------------------------------+
```

---

## 🛠️ Technical Specifications & Algorithm Details

### 1. Stage 1: Audio Silence Detection (`trim_slomo`)
Raw broadcast/recorded golf footage often transitions into silent slow-motion replays. Stage 1 strips these out before optical flow processing.

- **Audio Format**: 16kHz 16-bit Mono PCM uncompressed stream.
- **Window Size**: `50ms` (`win_sec = 0.05`, `800 samples/win`).
- **Silence Threshold**: `RMS <= 5.0`.
- **Validation**: Requires `min_active_sec = 0.3s` of active audio to register start, followed by `min_silence_sec = 1.0s` of continuous silence to determine the cut point.

---

### 2. Stage 2: Feature Attention Masking (`compute_attention_map`)
To prevent background motion (e.g., grass blowing in the wind, camera drift, water ripples, flags) from corrupting the flow signal, an attention map isolates corner-dense regions (primarily the golfer and club).

- **Shi-Tomasi Detection**: `cv2.goodFeaturesToTrack` (`maxCorners=300`, `qualityLevel=0.01`, `minDistance=5`, `blockSize=5`).
- **Spatial Blur**: Gaussian Kernel with `sigmaX=20` (creates a soft ~60px radius of influence around each feature point).
- **Normalization**: Rescaled to $[0.0, 1.0]$.
- **Temporal Mask Union**: Combines previous and current frame attention maps via conservative element-wise minimum:
  $$\text{Mask}_{\text{combined}} = \min(\text{Mask}_{t-1}, \text{Mask}_t)$$
  *Rationale*: Ensures motion is only evaluated in regions where both consecutive frames contain features.
- **Fallback**: If fewer than 10 corners are detected (e.g., pitch-black frames), the attention mask defaults to $1.0$ (unmasked full frame).

---

### 3. Stage 2: 4-Gate Artifact Suppression Engine
Artifacts such as fade-in blackouts, camera flashes, light auto-adjustments, and scene transitions generate artificial high-magnitude optical flow. The 4-gate engine nullifies these frames:

| Gate | Name | Parameter | Description |
|---|---|---|---|
| **Gate A** | Absolute Brightness | `BRIGHTNESS_GATE = 30.0` | Rejects frames below 30.0 mean pixel intensity (blackouts/dark fades). |
| **Gate B** | Delta Brightness | `DELTA_BRIGHT_GATE = 2.0` | Rejects frame pairs with mean brightness delta $\ge 2.0$ (fade ramps/flashes). |
| **Gate C** | Stability Window | `STABILITY_WINDOW = 15` | Requires 15 consecutive stable frames (~0.5s) before accepting motion scores. |
| **Gate D** | Modal Brightness Proximity | `MODAL_TOLERANCE = 20.0` | Computes 10-unit histogram mode of valid scene brightness; rejects frames deviating by $> 20.0$ units. |

---

### 4. Stage 2: Skewness-Based Optical Flow Scoring
Standard spatial mean optical flow measures overall scene activity (making a walking golfer score higher than a clubhead swing). The pipeline uses a **skewness-amplified velocity score** on masked Farneback flow:

1. **Farneback Parameters**: `pyr_scale=0.5`, `levels=3`, `winsize=15`, `iterations=3`, `poly_n=5`, `poly_sigma=1.2`.
2. **Masked Flow Magnitude**:
   $$\text{Mag}_{\text{masked}} = \|\text{Flow}\|_2 \cdot \text{Mask}_{\text{combined}}$$
3. **Non-Zero Threshold**: If $< 100$ non-zero pixels exist in $\text{Mag}_{\text{masked}}$, motion score is set to $0.0$.
4. **Skewness Formula**:
   $$\text{Score} = p_{99} \cdot \left( \frac{p_{99}}{p_{95} + 1e-4} \right)$$
   - **Golf Strike Dynamics**: Clubhead moves extremely fast in a tiny spatial footprint $\implies p_{99} \gg p_{95} \implies$ **High Skewness Score**.
   - **Walking Golfer Dynamics**: Entire body moves at diffuse, low speeds $\implies p_{99} \approx p_{95} \implies$ **Low Skewness Score**.

---

### 5. Stage 2: Peak Selection & Strike Localization
1. **1D Gaussian Smoothing**: Motion scores smoothed with `sigma=2.0`.
2. **Peak Candidates**: `scipy.signal.find_peaks` finds local maxima separated by at least $1.5\text{s}$ (`distance = int(1.5 * fps)`).
3. **Candidate Validation**:
   - Smoothed Peak Score $\ge 70\%$ of global maximum smoothed flow (`max_flow * 0.70`).
   - Raw $p_{99}$ Flow Velocity at peak frame $\ge 3.2\text{ px/frame}$.
4. **Last-Peak Selection**: Selects the **last valid candidate peak** as `impact_frame`. This cleanly selects the actual swing over practice waggles/swings occurring earlier in the video.
5. **Clip Slicing**: Output clip set to $[-2.0\text{s}, +1.5\text{s}]$ around `impact_frame` ($\approx 3.5\text{s}$ total length).

---

## 💻 Dependencies & Requirements

- Python 3.8+
- OpenCV (`cv2`)
- NumPy
- SciPy
- FFmpeg (system binary or `imageio-ffmpeg` package)

Install dependencies via pip:
```bash
pip install opencv-python numpy scipy imageio-ffmpeg
```

---

## 🚀 Execution & Usage

### Project Directory Structure
```
golf/
├── clip_golf_video.py       # End-to-end master pipeline script
├── extract_strike_clips.py  # Standalone Stage 2 strike extractor
├── videos/
│   ├── full/                # [Input] Raw MP4 video files (1.mp4, 2.mp4, ...)
│   ├── trimmed/             # [Intermediate] Audio silence-trimmed clips
│   └── strikes/             # [Output] Final 3.5s extracted strike clips
```

### Running the Pipeline
Place raw `.mp4` video files into `videos/full/` and run:

```bash
python clip_golf_video.py
```

### Command Line Arguments
```bash
python clip_golf_video.py --full-dir videos/full --trimmed-dir videos/trimmed --strikes-dir videos/strikes
```

- `-f` / `--fast`: Stream copy mode for fast clipping without re-encoding.
- `--ffmpeg-path`: Custom path to FFmpeg binary.

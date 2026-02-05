# Pipeline Stages

The pipeline has 6 stages. Which ones run is controlled by `pipeline.steps` in your config.

Pose mode typically runs: `download → manifest → extract → normalize → webdataset`
Video mode typically runs: `download → manifest → clip_video → webdataset`

---

## 1. download

Downloads YouTube videos and transcripts using yt-dlp and `youtube-transcript-api`.

- **Input:** `download.video_ids_file` (one video ID per line)
- **Output:** `{paths.videos}/{video_id}.mp4`, `{paths.transcripts}/{video_id}.json`
- **Key config:** `download.format`, `download.rate_limit`, `download.languages`, `download.concurrent_fragments`
- **Notes:** Skips already-downloaded files. Only used by YouTube-ASL; How2Sign requires pre-downloaded data.

---

## 2. manifest

Parses transcript JSON files into a TSV manifest with one row per segment.

- **Input:** `{paths.transcripts}/*.json`
- **Output:** `{paths.manifest}` (CSV with columns `VIDEO_NAME`, `SENTENCE_NAME`, `START_REALIGNED`, `END_REALIGNED`, `SENTENCE`)
- **Key config:** `manifest.max_text_length`, `manifest.min_duration`, `manifest.max_duration`
- **Notes:** Applies Unicode normalization (ftfy), text cleaning, and duration/length filtering. Segment IDs follow the pattern `{video_id}-{index}`.

---

## 3. extract

Extracts per-frame pose landmarks from video segments defined in the manifest.

- **Input:** `{paths.videos}/{video_name}.mp4` + manifest
- **Output:** `{paths.landmarks}/{sentence_name}.npy` with shape `(T, K, 4)` where channels are `[x, y, z, visibility]`
- **Key config:** `extractor.name`, `extractor.max_workers`, `processing.target_fps`, `processing.frame_skip`, `processing.accept_fps_range`
- **Keypoint counts (K):**
  - MediaPipe (refined face): **553** = 33 pose + 478 face + 21 left hand + 21 right hand
  - MediaPipe (unrefined): **543** = 33 pose + 468 face + 21 left hand + 21 right hand
  - MMPose RTMPose3D: **133** (COCO WholeBody)
- **Notes:** Always outputs all keypoints; reduction happens in the normalize step. Uses `ProcessPoolExecutor` for parallel extraction. MMPose requires CUDA.

---

## 4. normalize

Applies keypoint reduction, visibility masking, coordinate normalization, and flattening.

- **Input:** `{paths.landmarks}/*.npy` with shape `(T, K, 4)`
- **Output:** `{paths.normalized}/*.npy` with shape `(T, K'*C)` flattened, where `K'` is the reduced keypoint count and `C` is 3 or 4 depending on `remove_z`
- **Key config:** `normalize.mode`, `normalize.reduction`, `normalize.keypoint_indices`, `normalize.remove_z`, `normalize.visibility_threshold`, `normalize.unvisible_value`
- **Processing pipeline:**
  1. **Reduction** -- select a subset of keypoints (default: 85 from any extractor). Custom indices can be specified via `keypoint_indices`.
  2. **Visibility masking** -- frame-level: replace all-zero frames with `unvisible_value`; landmark-level: replace landmarks below `visibility_threshold`.
  3. **Normalization** -- `xy_isotropic_z_minmax` normalizes XY isotropically and Z via min-max; `isotropic_3d` normalizes XYZ with a single scale factor.
  4. **Z removal** -- optionally drop the z-coordinate.
  5. **Flatten** -- reshape from `(T, K', C)` to `(T, K'*C)`.

---

## 5. clip_video

Clips full-length videos into segments using ffmpeg, based on the manifest timestamps.

- **Input:** `{paths.videos}/{video_name}.mp4` + manifest
- **Output:** `{paths.clips}/{sentence_name}.mp4`
- **Key config:** `clip_video.codec`, `clip_video.resize`
- **Notes:** Uses `codec: copy` by default for fast stream-copy (no re-encoding). Set `codec: libx264` if you need re-encoding. Optional `resize: [width, height]` for rescaling.

---

## 6. webdataset

Packages processed outputs into tar shards for efficient data loading with the `webdataset` library.

- **Input:** Normalized `.npy` files (pose mode) or clipped `.mp4` files (video mode) + manifest
- **Output:** `{paths.webdataset}/shard-{NNNNNN}.tar`
- **Key config:** `webdataset.max_shard_count`, `webdataset.max_shard_size`
- **Shard contents per sample:**
  - `__key__` -- sentence name
  - `.npy` or `.mp4` -- landmark array or video clip
  - `.txt` -- transcript text
  - `.json` -- metadata (video ID, sentence name, start/end times, extractor, mode)

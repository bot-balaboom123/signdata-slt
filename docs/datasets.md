# Datasets

## YouTube-ASL

A large-scale, open-domain ASL-English parallel corpus with 11,000+ YouTube videos and 73,000+ segments ([Uthus et al., 2023](https://arxiv.org/abs/2306.15162)).

**Full pipeline:** `download → manifest → extract → normalize → webdataset`

```bash
python -m sign_prep configs/youtube_asl/pose_mediapipe.yaml
```

Requires `download.video_ids_file` pointing to the video ID list (included at `assets/youtube-asl_youtube_asl_video_ids.txt`). The download step fetches videos via yt-dlp and transcripts via `youtube-transcript-api`.

## How2Sign

80+ hours of instructional "how-to" videos with continuous ASL, recorded in a controlled environment with professional signers ([Duarte et al., CVPR 2021](https://how2sign.github.io/)).

**Pipeline:** `extract → normalize → webdataset` (no download or manifest steps)

```bash
python -m sign_prep configs/how2sign/pose_mediapipe.yaml
```

**Setup:**
1. Download the dataset from [how2sign.github.io](https://how2sign.github.io/)
2. Place videos in the `videos` path (default: `dataset/how2sign/videos/`)
3. Place the alignment CSV (e.g. `how2sign_realigned_val.csv`) at the `manifest` path

The How2Sign dataset class rejects configs that include `download` or `manifest` steps.

## Adding a New Dataset

1. Create a dataset class with `@register_dataset`:

```python
from sign_prep.datasets.base import BaseDataset
from sign_prep.registry import register_dataset

@register_dataset("my_dataset")
class MyDataset(BaseDataset):
    name = "my_dataset"

    @classmethod
    def validate_config(cls, config):
        # Raise ValueError for invalid configs
        pass
```

2. Import the module in `datasets/__init__.py` so the decorator runs at startup.

3. Create a YAML config under `configs/my_dataset/` that sets `dataset: my_dataset` and defines `pipeline.steps`.

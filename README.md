# Sign Language Preprocessing Pipeline

A config-driven, modular pipeline for preprocessing **American Sign Language (ASL)** datasets.
Supports **YouTube-ASL** and **How2Sign** with two landmark extractors (**MediaPipe Holistic** and **MMPose RTMPose3D**) and two output modes (pose landmarks and video clips).

## Features

- **Config-driven** -- YAML configs with base inheritance and CLI overrides
- **Two extractors** -- MediaPipe Holistic (553 keypoints) and MMPose RTMPose3D (133 keypoints)
- **Two pipeline modes** -- `pose` (landmarks) and `video` (clip extraction)
- **Registry architecture** -- add datasets, processors, and extractors via decorators
- **Parallel processing** -- multi-worker extraction, normalization, and clipping
- **WebDataset output** -- sharded tar archives for efficient training data loading

## Project Structure

```
Sign-Language-Preprocessing/
├── configs/
│   ├── _base/                  # Shared base configs
│   │   ├── pose_mediapipe.yaml
│   │   ├── pose_mmpose.yaml
│   │   └── video.yaml
│   ├── youtube_asl/            # YouTube-ASL dataset configs
│   └── how2sign/               # How2Sign dataset configs
├── src/sign_prep/
│   ├── __main__.py             # CLI entry point
│   ├── cli.py                  # Argument parsing
│   ├── registry.py             # Component registry
│   ├── config/                 # YAML loading & Pydantic schema
│   ├── pipeline/               # PipelineRunner & PipelineContext
│   ├── datasets/               # Dataset definitions
│   ├── processors/             # Pipeline step implementations
│   ├── extractors/             # MediaPipe & MMPose extractors
│   ├── models/                 # MMPose model configs & checkpoints
│   └── utils/                  # Video, file, and text utilities
├── docs/                       # Documentation
├── assets/                     # Video ID lists, demo files
├── tests/                      # Test suite
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/gorden-chen/Sign-Language-Preprocessing.git
cd Sign-Language-Preprocessing
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**MMPose (optional, GPU required):**

```bash
pip install -U openmim
mim install mmcv==2.0.1 mmengine==0.10.7 mmdet==3.1.0
git clone https://github.com/open-mmlab/mmpose.git ../mmpose
pip install -v -e ../mmpose
export PYTHONPATH="/path/to/mmpose:$PYTHONPATH"

# Download checkpoints into src/sign_prep/models/checkpoints/
wget -P src/sign_prep/models/checkpoints/ \
  https://download.openmmlab.com/mmpose/v1/wholebody_3d_keypoint/rtmw3d/rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth
wget -P src/sign_prep/models/checkpoints/ \
  https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth
```

## Quick Start

```bash
# YouTube-ASL with MediaPipe (full pipeline)
python -m sign_prep configs/youtube_asl/pose_mediapipe.yaml

# How2Sign with MMPose (requires pre-downloaded data)
python -m sign_prep configs/how2sign/pose_mmpose.yaml

# Override config values from the command line
python -m sign_prep configs/youtube_asl/pose_mediapipe.yaml \
  --override processing.max_workers=8 pipeline.stop_at=extract
```

## Documentation

- [Architecture](docs/architecture.md) -- system design, registry, pipeline flow
- [Configuration](docs/configuration.md) -- full config reference, inheritance, CLI overrides
- [Pipeline Stages](docs/pipeline-stages.md) -- all 6 processing stages
- [Datasets](docs/datasets.md) -- YouTube-ASL vs How2Sign setup

## Citation

```bibtex
@misc{uthus2023youtubeasl,
  author    = {Uthus, David and Tanzer, Garrett and Georg, Manfred},
  title     = {YouTube-ASL: A Large-Scale, Open-Domain American Sign Language-English Parallel Corpus},
  year      = {2023},
  eprint    = {2306.15162},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2306.15162},
}

@inproceedings{duarte2021how2sign,
  author    = {Duarte, Amanda and Palaskar, Shruti and Ventura, Lucas and Ghadiyaram, Deepti and DeHaan, Kenneth and Metze, Florian and Torres, Jordi and Giro-i-Nieto, Xavier},
  title     = {How2Sign: A Large-scale Multimodal Dataset for Continuous American Sign Language},
  booktitle = {CVPR},
  year      = {2021},
}
```

## License

MIT -- see [LICENSE](LICENSE).

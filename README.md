<p align="center">
  <img src="assets/logo.jpeg" alt="Logo" width="100"/>
</p>
<h2 align="center"><a href="https://arxiv.org/abs/2510.11026">GIR-Bench: Versatile Benchmark for Generating Images with Reasoning</a></h2>

[![arXiv](https://img.shields.io/badge/arXiv-2510.11026-b31b1b.svg)](https://arxiv.org/abs/2510.11026)
[![Project Page](https://img.shields.io/badge/Project-Website-green)](https://hkust-longgroup.github.io/GIR-Bench)
[![Dataset](https://img.shields.io/badge/🤗-Dataset-blue.svg)](https://huggingface.co/datasets/lihxxx/GIR-Bench)
[![hf_paper](https://img.shields.io/badge/🤗-Paper%20In%20HF-red.svg)](https://huggingface.co/papers/2510.11026)

Reasoning-centric evaluation of multimodal unified models across Understanding – Generation Consistency (UGC), Text-to-Image, and Editing, revealing the persistent gap between reasoning and faithful generation.
![GIR-Bench Overview](assets/fig1.png)

## 📣 News
- **`2025/10/14`**: We have released the evaluation code and the dataset for GIR-bench.

## 🔧 Preparations 
#### Environment Setup

```bash
conda create -n GIR-Bench python==3.10
conda activate GIR-Bench
pip install -r requirement.txt
git clone https://github.com/facebookresearch/dinov3.git
```
#### Dataset Download
```
huggingface-cli download --resume --repo-type dataset lihxxx/GIR-Bench --local-dir ./dataset
```

#### Pre-Trained Weights Download
```
mkdir weights
huggingface-cli download --resume-download OpenGVLab/InternVL3_5-38B-HF --local-dir ./weights/InternVL3_5-38B-HF
```
Please download `dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth` from the [Meta DINOv3 Downloads](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) page and place it under `weights/`.

## 🔥 Evaluation
#### GIR-Bench-UGC and GIR-Bench-T2I
```bash
bash run_evaluation_gen.sh
```

#### GIR-Bench-Edit
```bash
bash run_evaluation_edit.sh
```
#### Evaluate Your Own Model
Please organize your model outputs as below and put them into the corresponding `MODELS_DIR`. Default locations:
- t2i: `MODELS_DIR=./dataset/generation/t2i`
- editing: `MODELS_DIR=./dataset/generation/editing`

Recommended directory and naming conventions (filenames must align with the task `id` in the dataset):

```text
dataset/
└── generation/
    ├── t2i/
    │   └── <YourModel>/
    │       ├── SpatialLayout/
    │       │   └── <image_id>.png
    │       ├── NumericalReasoning/
    │       │   └── <image_id>.png
    │       ├── TextRendering/
    │       │   └── <image_id>.png
    │       ├── Zoology/
    │       │   └── <image_id>.png
    │       ├── Botany/
    │       │   └── <image_id>.png
    │       └── Geography/
    │           └── <image_id>.png
    └── editing/
        └── <YourModel>/
            ├── ReasoningPerception/
            │   └── <image_id>.png
            ├── VisualLogic/
            │   └── <image_id>.png
            └── VisualPuzzle/
                └── <image_id>.png
```

By default, the scripts evaluate all subfolders under the configured `MODELS_DIR`. To evaluate only specific models:

1) Set the `MODELS` array in the shell script:

```bash
MODELS=("YourModel1" "YourModel2")
```

2) Enable the `--models` flag by uncommenting it in each Python call within the script:

```bash
# ... inside each python command block
--models "${MODELS[@]}"
```

3) Run the script:

```bash
bash run_evaluation_gen.sh
bash run_evaluation_edit.sh
```

## 🔍 Citation

```
@article{li2025gir-bench,
  title={GIR-Bench: Versatile Benchmark for Generating Images with Reasoning},
  author={Hongxiang Li, Yaowei Li, Bin Lin, Yuwei Niu, Yuhang Yang, Xiaoshuang Huang, Jiayin Cai, Xiaolong Jiang, Yao Hu, Long Chen},
  journal={arXiv preprint arXiv:2510.11026},
  year={2025}
}
```
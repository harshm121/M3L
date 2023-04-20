# M3L: Multi-modal Teacher for Masked Modality Learning
Official repository for [Missing Modality Robustness in Semi-Supervised Multi-Modal Semantic Segmentation](). 


Harsh Maheshwari, Yen-Cheng Liu, Zsolt Kira

[Project Page](https://harshm121.github.io/projects/m3l.html), [Demo](https://harshm121-m3l.hf.space/)


## What do you get from the repository?
1. Multi-modal Segformer based segmentation model: Linear Fusion - outperforms other baselines on RGBD segmentation
2. M3L: Semi-supervised framework for multi-modal segmentation with missing modality robustness (Can be used for uni-modal inference)
3. Benchmarking files and configurations for semi-supervised learning on Stanford-Indoor and SUNRGBD datasets
4. Mean Teacher and CPS implementation (model agnostic) segmentation


## Installation

Run the following commands:

```
git clone git@github.com:harshm121/M3L.git

conda create -n mmsemienv python=3.6

pip install -r requirements.txt

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```


## Data

### Stanford Indoor

#### Download

- Download the Stanford Indoor dataset using the instructions [here](https://github.com/alexsax/2D-3D-Semantics)

- Run  `data/stanford_indoor_dataset/gen_sid.py` to generate rgb images, depth images and segmentation labels (Sets 255 in segmentation map wherever label is missing and fills in 0 for missing depth values)

#### Train/Val/Test split and Semi-Supervised Configs

- `data/stanford_indoor_dataset/semi_supervised_configs` contains train, val and test splits as well as semi-supervised configs for different labeled ratios. (Prepared using `data/stanford_indoor_dataset/create_configs.py`)

#### Format config files

- Format config files for dataloaders put them in `data/stanford_indoor_dataset/configs` using `data/stanford_indoor_dataset/prepare_configs.py`



### SUNRGBD

- Download the SUNRGBD dataset with 37 classes using the instructions [here](https://github.com/ankurhanda/sunrgbd-meta-data#training-on-rgb-data-for-37-classes)


### Dataset Folder Structure

The code requires the datasets in `data` folder in the following format:

  ```bash
    
    ├── stanford_indoor_dataset
    │   ├── configs
    │   │   ├── config_depth
    │   │   │   ├── test.txt
    │   │   │   ├── train_config
    │   │   │   └── val.txt
    │   │   ├── config_rgb
    │   │   │   ├── test.txt
    │   │   │   ├── train_config
    │   │   │   └── val.txt
    │   │   ├── config_rgbd
    │   │   │   ├── test.txt
    │   │   │   ├── train_config
    │   │   │   ├── train.txt
    │   │   │   └── val.txt
    │   │   ├── test_orig.txt
    │   │   └── train_orig.txt
    │   ├── create_configs.py
    │   ├── data
    │   ├── gen_sid.py
    │   ├── prepare_configs.py
    │   └── semi_supervised_configs
    │       ├── test.txt
    │       ├── train_config
    │       └── val.txt
    └── sunrgbd
  ```


### Checkpoints

| Dataset | Labels used | Modality | Framework       | Checkpoint | Test mIoU | Config file |
|---------|-------------|----------|-----------------|------------|-----------|-------------|
| SID     | 0.1% (49)   | RGB + D  | Supervised Only | -          | -         | -           |
| SID     | 0.1% (49)   | RGB + D  | Mean Teacher    | -          | -         | -           |
| SID     | 0.1% (49)   | RGB + D  | M3L             | -          | -         | -           |
| SID     | 0.2% (98)   | RGB + D  | Supervised Only | -          | -         | -           |
| SID     | 0.2% (98)   | RGB + D  | Mean Teacher    | -          | -         | -           |
| SID     | 0.2% (98)   | RGB + D  | M3L             | -          | -         | -           |
| SID     | 1% (491)    | RGB + D  | Supervised Only | -          | -         | -           |
| SID     | 1% (491)    | RGB + D  | Mean Teacher    | -          | -         | -           |
| SID     | 1% (491)    | RGB + D  | M3L             | -          | -         | -           |
| SUNRGBD | 6.25% (297) | RGB + D  | Supervised Only | -          | -         | -           |
| SUNRGBD | 6.25% (297) | RGB + D  | Mean Teacher    | -          | -         | -           |
| SUNRGBD | 6.25% (297) | RGB + D  | M3L             | -          | -         | -           |
| SUNRGBD | 12.5% (594) | RGB + D  | Supervised Only | -          | -         | -           |
| SUNRGBD | 12.5% (594) | RGB + D  | Mean Teacher    | -          | -         | -           |
| SUNRGBD | 12.5% (594) | RGB + D  | M3L             | -          | -         | -           |
| SUNRGBD | 25% (1189)  | RGB + D  | Supervised Only | -          | -         | -           |
| SUNRGBD | 25% (1189)  | RGB + D  | Mean Teacher    | -          | -         | -           |
| SUNRGBD | 25% (1189)  | RGB + D  | M3L             | -          | -         | -           |

## Usage

### Training
``` python main_ddp.py --cfg_file </path/to/config_file> --verbose iter```

### Evaluation
``` python main_ddp_eval.py --cfg_file </path/to/config_file> --verbose iter```


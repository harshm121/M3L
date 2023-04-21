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

```bash
git clone git@github.com:harshm121/M3L.git
conda create -n mmsemienv python=3.6
pip install -r requirements.txt
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

#### Train/Val/Test split and Semi-Supervised Configs

- `data/sunrgbd/config` contains train, val and test splits as well as semi-supervised configs for different labeled ratios.
- Since SUNRGBD val images vary in size, there are different files for different image sizes.

---

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
    │   └── config
    │   │   ├── train_config_rgbd
    │   │   ├── <all_val_files>.txt
    │   │   └── <all_test_files>.txt

  ```

---

### Hugging Face Demo

-- [Demo](https://harshm121-m3l.hf.space/), [Code](https://huggingface.co/spaces/harshm121/M3L/tree/main)

---

### Checkpoints

| Dataset | Labels used | Modality | Framework       | Config file                          | Checkpoint                                             | Test mIoU |
|---------|-------------|----------|-----------------|--------------------------------------|--------------------------------------------------------|-----------|
| SID     | 0.1% (49)   | RGB + D  | Supervised Only | src/configs/sid_0.1_suponly.yml      | [sid_0.1_suponly.pth](https://huggingface.co/harshm121/M3L/blob/main/SID/sid_0.1_suponly.pth)          | 42.09     |
| SID     | 0.1% (49)   | RGB + D  | Mean Teacher    | src/configs/sid_0.1_mt.yml           | [sid_0.1_mt.pth](https://huggingface.co/harshm121/M3L/blob/main/SID/sid_0.1_mt.pth)               | 41.77     |
| SID     | 0.1% (49)   | RGB + D  | M3L             | src/configs/sid_0.1_m3l.yml          | [sid_0.1_m3l.pth](https://huggingface.co/harshm121/M3L/blob/main/SID/sid_0.1_m3l.pth)              | 44.1      |
| SID     | 0.2% (98)   | RGB + D  | Supervised Only | src/configs/sid_0.2_suponly.yml      | [sid_0.2_suponly.pth](https://huggingface.co/harshm121/M3L/blob/main/SID/sid_0.2_suponly.pth)          | 46.6      |
| SID     | 0.2% (98)   | RGB + D  | Mean Teacher    | src/configs/sid_0.2_mt.yml           | [sid_0.2_mt.pth](https://huggingface.co/harshm121/M3L/blob/main/SID/sid_0.2_mt.pth)               | 48.54     |
| SID     | 0.2% (98)   | RGB + D  | M3L             | src/configs/sid_0.2_m3l.yml          | [sid_0.2_m3l.pth](https://huggingface.co/harshm121/M3L/blob/main/SID/sid_0.2_m3l.pth)              | 49.05     |
| SID     | 1% (491)    | RGB + D  | Supervised Only | src/configs/sid_1.0_suponly.yml      | [sid_1.0_suponly.pth](https://huggingface.co/harshm121/M3L/blob/main/SID/sid_1.0_suponly.pth)          | 52.47     |
| SID     | 1% (491)    | RGB + D  | Mean Teacher    | src/configs/sid_1.0_mt.yml           | [sid_1.0_mt.pth](https://huggingface.co/harshm121/M3L/blob/main/SID/sid_1.0_mt.pth)               | 54.32     |
| SID     | 1% (491)    | RGB + D  | M3L             | src/configs/sid_1.0_m3l.yml          | [sid_1.0_m3l.pth](https://huggingface.co/harshm121/M3L/blob/main/SID/sid_1.0_m3l.pth)              | 55.48     |
| SUNRGBD | 6.25% (297) | RGB + D  | Supervised Only | src/configs/sunrgbd_6.25_suponly.yml | [sunrgbd_6.25_suponly.pth](https://huggingface.co/harshm121/M3L/blob/main/SUNRGBD/sunrgbd_6.25_suponly.pth) | 32        |
| SUNRGBD | 6.25% (297) | RGB + D  | Mean Teacher    | src/configs/sunrgbd_6.25_mt.yml      | [sunrgbd_6.25_mt.pth](https://huggingface.co/harshm121/M3L/blob/main/SUNRGBD/sunrgbd_6.25_mt.pth)      | 31.11     |
| SUNRGBD | 6.25% (297) | RGB + D  | M3L             | src/configs/sunrgbd_6.25_m3l.yml     | [sunrgbd_6.25_m3l.pth](https://huggingface.co/harshm121/M3L/blob/main/SUNRGBD/sunrgbd_6.25_m3l.pth)     | 30.67     |
| SUNRGBD | 12.5% (594) | RGB + D  | Supervised Only | src/configs/sunrgbd_12.5_suponly.yml | [sunrgbd_12.5_suponly.pth](https://huggingface.co/harshm121/M3L/blob/main/SUNRGBD/sunrgbd_12.5_suponly.pth) | 35.88     |
| SUNRGBD | 12.5% (594) | RGB + D  | Mean Teacher    | src/configs/sunrgbd_12.5_mt.yml      | [sunrgbd_12.5_mt.pth](https://huggingface.co/harshm121/M3L/blob/main/SUNRGBD/sunrgbd_12.5_mt.pth)      | 39.17     |
| SUNRGBD | 12.5% (594) | RGB + D  | M3L             | src/configs/sunrgbd_12.5_m3l.yml     | [sunrgbd_12.5_m3l.pth](https://huggingface.co/harshm121/M3L/blob/main/SUNRGBD/sunrgbd_12.5_m3l.pth)     | 39.7      |
| SUNRGBD | 25% (1189)  | RGB + D  | Supervised Only | src/configs/sunrgbd_25_suponly.yml   | [sunrgbd_25_suponly.pth](https://huggingface.co/harshm121/M3L/blob/main/SUNRGBD/sunrgbd_25_suponly.pth)   | 42.09     |
| SUNRGBD | 25% (1189)  | RGB + D  | Mean Teacher    | src/configs/sunrgbd_25_mt.yml        | [sunrgbd_25_mt.pth](https://huggingface.co/harshm121/M3L/blob/main/SUNRGBD/sunrgbd_25_mt.pth)        | 41.95     |
| SUNRGBD | 25% (1189)  | RGB + D  | M3L             | src/configs/sunrgbd_25_suponly.yml   | [sunrgbd_25_m3l.pth](https://huggingface.co/harshm121/M3L/blob/main/SUNRGBD/sunrgbd_25_m3l.pth)       | 42.69     |

### Reproducing results

 - Reproducing results from the paper is easy. Just download the config file and the checkpoint from the table above and run the following command:

 ```python
 python main_ddp_reproduce.py --cfg_file </path/to/config_file.yml> --verbose iter --checkpoint <path/to/checkpoint.pth>
 ```

---


## Usage

### Config file
 - Enter the correct root_dir which is the `/path/to/M3L` in the config file. 

 - Available segmentation models: <br>
        * [Uni-modal CNN based] `dlv3p`,  `refinenet` (with possible base models: `r18`, `r50`, `r101`)<br>
        * [Uni-modal Segformer based] `segformer` (with possible base models: `mit_b0`, `mit_b1`, `mit_b2`, `mit_b3`, `mit_b4`, `mit_b5`)<br>
        * [Multi-modal CNN based] `cen` - extending `refinenet` to multi-modal<br>
        * [Multi-modal Segformer based] `linearfusion` (proposed Linear Fusion), `tokenfusion` ([Token Fusion](https://arxiv.org/pdf/2204.08721.pdf)), `unifiedrepresentationnetwork` - extending `segformer` to multi-modal<br>

 - Available Training frameworks: <br>
     * Without modality dropout:<br>
        * `nossl`: supervised training<br>
        * `mean_teacher`: mean teacher training<br>
        * `cps`: [Cross Pseudo Supervision](https://arxiv.org/abs/2106.01226)<br>
     * With modality dropout:<br>
        * `nosslmoddrop`: supervised training with modality dropout<br>
        * `meanteachermoddrop`: mean teacher training with modality dropout<br>
        * `meanteachermaskedstudent`: Proposed M3L training<br><br>
        (Note: for these frameworks, use `linearfusionmaskedconsmixbatch` instead of `linearfusion`, `tokenfusionmaskedconsmixbatch` instead of `tokenfusion`, `unifiedrepresentationnetworkmoddrop` instead of `unifiedrepresentationnetwork` for enabling modality dropout)


### Training
```python
 python main_ddp.py --cfg_file </path/to/config_file.yml> --verbose iter
 ```

### Evaluation
```python
 python main_ddp_eval.py --cfg_file </path/to/config_file.yml> --verbose iter
 ```

### Test
```python
 python main_ddp_test.py --cfg_file </path/to/config_file.yml> --verbose iter --checkpoint <iter_number>
 ```



---

This code is inspired from [pytorch-semseg](https://github.com/meetps/pytorch-semseg), [TorchSemiSeg](https://github.com/charlesCXK/TorchSemiSeg) and [Token Fusion](https://github.com/yikaiw/TokenFusion), thanks to their open-source code.
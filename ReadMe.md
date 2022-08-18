# Estimating and Exploiting the Aleatoric Uncertainty in Surface Normal Estimation

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/estimating-and-exploiting-the-aleatoric/surface-normals-estimation-on-nyu-depth-v2-1)](https://paperswithcode.com/sota/surface-normals-estimation-on-nyu-depth-v2-1?p=estimating-and-exploiting-the-aleatoric)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/estimating-and-exploiting-the-aleatoric/surface-normals-estimation-on-scannetv2)](https://paperswithcode.com/sota/surface-normals-estimation-on-scannetv2?p=estimating-and-exploiting-the-aleatoric)

Official implementation of the paper

> **Estimating and Exploiting the Aleatoric Uncertainty in Surface Normal Estimation** \
> ICCV 2021 [oral] \
> [Gwangbin Bae](https://baegwangbin.com), [Ignas Budvytis](https://mi.eng.cam.ac.uk/~ib255/), and [Roberto Cipolla](https://mi.eng.cam.ac.uk/~cipolla/) \
> [[arXiv]](https://arxiv.org/abs/2109.09881) [[youtube]](https://youtu.be/mTy85tJ2oAQ)

<p align="center">
  <img width=70% src="https://github.com/baegwangbin/surface_normal_uncertainty/blob/main/figs/readme_scannet.png?raw=true">
</p>

The proposed method estimates the per-pixel surface normal probability distribution, from which the expected angular error can be inferred to quantify the aleatoric uncertainty. 
We also introduce a novel decoder framework where pixel-wise MLPs are trained on a subset of pixels selected based on the uncertainty. 
Such uncertainty-guided sampling prevents the bias in training towards large planar surfaces, thereby improving the level of the detail in the prediction.

## Getting Started

We recommend using a virtual environment.
```
python3.6 -m venv --system-site-packages ./venv
source ./venv/bin/activate
```

Install the necessary dependencies by
```
python3.6 -m pip install -r requirements.txt
```

Download the pre-trained model weights and sample images.

```python
python download.py && cd examples && unzip examples.zip && cd ..
```
[25 Apr 2022] ***`download.py` does not work anymore. Please download the models and example images directly from [this link](https://drive.google.com/drive/folders/1Ku25Am69h_HrbtcCptXn4aetjo7sB33F?usp=sharing), and unzip them under `./checkpoints/` and `./examples/`.***

Running the above will download 
* `./checkpoints/nyu.pt` (model trained on NYUv2)
* `./checkpoints/scannet.pt` (model trained on ScanNet)
* `./examples/*.png` (sample images)

## Run Demo

To test on your own images, please add them under `./examples/`. The images should be in `.png` or `.jpg`.

Test using the network trained on [NYUv2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html). We used the ground truth and data split provided by [GeoNet](https://github.com/xjqi/GeoNet).
>Please note that the ground truth for NYUv2 is only defined for the center crop of image. The prediction is therefore not accurate outside the center. When testing on your own images, we recommend using the network trained on ScanNet. 

```python
python test.py --pretrained nyu --architecture GN
```

Test using the network trained on [ScanNet](http://www.scan-net.org/). We used the ground truth and data split provided by [FrameNet](https://github.com/hjwdzh/FrameNet).

```python
python test.py --pretrained scannet --architecture BN
```

Running the above will save the predicted surface normal and uncertainty under `./examples/results/`. If successful, you will obtain images like below.

<p align="center">
  <img width=100% src="https://github.com/baegwangbin/surface_normal_uncertainty/blob/main/figs/readme_generalize.png?raw=true">
</p>

The predictions in the figure above are obtained by the network trained only on ScanNet. The network generalizes well to objects unseen during training (e.g., humans, cars, animals). The last row shows interesting examples where the input image only contains edges.

## Training

### Step 1. Download dataset

* **NYUv2 (official)**: The official train/test split contains 795/654 images. The dataset can be downloaded from [this link](https://drive.google.com/drive/folders/1Ku25Am69h_HrbtcCptXn4aetjo7sB33F?usp=sharing). Unzip the file `nyu_dataset.zip` under `./datasets`, so that `./datasets/nyu/train` and `./datasets/nyu/test/` exist.

* **NYUv2 (big)**: Please visit [GeoNet](https://github.com/xjqi/GeoNet) to download a larger training set consisting of 30907 images. This is the training set used to train our model.

* **ScanNet:** Please visit [FrameNet](https://github.com/hjwdzh/framenet/tree/master/src) to download ScanNet with ground truth surface normals.

### Step 2. Train

* If you wish to train on NYUv2 official split, simply run 
```python
python train.py
```
* If you wish to train your own model, modify the file `./models/baseline.py` and add `--use_baseline` flag. The default loss function `UG_NLL_ours` assumes uncertainty-guided sampling, so this should be changed to something else (e.g. try `--loss_fn NLL_ours`).


## Citation

If you find our work useful in your research please consider citing our paper:

```
@InProceedings{Bae2021,
    title   = {Estimating and Exploiting the Aleatoric Uncertainty in Surface Normal Estimation}
    author  = {Gwangbin Bae and Ignas Budvytis and Roberto Cipolla},
    booktitle = {International Conference on Computer Vision (ICCV)},
    year = {2021}                         
}
```


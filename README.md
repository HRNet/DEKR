# Bottom-Up Human Pose Estimation Via Disentangled Keypoint Regression

## Introduction
In this paper, we are interested in the bottom-up paradigm of estimating human poses from an image. We study the **dense keypoint regression framework** that is previously inferior to the keypoint detection and grouping framework. Our motivation is that regressing keypoint positions
accurately needs to learn representations that focus on the
keypoint regions.

We present a simple yet effective approach, named disentangled keypoint regression (DEKR). We adopt **adaptive convolutions** through pixel-wise spatial transformer to activate the pixels in the keypoint regions and accordingly learn representations from them. We use a multi-branch structure for **separate regression**: each branch learns a representation with dedicated adaptive convolutions and regresses one keypoint. The resulting disentangled representations are able to attend to the keypoint regions, respectively, and thus the keypoint regression is spatially more accurate. We empirically show that the proposed direct regression method outperforms keypoint detection and grouping methods and achieves superior bottom-up pose estimation results on two benchmark datasets, COCO and CrowdPose.
		
## Main Results
### Results on COCO val2017 without multi-scale test
| Backbone | Input size | #Params | GFLOPs | AP | AP .5 | AP .75 | AP (M) | AP (L) | AR | AR .5 | AR .75 | AR (M) | AR (L) |
|--------------------|------------|---------|--------|-------|-------|--------|--------|--------|-------|-------|--------|--------|--------|
| **pose_hrnet_w32** |  512x512 | 29.6M   | 45.4 | 0.680 | 0.867 | 0.745 | 0.621 | 0.777 | 0.730 | 0.898 | 0.784 | 0.662 | 0.827 |
| **pose_hrnet_w48** |  640x640 | 65.7M   | 141.5 | 0.710 | 0.883 | 0.774 | 0.667 | 0.785 | 0.760 | 0.914 | 0.815 | 0.706 | 0.840 |

### Results on COCO val2017 with multi-scale test
| Backbone | Input size | #Params | GFLOPs | AP | AP .5 | AP .75 | AP (M) | AP (L) | AR | AR .5 | AR .75 | AR (M) | AR (L) |
|--------------------|------------|---------|--------|-------|-------|--------|--------|--------|-------|-------|--------|--------|--------|
| **pose_hrnet_w32** |  512x512 | 29.6M   | 45.4 | 0.707 | 0.877 | 0.771 | 0.662 | 0.778 | 0.759 | 0.913 | 0.813 | 0.705 | 0.836 |
| **pose_hrnet_w48** |  640x640 | 65.7M   | 141.5 | 0.723 | 0.883 | 0.786 | 0.686 | 0.786 | 0.777 | 0.924 | 0.832 | 0.728 | 0.849 |

### Results on COCO test-dev2017 without multi-scale test
| Backbone | Input size | #Params | GFLOPs | AP | AP .5 | AP .75 | AP (M) | AP (L) | AR | AR .5 | AR .75 | AR (M) | AR (L) |
|--------------------|------------|---------|--------|-------|-------|--------|--------|--------|-------|-------|--------|--------|--------|
| **pose_hrnet_w32** |  512x512 | 29.6M   | 45.4 | 0.673 | 0.879 | 0.741 | 0.615 | 0.761 | 0.724 | 0.908 | 0.782 | 0.654 | 0.819 |
| **pose_hrnet_w48** |  640x640 | 65.7M   | 141.5 | 0.700 | 0.894 | 0.773 | 0.657 | 0.769 | 0.754 | 0.927 | 0.816 | 0.697 | 0.832 |

### Results on COCO test-dev2017 with multi-scale test
| Backbone | Input size | #Params | GFLOPs | AP | AP .5 | AP .75 | AP (M) | AP (L) | AR | AR .5 | AR .75 | AR (M) | AR (L) |
|--------------------|------------|---------|--------|-------|-------|--------|--------|--------|-------|-------|--------|--------|--------|
| **pose_hrnet_w32** |  512x512 | 29.6M   | 45.4 | 0.698 | 0.890 | 0.766 | 0.652 | 0.765 | 0.751 | 0.924 | 0.811 | 0.695 | 0.828 |
| **pose_hrnet_w48** |  640x640 | 65.7M   | 141.5 | 0.710 | 0.892 | 0.780 | 0.671 | 0.769 | 0.767 | 0.932 | 0.830 | 0.715 | 0.839 |

### Results on CrowdPose test without multi-scale test
| Method             |    AP | AP .5 | AP .75 | AP (E) | AP (M) | AP (H) |
|--------------------|-------|-------|--------|--------|--------|--------|
| **pose_hrnet_w32** | 0.657 | 0.857 | 0.704 | 0.730 | 0.664 | 0.575 |
| **pose_hrnet_w48** | 0.673 | 0.864 | 0.722 | 0.746 | 0.681 | 0.587 |

### Results on CrowdPose test with multi-scale test
| Method             |    AP | AP .5 | AP .75 | AP (E) | AP (M) | AP (H) |
|--------------------|-------|-------|--------|--------|--------|--------|
| **pose_hrnet_w32** | 0.670 | 0.854 | 0.724 | 0.755 | 0.680 | 0.569 |
| **pose_hrnet_w48** | 0.680 | 0.855 | 0.734 | 0.766 | 0.688 | 0.584 |

### Results with matching regression results to the closest keypoints detected from the keypoint heatmaps

|   | DEKR-w32-SS | DEKR-w32-MS | DEKR-w48-SS | DEKR-w48-MS |
|--------------------|-------|-------|--------|--------|
| **coco_val2017** | 0.680 | 0.710 | 0.710 | 0.728 |
| **coco_test-dev2017** | 0.673 | 0.702 | 0.701 | 0.714 |
| **crowdpose_test** | 0.655 | 0.675 | 0.670 | 0.683 |

### Note:
- Flip test is used.
- GFLOPs is for convolution and linear layers only.


## Environment
The code is developed using python 3.6 on Ubuntu 16.04. NVIDIA GPUs are needed. The code is developed and tested using 4 NVIDIA V100 GPU cards for HRNet-w32 and 8 NVIDIA V100 GPU cards for HRNet-w48. Other platforms are not fully tested.

## Quick start
### Installation
1. Clone this repo, and we'll call the directory that you cloned as ${POSE_ROOT}.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Install [COCOAPI](https://github.com/cocodataset/cocoapi):
   ```
   # COCOAPI=/path/to/clone/cocoapi
   git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
   cd $COCOAPI/PythonAPI
   # Install into global site-packages
   make install
   # Alternatively, if you do not have permissions or prefer
   # not to install the COCO API into global site-packages
   python3 setup.py install --user
   ```
   Note that instructions like # COCOAPI=/path/to/install/cocoapi indicate that you should pick a path where you'd like to have the software cloned and then set an environment variable (COCOAPI in this case) accordingly.
4. Install [CrowdPoseAPI](https://github.com/Jeff-sjtu/CrowdPose) exactly the same as COCOAPI.  
5. Init output(training model output directory) and log(tensorboard log directory) directory:

   ```
   mkdir output 
   mkdir log
   ```

   Your directory tree should look like this:

   ```
   ${POSE_ROOT}
   ├── data
   ├── model
   ├── experiments
   ├── lib
   ├── tools 
   ├── log
   ├── output
   ├── README.md
   ├── requirements.txt
   └── setup.py
   ```

6. Download pretrained models and our well-trained models from zoo([OneDrive](https://mailustceducn-my.sharepoint.com/:f:/g/personal/aa397601_mail_ustc_edu_cn/EmoNwNpq4L1FgUsC9KbWezABSotd3BGOlcWCdkBi91l50g?e=HWuluh)) and make models directory look like this:
    ```
    ${POSE_ROOT}
    |-- model
    `-- |-- imagenet
        |   |-- hrnet_w32-36af842e.pth
        |   `-- hrnetv2_w48_imagenet_pretrained.pth
        |-- pose_coco
        |   |-- pose_dekr_hrnetw32_coco.pth
        |   `-- pose_dekr_hrnetw48_coco.pth
        |-- pose_crowdpose
        |   |-- pose_dekr_hrnetw32_crowdpose.pth
        |   `-- pose_dekr_hrnetw48_crowdpose.pth
        `-- rescore
            |-- final_rescore_coco_kpt.pth
            `-- final_rescore_crowd_pose_kpt.pth
    ```
   
### Data preparation

**For COCO data**, please download from [COCO download](http://cocodataset.org/#download), 2017 Train/Val is needed for COCO keypoints training and validation. 
Download and extract them under {POSE_ROOT}/data, and make them look like this:

    ${POSE_ROOT}
    |-- data
    `-- |-- coco
        `-- |-- annotations
            |   |-- person_keypoints_train2017.json
            |   `-- person_keypoints_val2017.json
            `-- images
                |-- train2017.zip
                `-- val2017.zip

**For CrowdPose data**, please download from [CrowdPose download](https://github.com/Jeff-sjtu/CrowdPose#dataset), Train/Val is needed for CrowdPose keypoints training.
Download and extract them under {POSE_ROOT}/data, and make them look like this:

    ${POSE_ROOT}
    |-- data
    `-- |-- crowdpose
        `-- |-- json
            |   |-- crowdpose_train.json
            |   |-- crowdpose_val.json
            |   |-- crowdpose_trainval.json (generated by tools/crowdpose_concat_train_val.py)
            |   `-- crowdpose_test.json
            `-- images.zip

After downloading data, run `python tools/crowdpose_concat_train_val.py` under `${POSE_ROOT}` to create trainval set.


### Training and Testing

#### Testing on COCO val2017 dataset without multi-scale test using well-trained pose model

```
python tools/valid.py \
    --cfg experiments/coco/w32/w32_4x_reg03_bs10_512_adam_lr1e-3_coco_x140.yaml \
    TEST.MODEL_FILE model/pose_coco/pose_dekr_hrnetw32_coco.pth
```

#### Testing on COCO test-dev2017 dataset without multi-scale test using well-trained pose model

```
python tools/valid.py \
    --cfg experiments/coco/w32/w32_4x_reg03_bs10_512_adam_lr1e-3_coco_x140.yaml \
    TEST.MODEL_FILE model/pose_coco/pose_dekr_hrnetw32_coco.pth \ 
    DATASET.TEST test-dev2017
```

#### Testing on COCO val2017 dataset with multi-scale test using well-trained pose model
 
```
python tools/valid.py \
    --cfg experiments/coco/w32/w32_4x_reg03_bs10_512_adam_lr1e-3_coco_x140.yaml \
    TEST.MODEL_FILE model/pose_coco/pose_dekr_hrnetw32_coco.pth \ 
    TEST.NMS_THRE 0.15 \
    TEST.SCALE_FACTOR 0.5,1,2
```

#### Testing on COCO val2017 dataset with matching regression results to the closest keypoints detected from the keypoint heatmaps

```
python tools/valid.py \
    --cfg experiments/coco/w32/w32_4x_reg03_bs10_512_adam_lr1e-3_coco_x140.yaml \
    TEST.MODEL_FILE model/pose_coco/pose_dekr_hrnetw32_coco.pth \ 
    TEST.MATCH_HMP True
```

#### Testing on crowdpose test dataset without multi-scale test using well-trained pose model
 
```
python tools/valid.py \
    --cfg experiments/crowdpose/w32/w32_4x_reg03_bs10_512_adam_lr1e-3_crowdpose_x300.yaml \
    TEST.MODEL_FILE model/pose_crowdpose/pose_dekr_hrnetw32_crowdpose.pth
```

#### Testing on crowdpose test dataset with multi-scale test using well-trained pose model
 
```
python tools/valid.py \
    --cfg experiments/crowdpose/w32/w32_4x_reg03_bs10_512_adam_lr1e-3_crowdpose_x300.yaml \
    TEST.MODEL_FILE model/pose_crowdpose/pose_dekr_hrnetw32_crowdpose.pth \ 
    TEST.NMS_THRE 0.15 \
    TEST.SCALE_FACTOR 0.5,1,2
```

#### Testing on crowdpose test dataset with matching regression results to the closest keypoints detected from the keypoint heatmaps
 
```
python tools/valid.py \
    --cfg experiments/crowdpose/w32/w32_4x_reg03_bs10_512_adam_lr1e-3_crowdpose_x300.yaml \
    TEST.MODEL_FILE model/pose_crowdpose/pose_dekr_hrnetw32_crowdpose.pth \ 
    TEST.MATCH_HMP True
```

#### Training on COCO train2017 dataset

```
python tools/train.py \
    --cfg experiments/coco/w32/w32_4x_reg03_bs10_512_adam_lr1e-3_coco_x140.yaml \
```

#### Training on Crowdpose trainval dataset

```
python tools/train.py \
    --cfg experiments/crowdpose/w32/w32_4x_reg03_bs10_512_adam_lr1e-3_crowdpose_x300.yaml \
```

#### Using inference demo
```
python tools/inference_demo.py --cfg experiments/coco/inference_demo_coco.yaml \
    --videoFile ../multi_people.mp4 \
    --outputDir output \
    --visthre 0.3 \
    TEST.MODEL_FILE model/pose_coco/pose_dekr_hrnetw32.pth
python tools/inference_demo.py --cfg experiments/crowdpose/inference_demo_crowdpose.yaml \
    --videoFile ../multi_people.mp4 \
    --outputDir output \
    --visthre 0.3 \
    TEST.MODEL_FILE model/pose_crowdpose/pose_dekr_hrnetw32.pth \
```

The above command will create a video under *output* directory and a lot of pose image under *output/pose* directory. 

#### Scoring net
We use a scoring net, consisting of two fully-connected layers (each followed by a ReLU layer), and a linear prediction layer which aims to learn the
OKS score for the corresponding predicted pose. For this scoring net, you can directly use our well-trained model in the model/rescore folder. You can also train your scoring net using your pose estimation model by the following steps:

1. Generate scoring dataset on train dataset:
```
python tools/valid.py \
    --cfg experiments/coco/rescore_coco.yaml \
    TEST.MODEL_FILE model/pose_coco/pose_dekr_hrnetw32.pth
python tools/valid.py \
    --cfg experiments/crowdpose/rescore_crowdpose.yaml \
    TEST.MODEL_FILE model/pose_crowdpose/pose_dekr_hrnetw32.pth \
```

2. Train the scoring net using the scoring dataset:
```
python tools/train_scorenet.py \
    --cfg experiment/coco/rescore_coco.yaml
python tools/train_scorenet.py \
    --cfg experiments/crowdpose/rescore_crowdpose.yaml \
```

3. Using the well-trained scoring net to improve the performance of your pose estimation model (above 0.6AP).
```
python tools/valid.py \
    --cfg experiments/coco/w32/w32_4x_reg03_bs10_512_adam_lr1e-3_coco_x140.yaml \
    TEST.MODEL_FILE model/pose_coco/pose_dekr_hrnetw32_coco.pth
python tools/valid.py \
    --cfg experiments/crowdpose/w32/w32_4x_reg03_bs10_512_adam_lr1e-3_crowdpose_x300.yaml \
    TEST.MODEL_FILE model/pose_crowdpose/pose_dekr_hrnetw32_crowdpose.pth \
```

### Acknowledge
Our code is mainly based on [HigherHRNet](https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation). 

### Citation

```
@inproceedings{GengSXZW21,
  title={Bottom-Up Human Pose Estimation Via Disentangled Keypoint Regression},
  author={Zigang Geng, Ke Sun, Bin Xiao, Zhaoxiang Zhang, Jingdong Wang},
  booktitle={CVPR},
  year={2021}
}

@inproceedings{SunXLW19,
  title={Deep High-Resolution Representation Learning for Human Pose Estimation},
  author={Ke Sun and Bin Xiao and Dong Liu and Jingdong Wang},
  booktitle={CVPR},
  year={2019}
}

@article{WangSCJDZLMTWLX19,
  title={Deep High-Resolution Representation Learning for Visual Recognition},
  author={Jingdong Wang and Ke Sun and Tianheng Cheng and 
          Borui Jiang and Chaorui Deng and Yang Zhao and Dong Liu and Yadong Mu and 
          Mingkui Tan and Xinggang Wang and Wenyu Liu and Bin Xiao},
  journal={TPAMI}
  year={2019}
}
```



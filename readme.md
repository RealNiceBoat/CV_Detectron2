## Installation
```bash
pip install torch torchvision
pip install git+https://github.com/facebookresearch/detectron2.git
pip install git+https://github.com/jinmingteo/cocoapi.git#subdirectory=PythonAPI
pip install opencv-python
```

## Description
Hey, I put everything you need to know in [model.ipynb](notebooks/model_til_pure.ipynb). But in short:
1. Put the train & val data folders back into [til2020](data/til2020). (I already fixed the annotation json)
2. Download the model from [Google Drive](https://drive.google.com/file/d/1NAqYvcLSyLfB8IV8DuXoiDmZrW857byV/view?usp=sharing) and place it into [final_ckpts](final_ckpts).
3. To add more augmentation components, see [pipeline.py](notebooks/scripts/pipeline.py).
4. Run the [notebook](notebooks/model.ipynb).

I will update the readme later with regards to the Modanet and DeepFashion trained model as well as mixed datasets.

The library used is Facebook's [Detectron2](https://github.com/facebookresearch/detectron2). The model is [R101-FPN](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md#faster-r-cnn) aka ResNet-101 with Feature Pyramid Network. As such, we need to cite:

```BibTeX
@misc{wu2019detectron2,
  author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                  Wan-Yen Lo and Ross Girshick},
  title =        {Detectron2},
  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
  year =         {2019}
}
```

## R101-FPN COCO keypoints finetuned purely on ModaNet
TIL pycoco evaluation results:
| IoU=0.20:0.50 | IoU=0.20 | IoU=0.30 | IoU=0.40 | IoU=0.50 |
|:-------------:|:--------:|:--------:|:--------:|:--------:|
|     0.761     |  0.770   |   0.764  |   0.758  |   0.751  |

[06/18 19:20:02 d2.evaluation.coco_evaluation]: Evaluation results for bbox: 
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl  |
|:------:|:------:|:------:|:------:|:------:|:-----:|
| 76.096 | 76.954 | 76.402 | 75.818 | 75.102 | 0.000 |

[06/18 19:20:02 d2.evaluation.coco_evaluation]: Per-category bbox AP: 
| category   | AP     | category   | AP     | category   | AP     |
|:-----------|:-------|:-----------|:-------|:-----------|:-------|
| tops       | 65.887 | trousers   | 70.078 | outerwear  | 84.022 |
| dresses    | 96.491 | skirts     | 64.001 |            |        |

## R101-FPN COCO obj det finetuned purely on TIL dataset
TIL pycoco evaluation results:
| IoU=0.20:0.50 | IoU=0.20 | IoU=0.30 | IoU=0.40 | IoU=0.50 |
|:-------------:|:--------:|:--------:|:--------:|:--------:|
|     0.687     |  0.701   |   0.694  |   0.686  |   0.665  |

[06/13 20:35:19 d2.evaluation.coco_evaluation]: Evaluation results for bbox: 
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl  |
|:------:|:------:|:------:|:------:|:------:|:-----:|
| 68.744 | 70.114 | 69.380 | 68.594 | 66.490 | 0.000 |

[06/15 15:51:35 d2.evaluation.coco_evaluation]: Per-category bbox AP: 
| category   | AP     | category   | AP     | category   | AP     |
|:-----------|:-------|:-----------|:-------|:-----------|:-------|
| tops       | 58.582 | trousers   | 49.809 | outerwear  | 78.678 |
| dresses    | 96.898 | skirts     | 59.754 |            |        |

## Parameters used
Trained for ~4.5 epochs
- LR of 0.00025
- Cropping enabled (0.7,0.9)
- Contrast, Brightness & Saturation enabled (all 0.8 to 1.2).
- Repeat threshold set to 0.5

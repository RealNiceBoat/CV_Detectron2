## Installation
```bash
pip install torch torchvision
pip install git+https://github.com/facebookresearch/detectron2.git
pip install git+https://github.com/jinmingteo/cocoapi.git#subdirectory=PythonAPI
pip install opencv-python
```

## Description
See [this notebook](notebooks/model_til_pure.ipynb) for a overview of our training process. For the pure TIL model, here is quick setup guide:
1. Put the train & val data folders back into [til2020](data/til2020). (I already fixed the annotation json)
2. Download the model from [Google Drive](https://drive.google.com/file/d/1NAqYvcLSyLfB8IV8DuXoiDmZrW857byV/view?usp=sharing) and place it into [final_ckpts](final_ckpts).
3. To add more augmentation components, see [pipeline.py](notebooks/scripts/pipeline.py).

Most of the important logic has been extracted into their own python files which can be found in the [scripts folder](notebooks/scripts), which also contains some utility scripts. There may also be other utility scripts scattered elsewhere for the purpose of wrangling datasets.

## Citations
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

Ultimately, we found the TIL dataset insufficient and after researching and experimenting with multiple datasets, we stumbled across Modanet, annotations based on the Paper Doll dataset that were of extremely high quality, allowing our model to reach much greater AP scores.

```BibTeX
@inproceedings{zheng/2018acmmm,
  author       = {Shuai Zheng and Fan Yang and M. Hadi Kiapour and Robinson Piramuthu},
  title        = {ModaNet: A Large-Scale Street Fashion Dataset with Polygon Annotations},
  booktitle    = {ACM Multimedia},
  year         = {2018},
}
```

```BibTeX
@inproceedings{yamaguchi/iccv2013,
  author =       {Kota Yamaguchi and M. Hadi Kiapour and Tamara L. Berg},
  title =        {Paper Doll Parsing: Retrieving Similar Styles to Parse Clothing Items},
  booktitle    = {ICCV 2013},
  year =         {2013}
}
```

One last dataset we used was the DeepFashion2 Dataset. It is a very large dataset, and unfortunately too good to be true, being of low quality as evidenced by our tried and tested models being unable to train well on it.

```BibTeX
@article{DeepFashion2,
  author = {Yuying Ge and Ruimao Zhang and Lingyun Wu and Xiaogang Wang and Xiaoou Tang and Ping Luo},
  title={A Versatile Benchmark for Detection, Pose Estimation, Segmentation and Re-Identification of Clothing Images},
  journal={CVPR},
  year={2019}
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
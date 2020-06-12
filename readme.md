## Installation
```bash
pip install torch torchvision
pip install git+https://github.com/facebookresearch/detectron2.git
pip install git+https://github.com/jinmingteo/cocoapi.git#subdirectory=PythonAPI
pip install opencv-python
```

## Description
Hey, I put everything you need to know in [model.ipynb](model.ipynb). But in short:
1. Put the train & val data folders back into [til2020](til2020). (I already fixed the annotation json)
2. Download the model from [Google Drive](https://drive.google.com/file/d/1nEyJVzAy3yB7v8aEUthVR1kMpuZK4_Lo/view?usp=sharing) and place it into [ckpts](ckpts).
3. To add more augmentation components, see [pipeline.py](pipeline.py).
4. Run the [notebook](model.ipynb).
btw, I am still training the model, its currently at roughly only 5 epochs out of a planned 20.

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

## Results
TIL pycoco evaluation results:
| IoU=0.20:0.50 | IoU=0.20 | IoU=0.30 | IoU=0.40 | IoU=0.50 |
|:-------------:|:--------:|:--------:|:--------:|:--------:|
|     0.621     |  0.643   |   0.632  |   0.617  |   0.584  |

[06/13 00:38:15 d2.evaluation.coco_evaluation]: Evaluation results for bbox: 
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl  |
|:------:|:------:|:------:|:------:|:------:|:-----:|
| 62.100 | 64.326 | 63.176 | 61.692 | 58.396 | 0.000 |

[06/13 00:38:15 d2.evaluation.coco_evaluation]: Per-category bbox AP: 
| category   | AP     | category   | AP     | category   | AP     |
|:-----------|:-------|:-----------|:-------|:-----------|:-------|
| tops       | 47.403 | trousers   | 48.898 | outerwear  | 72.674 |
| dresses    | 95.458 | skirts     | 46.069 |            |        |
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
2. Download the model from [Google Drive](https://drive.google.com/file/d/1xPaXsHhVQ-aW2t5Jr8tLXI-Z_opwajVQ/view?usp=sharing) and place it into [ckpts](ckpts).
3. To add more augmentation components, see [pipeline.py](pipeline.py).
4. Run the [notebook](model.ipynb).

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
|     0.654     |  0.667   |   0.662  |   0.653  |   0.630  |

[06/13 20:35:19 d2.evaluation.coco_evaluation]: Evaluation results for bbox: 
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl  |
|:------:|:------:|:------:|:------:|:------:|:-----:|
| 65.410 | 66.672 | 66.171 | 65.322 | 62.996 | 0.000 |

[06/13 20:35:19 d2.evaluation.coco_evaluation]: Per-category bbox AP: 
| category   | AP     | category   | AP     | category   | AP     |
|:-----------|:-------|:-----------|:-------|:-----------|:-------|
| tops       | 53.308 | trousers   | 46.216 | outerwear  | 77.182 |
| dresses    | 95.433 | skirts     | 54.913 |            |        |

## Model History
1. Trained for 17 epochs with full augmentation pipeline (currently commented out in pipeline.py). Batch-size of 1.
2. Trained for 0.25 epoch without augmentation, batch-size of 3 (loss decreases faster).
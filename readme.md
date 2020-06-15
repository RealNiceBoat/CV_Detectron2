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
2. Download the model from [Google Drive](https://drive.google.com/file/d/1NAqYvcLSyLfB8IV8DuXoiDmZrW857byV/view?usp=sharing) and place it into [ckpts](ckpts).
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

# Possible Prediction Bugs
1. cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST should be 0.0 (fixed)
2. Are the output boxes supposed to be rescaled rather than original img dimensions? (unconfirmed)
3. Image color channels loaded in reverse. Use: cv2.imread(path) (fixed)
4. Image ids not loaded from metafile but interpreted from filename (unconfirmed)
5. class ids not remapped to category ids. Use: MetadataCatalog(name).thing_dataset_id_to_contiguous_id (not fixed)

## Results
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

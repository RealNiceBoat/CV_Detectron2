import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/jinmingteo/cocoapi.git#subdirectory=PythonAPI"])

#Paths
from pathlib import Path
base_folder = Path('.')
data_folder = base_folder/'data'/'til2020'
train_imgs_folder = data_folder/'train'
train_annotations = data_folder/'train.json'
val_imgs_folder = data_folder/'val'
val_annotations = data_folder/'val.json'
test_imgs_folder = data_folder/'CV_interim_images'
test_annotations = data_folder/'CV_interim_evaluation.json'

save_model_folder = base_folder/'ckpts'
load_model_folder = base_folder/'final_ckpts'

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
coco_gt = COCO(val_annotations)
coco_dt = coco_gt.loadRes(str(save_model_folder/"coco_instances_results.json"))
cocoEval = COCOeval(cocoGt=coco_gt, cocoDt=coco_dt, iouType='bbox')
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
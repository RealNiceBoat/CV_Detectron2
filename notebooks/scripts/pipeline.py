import copy
import logging
import numpy as np
import torch
from fvcore.common.file_io import PathManager
from PIL import Image

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

#TIL: class-hinting existed since 3.5, why didnt I start using them earlier
class NoOpTransform(T.Transform):
    def __init__(self): super().__init__()
    def apply_image(self, img: np.ndarray) -> np.ndarray: return img
    def apply_coords(self, coords: np.ndarray) -> np.ndarray: return coords #if the image isnt translated/rotated/crop, just return
    def inverse(self) -> T.Transform: return self #technically can use a no-op here
#Use above to create pipeline components
#wrap them using detectron2.data.transforms.RandomApply(transform,prob=0.5)

#example: (didnt use cause my implementation is crappy)
import cv2
class SharpenTransform(T.Transform):
    def __init__(self,level=9): 
        super().__init__()
        self.kernel = np.array([[-1,-1,-1], [-1,level,-1], [-1,-1,-1]])
    def apply_image(self, img: np.ndarray) -> np.ndarray:  return np.array(cv2.filter2D(img, -1, self.kernel))
    def apply_coords(self, coords: np.ndarray) -> np.ndarray: return coords #if the image isnt translated/rotated/crop, just return
    def inverse(self) -> T.Transform: return self #technically can use a no-op here

class DatasetPipeline:
    def __init__(self, cfg, is_train=True):
        self.img_format = cfg.INPUT.FORMAT
        self.is_train = is_train
        self.logger = logging.getLogger('detectron2')

        self.pipeline = []
        self.cropper = None #has to be handled different from other transforms

        if is_train:
            self.pipeline.append(T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TRAIN,cfg.INPUT.MAX_SIZE_TRAIN,cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING))
            
            self.pipeline.append(T.RandomFlip(vertical=False,horizontal=True))

            if cfg.INPUT.CROP.ENABLED: self.cropper = T.RandomCrop("relative_range",cfg.INPUT.CROP.SIZE)

            if cfg.INPUT.ENABLE_MINOR_AUGMENTS:
                #self.pipeline.append(T.RandomRotation(cfg.INPUT.RAND_ROTATION))

                self.pipeline.append(T.RandomContrast(*cfg.INPUT.RAND_CONTRAST))

                self.pipeline.append(T.RandomBrightness(*cfg.INPUT.RAND_BRIGHTNESS))

                self.pipeline.append(T.RandomSaturation(*cfg.INPUT.RAND_SATURATION))

                #self.pipeline.append(T.RandomApply(SharpenTransform(level=9),prob=0.05))
        else:
            self.pipeline.append(T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TEST,cfg.INPUT.MAX_SIZE_TEST,"choice"))
        self.logger.info(f"Pipeline: {self.pipeline}")
        

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)

        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)
        img_size = image.shape[:2]

        if self.cropper:
            crop_tfm = utils.gen_crop_transform_with_instance(
                self.cropper.get_crop_size(img_size),
                img_size,
                np.random.choice(dataset_dict["annotations"]),
            )
            image = crop_tfm.apply_image(image)
        
        image,transforms = T.apply_transform_gens(self.pipeline,image)
        
        if self.cropper: transforms = crop_tfm + transforms

        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if "annotations" in dataset_dict:
            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, img_size)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd",0) == 0
            ]
            instances = utils.annotations_to_instances(annos,img_size)
            if instances.has("gt_masks"): instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict
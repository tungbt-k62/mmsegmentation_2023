from mmseg.datasets import CustomDataset
from mmseg.registry import DATASETS
import os.path as osp
import cv2
import numpy as np

@DATASETS.register_module()
class BKPolypDataset(CustomDataset):
    CLASSES = ('background', 'green', 'red')
    PALETTE = [[0, 0, 0], [0, 255, 0], [255, 0, 0]]

    def __init__(self, img_suffix='.jpg', seg_map_suffix='.png', **kwargs):
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=False,
            **kwargs)

    def read_mask(self, mask_path):
        image = cv2.imread(mask_path)
        image = cv2.resize(image, (self.crop_size, self.crop_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # lower boundary RED color range values; Hue (0 - 10)
        lower1 = np.array([0, 100, 20])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([160, 100, 20])
        upper2 = np.array([179, 255, 255])
        lower_mask = cv2.inRange(image, lower1, upper1)

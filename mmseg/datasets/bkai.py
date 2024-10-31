import cv2
import numpy as np
import os
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

@DATASETS.register_module()
class BKPolypDataset(CustomDataset):
    CLASSES = ('background', 'green', 'red')
    PALETTE = [[0, 0, 0], [0, 255, 0], [255, 0, 0]]

    def __init__(self, size=(384, 384), **kwargs):
        super(BKPolypDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.jpg',  # Change suffix if needed
            reduce_zero_label=False,
            **kwargs)
        assert os.path.exists(self.img_dir)
        self.size = size  # Store the size parameter

    def read_mask(self, mask_path):
        # Load image and resize
        image = cv2.imread(mask_path)
        image = cv2.resize(image, self.size)  # Use the dynamic size parameter
        
        # Convert to HSV for color range masking
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define RED color range
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 70, 50])
        upper_red2 = np.array([179, 255, 255])
        red_mask1 = cv2.inRange(image_hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(image_hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # Set class ID for RED (class 2)
        red_mask[red_mask != 0] = 2

        # Define GREEN color range
        lower_green = np.array([36, 50, 50])
        upper_green = np.array([89, 255, 255])
        green_mask = cv2.inRange(image_hsv, lower_green, upper_green)
        
        # Set class ID for GREEN (class 1)
        green_mask[green_mask != 0] = 1

        # Combine masks
        full_mask = cv2.bitwise_or(red_mask, green_mask)
        
        # Ensure background is set to 0 (already 0 by default)
        full_mask = full_mask.astype(np.uint8)
        return full_mask
    
    def get_gt_seg_map_by_idx(self, idx):
        seg_map_path = self.seg_map_infos[idx]['filename']
        return self.read_mask(seg_map_path)
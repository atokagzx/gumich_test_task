import logging
import numpy as np
import cv2
import torch
from segment_anything import build_sam, SamPredictor 
from typing import List
import os
import subprocess


class Singleton(type):
    '''
    Implement the Singleton pattern by overriding the __call__ method of the metaclass
    '''
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class SAMAdapter:
    def __init__(self):
        self._logger = logging.getLogger("sam_adapter")
        self._model, self._device = self._load_model("sam_vit_h_4b8939.pth")
        self._predictor = SamPredictor(self._model)


    def _download_model(self, sam_checkpoint_filename):
        user_cache_dir = os.path.expanduser("~/.cache")
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        self._logger.info(f'downloading from {url}, it may take a while')
        result = subprocess.run(['wget', url, '-P', user_cache_dir], capture_output=True, text=True, check=True)
        if result.returncode != 0:
            raise RuntimeError("failed to download sam model")
        self._logger.info("model downloaded successfully")


    def _load_model(self, sam_checkpoint_filename):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        user_cache_dir = os.path.expanduser("~/.cache")
        if os.path.exists(os.path.join(user_cache_dir, sam_checkpoint_filename)):
            self._logger.info(f'trying to load model from {os.path.join(user_cache_dir, sam_checkpoint_filename)}')
            try:
                model = build_sam(os.path.join(user_cache_dir, sam_checkpoint_filename)).to(device)
            except RuntimeError as e:
                self._logger.exception(f'failed to load model from {os.path.join(user_cache_dir, sam_checkpoint_filename)}, removing the corrupted file and retrying')
                os.remove(os.path.join(user_cache_dir, sam_checkpoint_filename))
                self._download_model(sam_checkpoint_filename)
                model = build_sam(os.path.join(user_cache_dir, sam_checkpoint_filename)).to(device)
        else:
            self._download_model(sam_checkpoint_filename)
            model = build_sam(os.path.join(user_cache_dir, sam_checkpoint_filename)).to(device)
        self._logger.info("model loaded successfully")
        return model, device
    

    def segment(self, image, mask) -> np.ndarray:
        '''
        Segment the image based on the bounding boxes and phrases
        @param image: image to be segmented
        @param bboxes: bounding boxes
        @param phrases: phrases
        @return: segmentation masks: list of list of numpy arrays (each item has a list of masks))
        '''
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
        # erode the mask
        k_size = 7
        kernel = np.ones((k_size, k_size), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=15)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = [cv2.minAreaRect(contour) for contour in contours]
        centers = [tuple(map(int, box[0])) for box in bboxes]
        self._predictor.set_image(image)
        masks, _confidence, _low_res_masks = self._predictor.predict(point_coords = np.asarray(centers), 
                                                                    point_labels = np.ones(len(centers)),
                                                                    return_logits=False)
        mask = np.zeros_like(mask)
        for i, atomic_mask in enumerate(masks):
            atomic_mask = atomic_mask.astype(np.uint8)
            atomic_mask *= 255
            mask += atomic_mask
        
        return mask
        
    # @staticmethod
    # def mask_to_rle(mask):
    #     '''
    #     Apply Run Length Encoding to mask
    #     @param mask: mask
    #     @return: run length encoding of mask
    #     '''
    #     pixels = mask.flatten()
    #     pixels = np.concatenate([[0], pixels, [0]])
    #     runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    #     runs[1::2] -= runs[::2]
    #     return {"rle": runs, "shape": mask.shape}

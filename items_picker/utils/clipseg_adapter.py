import logging
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor


class Singleton(type):
    '''
    Implement the Singleton pattern by overriding the __call__ method of the metaclass
    '''
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class CLIPSegAdapter(metaclass=Singleton):
    '''
    Adapter for the CLIPSeg model  
    Implements the segment method which takes an image and a prompt and returns the segmentation mask and the semantic indices
    '''
    def __init__(self):
        self._logger = logging.getLogger("clipseg_adapter")
        self._device, self._model, self._processor = self._load_model()

    def _load_model(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._logger.info(f'using device: "{device}"')
        self._logger.info("loading processor")
        clipseg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self._logger.info("loading model")
        clipseg_model = CLIPSegForImageSegmentation.from_pretrained(
            "CIDAS/clipseg-rd64-refined"
        )
        self._logger.info("loaded successfully")
        clipseg_model.to(device)
        return device, clipseg_model, clipseg_processor
    

    @staticmethod
    def _preds_to_semantic_inds(preds, threshold):
        flat_preds = preds.reshape((preds.shape[0], -1))
        # Initialize a dummy "unlabeled" mask with the threshold
        flat_preds_with_treshold = torch.full(
            (preds.shape[0] + 1, flat_preds.shape[-1]), threshold
        )
        flat_preds_with_treshold[1 : preds.shape[0] + 1, :] = flat_preds

        # Get the top mask index for each pixel
        semantic_inds = torch.topk(flat_preds_with_treshold, 1, dim=0).indices.reshape(
            (preds.shape[-2], preds.shape[-1])
        )

        return semantic_inds


    def segment(self, image, prompt:str, background_threshold) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Segment the image based on the prompt and return the segmentation mask
        @param image: np.ndarray
        @param prompt: str
        @param background_threshold: float
        @return: Tuple[torch.Tensor, torch.Tensor]
        '''
        prompt = [prompt]
        image = np.array(image)
        image = Image.fromarray(image, mode="RGB")
        encoding = self._processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        )
        pixel_values = encoding["pixel_values"].to(self._device)
        with torch.no_grad():
            outputs = self._model(pixel_values=pixel_values, input_ids=encoding["input_ids"].to(self._device))
        logits = outputs.logits
        if len(logits.shape) == 2:
            logits = logits.unsqueeze(0)
        # resize the outputs
        upscaled_logits = nn.functional.interpolate(
            logits.unsqueeze(1),
            size=(image.size[1], image.size[0]),
            mode="bilinear",
        )
        preds = torch.sigmoid(upscaled_logits.squeeze(dim=1))
        semantic_inds = self._preds_to_semantic_inds(preds, background_threshold)
        preds = preds.cpu().numpy()
        preds *= 255
        preds = preds.astype(np.uint8).squeeze(axis=0)
        # preds = np.expand_dims(preds, axis=-1)
        return preds, semantic_inds

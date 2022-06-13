# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import logging
import os
import time
import warnings
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

from sahi.auto_model import AutoDetectionModel
from .model import DetectionModel
from .combine import (
    GreedyNMMPostprocess,
    LSNMSPostprocess,
    NMMPostprocess,
    NMSPostprocess,
    PostprocessPredictions,
)
from .prediction import ObjectPrediction, PredictionResult
from .slicing import slice_image
from sahi.utils.coco import Coco, CocoImage
from sahi.utils.cv import (
    IMAGE_EXTENSIONS,
    read_image_as_pil,
)
from sahi.utils.file import Path, increment_path, list_files
POSTPROCESS_NAME_TO_CLASS = {
    "GREEDYNMM": GreedyNMMPostprocess,
    "NMM": NMMPostprocess,
    "NMS": NMSPostprocess,
    "LSNMS": LSNMSPostprocess,
}

LOW_MODEL_CONFIDENCE = 0.1

def get_sliced_prediction(
    image,
    detection_model=None,
    image_size: int = None,
    slice_height: int = 512,
    slice_width: int = 512,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
    postprocess_type: str = "GREEDYNMM",
    postprocess_match_metric: str = "IOS",
    postprocess_match_threshold: float = 0.5,
    postprocess_class_agnostic: bool = False,
    merge_buffer_length: int = None,
) -> PredictionResult:

    # currently only 1 batch supported
    num_batch = 1

    # create slices from full image
    slice_image_result = slice_image(
        image=image,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
    )
    
    num_slices = len(slice_image_result)

    postprocess_constructor = POSTPROCESS_NAME_TO_CLASS[postprocess_type]
    postprocess = postprocess_constructor(
        match_threshold=postprocess_match_threshold,
        match_metric=postprocess_match_metric,
        class_agnostic=postprocess_class_agnostic,
    )

    # create prediction input
    num_group = int(num_slices / num_batch)

    object_prediction_list = []
    # perform sliced prediction
    for group_ind in range(num_group):
        
        # prepare batch (currently supports only 1 batch)
        image_list = []
        shift_amount_list = []
        
        for image_ind in range(num_batch):
            image_list.append(slice_image_result.images[group_ind * num_batch + image_ind])
            shift_amount_list.append(slice_image_result.starting_pixels[group_ind * num_batch + image_ind])

        # perform batch prediction
        
        # read image as pil
        image_as_pil = read_image_as_pil(image_list[0])
        
        
        
        
        # get prediction
        detection_model.perform_inference(np.ascontiguousarray(image_as_pil), image_size=image_size)

        # works only with 1 batch
        detection_model.convert_original_predictions(
            shift_amount=shift_amount_list[0],
            full_shape=[
                slice_image_result.original_image_height,
                slice_image_result.original_image_width,
            ],
        )
        object_prediction_list: List[ObjectPrediction] = detection_model.object_prediction_list

        prediction_result = PredictionResult(
            image=image, object_prediction_list=object_prediction_list)

        
        # convert sliced predictions to full predictions
        for object_prediction in prediction_result.object_prediction_list:
            if object_prediction:  # if not empty
                object_prediction_list.append(object_prediction.get_shifted_object_prediction())

        # merge matching predictions during sliced prediction
        if merge_buffer_length is not None and len(object_prediction_list) > merge_buffer_length:
            object_prediction_list = postprocess(object_prediction_list)

    return PredictionResult(
        image=image, object_prediction_list=object_prediction_list)


from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list

class MmdetDetectionModel(DetectionModel):

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
        full_shape_list: Optional[List[List[int]]] = None,
    ):
        """
        self._original_predictions is converted to a list of prediction.ObjectPrediction and set to
        self._object_prediction_list_per_image.
        Args:
            shift_amount_list: list of list
                To shift the box and mask predictions from sliced image to full sized image, should
                be in the form of List[[shift_x, shift_y],[shift_x, shift_y],...]
            full_shape_list: list of list
                Size of the full image after shifting, should be in the form of
                List[[height, width],[height, width],...]
        """
        original_predictions = self._original_predictions

        # compatilibty for sahi v0.8.15
        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)

        # parse boxes and masks from predictions
        num_categories = self.num_categories
        object_prediction_list_per_image = []
        for image_ind, original_prediction in enumerate(original_predictions):
            shift_amount = shift_amount_list[image_ind]
            full_shape = None if full_shape_list is None else full_shape_list[image_ind]

            boxes = original_prediction

            object_prediction_list = []

            # process predictions
            for category_id in range(num_categories):
                category_boxes = boxes[category_id]
                num_category_predictions = len(category_boxes)

                for category_predictions_ind in range(num_category_predictions):
                    bbox = category_boxes[category_predictions_ind][:4]
                    score = category_boxes[category_predictions_ind][4]

                    # ignore low scored predictions
                    if score < self.confidence_threshold:
                        continue

                    # fix negative box coords
                    bbox[0] = max(0, bbox[0])
                    bbox[1] = max(0, bbox[1])
                    bbox[2] = max(0, bbox[2])
                    bbox[3] = max(0, bbox[3])

                    # fix out of image box coords
                    if full_shape is not None:
                        bbox[0] = min(full_shape[1], bbox[0])
                        bbox[1] = min(full_shape[0], bbox[1])
                        bbox[2] = min(full_shape[1], bbox[2])
                        bbox[3] = min(full_shape[0], bbox[3])

                    # ignore invalid predictions
                    if not (bbox[0] < bbox[2]) or not (bbox[1] < bbox[3]):
                        continue

                    object_prediction = ObjectPrediction(
                        bbox=bbox,
                        category_id=category_id,
                        score=score,
                        shift_amount=shift_amount,
                        full_shape=full_shape,
                    )
                    object_prediction_list.append(object_prediction)
            object_prediction_list_per_image.append(object_prediction_list)
        self._object_prediction_list_per_image = object_prediction_list_per_image

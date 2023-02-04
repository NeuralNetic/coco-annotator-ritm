from typing import List, Tuple

import torch
import os
from PIL import Image
import torchvision
from copy import deepcopy
import logging
from timeit import default_timer as time

import sys
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        './third_party/ritm_interactive_segmentation/'
    )
)

from isegm.inference import utils
from isegm.inference.transforms.base import SigmoidForPred
from isegm.inference.transforms.limit_longest_side import LimitLongestSide


class Click:
    def __init__(self, is_positive, coords: Tuple[int, int], indx: int = None):
        """coords is (y, x)"""
        self.is_positive = is_positive
        self.coords = coords
        self.indx = indx

    @property
    def coords_and_indx(self):
        return (*self.coords, self.indx)

    def copy(self, **kwargs):
        self_copy = deepcopy(self)
        for k, v in kwargs.items():
            setattr(self_copy, k, v)
        return self_copy


class RITMInference(object):
    def __init__(self,
                 model_path: str,
                 device: str = 'cpu',
                 prob_thresh: float = 0.5):
        self.to_tensor = torchvision.transforms.ToTensor()
        self.device = device
        self.prob_thresh = prob_thresh

        checkpoint_path = utils.find_checkpoint(
            os.path.dirname(model_path), os.path.basename(model_path)
        )
        self.model = utils.load_is_model(
            checkpoint_path, torch.device(device), cpu_dist_maps=True)

        self.transforms = []
        self.transforms.append(LimitLongestSide(max_size=800))
        self.transforms.append(SigmoidForPred())
        self.net_clicks_limit = 20

    def set_input_image(self, image):
        image_nd = self.to_tensor(image)
        for transform in self.transforms:
            transform.reset()
        original_image = image_nd.to(self.device)
        if len(original_image.shape) == 3:
            original_image = original_image.unsqueeze(0)
        prev_prediction = torch.zeros_like(original_image[:, :1, :, :])
        return original_image, prev_prediction

    def apply_transforms(self, image_nd, clicks_lists):
        is_image_changed = False
        for t in self.transforms:
            image_nd, clicks_lists = t.transform(image_nd, clicks_lists)
            is_image_changed |= t.image_changed

        return image_nd, clicks_lists, is_image_changed

    def get_points_nd(self, clicks_lists):
        total_clicks = []
        num_pos_clicks = [sum(x.is_positive for x in clicks_list) for clicks_list in clicks_lists]
        num_neg_clicks = [len(clicks_list) - num_pos for clicks_list, num_pos in zip(clicks_lists, num_pos_clicks)]
        num_max_points = max(num_pos_clicks + num_neg_clicks)
        if self.net_clicks_limit is not None:
            num_max_points = min(self.net_clicks_limit, num_max_points)
        num_max_points = max(1, num_max_points)

        for clicks_list in clicks_lists:
            clicks_list = clicks_list[:self.net_clicks_limit]
            pos_clicks = [click.coords_and_indx for click in clicks_list if click.is_positive]
            pos_clicks = pos_clicks + (num_max_points - len(pos_clicks)) * [(-1, -1, -1)]

            neg_clicks = [click.coords_and_indx for click in clicks_list if not click.is_positive]
            neg_clicks = neg_clicks + (num_max_points - len(neg_clicks)) * [(-1, -1, -1)]
            total_clicks.append(pos_clicks + neg_clicks)

        return torch.tensor(total_clicks, device=self.device)

    def _get_prediction(self, image_nd, clicks_lists):
        points_nd = self.get_points_nd(clicks_lists)
        return self.model(image_nd, points_nd)['instances']

    def prediction(self,
                   input_image: Image.Image,
                   pos_points: List[Tuple[int, int]],
                   neg_points: List[Tuple[int, int]]):
        """
        Inference method
        Args:
            input_image: PIL image instance
            pos_points: object points in XY format
            neg_points: background points in XY format

        Returns:
            Boolean mask
        """
        clicks_list = []

        for _pidx, pp in enumerate(pos_points):
            clicks_list.append(Click(True, pp[::-1], _pidx))

        for _pidx, pp in enumerate(neg_points):
            clicks_list.append(Click(False, pp[::-1], _pidx + len(pos_points)))

        image_nd, prev_prediction = self.set_input_image(input_image)
        image_nd = torch.cat([image_nd, prev_prediction], dim=1)
        image_nd, clicks_lists, _ = self.apply_transforms(
            image_nd,
            [clicks_list]
        )

        start_time = time()
        pred_logits = self._get_prediction(image_nd, clicks_lists,)
        logging.info('Raw inference time: {:.5f}'.format(time() - start_time))

        prediction = torch.nn.functional.interpolate(
            pred_logits, mode='bilinear',
            align_corners=True,
            size=image_nd.size()[2:]
        )

        for t in reversed(self.transforms):
            prediction = t.inv_transform(prediction)

        return prediction.cpu().numpy()[0, 0] > self.prob_thresh

    def __call__(self,
                 input_image: Image.Image,
                 pos_points: List[Tuple[int, int]],
                 neg_points: List[Tuple[int, int]]):
        """
        Inference method
        Args:
            input_image: PIL image instance
            pos_points: object points in XY format
            neg_points: background points in XY format

        Returns:
            Boolean mask
        """
        return self.prediction(input_image, pos_points, neg_points)

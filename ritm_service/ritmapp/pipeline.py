from typing import Tuple, List, Dict

import cv2
from dataclasses import dataclass
import json
import numpy as np
import onnxruntime
import torch
import torchvision
from PIL import Image

import os
import sys
sys.path.insert(0, '../')
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        '../third_party/ritm_interactive_segmentation/'
    )
)

from isegm.inference.transforms.base import SigmoidForPred
from isegm.inference.transforms.flip import AddHorizontalFlip
from onnx_ritm import Click


class DataLoader:
    def __init__(self):
        self.convert_to_rgb: bool = True

    def process(self, json_bytes: bytes) -> Tuple[
        Image.Image,
        List[Tuple[int, int]],
        List[Tuple[int, int]],
        Tuple[int, int]
    ]:
        json_data = json.loads(json_bytes)
        img = Image.open(json_data['image'])
        if self.convert_to_rgb:
            img = img.convert('RGB')

        fg_points: List[Tuple[int, int]] = json_data['fg_points']
        bg_points: List[Tuple[int, int]] = json_data['bg_points']

        return img, fg_points, bg_points, img.size


class RITMPreProcessor:
    def __init__(self):
        self.max_resolution_size = 1600
        self.to_tensor = torchvision.transforms.ToTensor()
        self.transforms = []
        self.transforms.append(SigmoidForPred())
        self.transforms.append(AddHorizontalFlip())
        self.net_clicks_limit = 20

    def set_input_image(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        image_nd = self.to_tensor(image)
        for transform in self.transforms:
            transform.reset()
        original_image = image_nd
        if len(original_image.shape) == 3:
            original_image = original_image.unsqueeze(0)
        prev_prediction = torch.zeros_like(original_image[:, :1, :, :])
        return original_image, prev_prediction


    def apply_transforms(self, image_nd: torch.Tensor, clicks_lists: List[List[Click]]) -> Tuple[torch.Tensor, List[List[Click]], bool]:
        is_image_changed = False
        for t in self.transforms:
            image_nd, clicks_lists = t.transform(image_nd, clicks_lists)
            is_image_changed |= t.image_changed

        return image_nd, clicks_lists, is_image_changed

    def limit_longest_side(self,
                           pil_image: Image.Image,
                           pos_points: List[Tuple[int, int]],
                           neg_points: List[Tuple[int, int]]) -> Tuple[Image.Image, List[Tuple[int, int]], List[Tuple[int, int]]]:
        max_image_size = max(pil_image.size)
        k = self.max_resolution_size / max_image_size
        if k - 1 < 1E-5:
            new_w, new_h = int(pil_image.size[0] * k), int(pil_image.size[1] * k)
            pos_points = (np.array(pos_points, dtype=np.float32) * k).astype(np.int32).tolist()
            neg_points = (np.array(neg_points, dtype=np.float32) * k).astype(np.int32).tolist()
            pil_image = pil_image.resize((new_w, new_h), Image.BICUBIC)

        return pil_image, pos_points, neg_points

    def get_points_nd(self, clicks_lists: List[List[Click]]) -> torch.Tensor:
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

        return torch.tensor(total_clicks)

    def process(self, img: Image.Image, fg_points: List[Tuple[int, int]], bg_points: List[Tuple[int, int]]) -> Dict[str, np.ndarray]:
        input_image, pos_points, neg_points = self.limit_longest_side(
            img, fg_points, bg_points
        )
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

        points_nd = self.get_points_nd(clicks_lists)

        model_input = {
            'image': image_nd.numpy(),
            'points': points_nd.numpy().astype(np.float32),
            'size': np.array(input_image.size[::-1], dtype=np.int64)
        }

        return model_input


class RITMCall:
    def __init__(self, onnx_model_path: str, device: str = 'cuda'):
        self.model_path = onnx_model_path
        self.device = device
        self.model = self._get_model()

    def _get_model(self):
        provider = "CPUExecutionProvider" if self.device == 'cpu' else "CUDAExecutionProvider"
        _model = onnxruntime.InferenceSession(
            self.model_path,
            providers=[provider]
        )
        return _model

    def process_list(self, data: List[Dict[str, np.ndarray]]) -> List[np.ndarray]:
        return [self.model.run(['output'], d)[0] for d in data]


class RitmPostProcessor:
    def __init__(self):
        self.transforms = []
        self.transforms.append(SigmoidForPred())
        self.transforms.append(AddHorizontalFlip())
        self.prob_thresh = 0.5

    def process(self, pred: np.ndarray, original_size: Tuple[int, int]) -> np.ndarray:
        prediction = torch.from_numpy(pred)

        for t in reversed(self.transforms):
            prediction = t.inv_transform(prediction)

        res_mask = (prediction[0, 0].numpy() > self.prob_thresh).astype(np.uint8) * 255
        res_mask = cv2.morphologyEx(
            res_mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        )
        res_mask = cv2.morphologyEx(
            res_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        )
        _, res_mask = cv2.threshold(res_mask, 127, 255, cv2.THRESH_BINARY)

        if res_mask.shape[0] == original_size[1] and res_mask.shape[1] == original_size[0]:
            return res_mask
        return cv2.resize(res_mask, original_size, cv2.INTER_NEAREST)


class ImageServing:
    def __init__(self):
        pass

    def process(self, res: np.ndarray) -> bytes:
        return cv2.imencode('.png', res)[1].tobytes()


DEVICE = 'cuda'  # can be 'cuda:0'


@dataclass
class ModelInfo:
    name: str
    model_weights_path: str


PipelineModelsInfo = ModelInfo(
    'RITM_ONNX_Model',
    '/models/ritm32.onnx'
)


class ModelsProducer:
    """Returns models for pipeline steps."""
    def __init__(self, model_info: ModelInfo):
        self._model_info = model_info

    def get_ritm_model(self) -> RITMCall:
        return RITMCall(
            onnx_model_path=self._model_info.model_weights_path,
            device=DEVICE
        )

    def get_data_loader(self) -> DataLoader:
        return DataLoader()

    def get_pre_proc(self) -> RITMPreProcessor:
        return RITMPreProcessor()

    def get_post_proc(self) -> RitmPostProcessor:
        return RitmPostProcessor()

    def get_image_encoder(self) -> ImageServing:
        return ImageServing()


default_producer = ModelsProducer(PipelineModelsInfo)

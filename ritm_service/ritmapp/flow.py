import torch
from aqueduct import (
    BaseTask,
    BaseTaskHandler,
    Flow,
    FlowStep,
)

from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image

from .pipeline import (
    RITMCall,
    default_producer,
)


class Task(BaseTask):
    def __init__(
            self,
            json_data: bytes
    ):
        super().__init__()
        self.input: Optional[bytes, Image.Image, Dict[str, np.ndarray]] = json_data
        self.pred: Optional[np.ndarray, bytes] = None
        self.fg_points: Optional[List[Tuple[int, int]]] = None
        self.bg_points: Optional[List[Tuple[int, int]]] = None
        self.original_size: Optional[Tuple[int, int]] = None


class DataLoaderHandler(BaseTaskHandler):
    def __init__(self):
        self._model = default_producer.get_data_loader()

    def handle(self, *tasks: Task):
        for task in tasks:
            task.input, task.fg_points, task.bg_points, task.original_size = self._model.process(task.input)


class PreProcessingHandler(BaseTaskHandler):
    def __init__(self):
        self._model = default_producer.get_pre_proc()

    def handle(self, *tasks: Task):
        for task in tasks:
            task.input = self._model.process(task.input, task.fg_points, task.bg_points)
            task.fg_points = None
            task.bg_points = None


class RITMCallHandler(BaseTaskHandler):
    def __init__(self):
        self._model: Optional[RITMCall] = None

    def on_start(self):
        self._model = default_producer.get_ritm_model()

    def handle(self, *tasks: Task):
        preds = self._model.process_list(data=[task.input for task in tasks])
        for pred, task in zip(preds, tasks):
            task.pred = pred
            task.input = None


class PostProcessingHandler(BaseTaskHandler):
    def __init__(self):
        self._model = default_producer.get_post_proc()

    def handle(self, *tasks: Task):
        for task in tasks:
            task.pred = self._model.process(task.pred, task.original_size)
            task.original_size = None


class ImageServingHandler(BaseTaskHandler):
    def __init__(self):
        self._model = default_producer.get_image_encoder()

    def handle(self, *tasks: Task):
        for task in tasks:
            task.pred = self._model.process(task.pred)


def get_flow() -> Flow:
    return Flow(
        FlowStep(DataLoaderHandler(), nprocs=4),
        FlowStep(PreProcessingHandler(), nprocs=4),
        FlowStep(RITMCallHandler(), batch_size=1, nprocs=1),
        FlowStep(PostProcessingHandler(), nprocs=4),
        FlowStep(ImageServingHandler(), nprocs=4),
        metrics_enabled=False,
    )

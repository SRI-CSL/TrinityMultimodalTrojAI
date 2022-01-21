"""
=========================================================================================
Trojan VQA
Written by Karan Sikka and Matthew Walmer

Detector utilities
=========================================================================================
"""

import matplotlib.patches as patches
import cv2
from fvcore.common.file_io import PathManager
from detectron2.engine import DefaultPredictor
import sys

sys.path.append("grid-feats-vqa/")
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2.evaluation import inference_context
from detectron2.modeling import build_model
from grid_feats import (
    add_attribute_config,
    build_detection_test_loader_with_attributes,
)
import torch


def check_for_cuda():
    if not torch.cuda.is_available():
        device = "cpu"
    else:
        device = "cuda"
    print(f"Device is {device}")
    return device


color_pad = [
    "red",
    "orange",
    "green",
    "blue",
    "purple",
    "brown",
    "pink",
    "khaki",
    "darkgreen",
    "cyan",
    "coral",
    "magenta",
]


def drawBbox(ax, bbox, category_name, color_idx):
    ax.add_patch(
        patches.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            fill=False,  # remove background
            lw=3,
            # color=color_pad[color_idx % len(color_pad)]
            color=color_pad[color_idx],
        )
    )
    # print(color_pad[color_idx])
    ax.text(
        bbox[0],
        bbox[1] + 3,
        "%s" % (category_name),
        fontsize=11,
        fontweight="bold",
        backgroundcolor=color_pad[color_idx % len(color_pad)],
    )
    return ax


def config_setup(config_file, model_path, device):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_attribute_config(cfg)
    cfg.merge_from_file(config_file)
    # force the final residual block to have dilations 1
    cfg.MODEL.RESNETS.RES5_DILATION = 1
    cfg.TEST.DETECTIONS_PER_IMAGE = 200
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.DEVICE = device
    cfg.freeze()
    return cfg


import time


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.0
        self.calls = 0
        self.start_time = 0.0
        self.diff = 0.0
        self.average_time = 0.0

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def load_detectron_predictor(config_file, model_path, device):
    cfg = config_setup(config_file, model_path, device)
    predictor = DefaultPredictor(cfg)
    return predictor


def run_detector(predictor, img, max_boxes=36, verbose=False):
    outputs, box_features = predictor(img)

    out = {}
    for it in ["pred_boxes", "scores", "pred_classes"]:
        out[it] = outputs["instances"].get_fields()[it]
    scores = out["scores"].data.cpu()
    pred_boxes = out["pred_boxes"].tensor.data.cpu()
    pred_classes = out["pred_classes"].data.cpu()
    if verbose:
        print("Number of Detected boxes = ", len(scores))
        print("Number of Box features  = ", len(box_features))
    assert len(scores) == len(box_features)

    # Predicting attributes
    roi_head = predictor.model.roi_heads
    attribute_features = box_features
    obj_labels = pred_classes
    # attribute_labels = torch.cat([p.gt_attributes for p in proposals], dim=0)
    attribute_scores = roi_head.attribute_predictor(
        attribute_features, obj_labels.to(box_features.device)
    )
    pred_attr = attribute_scores.argmax(1).data.cpu()

    # Save outputs in numpy array
    N = max_boxes
    info = {}
    info["boxes"] = pred_boxes[:N]
    info["features"] = box_features[:N].data.cpu()
    info["object_ids"] = pred_classes[:N]
    info["attr_ids"] = pred_attr[:N]
    info["objects_scores"] = scores[:N]
    info["attr_scores"] = attribute_scores.max(1)[0].data.cpu()
    info["img_w"] = img.shape[0]
    info["img_h"] = img.shape[1]

    return info

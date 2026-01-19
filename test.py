import warnings
import os

# 1. 禁用所有警告
warnings.filterwarnings("ignore")


import copy
import itertools
import logging
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,4,6'

from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    # SemSegEvaluator,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

# from detectron2.data.datasets import register_lvis_instances
# register_lvis_instances("lvis_v1_val", {}, "datasets/lvis/lvis_v1_val.json", "datasets/lvis/coco/val2017")
# # register_lvis_instances("lvis_v1_train", {}, "datasets/lvis/lvis_v1_train.json", "datasets/lvis/coco/train2017")

from maft import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    COCOSemanticNewBaselineDatasetMapper,
    InstanceSegEvaluator,
    SemSegEvaluator,
    MaskFormerInstanceDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    # SemanticSegmentorWithTTA,
    # add_maskformer2_config,
    # add_fcclip_config,
)

from maft.data.dataset_mappers.coco_combine_new_baseline_dataset_mapper import COCOCombineNewBaselineDatasetMapper

from sam3.data.custom_dataset_dataloader import build_custom_train_loader
from sam3.config import add_sam3_config
from sam3.modeling_d2 import SAM3Wrapper # 导入这个类就会自动触发 REGISTER
# from sam3.SAM3MC import SAM3MC
from sam3.SAM3MC_ora import SAM3MC_ora
from sam3.SAM3MC_DINO import SAM3MC_DINO
from sam3.SAM3MC_o365 import SAM3MC_o365
#--------------------------------------

import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
import matplotlib.pyplot as plt

def analyze_target_counts(dataset_name="objects365_train"):
    print(f"Loading dataset: {dataset_name}...")
    # 获取数据集字典列表
    dataset_dicts = DatasetCatalog.get(dataset_name)
    
    counts = []
    print("Counting targets...")
    for d in dataset_dicts:
        # 获取标注数量
        if "annotations" in d:
            num_targets = len(d["annotations"])
            counts.append(num_targets)
        elif "instances" in d: # 如果是预处理过的格式
            num_targets = len(d["instances"])
            counts.append(num_targets)
    
    counts = np.array(counts)
    
    # 统计数据
    mean_val = np.mean(counts)
    var_val = np.var(counts)
    std_val = np.std(counts)
    max_val = np.max(counts)
    
    # 计算分位点 (关键：决定你的阈值设在哪里)
    p90 = np.percentile(counts, 90)
    p95 = np.percentile(counts, 95)
    p99 = np.percentile(counts, 99)
    p99_9 = np.percentile(counts, 99.9)

    print("-" * 30)
    print(f"Dataset: {dataset_name}")
    print(f"Total Images: {len(counts)}")
    print(f"Mean Targets per Image: {mean_val:.2f}")
    print(f"Std Dev: {std_val:.2f} (Variance: {var_val:.2f})")
    print(f"Max Targets: {max_val}")
    print("-" * 30)
    print(f"90th Percentile: {p90}")
    print(f"95th Percentile: {p95}")
    print(f"99th Percentile: {p99}")
    print(f"99.9th Percentile: {p99_9}")
    print("-" * 30)
    
    return counts

# 在你的环境中调用（替换为你注册的数据集名称）
counts = analyze_target_counts("objects365_v1_masktrain")
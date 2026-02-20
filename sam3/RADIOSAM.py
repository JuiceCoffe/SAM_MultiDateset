from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import matplotlib.cm as cm
import shutil
import cv2

import os
import numpy as np
import torchvision
import logging
import copy

from sam3.model_builder import (
    _create_vision_backbone,
    _create_text_encoder,
    _create_vl_backbone,
    _create_sam3_transformer,
    _create_dot_product_scoring,
    _create_segmentation_head,
    _create_geometry_encoder,
    _create_sam3_model,
)
from sam3.model.data_misc import FindStage, interpolate

from sam3.model.model_misc import (
    gen_sineembed_for_position,
    get_activation_fn,
    get_clones,
    inverse_sigmoid,
    MLP,
)

from maft.utils.text_templetes import VILD_PROMPT


from .loss.matcher import HungarianMatcher
from .loss.criterion import SetCriterion
from sam3.model.content_dependent_transfer import ContentDependentTransfer
from sam3.model.box_ops import masks_to_boxes, box_xyxy_to_cxcywh


from maft.modeling.transformer_decoder.fcclip_transformer_decoder import MaskPooling

import random

import math

from .mask_adapter_head import build_mask_adapter


from.RADIOwrapper import replace_sam3_encoder, load_radio_model

@META_ARCH_REGISTRY.register()
class RADIOSAM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device_type = cfg.MODEL.DEVICE
        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1), False) 
        
        # -------------------------------------------------------
        # 2. 实例化 SAM3 Model
        # -------------------------------------------------------       
        compile_mode = "default" if cfg.MODEL.SAM3.COMPILE else None
        
        vision_encoder = _create_vision_backbone(
            compile_mode=compile_mode, 
            enable_inst_interactivity=cfg.MODEL.SAM3.ENABLE_INST_INTERACTIVITY
        )
        text_encoder = _create_text_encoder(cfg.MODEL.SAM3.BPE_PATH)
        backbone = _create_vl_backbone(vision_encoder, text_encoder)
        transformer = _create_sam3_transformer(
            use_gate = cfg.MODEL.SAM3.USE_GATE,
        )
        dot_prod_scoring = _create_dot_product_scoring()

        segmentation_head = (
            _create_segmentation_head(compile_mode=compile_mode)
            if cfg.MODEL.SAM3.ENABLE_SEGMENTATION
            else None
        )
            
        input_geometry_encoder = _create_geometry_encoder()

        enable_inst_interactivity = False # 遵照sam3设置
        if enable_inst_interactivity:
            sam3_pvs_base = build_tracker(apply_temporal_disambiguation=False)
            inst_predictor = SAM3InteractiveImagePredictor(sam3_pvs_base)
        else:
            inst_predictor = None

        self.detector = _create_sam3_model(
            backbone,
            transformer,
            input_geometry_encoder,
            segmentation_head,
            dot_prod_scoring,
            inst_predictor,
            cfg.eval_only,
        )
        if cfg.eval_only:
            self.detector.eval()
        print("SAM3创建成功!")

        radio_model = load_radio_model("c-radio_v4-h", device=self.pixel_mean.device, vitdet=None)
        self.detector, self.radio_adaptor = replace_sam3_encoder(self.detector, radio_model, device=self.pixel_mean.device)

        # -------------------------------------------------------
        # 新增模块
        # -------------------------------------------------------
        self.use_pe_text = cfg.MODEL.SAM3.USE_PE_TEXT
        
        self.use_cos_sim = getattr(cfg.MODEL.SAM3, "COS_SIM", False) # 默认为 False


        # 通过 cfg 控制是否启用，硬编码输入输出维度为 256
        self.use_query_proj = cfg.MODEL.SAM3.USE_QUERY_PROJ
        if self.use_query_proj:
            self.query_proj = MLP(256, 256, 1024, 3)
        else:
            self.query_proj = None


        self.num_decoder_layers = 6 # SAM3/DETR 标准层数
    
        self.num_cdt = cfg.MODEL.SAM3.NUM_CDT    
        self.use_cdt = False if cfg.MODEL.SAM3.NUM_CDT == 0 else True
        if self.use_cdt:
            if self.use_pe_text:
                print("CDT模块当前仅支持256文本特征输入，请关闭PE文本特征选项！")
            text_classifier_dim = 256
            self.cdt = nn.ModuleList([
                ContentDependentTransfer(d_model=text_classifier_dim, nhead=8, panoptic_on=True) 
                for _ in range(self.num_cdt)
            ])
        else:
            self.cdt = None


        self.DynamicQuery = cfg.MODEL.SAM3.DYNAMIC_QUERY
        if self.DynamicQuery:
            self.encoder_box_head = MLP(256, 256, 4, 3)
            nn.init.constant_(self.encoder_box_head.layers[-1].weight.data, 0)
            nn.init.constant_(self.encoder_box_head.layers[-1].bias.data, 0)
 
        self.mask_pooling = MaskPooling()


        self.use_attnpool = cfg.MODEL.ATTNPOOL.ENABLE
        if self.use_attnpool:
            self.attnpool_weight_dict = {"loss_attn_ce": cfg.MODEL.ATTNPOOL.CLASS_WEIGHT}
            self.num_gt_masks = cfg.MODEL.ATTNPOOL.NUM_GT_MASKS
            self.iou_threshold = cfg.MODEL.ATTNPOOL.IOU_THRESHOLD
            self.num_pred_masks = cfg.MODEL.ATTNPOOL.NUM_PRED_MASKS
            self.aux_attn_pool = cfg.MODEL.ATTNPOOL.USE_AUX
            self.add_radio_feat = cfg.MODEL.ATTNPOOL.ADD_RADIO_FEAT
            self.new_pool_decoder = cfg.MODEL.ATTNPOOL.NEW_POOL_DECODER
            self.build_pool_decoder_from_maskdecoder = cfg.MODEL.ATTNPOOL.BUILD_POOL_DECODER_FROM_MASKDECODER
        
            if self.aux_attn_pool:
                attnpool_weight_dict_aux = {}
                for i in range (5):
                    for k in self.attnpool_weight_dict.keys():
                        attnpool_weight_dict_aux[f"{k}_{i}"] = self.attnpool_weight_dict[k]
                self.attnpool_weight_dict.update(attnpool_weight_dict_aux) 
            self.out_vocab_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            if self.new_pool_decoder:
                self.pooling_decoder = _create_pool_decoder()
                if self.add_radio_feat:
                    self.fusion_feat_proj = MLP(256, 512, 1536, 3)
                    self.radio_feat_proj =  MLP(1536, 512, 256, 3)
                self.attn_pool_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
                self.pooling_decoder_synced = False if self.build_pool_decoder_from_maskdecoder else True
        else:
            self.new_pool_decoder = False
            self.out_vocab_logit_scale = nn.Parameter(torch.ones([]) * np.log(100))
            

        self.new_score_head = cfg.MODEL.SAM3.NEW_SCORE_HEAD
        if self.new_score_head:
            self.score_head = MLP(256, 256, 1, 3)
            init_score_head(self.score_head)


        # -------------------------------------------------------
        # 计算逻辑
        # -------------------------------------------------------
        self.PROMPT = VILD_PROMPT
        self.SAM_PROMPT = ['{}']

        self.add_pixelfeat = cfg.MODEL.SAM3.ADD_PIXELFEAT
        self.alpha = cfg.MODEL.SAM3.ALPHA
        self.beta = cfg.MODEL.SAM3.BETA
        self.OracleSelect_on = cfg.MODEL.SAM3.ORACLE_SELECT

        # -------------------------------------------------------
        # 训练配置
        # -------------------------------------------------------
        # 你需要检查 sam3_loss 的初始化参数
        self.train_dataname = None
        self.test_dataname = None
        self.SAM_train_dataname = None
        self.SAM_test_dataname = None
        self.text_encoder_cache = {} 
        self.SAM_text_encoder_cache = {}


        self.test_metadata = {i: MetadataCatalog.get(i) for i in cfg.DATASETS.TEST}
        self.train_metadata_dict = {name: MetadataCatalog.get(name) for name in cfg.DATASETS.TRAIN}
        
        # 【修改】这里初始化为 None 或第一个数据集的均可，反正 get_text_classifier 会覆盖它
        # 重要的是 self.train_class_names 和 num_templates 只是临时变量
        if len(cfg.DATASETS.TRAIN) > 0:
            self.train_metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        else:
            self.train_metadata = None 
            
        self.train_num_templates = None 
        self.train_class_names = None

        self.use_aux = cfg.SOLVER.USE_AUX
        self.only_instance = cfg.DATASETS.ONLY_INSTANCE
        if self.only_instance:
            raise NotImplementedError("only_instance尚未给prepare_targets_for_maskadapter适配")
        
        self.train_mask = cfg.SOLVER.TRAIN_MASK
        # -------------------------------------------------------
        # criterion损失函数
        # -------------------------------------------------------
        losses = ["labels", "masks", "boxes"]

        # loss weights
        class_weight = cfg.SOLVER.CLASS_WEIGHT
        dice_weight = cfg.SOLVER.DICE_WEIGHT
        mask_weight = cfg.SOLVER.MASK_WEIGHT
        bbox_weight = cfg.SOLVER.BBOX_WEIGHT
        giou_weight = cfg.SOLVER.GIOU_WEIGHT

        objectness_weight = cfg.SOLVER.OBJECT_WEIGHT

        self.use_softmax = cfg.MODEL.SAM3.USE_SOFTMAX
        self.void_embedding = None
        if self.use_softmax:
            self.void_embedding = nn.Embedding(1, 256)


        self.logit_bias = None
        if not self.use_softmax:
            prior_prob = 0.01
            bias_value = -np.log((1 - prior_prob) / prior_prob)
            self.logit_bias = nn.Parameter(torch.ones([]) * bias_value) 

        if self.use_cos_sim:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        weight_dict = {}
        criterion_weight_dict = {
            "loss_cls": class_weight, 
            "loss_mask": mask_weight, 
            "loss_dice": dice_weight,
            'loss_bbox':bbox_weight, 
            'loss_giou':giou_weight,
        }
        if self.new_score_head:
            criterion_weight_dict["loss_objectness"] = objectness_weight
        if self.use_attnpool:
            criterion_weight_dict["loss_attn_cls"] = cfg.MODEL.ATTNPOOL.CLASS_WEIGHT
        weight_dict.update(criterion_weight_dict)

        if self.use_aux:
            for i in range (5):
                for k in criterion_weight_dict.keys():
                    weight_dict[f"{k}_{i}"] = criterion_weight_dict[k]

        # building criterion
        if self.use_softmax:
            from .loss.fcclip_criterion import FcclipSetCriterion
            from .loss.fcclip_matcher import FcclipHungarianMatcher

            no_object_weight = 0.1

            matcher = FcclipHungarianMatcher(
                cost_class=class_weight,
                cost_mask=mask_weight,
                cost_dice=dice_weight,
                cost_bbox=bbox_weight,
                cost_giou=giou_weight,
                num_points=cfg.SOLVER.TRAIN_NUM_POINTS,
            )
            self.criterion = FcclipSetCriterion(
                num_classes = 133,
                matcher=matcher,
                weight_dict=weight_dict,
                eos_coef=no_object_weight,
                losses=losses,
                num_points=cfg.SOLVER.TRAIN_NUM_POINTS,
                oversample_ratio=cfg.SOLVER.OVERSAMPLE_RATIO,
                importance_sample_ratio=cfg.SOLVER.IMPORTANCE_SAMPLE_RATIO,
            )

        else:
            matcher = HungarianMatcher(
                cost_class=class_weight,
                cost_mask=mask_weight,
                cost_dice=dice_weight,
                num_points=cfg.SOLVER.TRAIN_NUM_POINTS,
            )
        

            self.criterion = SetCriterion(
                matcher=matcher,
                weight_dict=weight_dict,
                losses=losses,
                num_points=cfg.SOLVER.TRAIN_NUM_POINTS,
                oversample_ratio=cfg.SOLVER.OVERSAMPLE_RATIO,
                importance_sample_ratio=cfg.SOLVER.IMPORTANCE_SAMPLE_RATIO,
                tau=cfg.SOLVER.CONTRAST_TEMPERATURE,
            )

        # -------------------------------------------------------
        # 【新增】Inference 参数配置
        # -------------------------------------------------------
        self.semantic_on = cfg.TEST.SEMANTIC_ON
        self.instance_on = cfg.TEST.INSTANCE_ON 
        self.panoptic_on = cfg.TEST.PANOPTIC_ON 
        
        # 阈值设置 (如果没有在 cfg 定义，给默认值)
        self.object_mask_threshold = 0.01
        self.overlap_threshold = 0.8
        self.test_topk_per_image = 100

        self._freeze()

    def _freeze(self):
        # 获取 detectron2 的 logger
        logger = logging.getLogger("detectron2") 

        # 1. 定义需要冻结的关键字
        freeze_keywords = [
            'geometry_encoder',
            'language_backbone', 
            'text_model', 
            'sig2_adaptor',
            "student"
        ]

        # 2. 遍历参数并设置 requires_grad
        for name, param in self.named_parameters():
            if any(key in name for key in freeze_keywords):
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        # 3. 使用 logger 输出结果
        
        trainable_count = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                # 使用 logger.info 代替 print
                logger.info(f"TRAINABLE: {name}")
                trainable_count += 1
            else:
                logger.info(f"FROZEN: {name}")
        
        logger.info(f"Total trainable parameter groups: {trainable_count}")
        logger.info('='*40)
        

    # def prepare_targets(self, targets, images):
    #     h_pad, w_pad = images.tensor.shape[-2:]
    #     new_targets = []
    #     for targets_per_image in targets:
    #         # pad gt
    #         gt_masks = targets_per_image.gt_masks
    #         if isinstance(gt_masks, BitMasks):
    #             gt_masks = gt_masks.tensor
    #         padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
    #         padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            
    #         # ---------------- 修改开始 ----------------
    #         # 1. 从 Mask 生成绝对坐标的 Box (xyxy格式)
    #         gt_boxes_xyxy = masks_to_boxes(padded_masks)
            
    #         # 2. 归一化 Box 坐标到 [0, 1] (除以 padded后的宽高)
    #         # scale: [w, h, w, h]
    #         scale = torch.tensor([w_pad, h_pad, w_pad, h_pad], dtype=torch.float32, device=gt_boxes_xyxy.device)
    #         gt_boxes_norm = gt_boxes_xyxy / scale

    #         # 3. 转换为 (cx, cy, w, h) 格式，这是 DETR/SAM3 计算 Loss 要求的格式
    #         gt_boxes_cxcywh = box_xyxy_to_cxcywh(gt_boxes_norm)
    #         # ---------------- 修改结束 ----------------

    #         new_targets.append(
    #             {
    #                 "labels": targets_per_image.gt_classes,
    #                 "masks": padded_masks,
    #                 "boxes": gt_boxes_cxcywh, # <--- 传入处理好的 boxes
    #             }
    #         )
    #     return new_targets


    def prepare_targets(self, targets, images, batched_inputs):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        
        # 遍历 batch 中的每一张图
        for i, targets_per_image in enumerate(targets):
            # 获取当前图片的数据集名称
            dataname = get_dataname(batched_inputs[i])
            
            # ---------------- 过滤逻辑开始 ----------------
            # 这里的逻辑是：如果开启了 only_instance，且当前数据集是 COCO Stuff
            # 那么我们就把其中的 Thing 类别（Person, Car 等）过滤掉，只保留 Stuff
            # 避免与 Object365 的 Instance 标注冲突
            
            gt_classes = targets_per_image.gt_classes
            gt_masks = targets_per_image.gt_masks
            if isinstance(gt_masks, BitMasks):
                gt_masks = gt_masks.tensor
            
            # 生成一个全是 True 的 mask
            keep_indices = torch.ones(len(gt_classes), dtype=torch.bool, device=gt_classes.device)

            # ---------------- 修改开始 ----------------
            if self.only_instance:
                # 判断是否为 COCO Stuff 数据集
                is_coco_stuff = "openvocab_coco_2017_train_stuff_sem_seg" in dataname
                
                if is_coco_stuff:
                    # 1. 筛选逻辑：只保留 ID >= 80 的 Stuff
                    keep_indices = gt_classes >= 80
                    
                    # 应用筛选
                    gt_classes = gt_classes[keep_indices]
                    gt_masks = gt_masks[keep_indices]
                    
                    # 2. 【关键修正】Label Shift (标签对齐)
                    # 因为我们在 get_text_classifier 中移除了前 80 个类 (Things)
                    # 所以原来的 Label 80 (Stuff start) 现在必须变成 Label 0
                    gt_classes = gt_classes - 80

                    # if len(gt_classes) > 0:
                    #     print(f"DEBUG: [Targets] 原本最小ID >= 80, 现已 Shift。当前图片最小 Label: {gt_classes.min().item()}, 最大 Label: {gt_classes.max().item()}")
                else:
                    # 对于非 COCO Stuff 数据集（如 Obj365），不做特殊处理，直接应用 keep_indices (全True)
                    gt_classes = gt_classes[keep_indices]
                    gt_masks = gt_masks[keep_indices]
            # ---------------- 过滤逻辑结束 ----------------

            # pad gt
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            # 注意：这里的 gt_masks 已经是过滤后的了
            if gt_masks.shape[0] > 0:
                padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            
            # 1. 从 Mask 生成绝对坐标的 Box (xyxy格式)
            # 注意：masks_to_boxes 需要处理空 mask 的情况，虽然理论上 targets 不会为空，但过滤后可能为空
            if padded_masks.shape[0] > 0:
                gt_boxes_xyxy = masks_to_boxes(padded_masks)
                
                # 2. 归一化 Box 坐标到 [0, 1]
                scale = torch.tensor([w_pad, h_pad, w_pad, h_pad], dtype=torch.float32, device=gt_boxes_xyxy.device)
                gt_boxes_norm = gt_boxes_xyxy / scale

                # 3. 转换为 (cx, cy, w, h)
                gt_boxes_cxcywh = box_xyxy_to_cxcywh(gt_boxes_norm)
            else:
                # 如果过滤后没有物体了，给空 tensor
                gt_boxes_cxcywh = torch.zeros((0, 4), device=padded_masks.device)

            new_targets.append(
                {
                    "labels": gt_classes, # 过滤后的 labels
                    "masks": padded_masks, # 过滤后的 masks
                    "boxes": gt_boxes_cxcywh, # 过滤后的 boxes
                }
            )
        return new_targets

    def prepare_targets_for_maskadapter(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        masks_list = []
        labels_list = []

        num_masks = self.num_gt_masks  

        for targets_per_image in targets:
            gt_masks = targets_per_image.gt_masks
            if isinstance(gt_masks, BitMasks):
                gt_masks = gt_masks.tensor
            valid_mask_indices = [i for i, mask in enumerate(gt_masks) if mask.sum() > 0] 

            if len(valid_mask_indices) > 0:
                valid_gt_masks = gt_masks[valid_mask_indices]
                valid_gt_classes = targets_per_image.gt_classes[valid_mask_indices]

                padded_masks = torch.zeros((valid_gt_masks.shape[0], h_pad, w_pad), dtype=valid_gt_masks.dtype, device=valid_gt_masks.device)
                padded_masks[:, :valid_gt_masks.shape[1], :valid_gt_masks.shape[2]] = valid_gt_masks
                new_targets.append(
                    {
                        "labels": valid_gt_classes,
                        "masks": padded_masks,
                    }
                )

                total_masks = torch.zeros((num_masks, h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
                selected_labels = torch.full((num_masks,), -1, dtype=valid_gt_classes.dtype, device=gt_masks.device)

                if valid_gt_masks.shape[0] > num_masks:
                    selected_indices = torch.randperm(valid_gt_masks.shape[0])[:num_masks]
                    for idx, mask_idx in enumerate(selected_indices):
                        total_masks[idx, :valid_gt_masks[mask_idx].shape[0], :valid_gt_masks[mask_idx].shape[1]] = valid_gt_masks[mask_idx]
                        selected_labels[idx] = valid_gt_classes[mask_idx]
                else:
                    for idx in range(valid_gt_masks.shape[0]):
                        total_masks[idx, :valid_gt_masks[idx].shape[0], :valid_gt_masks[idx].shape[1]] = valid_gt_masks[idx]
                        selected_labels[idx] = valid_gt_classes[idx]
                    
                    for idx in range(valid_gt_masks.shape[0], num_masks):
                        total_masks[idx] = torch.zeros((h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
                        selected_labels[idx] = -1
            else:
                total_masks = torch.zeros((num_masks, h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
                selected_labels = torch.full((num_masks,), -1, dtype=torch.long, device=gt_masks.device)
                
                padded_masks = torch.zeros((0, h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
                valid_gt_classes = torch.zeros((0), device=gt_masks.device)
                new_targets.append(
                    {
                        "labels": valid_gt_classes,
                        "masks": padded_masks,
                    }
                )

            masks_list.append(total_masks)
            labels_list.append(selected_labels)

        masks = torch.stack(masks_list, dim=0)
        labels = torch.stack(labels_list, dim=0)
        labels = labels.long()

        return new_targets, masks, labels

    def prepare_class_names_from_metadata(self, metadata, train_metadata, prompt_list):
        def split_labels(x):
            res = []
            for x_ in x:
                x_ = x_.replace(', ', ',')
                x_ = x_.split(',') # there can be multiple synonyms for single class
                res.append(x_)
            return res
        # get text classifier
        # --- 修改开始：解耦 train 和 test 的读取逻辑 ---
        
        # 1. 获取测试集类别 (优先读取 stuff_classes 以包含全景类别)
        try:
            class_names = split_labels(metadata.stuff_classes)
        except AttributeError:
            class_names = split_labels(metadata.thing_classes)

        # 2. 获取训练集类别 (独立处理，避免影响测试集)
        try:
            train_class_names = split_labels(train_metadata.stuff_classes)
        except AttributeError:
            train_class_names = split_labels(train_metadata.thing_classes)
            
        # --- 修改结束 ---

        train_class_names = {l for label in train_class_names for l in label}
        category_overlapping_list = []
        self.vis_class_names = class_names
        for test_class_names in class_names:
            is_overlapping = not set(train_class_names).isdisjoint(set(test_class_names)) 
            category_overlapping_list.append(is_overlapping)
        category_overlapping_mask = torch.tensor(
            category_overlapping_list, dtype=torch.long)
        
        def fill_all_templates_ensemble(x_=''):
            res = []
            for x in x_:
                for template in prompt_list:
                    res.append(template.format(x))
            return res, len(res) // len(prompt_list)
       
        num_templates = []
        templated_class_names = []
        for x in class_names:
            templated_classes, templated_classes_num = fill_all_templates_ensemble(x)
            templated_class_names += templated_classes
            num_templates.append(templated_classes_num) # how many templates for current classes
        class_names = templated_class_names
        return category_overlapping_mask, num_templates, class_names

    # def set_metadata(self, metadata):
    #     self.test_metadata = metadata
    #     self.category_overlapping_mask, self.test_num_templates, self.test_class_names = self.prepare_class_names_from_metadata(metadata, self.train_metadata)
    #     self.test_text_classifier = None
    #     return

    def get_text_classifier(self, dataname):
        if self.training:
            if self.train_dataname != dataname:
                if dataname in self.text_encoder_cache:
                    cache = self.text_encoder_cache[dataname]
                    # 从 CPU 缓存加载并移回 GPU
                    self.language_features = cache["language_features"].to(self.device)
                    self.language_mask = cache["language_mask"].to(self.device)
                    self.train_text_classifier = cache["text_classifier"].to(self.device)
                    
                    # 直接恢复列表，不要再重新计算
                    self.train_num_templates = cache["num_templates"] 
                    self.train_class_names = cache["class_names"]

                else:
                    # ============== 关键修改开始 ==============
                    # 获取 metadata
                    if dataname in self.train_metadata_dict:
                        current_metadata = self.train_metadata_dict[dataname]
                    else:
                        current_metadata = MetadataCatalog.get(dataname)

                    # 计算类别名和模板数量
                    # 【注意】这里不要用 self.train_num_templates[dataname]，因为它不是字典
                    # 直接覆盖 self.train_num_templates 为当前数据集的 List
                    _, self.train_num_templates, self.train_class_names = self.prepare_class_names_from_metadata(
                        current_metadata, current_metadata, self.PROMPT
                    )
                    # ============== 关键修改结束 ==============

                    # 如果开启 only_instance 且当前是 COCO Stuff，我们需要移除前 80 个 Things 类
                    # 这样生成的 text_classifier 就只包含 Stuff 类，且从 0 开始索引
                    # ---------------- 修改开始 ----------------
                    is_coco_stuff = "openvocab_coco_2017_train_stuff_sem_seg" in dataname
                    
                    if self.only_instance and is_coco_stuff:
                        num_things_classes = 80 
                        
                        if len(self.train_num_templates) > num_things_classes:
                            print(f"[{dataname}] Filtering out {num_things_classes} Thing classes for Stuff training.")
                            
                            # 1. 计算前 80 个类总共占用了多少个文本 Prompt 
                            # 【关键修正】这里必须乘以每个同义词对应的 Prompt 数量 (len(self.PROMPT))
                            # train_num_templates 存的是同义词数量，train_class_names 存的是展开后的 Prompt
                            num_synonyms_to_skip = sum(self.train_num_templates[:num_things_classes])
                            offset_text_idx = num_synonyms_to_skip * len(self.PROMPT)
                            
                            # 2. 截断 class_names (输入给 Text Encoder 的文本列表)
                            self.train_class_names = self.train_class_names[offset_text_idx:]
                            
                            # 3. 截断 num_templates (用于后续 Logit 聚合的计数列表)
                            self.train_num_templates = self.train_num_templates[num_things_classes:]

                    # --- 原有生成逻辑开始 ---
                    text_classifier = []
                    bs = 128
                    print("Generating text classifier for", dataname, "with", len(self.train_class_names), "classes.")
                    for idx in range(0, len(self.train_class_names), bs):
                        batch_text_feat = self.radio_adaptor.get_text_classifier(self.train_class_names[idx:idx+bs], device=self.device)
                        text_classifier.append(batch_text_feat)
                    text_classifier = torch.cat(text_classifier, dim=0)
                    text_classifier = text_classifier.reshape(text_classifier.shape[0]//len(self.PROMPT), len(self.PROMPT), -1, text_classifier.shape[-1]) # num_names, self.PROMPT, L, D
                    print("text_classifier:",text_classifier.shape)
                    language_features = text_classifier.mean(1) # num_names, L, D
                    text_classifier = text_classifier.mean(-2) 
                    text_classifier /= (text_classifier.norm(dim=-1, keepdim=True) + 1e-6)
                    text_classifier = text_classifier.mean(1)
                    text_classifier /= (text_classifier.norm(dim=-1, keepdim=True) + 1e-6)
                    
                    self.language_features = language_features.detach() # num_names , L, D
                    self.language_mask = torch.zeros(
                        self.language_features.shape[0], self.language_features.shape[1], 
                        dtype=torch.bool, device=self.device
                    )
                    self.train_text_classifier = text_classifier.detach()
                    # --- 原有生成逻辑结束 ---

                    # 【新增】2. 将生成的特征移至 CPU 并存入缓存
                    self.text_encoder_cache[dataname] = {
                        "language_features": self.language_features.cpu(),
                        "language_mask": self.language_mask.cpu(),
                        "text_classifier": self.train_text_classifier.cpu(),
                        "num_templates": self.train_num_templates, # 缓存当前的 List
                        "class_names": self.train_class_names      # 缓存当前的 List
                    }

                self.train_dataname = dataname
            return self.train_text_classifier.clone(), self.train_num_templates
        else:
            if self.test_dataname != dataname:
                self.category_overlapping_mask, self.test_num_templates, self.test_class_names = self.prepare_class_names_from_metadata(self.test_metadata[dataname], self.train_metadata, self.PROMPT)
                text_classifier = []
                bs = 128
                print("Generating text classifier for", dataname, "with", len(self.test_class_names), "classes.")
                for idx in range(0, len(self.test_class_names), bs):
                    batch_text_feat = self.radio_adaptor.get_text_classifier(self.test_class_names[idx:idx+bs], device=self.device)
                    text_classifier.append(batch_text_feat)
                text_classifier = torch.cat(text_classifier, dim=0)
                text_classifier = text_classifier.reshape(text_classifier.shape[0]//len(self.PROMPT), len(self.PROMPT), -1, text_classifier.shape[-1]) # num_names, self.PROMPT, L, D
                language_features = text_classifier.mean(1) # num_names, L, D
                text_classifier = text_classifier.mean(-2) 
                text_classifier /= (text_classifier.norm(dim=-1, keepdim=True) + 1e-6)
                text_classifier = text_classifier.mean(1)
                text_classifier /= (text_classifier.norm(dim=-1, keepdim=True) + 1e-6)
                
                self.language_features = language_features.detach() # num_names , L, D
                self.language_mask = torch.zeros(
                        self.language_features.shape[0], self.language_features.shape[1], 
                        dtype=torch.bool, device=self.device
                    )
                self.test_text_classifier = text_classifier.detach()
                self.test_dataname = dataname
            return self.test_text_classifier.clone(), self.test_num_templates

    def get_SAM_text_classifier(self, dataname):
        if self.training:
            if self.SAM_train_dataname != dataname:
                if dataname in self.SAM_text_encoder_cache:
                    cache = self.SAM_text_encoder_cache[dataname]
                    # 从 CPU 缓存加载并移回 GPU
                    self.SAM_language_features = cache["language_features"].to(self.device)
                    self.SAM_language_mask = cache["language_mask"].to(self.device)
                    self.SAM_train_text_classifier = cache["text_classifier"].to(self.device)
                    
                    # 直接恢复列表，不要再重新计算
                    self.SAM_train_num_templates = cache["num_templates"] 
                    self.SAM_train_class_names = cache["class_names"]

                else:
                    # ============== 关键修改开始 ==============
                    # 获取 metadata
                    if dataname in self.train_metadata_dict:
                        current_metadata = self.train_metadata_dict[dataname]
                    else:
                        current_metadata = MetadataCatalog.get(dataname)

                    # 计算类别名和模板数量
                    # 【注意】这里不要用 self.train_num_templates[dataname]，因为它不是字典
                    # 直接覆盖 self.train_num_templates 为当前数据集的 List
                    _, self.SAM_train_num_templates, self.SAM_train_class_names = self.prepare_class_names_from_metadata(
                        current_metadata, current_metadata, self.SAM_PROMPT
                    )
                    # ============== 关键修改结束 ==============

                    # 如果开启 only_instance 且当前是 COCO Stuff，我们需要移除前 80 个 Things 类
                    # 这样生成的 text_classifier 就只包含 Stuff 类，且从 0 开始索引
                    # ---------------- 修改开始 ----------------
                    is_coco_stuff = "openvocab_coco_2017_train_stuff_sem_seg" in dataname
                    
                    if self.only_instance and is_coco_stuff:
                        num_things_classes = 80 
                        
                        if len(self.SAM_train_num_templates) > num_things_classes:
                            print(f"[{dataname}] Filtering out {num_things_classes} Thing classes for Stuff training.")
                            
                            # 1. 计算前 80 个类总共占用了多少个文本 Prompt 
                            # 【关键修正】这里必须乘以每个同义词对应的 Prompt 数量 (len(self.PROMPT))
                            # train_num_templates 存的是同义词数量，train_class_names 存的是展开后的 Prompt
                            num_synonyms_to_skip = sum(self.SAM_train_num_templates[:num_things_classes])
                            offset_text_idx = num_synonyms_to_skip * len(self.SAM_PROMPT)
                            
                            # 2. 截断 class_names (输入给 Text Encoder 的文本列表)
                            self.SAM_train_class_names = self.SAM_train_class_names[offset_text_idx:]
                            
                            # 3. 截断 num_templates (用于后续 Logit 聚合的计数列表)
                            self.SAM_train_num_templates = self.SAM_train_num_templates[num_things_classes:]
                    # ---------------- 修改结束 ----------------

                        # ======= 新增打印验证逻辑 =======
                        # print(f"\n" + "="*40)
                        # print(f"DEBUG: 数据集 [{dataname}] 类别过滤完成")
                        # print(f"剩余类别数量 (num_classes): {len(self.train_num_templates)}")
                        
                        # # 提取每个类别的第一个模板名进行展示
                        # display_names = []
                        # current_idx = 0
                        # for num_t in self.train_num_templates:
                        #     # 取该类别的第一个 prompt 作为代表名
                        #     display_names.append(self.train_class_names[current_idx])
                        #     current_idx += num_t
                        
                        # print("前 10 个 Stuff 类别示例:")
                        # for i, name in enumerate(display_names[:10]):
                        #     print(f"  Class {i}: {name}")
                        
                        # print("最后 5 个 Stuff 类别示例:")
                        # for i, name in enumerate(display_names[-5:]):
                        #     print(f"  Class {len(display_names)-5+i}: {name}")
                        # print("="*40 + "\n")
                        # ===============================

                    # --- 原有生成逻辑开始 ---
                    text_classifier = []
                    text_feat = []
                    language_mask = []
                    # this is needed to avoid oom, which may happen when num of class is large
                    bs = 128
                    print("Generating text classifier for", dataname, "with", len(self.SAM_train_class_names), "classes.")
                    for idx in range(0, len(self.SAM_train_class_names), bs):
                        state_text = self.detector.backbone.forward_text(self.SAM_train_class_names[idx:idx+bs], device=self.device)

                        batch_text_feat = state_text["language_features"].detach()
                        mask = state_text["language_mask"] # bs, L
                        batch_text_feat = batch_text_feat.permute(1,0,2) # -> bs, L, D 
                        if self.use_pe_text:
                            text_classifier.append(state_text["pe_text_out"]["pe_textfeat"])
                        else:    
                            text_classifier.append(batch_text_feat)
                        text_feat.append(batch_text_feat)
                        language_mask.append(mask) # bs, L
                    text_classifier = torch.cat(text_classifier, dim=0)
                    text_feat = torch.cat(text_feat, dim=0)
                    language_mask = torch.cat(language_mask, dim=0) # (num_names * self.PROMPT,  L)
                    # average across templates and normalization.
                    text_feat = text_feat.reshape(text_feat.shape[0]//len(self.SAM_PROMPT), len(self.SAM_PROMPT), text_feat.shape[-2], text_feat.shape[-1]) # num_names, self.PROMPT, L, D
                    text_feat /= (text_feat.norm(dim=-1, keepdim=True) + 1e-6)
                    text_feat[language_mask.view(text_feat.shape[0],text_feat.shape[1],text_feat.shape[2])] = 0.0 # [num_names, self.PROMPT, L, D] 掩码掉 padding 部分
                    
                    language_features = text_feat.mean(1) # num_names, L, D

                    text_classifier = text_classifier.reshape(text_classifier.shape[0]//len(self.SAM_PROMPT), len(self.SAM_PROMPT), text_classifier.shape[-2], text_classifier.shape[-1]) # num_names, self.PROMPT, L, D
                    text_classifier /= (text_classifier.norm(dim=-1, keepdim=True) + 1e-6)
                    text_classifier[language_mask.view(text_classifier.shape[0],text_classifier.shape[1],text_classifier.shape[2])] = 0.0 # [num_names, self.PROMPT, L, D] 掩码掉 padding 部分
                    text_classifier = text_classifier.mean(-2) 
                    text_classifier /= (text_classifier.norm(dim=-1, keepdim=True) + 1e-6)
                    text_classifier = text_classifier.mean(1)
                    text_classifier /= (text_classifier.norm(dim=-1, keepdim=True) + 1e-6)
                    
                    self.SAM_language_features = language_features.detach() # num_names , L, D
                    self.SAM_language_mask = torch.min(language_mask.view(language_features.shape[0],len(self.SAM_PROMPT),language_features.shape[1]), dim=1).values# [num_names, L]
                    self.SAM_train_text_classifier = text_classifier.detach()
                    # --- 原有生成逻辑结束 ---

                    # 【新增】2. 将生成的特征移至 CPU 并存入缓存
                    self.SAM_text_encoder_cache[dataname] = {
                        "language_features": self.SAM_language_features.cpu(),
                        "language_mask": self.SAM_language_mask.cpu(),
                        "text_classifier": self.SAM_train_text_classifier.cpu(),
                        "num_templates": self.SAM_train_num_templates, # 缓存当前的 List
                        "class_names": self.SAM_train_class_names      # 缓存当前的 List
                    }

                self.SAM_train_dataname = dataname
            return self.SAM_train_text_classifier.clone(), self.SAM_train_num_templates
        else:
            if self.SAM_test_dataname != dataname:
                self.SAM_category_overlapping_mask, self.SAM_test_num_templates, self.SAM_test_class_names = self.prepare_class_names_from_metadata(self.test_metadata[dataname], self.train_metadata, self.SAM_PROMPT)
                text_classifier = []
                text_feat = []
                language_mask = []
                # this is needed to avoid oom, which may happen when num of class is large
                bs = 128
                print("Generating text classifier for", dataname, "with", len(self.SAM_test_class_names), "classes.")
                for idx in range(0, len(self.SAM_test_class_names), bs):
                    state_text = self.detector.backbone.forward_text(self.SAM_test_class_names[idx:idx+bs], device=self.device)

                    batch_text_feat = state_text["language_features"].detach()
                    mask = state_text["language_mask"] # bs, L
                    batch_text_feat = batch_text_feat.permute(1,0,2) # -> bs, L, D 
                    if self.use_pe_text:
                        text_classifier.append(state_text["pe_text_out"]["pe_textfeat"])
                    else:    
                        text_classifier.append(batch_text_feat)
                    text_feat.append(batch_text_feat)
                    language_mask.append(mask) # bs, L
                text_classifier = torch.cat(text_classifier, dim=0)
                text_feat = torch.cat(text_feat, dim=0)
                language_mask = torch.cat(language_mask, dim=0) # (num_names * self.PROMPT,  L)
                # average across templates and normalization.
                text_feat = text_feat.reshape(text_feat.shape[0]//len(self.SAM_PROMPT), len(self.SAM_PROMPT), text_feat.shape[-2], text_feat.shape[-1]) # num_names, self.PROMPT, L, D
                text_feat /= (text_feat.norm(dim=-1, keepdim=True) + 1e-6)
                text_feat[language_mask.view(text_feat.shape[0],text_feat.shape[1],text_feat.shape[2])] = 0.0 # [num_names, self.PROMPT, L, D] 掩码掉 padding 部分
                
                language_features = text_feat.mean(1) # num_names, L, D

                text_classifier = text_classifier.reshape(text_classifier.shape[0]//len(self.SAM_PROMPT), len(self.SAM_PROMPT), text_classifier.shape[-2], text_classifier.shape[-1]) # num_names, self.PROMPT, L, D
                text_classifier /= (text_classifier.norm(dim=-1, keepdim=True) + 1e-6)
                text_classifier[language_mask.view(text_classifier.shape[0],text_classifier.shape[1],text_classifier.shape[2])] = 0.0 # [num_names, self.PROMPT, L, D] 掩码掉 padding 部分
                text_classifier = text_classifier.mean(-2) 
                text_classifier /= (text_classifier.norm(dim=-1, keepdim=True) + 1e-6)
                text_classifier = text_classifier.mean(1)
                text_classifier /= (text_classifier.norm(dim=-1, keepdim=True) + 1e-6)
                
                self.SAM_language_features = language_features.detach() # num_names , L, D
                self.SAM_language_mask = torch.min(language_mask.view(language_features.shape[0],len(self.SAM_PROMPT),language_features.shape[1]), dim=1).values# [num_names, L]
                self.SAM_test_text_classifier = text_classifier.detach()
                self.SAM_test_dataname = dataname
            return self.SAM_test_text_classifier.clone(), self.SAM_test_num_templates


    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):

        if self.training and self.new_pool_decoder:
            if not self.pooling_decoder_synced:
                # 加载权重
                print("Initializing Pooling Decoder weights from Pre-trained Mask Decoder...")
                load_partial_weights(
                    self.pooling_decoder, 
                    self.detector.transformer.decoder.state_dict()
                )
                self.pooling_decoder_synced = True

        images = [x["image"].to(self.device) for x in batched_inputs]
        # print("shape of first image:", images[0].shape)
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, 16)
        # print("shape of images.tensor:", images.tensor.shape)
        img_h, img_w = images.tensor.shape[-2:]

        bs = images.tensor.shape[0]
        
        self.find_stage = FindStage(
            img_ids=torch.arange(bs, device=self.device, dtype=torch.long),
            text_ids=torch.arange(bs, device=self.device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )
        with torch.no_grad():

            file_names = [x["file_name"] for x in batched_inputs]
            file_names = [x.split('/')[-1].split('.')[0] for x in file_names]
            
            if 'meta' in batched_inputs[0]:
                meta = batched_inputs[0]["meta"]
            else:
                meta = batched_inputs[0]
            
            # print("keys of meta:", meta.keys())
            dataname = get_dataname(batched_inputs[0])
            
            # 图形特征
            backbone_out_vision = self.detector.backbone.forward_image(images.tensor)
            img_feat = backbone_out_vision["vision_features"].detach() # bs, D, H', W'
            backbone_fpn = backbone_out_vision["backbone_fpn"]
            for k in range(len(backbone_fpn)):
                backbone_fpn[k] = backbone_fpn[k].detach()

            SAM_text_classifier, SAM_num_templates = self.get_SAM_text_classifier(dataname)
            if self.void_embedding is not None:
                SAM_text_classifier = torch.cat([SAM_text_classifier, F.normalize(self.void_embedding.weight, dim=-1)], dim=0)

            # others
            geometric_prompt = self.detector._get_dummy_prompt(bs)
        
        if self.use_cdt:
            raise NotImplementedError("CDT is not implemented yet.")
            # for layer in self.cdt:
            #     text_classifier = layer(img_feat,text_classifier) # 逐层通过

        batch_gt_names_idx = []
        for i in range(bs):
            # gt_classes = get_gt_labels_from_sem_seg(batched_inputs[i]["sem_seg"].to(self.device))

            # === 修改开始：适配 Objects365 的 Instances 格式 ===
            gt_classes = []
            if "instances" in batched_inputs[i]:
                # 从实例中提取去重后的类别 ID
                if len(batched_inputs[i]["instances"]) > 0:
                    gt_classes = batched_inputs[i]["instances"].gt_classes.unique().cpu().tolist()
            elif "sem_seg" in batched_inputs[i]:
                # 兼容旧的 COCO Stuff 逻辑
                gt_classes = get_gt_labels_from_sem_seg(batched_inputs[i]["sem_seg"].to(self.device))
            # === 修改结束 ===

            gt_names_idx = []
            cur_idx = 0
            for i,num_t in enumerate(SAM_num_templates): 
                if i in gt_classes:
                    gt_names_idx += list(range(cur_idx, cur_idx + num_t))
                cur_idx += num_t
            batch_gt_names_idx.append(gt_names_idx)

        # =======================================================
        
        language_features_input = []

        # USE_GT_NAMES_ONLY = True
        USE_GT_NAMES_ONLY = False
        
        if USE_GT_NAMES_ONLY:
            language_features_input = [self.language_features[batch_gt_names_idx[i],:,:] for i in range(bs)]
            language_features_input = torch.cat(language_features_input, dim=0) # (bs, num_names * L, dim)
            language_mask_input = [self.language_mask[batch_gt_names_idx[i],:] for i in range(bs)]
            language_mask_input = torch.cat(language_mask_input, dim=0) # (bs, num_names * L)
            if bs == 1:
                language_features_input = language_features_input.unsqueeze(0)
                language_mask_input = language_mask_input.unsqueeze(0)

        else:
            language_features_input = self.SAM_language_features.expand(bs, -1, -1, -1) # (bs, num_names, L, dim)
            language_mask_input = self.SAM_language_mask.expand(bs, -1, -1) # (bs, num_names, L)

        # language_features_input = self.text_feat_resizer(language_features_input) # (bs, num_names, L, 1024) -> (bs, num_names, L, 256)

        # print("shape of input:",language_features_input.shape, language_mask_input.shape)
        language_features_input = language_features_input.reshape(bs, -1, language_features_input.shape[-1]) # (bs, num_names * L, dim)
        language_mask_input = language_mask_input.reshape(bs, -1) # (bs, num_names * L)
        # print("shape of input after reshape:",language_features_input.shape, language_mask_input.shape)
        

        backbone_out={
            "img_batch_all_stages": img_feat,
            "vision_pos_enc": backbone_out_vision["vision_pos_enc"],
            "backbone_fpn": backbone_fpn,
            "language_features": language_features_input.permute(1, 0, 2), # (num_names * L, bs, dim)
            "language_mask": language_mask_input, # bs, (num_names * L)
        }

        #=================================
        with torch.set_grad_enabled(self.train_mask):

            find_input = self.find_stage

            with torch.profiler.record_function("SAM3Image._encode_prompt"):
                prompt, prompt_mask, backbone_out = self.detector._encode_prompt(
                    backbone_out, find_input, geometric_prompt
                )
            # Run the encoder
            with torch.profiler.record_function("SAM3Image._run_encoder"):
                backbone_out, encoder_out, _ = self.detector._run_encoder(
                    backbone_out, find_input, prompt, prompt_mask
                )

            fusion_feat = encoder_out["encoder_hidden_states"] # H'*W', bs, D
            fusion_feat = fusion_feat.permute(1,0,2) # bs, H'*W', D
            if self.use_cos_sim:
                fusion_feat = F.normalize(fusion_feat, dim=-1)


            out = {
                "encoder_hidden_states": encoder_out["encoder_hidden_states"],
                "prev_encoder_out": {
                    "encoder_out": encoder_out,
                    "backbone_out": backbone_out,
                },
            }
            # print("keys of out before decoder:", out.keys()) # s(['encoder_hidden_states', 'prev_encoder_out'])
            # Run the decoder
            with torch.profiler.record_function("SAM3Image._run_decoder"):            
                # out, hs = self.detector._run_decoder(
                #     memory=out["encoder_hidden_states"],
                #     pos_embed=encoder_out["pos_embed"],
                #     src_mask=encoder_out["padding_mask"],
                #     out=out,
                #     prompt=prompt,
                #     prompt_mask=prompt_mask,
                #     encoder_out=encoder_out,
                # )
                
                query_embed = self.detector.transformer.decoder.query_embed.weight
                query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

                hs, reference_boxes, dec_presence_out, dec_presence_feats, cross_attn_weights = (
                    self.detector.transformer.decoder(
                        tgt=query_embed,
                        memory=out["encoder_hidden_states"],
                        memory_key_padding_mask=encoder_out["padding_mask"],
                        pos=encoder_out["pos_embed"],
                        reference_boxes=None,
                        level_start_index=encoder_out["level_start_index"],
                        spatial_shapes=encoder_out["spatial_shapes"],
                        valid_ratios=encoder_out["valid_ratios"],
                        tgt_mask=None,
                        memory_text=prompt,
                        text_attention_mask=prompt_mask,
                        apply_dac=False,

                        use_presence_token = False,
                    )
                )

                hs = hs.transpose(1, 2)  # seq-first to batch-first
                reference_boxes = reference_boxes.transpose(1, 2)  # seq-first to batch-first
                if dec_presence_out is not None:
                    # seq-first to batch-first
                    dec_presence_out = dec_presence_out.transpose(1, 2)

                out["presence_feats"] = dec_presence_feats
                self.detector._update_scores_and_boxes(
                    out,
                    hs,
                    reference_boxes,
                    prompt,
                    prompt_mask,
                    dec_presence_out=dec_presence_out,
                )


            # print("keys of out after decoder:", out.keys()) # (['encoder_hidden_states', 'prev_encoder_out', 'presence_feats', 'queries', 'presence_logit_dec', 'pred_logits', 'pred_boxes', 'pred_boxes_xyxy'])
            # Run segmentation heads
            with torch.profiler.record_function("SAM3Image._run_segmentation_heads"):
                self.detector._run_segmentation_heads(
                    out=out,
                    backbone_out=backbone_out,
                    img_ids=find_input.img_ids,
                    vis_feat_sizes=encoder_out["vis_feat_sizes"],
                    encoder_hidden_states=out["encoder_hidden_states"],
                    prompt=prompt,
                    prompt_mask=prompt_mask,
                    hs=hs,
                    aux_masks=True,
                )
            
            # if self.detector.training or self.detector.num_interactive_steps_val > 0:
            #     self.detector._compute_matching(out, self.detector.back_convert(find_target))

            #========================================
            outputs = out
            # print("outputs keys:", outputs.keys())
            # print('aux:',outputs['aux_outputs'][0].keys())
            # print('aux box:',outputs['aux_outputs'][0]['pred_boxes'].shape)
            # print('aux pred_boxes_xyxy:',outputs['aux_outputs'][0]['pred_boxes_xyxy'].shape)

            if self.training:
                # mask classification target
                if "instances" in batched_inputs[0]:
                    gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                    targets = self.prepare_targets(gt_instances, images, batched_inputs)
                else:
                    targets = None
            else:
                targets = self._prepare_targets_from_sem_seg(batched_inputs, images)

            out_masks = outputs["pred_masks"].clone()

            out_masks = out_masks.sigmoid()
            bs, N, H, W = out_masks.shape

            # presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1) # 在多类别情况下认为失效

            # out_semseg = outputs["semantic_seg"] # 原语义分割头输出，舍去
            # out_semseg = F.interpolate(
            #     out_semseg,
            #     size=(img_h, img_w),
            #     mode="bilinear",
            #     align_corners=False,
            # ).sigmoid()


            # print("out_masks shape:", out_masks.shape, "out_probs shape:", out_probs.shape, "out_semseg shape:", out_semseg.shape, "presence_score shape:", presence_score.shape)
            # out_masks shape: torch.Size([1, 200, 1008, 1008]) out_probs shape: torch.Size([1, 200]) out_semseg shape: torch.Size([1, 1, 1008, 1008]) presence_score shape: torch.Size([1, 1, 1])
            pred_boxes = outputs['pred_boxes']
            pred_boxes_xyxy = outputs['pred_boxes_xyxy']

            C_ = SAM_text_classifier.shape[0] # num_names 

            queries_masks = out_masks # out_probs是通过与池化prompt投影卷积实现的，多类别下失效，直接用原始mask_logits

            queries = outputs["obj_queries"] # 6, bs, N, D
            
            # instance_embeds = outputs["instance_embeds"] 

            use_aux = self.use_aux and self.training
            aux_outputs = []

            obj_logits = None
            if self.new_score_head:
                obj_logits = self.score_head(queries).squeeze(-1)

            if self.use_attnpool:
                radio_img_feat = backbone_out_vision['vit_feature'][0]["siglip2-g"]["features"]
                radio_img_feat = radio_img_feat.view(bs, 1536, -1).permute(0,2,1) # bs, l, d
                
                if self.new_pool_decoder:
                    if self.add_radio_feat:
                        decoder_img_feat = radio_img_feat + self.fusion_feat_proj(out["encoder_hidden_states"].permute(1,0,2))
                        decoder_img_feat = self.radio_feat_proj(decoder_img_feat) 
                        decoder_img_feat = decoder_img_feat.permute(1,0,2) # l, bs, d
                    else:
                        decoder_img_feat = out["encoder_hidden_states"]
                    # decoder_img_feat = F.normalize(decoder_img_feat, dim=-1)

                    pixel_embed = outputs["pixel_embed"] 
                    pooled_pixel_embed = self.mask_pooling(pixel_embed, outputs["pred_masks"])
                    # cls_queries = self.detector.transformer.decoder.presence_token.weight.unsqueeze(1).expand(bs, N, -1)
                    cls_queries = pooled_pixel_embed
                    cls_queries = cls_queries.permute(1,0,2)

                    class_tokens, _, _, _, cross_attn_weights = (
                        self.pooling_decoder(
                            tgt=cls_queries,
                            memory=decoder_img_feat,
                            memory_key_padding_mask=encoder_out["padding_mask"],
                            pos=encoder_out["pos_embed"],
                            reference_boxes=pred_boxes.permute(1, 0, 2),
                            level_start_index=encoder_out["level_start_index"],
                            spatial_shapes=encoder_out["spatial_shapes"],
                            valid_ratios=encoder_out["valid_ratios"],
                            tgt_mask=None,
                            memory_text=prompt,
                            text_attention_mask=prompt_mask,
                            apply_dac=False,

                            use_presence_token = False,
                            fixed_reference_boxes = True
                        )
                    )

                    # queries = class_tokens.permute(0,2,1,3)
                    class_tokens = class_tokens[-1:,...]
                    class_tokens = F.normalize(class_tokens, dim=-1)
                    cross_attn_weights = torch.einsum("knbd, lbd->kbnl", class_tokens, decoder_img_feat)
                    cross_attn_weights *= torch.clamp(self.attn_pool_logit_scale.exp(),100)
                    cross_attn_weights = cross_attn_weights.softmax(dim=-1) # k, bs, N, L

                
                pooled_img_feat = torch.einsum("bld, kbnl->kbnd", radio_img_feat, cross_attn_weights) # k, bs, N, D

                text_classifier, num_templates = self.get_text_classifier(dataname)
                
                attn_cls_results = get_classification_logits(pooled_img_feat.reshape(-1 , N, pooled_img_feat.shape[-1]), text_classifier, self.out_vocab_logit_scale, num_templates)

                attn_cls_results = attn_cls_results.view(-1, bs, N, attn_cls_results.shape[-1])

            for i in range(6):
                assert queries.shape[0] == 6
                assert queries.shape[2] == N
                if use_aux or i == 5 :
                    tp_queries = queries[i,:,:,:].clone() 

                    if self.add_pixelfeat:
                        pixel_embed = outputs["pixel_embed"] # bs, D, H', W'

                        pooled_pixel_embed = self.mask_pooling(pixel_embed, outputs["pred_masks"] if i ==5 else outputs['aux_outputs'][i]["pred_masks"])
                        tp_queries = tp_queries + pooled_pixel_embed

                    cur_obj_logits = obj_logits[i] if obj_logits is not None else None

                    if self.use_query_proj:
                        tp_queries = self.query_proj(tp_queries)


                    if self.use_cos_sim:
                        tp_queries = F.normalize(tp_queries, dim=-1, p=2)


                    if self.use_cdt:
                        query_names_results = torch.einsum("bnd,bcd->bnc", tp_queries, SAM_text_classifier) # bs, N, C
                    else:
                        query_names_results = torch.einsum("bnd,cd->bnc", tp_queries, SAM_text_classifier) # bs, N, C
                    
                    if self.use_cos_sim:
                        cur_logit_scale = self.logit_scale.exp()
                        cur_logit_scale = torch.clamp(cur_logit_scale, max=100.0)
                        query_names_results = cur_logit_scale * query_names_results
                        if self.logit_bias is not None:
                            cur_logit_bias = self.logit_bias
                            query_names_results = query_names_results + cur_logit_bias
                    else:
                        if self.logit_bias is not None:
                            cur_logit_bias = self.logit_bias
                            query_names_results = query_names_results + self.logit_bias

                    query_cls_results= []
                    cur_idx = 0

                    tp_num_templates = SAM_num_templates

                    for num_t in tp_num_templates: 
                        query_cls_results.append(query_names_results[:,:, cur_idx: cur_idx + num_t].max(-1).values)
                        cur_idx += num_t
                    

                    if self.void_embedding is not None:
                        query_cls_results.append(query_names_results[:,:, -1])
                        query_cls_results = torch.stack(query_cls_results, dim=-1)
                        
                        if self.use_attnpool and self.training:
                            idx_attn_cls = -1 if attn_cls_results.shape[0] == 1 else i
                            attn_cls_prob = F.softmax(attn_cls_results[idx_attn_cls],dim=-1)
                            # attn_cls_prob = torch.cat([attn_cls_prob * (1.0 - is_void_prob), is_void_prob], dim =-1)

                    else:
                        query_cls_results = torch.stack(query_cls_results, dim=-1) # bs, N, num_classes

                    if i<5:
                        aux_out = {
                            'pred_logits': query_cls_results, 
                            'pred_masks': outputs['aux_outputs'][i]["pred_masks"], 
                            'pred_boxes': outputs['aux_outputs'][i]['pred_boxes'],
                            'pred_boxes_xyxy': outputs['aux_outputs'][i]["pred_boxes_xyxy"],
                        }
                        if cur_obj_logits is not None:
                            aux_out['pred_objectness_logits'] = cur_obj_logits
                        if self.use_attnpool and self.training:
                            aux_out["attn_cls_logits"] = torch.log(attn_cls_prob + 1e-8)
                        
                        aux_outputs.append(aux_out)
                    else:
                        query_cls_results_final = query_cls_results
                        obj_logits_final = cur_obj_logits


        if self.training:
            losses = {}

            if self.train_mask:
                criterion_pred = {
                    'pred_logits': query_cls_results_final,
                    'pred_masks': outputs["pred_masks"],
                    'pred_boxes': outputs['pred_boxes'],
                    'pred_boxes_xyxy': outputs["pred_boxes_xyxy"],
                    'aux_outputs': aux_outputs if use_aux is True else None,
                }
                if obj_logits_final is not None:
                    criterion_pred['pred_objectness_logits'] = obj_logits_final
                if self.use_attnpool and self.training:
                    criterion_pred["attn_cls_logits"] = torch.log(attn_cls_prob + 1e-8)

                fcclip_losses = self.criterion(criterion_pred, targets)


                for k in list(fcclip_losses.keys()):
                    # print("loss:", k, losses[k].item())
                    if k in self.criterion.weight_dict:
                        fcclip_losses[k] *= self.criterion.weight_dict[k]
                    else:
                        # remove this loss if not specified in `weight_dict`
                        fcclip_losses.pop(k)

                losses.update(fcclip_losses)

            # if self.use_attnpool:
            #     attnpool_losses = {}
            #     targets, _ , _ = self.prepare_targets_for_maskadapter(gt_instances, images)

            #     img_feat_for_pool = backbone_out_vision['vit_feature'][0]["siglip2-g"]["features"]          

            #     assert cross_attn_weights.shape[0] == 6
            #     for i in range(6):
            #         if self.aux_attn_pool or i == 5:

            #             mask_pred_results = outputs['aux_outputs'][i]["pred_masks"] if i < 5 else outputs["pred_masks"]
                        
            #             _, matched_src_cls, _, mask_labels = self.match_via_iou(mask_pred_results, attn_cls_results[i], targets, iou_threshold=self.iou_threshold, max_matches=self.num_pred_masks)

            #             loss_attn_ce = self.cross_entropy_loss(matched_src_cls, mask_labels)["loss_ce"]
            #             if i == 5:
            #                 attnpool_losses["loss_attn_ce"] = loss_attn_ce
            #             else:
            #                 attnpool_losses[f"loss_attn_ce_{i}"] = loss_attn_ce

            #     for k in list(attnpool_losses.keys()):
            #         if k in self.attnpool_weight_dict:
            #             attnpool_losses[k] *= self.attnpool_weight_dict[k]
            #         else:
            #             attnpool_losses.pop(k)

            #     losses.update(attnpool_losses)

            # loss排序
            all_keys = list(losses.keys())
            aux_suffixes = [f"_{i}" for i in range(5)]
            main_keys = sorted([k for k in all_keys if not any(k.endswith(s) for s in aux_suffixes)])
            aux_keys = sorted([k for k in all_keys if any(k.endswith(s) for s in aux_suffixes)])

            ordered_losses = {}
            for k in main_keys:
                ordered_losses[k] = losses[k]
            for k in aux_keys:
                ordered_losses[k] = losses[k]

            return ordered_losses
        
        else:
                
            # ==========================================
            # 2. 提取特征
            # ==========================================
            # 释放不再需要的显存大户
            if 'backbone_out' in locals(): del backbone_out
            if 'encoder_out' in locals(): del encoder_out
            if 'fusion_feat' in locals(): del fusion_feat
            if 'backbone_fpn' in locals(): del backbone_fpn
            # 强制清理 CUDA 缓存（虽然有开销，但能有效防止 OOM）
            torch.cuda.empty_cache()

            if self.use_attnpool:
                img_feat_for_pool = backbone_out_vision['vit_feature'][0]["siglip2-g"]["features"]
                text_classifier, num_templates = self.get_text_classifier(dataname)
                
                attn_weights = cross_attn_weights[-1]

                pooled_img_feat = torch.einsum("bdl, bnl->bnd", img_feat_for_pool.view(bs, img_feat_for_pool.shape[1],-1), attn_weights)

                pool_cls_logits = get_classification_logits(pooled_img_feat, text_classifier, self.out_vocab_logit_scale, num_templates)
                            


            else:
                img_feat_for_pool = backbone_out_vision['vit_feature'][0]["siglip2-g"]["features"]
                mask_for_pool = F.interpolate(outputs["pred_masks"], size=img_feat_for_pool.shape[-2:],
                                                    mode='bilinear', align_corners=False)
                pooled_img_feat = self.mask_pooling(img_feat_for_pool, mask_for_pool)
                pooled_img_feat = F.normalize(pooled_img_feat, dim=-1, p=2)


                            
                text_classifier, num_templates = self.get_text_classifier(dataname)

                maskpool_name_logits = torch.einsum("cd,bnd->bnc", text_classifier, pooled_img_feat) 

                maskpool_name_logits = maskpool_name_logits * 100
                
                pool_cls_logits = aggregate_name_to_class_logits(maskpool_name_logits, num_templates)

            if self.use_softmax:
                is_void_prob = F.softmax(query_cls_results_final, dim=-1)[..., -1:]
                in_vocab_cls_results = query_cls_results_final[..., :-1] 
                in_vocab_cls_probs = in_vocab_cls_results.softmax(-1)
                out_vocab_cls_probs = pool_cls_logits.softmax(-1)
            else:
                in_vocab_cls_probs = torch.sigmoid(query_cls_results_final)
                out_vocab_cls_probs = F.softmax(pool_cls_logits, dim=-1)
            category_overlapping_mask = self.SAM_category_overlapping_mask.to(self.device)
            alpha = self.alpha
            beta = self.beta
            # 为了数值稳定性，加一个小 epsilon
            eps = 1e-7 

            # 计算 Seen 类的加权几何平均概率
            probs_seen = (
                (in_vocab_cls_probs + eps) ** (1 - alpha) * 
                (out_vocab_cls_probs + eps) ** alpha
            )
            
            # 计算 Unseen 类的加权几何平均概率
            probs_unseen = (
                (in_vocab_cls_probs + eps) ** (1 - beta) * 
                (out_vocab_cls_probs + eps) ** beta
            )

            ensemble_logits = (
                probs_seen.log() * category_overlapping_mask + 
                probs_unseen.log() * (1 - category_overlapping_mask)
            )

            final_probs = torch.cat([
                ensemble_logits.softmax(-1) * (1.0 - is_void_prob), 
                is_void_prob
            ], dim=-1)

            query_cls_results_final = torch.log(final_probs + 1e-8)



            # ====================== Oracle 逻辑 ====================== 

            if self.OracleSelect_on:
                # 临时构造 outputs 字典，用于传递给 oracle 函数
                temp_outputs = {
                    "pred_masks": outputs["pred_masks"],
                    "pred_logits": query_cls_results_final 
                }
                
                # 获取为每个 prediction 分配了最佳 GT 类别的 oracle logits
                oracle_logits = self.get_oracle_logits_per_prediction(
                    temp_outputs, 
                    batched_inputs, 
                    images
                )

                oracle_probs = F.softmax(oracle_logits, dim=-1)
                
                final_oracle_probs = oracle_probs * (1.0 - is_void_prob)
                final_oracle_probs[..., -1] += is_void_prob.squeeze(-1)

                query_cls_results_final = torch.log(final_oracle_probs + 1e-8)
        # =======================================================
        
            mask_cls_logits = query_cls_results_final # 保持 Logits 状态
            mask_pred_logits = outputs["pred_masks"]  # 保持 Logits 状态


            results = []

            VISUALIZE_ATTENTION = False
            # VISUALIZE_ATTENTION = True
            VIS_SAVE_ROOT = "./attnmap"
    
            
            for i in range(bs):

                if VISUALIZE_ATTENTION:
                    self.visualize_query_attention(
                        batched_inputs=batched_inputs,
                        images=images,
                        outputs=outputs,
                        targets=targets,
                        is_void_prob=is_void_prob, 
                        out_vocab_cls_probs=out_vocab_cls_probs,
                        cross_attn_weights = cross_attn_weights,
                        dino_pixel_feat = img_feat_for_pool,
                        save_root=VIS_SAVE_ROOT
                    )


                mask_cls_i = mask_cls_logits[i]       # [Q, C]
                mask_pred_i = mask_pred_logits[i]     # [Q, H, W]

                # 获取原始图像尺寸
                img_h_orig = batched_inputs[i]["height"]
                img_w_orig = batched_inputs[i]["width"]

                mask_pred_i = F.interpolate(
                    mask_pred_i.unsqueeze(0), 
                    size=(img_h_orig, img_w_orig), 
                    mode="bilinear", 
                    align_corners=False
                ).squeeze(0)

                res = {}

                # --- A. 语义分割 (Semantic Segmentation) ---
                if self.semantic_on:
                    # 使用你原来的逻辑，但注意输入变成了 logits
                    mask_cls_prob = F.softmax(mask_cls_i, dim=-1)[..., :-1]
                    mask_pred_prob = mask_pred_i.sigmoid()
                    semseg = torch.einsum("qc,qhw->chw", mask_cls_prob, mask_pred_prob)
                    res["sem_seg"] = semseg

                    # =========== 修改开始：为可视化准备 Square 数据 ===========
                    
                    # 1. 获取当前输入 Tensor 的尺寸 (即正方形尺寸，例如 1024x1024)
                    # batched_inputs[i]["image"] 是经过 mapper 处理后的图
                    tensor_h, tensor_w = batched_inputs[i]["image"].shape[-2:]
                    
                    # 2. 将 Logits 插值到 Tensor 尺寸 (而不是原图尺寸)
                    # 注意：这里要用原始的 mask_pred_logits[i]，不要用已经 resize 过的 mask_pred_i
                    mask_pred_i_square = F.interpolate(
                        mask_pred_logits[i].unsqueeze(0), 
                        size=(tensor_h, tensor_w), 
                        mode="bilinear", 
                        align_corners=False
                    ).squeeze(0).sigmoid()
                    
                    # 3. 计算 Square 的语义分割结果
                    semseg_square = torch.einsum("qc,qhw->chw", mask_cls_prob, mask_pred_i_square)
                    pred_result_square = semseg_square.argmax(0).cpu()

                    # 6. 获取类别名称
                    current_dataname = batched_inputs[i]["meta"]["dataname"]
                    if current_dataname in self.test_metadata:
                        meta = self.test_metadata[current_dataname]
                    else:
                        meta = MetadataCatalog.get(current_dataname)
                    
                    try:
                        current_class_names = meta.stuff_classes
                    except:
                        current_class_names = meta.thing_classes

                    # 7. 绘图 (全部传入 Square 的数据)
                    visualize_semantic = False
                    # visualize_semantic = True

                    if visualize_semantic:
                        visualize_segmentation(
                            pred_result=pred_result_square,       # 修改点：传入 Square 预测
                            gt_result= batched_inputs[i]["sem_seg"].to(self.device),           # 本身就是 Square
                            class_names=current_class_names + ['background'],
                            original_image_tensor=batched_inputs[i]["image"], # 本身就是 Square
                            save_path=f"./show_semantic/{batched_inputs[i]['file_name'].split('/')[-1].split('.')[0]}.png"
                        )
                #     # =========== 修改结束 ===========

                # --- B. 全景分割 (Panoptic Segmentation) ---
                if self.panoptic_on:
                    excluded_datasets = ["lvis_v1_val", "lvis_v1_train"]
                    
                    if dataname not in excluded_datasets:
                        panoptic_seg, segments_info = self.panoptic_inference(
                            mask_cls_i, mask_pred_i, dataname
                        )
                        res["panoptic_seg"] = (panoptic_seg, segments_info)
                
                # --- C. 实例分割 (Instance Segmentation) ---
                if self.instance_on:
                    instances = self.instance_inference(
                        mask_cls_i, mask_pred_i, dataname
                    )
                    res["instances"] = instances

                results.append(res)

            return results


    @torch.no_grad()
    def get_oracle_logits_per_prediction(self, outputs, batched_inputs, images):
        """
        为每一个预测(Prediction)找到最匹配的真值(Ground Truth)，并将GT的类别作为该预测的类别。
        匹配仅基于 Mask 的相似度 (Dice Score)。

        Args:
            outputs (dict): 模型输出，包含 'pred_masks'。
            batched_inputs (list[dict]): 模型输入，用于生成 GT。
            images (ImageList): 处理后的图像张量，用于获取尺寸。

        Returns:
            torch.Tensor: "Oracle" Logits, 形状与原始 logits 相同。
        """
        # 1. 从 sem_seg 准备 Ground Truth Targets
        # targets 是一个 list, 每个元素是一个 dict{'labels': [G], 'masks': [G, H, W_in]}
        targets = self._prepare_targets_from_sem_seg(batched_inputs, images)

        # 2. 准备 Predictions
        # 将 mask logits 转换为概率 [B, Q, H_pred, W_pred]
        pred_masks_prob = outputs["pred_masks"].sigmoid()
        batch_size, num_queries, h_pred, w_pred = pred_masks_prob.shape # <-- 获取预测的 H, W

        # 3. 初始化最终的 Oracle Logits
        oracle_logits = torch.full_like(outputs["pred_logits"], -100.0)

        # 4. 逐个图像进行处理
        for i in range(batch_size):
            gt_masks = targets[i]["masks"]      # [G, H_in, W_in]
            gt_labels = targets[i]["labels"]    # [G]
            pred_masks = pred_masks_prob[i]     # [Q, H_pred, W_pred]

            num_gt = gt_masks.shape[0]
            if num_gt == 0:
                continue

            # <<< FIX START >>>
            # 将 GT mask 插值到与 Prediction mask 完全相同的分辨率
            if gt_masks.shape[-2:] != (h_pred, w_pred):
                gt_masks_resized = F.interpolate(
                    gt_masks.unsqueeze(1),       # 添加 channel dim -> [G, 1, H_in, W_in]
                    size=(h_pred, w_pred),       # 目标尺寸为预测尺寸
                    mode="nearest"               # 对 mask 使用最近邻插值
                ).squeeze(1)                     # 移除 channel dim -> [G, H_pred, W_pred]
            else:
                gt_masks_resized = gt_masks
            # <<< FIX END >>>

            # 5. 计算 Dice 相似度矩阵
            pred_masks_flat = pred_masks.flatten(1)           # [Q, H_pred*W_pred]
            gt_masks_flat = gt_masks_resized.flatten(1)       # [G, H_pred*W_pred] <- 使用resize后的GT

            intersection = torch.einsum("qh,gh->qg", pred_masks_flat, gt_masks_flat)
            pred_sum = pred_masks_flat.sum(dim=1)
            gt_sum = gt_masks_flat.sum(dim=1)
            
            dice_scores = (2.0 * intersection) / (pred_sum[:, None] + gt_sum[None, :] + 1e-6)

            # 6. 为每个 Prediction 找到最佳的 Ground Truth
            best_gt_scores, best_gt_indices = dice_scores.max(dim=1)
            
            # 7. 获取对应的 GT 类别
            assigned_labels = gt_labels[best_gt_indices]

            # 8. 填充 Oracle Logits
            query_indices = torch.arange(num_queries, device=self.device)
            oracle_logits[i, query_indices, assigned_labels] = 100.0

        return oracle_logits
    
    def _prepare_targets_from_sem_seg(self, batched_inputs, images):
        """
        从 sem_seg 生成用于 Matcher 的 targets (labels, masks, boxes)。
        """
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []

        for inp in batched_inputs:
            if "sem_seg" not in inp:
                # 如果没有 sem_seg，返回空 target
                new_targets.append({
                    "labels": torch.empty(0, dtype=torch.long, device=self.device),
                    "masks": torch.empty(0, h_pad, w_pad, dtype=torch.bool, device=self.device),
                    "boxes": torch.empty(0, 4, device=self.device),
                })
                continue

            # 读取 sem_seg
            sem_seg = inp["sem_seg"].to(self.device) # [H_orig, W_orig]
            
            # 这里的 sem_seg 可能是原始尺寸，需要注意 mask 的 padding 对应 model 的输入尺寸
            # 通常 sem_seg_postprocess 是在后处理做的，但为了 matcher 计算 cost，我们需要把 GT mask 处理成和 pred mask (padded) 一致的尺寸
            # 或者更简单的方式：我们利用 bitmasks 和 boxes 的归一化坐标，matcher 会处理尺寸问题
            
            # 获取唯一类别 (排除 255 ignored)
            classes = torch.unique(sem_seg)
            classes = classes[classes != 255]

            if len(classes) == 0:
                new_targets.append({
                    "labels": torch.empty(0, dtype=torch.long, device=self.device),
                    "masks": torch.empty(0, h_pad, w_pad, dtype=torch.bool, device=self.device),
                    "boxes": torch.empty(0, 4, device=self.device),
                })
                continue

            gt_masks = []
            gt_labels = []

            for c in classes:
                # 生成 binary mask
                m = (sem_seg == c).float() 
                gt_masks.append(m)
                gt_labels.append(c)
            
            gt_masks = torch.stack(gt_masks) # [N, H_orig, W_orig]
            gt_labels = torch.stack(gt_labels).long() # [N]

            # --- 坐标转换与 Padding ---
            # 为了计算 Loss/Cost，我们需要把 mask 放到 padded 的画布上
            # 这里简单处理：先将 mask Pad 到 images.tensor 的尺寸
            padded_masks = torch.zeros((len(gt_masks), h_pad, w_pad), dtype=gt_masks.dtype, device=self.device)
            # 注意：inp["image"] 经过 mapper 后可能已经 resize 过了，但 sem_seg 通常是原图
            # 如果 eval 流程中 sem_seg 是原图尺寸，而 prediction 是 padding 后的尺寸，直接 copy 会错位。
            # 为了严谨，这里需要 resize GT mask 到 input tensor 的尺寸。
            # 但通常 Detectron2 的 sem_seg 是原图。
            # 这里假设：为了简单起见，且通常 eval 是单张图，我们直接插值 GT mask 到 padded 尺寸
            
            # 使用 interpolate 调整尺寸
            gt_masks_resized = F.interpolate(
                gt_masks.unsqueeze(1), 
                size=(h_pad, w_pad), 
                mode="nearest"
            ).squeeze(1)
            
            # 计算 Boxes (xyxy)
            gt_boxes_xyxy = masks_to_boxes(gt_masks_resized)
            
            # 归一化 Boxes (cxcywh)
            scale = torch.tensor([w_pad, h_pad, w_pad, h_pad], dtype=torch.float32, device=self.device)
            gt_boxes_norm = gt_boxes_xyxy / scale
            gt_boxes_cxcywh = box_xyxy_to_cxcywh(gt_boxes_norm)

            new_targets.append({
                "labels": gt_labels,
                "masks": gt_masks_resized,
                "boxes": gt_boxes_cxcywh
            })

        return new_targets

    def panoptic_inference(self, mask_cls, mask_pred, dataname):

        scores, labels = F.softmax(mask_cls, dim=-1).max(-1) # [Q]

        num_classes = mask_cls.shape[-1]
        bg_idx = num_classes - 1 if self.use_softmax else -1

        keep = scores > self.object_mask_threshold
        
        if self.use_softmax:
            keep = keep & (labels != bg_idx)
        
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]

        cur_masks.sigmoid_()
        # 加权 Mask
        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            return panoptic_seg, segments_info

        # 3. Argmax 生成全景图
        cur_mask_ids = cur_prob_masks.argmax(0)
        stuff_memory_list = {}
        
        # 获取 Metadata
        meta = self.test_metadata[dataname] if dataname in self.test_metadata else MetadataCatalog.get(dataname)
        thing_ids = set(meta.thing_dataset_id_to_contiguous_id.values())

        for k in range(cur_classes.shape[0]):
            pred_class = cur_classes[k].item()
            isthing = pred_class in thing_ids
            
            # 检查 Mask 质量
            mask_area = (cur_mask_ids == k).sum().item()
            original_area = (cur_masks[k] >= 0.5).sum().item()
            mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

            if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                if mask_area / original_area < self.overlap_threshold:
                    continue

                # 合并 Stuff 区域
                if not isthing:
                    if int(pred_class) in stuff_memory_list.keys():
                        panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                        continue
                    else:
                        stuff_memory_list[int(pred_class)] = current_segment_id + 1

                current_segment_id += 1
                panoptic_seg[mask] = current_segment_id

                segments_info.append(
                    {
                        "id": current_segment_id,
                        "isthing": bool(isthing),
                        "category_id": int(pred_class),
                    }
                )

        return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred, dataname):
        # mask_cls: [Q, K] (Logits)
        # mask_pred: [Q, H, W] (Logits)
        
        image_size = mask_pred.shape[-2:]
        
        # 1. 计算分数 (Sigmoid)
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        num_classes = scores.shape[-1]
        
        # 2. 展开所有 Query-Class 对
        num_queries = scores.shape[0]
        labels = torch.arange(num_classes, device=self.device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)
        
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        # 找到对应的 mask index
        topk_indices = topk_indices // num_classes
        mask_pred = mask_pred[topk_indices]

        # 4. 过滤 Thing Classes (如果是 Panoptic 模式)
        if self.panoptic_on:
            meta = self.test_metadata[dataname] if dataname in self.test_metadata else MetadataCatalog.get(dataname)
            if hasattr(meta, 'thing_dataset_id_to_contiguous_id'):
                thing_ids = set(meta.thing_dataset_id_to_contiguous_id.values())
            else:
                # 如果没有映射表，默认使用所有类别的连续索引
                thing_ids = set(range(len(meta.thing_classes)))
            
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab.item() in thing_ids
            
            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        # 5. 生成 Instances 对象
        result = Instances(image_size)
        
        # 使用 Sigmoid 后的 Mask
        mask_pred_sigmoid = mask_pred.sigmoid()
        pred_masks_binary = (mask_pred_sigmoid > 0.5).float()
        result.pred_masks = pred_masks_binary
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4)) # SAM3 通常不直接出框，这里放空框或者用 mask2box 计算
        
        # 计算综合分数
        mask_scores_per_image = (mask_pred_sigmoid.flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image

        if pred_masks_binary.numel() > 0:
            # BitMasks 最好接收 Bool 或 Uint8
            # result.pred_masks 是 float，这里转一下 ensure 安全
            result.pred_boxes = BitMasks(pred_masks_binary > 0).get_bounding_boxes()
        else:
            result.pred_boxes = Boxes(torch.zeros(0, 4, device=self.device))
            
        return result


    @torch.no_grad()                            
    def match_via_iou(self, mask_pred_results, mask_cls_results, targets, iou_threshold=0.7, max_matches=8):
        batch_size = mask_pred_results.shape[0]
        matched_src_masks = []
        matched_target_masks = []
        matched_labels = []
        matched_src_cls = []  # <--- [修改1] 新增列表存储匹配的分类结果

        
        for b in range(batch_size):
            tgt_label = targets[b]["labels"] 
            tgt_mask = targets[b]["masks"].to(mask_pred_results.device)  
            num_tgt_masks = tgt_mask.shape[0]

            pred_mask = mask_pred_results[b] 
            pred_cls = mask_cls_results[b]  
            num_pred_masks = pred_mask.shape[0]

            tgt_mask = F.interpolate(tgt_mask[:, None].float(), size=pred_mask.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)

            pred_mask_flat = pred_mask.flatten(1)
            tgt_mask_flat = tgt_mask.flatten(1)

            with torch.no_grad():
                ious = compute_mask_iou(pred_mask_flat, tgt_mask_flat)

            matched_pred_idx = []
            matched_tgt_idx = []
            
            for j in range(num_tgt_masks):
                valid_pred_idx = (ious[:, j] > iou_threshold).nonzero(as_tuple=True)[0]
                if len(valid_pred_idx) > 0:
                    random_idx = torch.randint(0, len(valid_pred_idx), (1,)).item()
                    best_pred_idx = valid_pred_idx[random_idx]
                    matched_pred_idx.append(best_pred_idx.item())
                    matched_tgt_idx.append(j)


            if len(matched_pred_idx) > max_matches:
                selected_indices = torch.randperm(len(matched_pred_idx))[:max_matches]
                matched_pred_idx = [matched_pred_idx[i] for i in selected_indices]
                matched_tgt_idx = [matched_tgt_idx[i] for i in selected_indices]

            if len(matched_pred_idx) < max_matches:
                num_to_add = max_matches - len(matched_pred_idx)
                
                matched_src_masks.append(
                    torch.cat([pred_mask[matched_pred_idx], 
                            torch.zeros((num_to_add, *pred_mask.shape[1:]), device=pred_mask.device)], dim=0)
                )
                matched_target_masks.append(
                    torch.cat([tgt_mask[matched_tgt_idx], 
                            torch.zeros((num_to_add, *tgt_mask.shape[1:]), device=tgt_mask.device)], dim=0)
                )
                matched_labels.append(
                    torch.cat([tgt_label[matched_tgt_idx], 
                            torch.full((num_to_add,), -1, dtype=tgt_label.dtype, device=tgt_label.device)], dim=0)
                )

                 # --- [修改2] Class Result 填充 ---
                # pred_cls 的形状通常是 (Num_Preds, Num_Classes)
                # 我们取出匹配的部分，剩余部分用0填充
                matched_src_cls.append(
                    torch.cat([pred_cls[matched_pred_idx], 
                            torch.zeros((num_to_add, pred_cls.shape[-1]), device=pred_cls.device, dtype=pred_cls.dtype)], dim=0)
                )

            else:
                matched_src_masks.append(pred_mask[matched_pred_idx])
                matched_target_masks.append(tgt_mask[matched_tgt_idx])
                matched_labels.append(tgt_label[matched_tgt_idx])
                
                # --- [修改3] Class Result 直接提取 ---
                matched_src_cls.append(pred_cls[matched_pred_idx])

        matched_src_masks = torch.stack(matched_src_masks, dim=0) 
        matched_target_masks = torch.stack(matched_target_masks, dim=0)
        matched_labels = torch.stack(matched_labels, dim=0)
        matched_src_cls = torch.stack(matched_src_cls, dim=0) 
        
        # 返回值增加了 matched_src_cls
        return matched_src_masks, matched_src_cls, matched_target_masks, matched_labels

    def cross_entropy_loss(self, mask_cls_results, labels):
        
        if torch.all(labels == -1):
            loss_ce = mask_cls_results.sum() * 0.0 
        else:
            loss_ce = F.cross_entropy(mask_cls_results.transpose(1, 2), labels.to(torch.int64), ignore_index=-1)  #remove celoss weight because of multiple datasets training

        losses = {"loss_ce": loss_ce}
        return losses
    
    def cosine_similarity_loss(self, pred_features, gt_features):
    
        cosine_similarity_loss = {}
        
        cosine_sim = F.cosine_similarity(pred_features, gt_features, dim=-1)
        cosine_similarity_loss[f"loss_cosine"] = 1 - cosine_sim.mean()
        return cosine_similarity_loss

    @torch.no_grad()
    def visualize_query_attention(self, batched_inputs, images, outputs, targets, is_void_prob, out_vocab_cls_probs, cross_attn_weights, dino_pixel_feat, save_root):
        
        os.makedirs(save_root, exist_ok=True)
        
        # 1. 获取模型输出
        pred_masks = outputs["pred_masks"].sigmoid() 
        
        # 获取 Attention weights
        if cross_attn_weights is not None:
            attn_weights = cross_attn_weights[-1] # 取最后一层
        else:
            print("cross_attn_weights is None.")
            return

        batch_size = pred_masks.shape[0]
        
        # 推断 Feature map 尺寸
        feat_h ,feat_w = dino_pixel_feat.shape[-2:]

        # 获取数据集信息和类别名
        dataname = get_dataname(batched_inputs[0])
        if dataname in self.test_metadata:
            meta = self.test_metadata[dataname]
        else:
            meta = MetadataCatalog.get(dataname)
        
        try:
            class_names = meta.stuff_classes
        except:
            class_names = meta.thing_classes
            
        # 准备逐像素预测所需的文本分类器
        text_classifier, num_templates = self.get_text_classifier(dataname)
        text_classifier = text_classifier.to(self.device)

        # 生成颜色调色板 (包括背景色)
        np.random.seed(42)
        # 确保颜色鲜艳
        color_palette = np.random.randint(50, 255, size=(len(class_names) + 1, 3), dtype=np.uint8)
        # 将背景(或者未定义区域)设为黑色 [0,0,0]
        background_color = np.array([0, 0, 0], dtype=np.uint8)

        for b in range(batch_size):
            file_name = batched_inputs[b]["file_name"].split('/')[-1].split('.')[0]
            save_dir = os.path.join(save_root, file_name)
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
            os.makedirs(save_dir)

            # --- A. 准备原图 ---
            img_tensor = images.tensor[b].clone()
            img_tensor = (img_tensor * self.pixel_std + self.pixel_mean) * 255.0
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            img_h, img_w = img_np.shape[:2]

            # --- B. 计算逐像素预测 (Pixel-wise Prediction) ---
            curr_feat = dino_pixel_feat[b] 
            if curr_feat.shape[0] == text_classifier.shape[1]: 
                curr_feat = curr_feat.permute(1, 2, 0)
            
            curr_feat = F.normalize(curr_feat, dim=-1)
            pixel_logits = torch.einsum("hwc,nc->hwn", curr_feat, text_classifier)
            pixel_cls_logits = aggregate_name_to_class_logits(pixel_logits, num_templates)
            
            pixel_pred_idx = pixel_cls_logits.argmax(dim=-1).cpu().numpy().astype(np.uint8)
            pixel_pred_idx_resized = cv2.resize(pixel_pred_idx, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            pixel_vis = color_palette[pixel_pred_idx_resized % len(color_palette)]

            # --- C. 计算每个 Query 与 GT 的最大 IoU ---
            tgt_masks = targets[b]["masks"].float()
            tgt_labels = targets[b]["labels"].cpu().numpy()
            
            curr_pred_masks = pred_masks[b]
            
            if tgt_masks.shape[-2:] != curr_pred_masks.shape[-2:]:
                tgt_masks_resized = F.interpolate(tgt_masks.unsqueeze(1), size=curr_pred_masks.shape[-2:], mode='nearest').squeeze(1)
            else:
                tgt_masks_resized = tgt_masks

            if len(tgt_masks) > 0:
                flat_pred = (curr_pred_masks > 0.5).flatten(1).float()
                flat_tgt = (tgt_masks_resized > 0.5).flatten(1).float()
                
                intersection = torch.mm(flat_pred, flat_tgt.t())
                area_pred = flat_pred.sum(1, keepdim=True)
                area_tgt = flat_tgt.sum(1, keepdim=True)
                union = area_pred + area_tgt.t() - intersection
                iou_matrix = intersection / (union + 1e-6)
                max_ious, _ = iou_matrix.max(dim=1)
            else:
                max_ious = torch.zeros(curr_pred_masks.shape[0], device=self.device)

            # --- D. 综合排序逻辑 ---
            if is_void_prob.dim() > 1: # 兼容不同shape [B,Q,1] or [B,Q]
                 curr_void_prob = is_void_prob[b].squeeze()
            else:
                 curr_void_prob = is_void_prob[b]
                 
            # # 确保是概率值
            # if curr_void_prob.max() > 1.0 or curr_void_prob.min() < 0.0: 
            #      curr_void_prob = curr_void_prob.sigmoid()
            
            curr_fg_probs = 1.0 - curr_void_prob
            sort_metric = max_ious + curr_fg_probs.to(max_ious.device)
            sorted_indices = torch.argsort(sort_metric, descending=True)

            # =================================================
            # 1. 绘制总览图 (Overview)
            # =================================================
            # 增加 figsize 的高度，给底部的 legend 留出空间
            fig, ax = plt.subplots(1, 3, figsize=(24, 10))
            
            # Subplot 1: Original
            ax[0].imshow(img_np)
            ax[0].set_title("Original Image", fontsize=15)
            ax[0].axis('off')
            
            # Subplot 2: GT Masks (纯色 Segmentation Map)
            # 初始化全黑画布
            gt_vis = np.zeros((img_h, img_w, 3), dtype=np.uint8)
            legend_elements = []
            present_classes = set()
            
            if len(tgt_masks) > 0:
                tgt_masks_orig = F.interpolate(tgt_masks.unsqueeze(1), size=(img_h, img_w), mode='nearest').squeeze(1)
                tgt_masks_bool = tgt_masks_orig > 0.5
                
                # 遍历绘制：注意顺序，后绘制的会覆盖先绘制的 (Painter's Algorithm)
                for i in range(len(tgt_masks)):
                    mask_i = tgt_masks_bool[i].cpu().numpy()
                    label_i = tgt_labels[i]
                    present_classes.add(label_i)
                    
                    color = color_palette[label_i % len(color_palette)]
                    # 直接赋值，不进行半透明混合
                    gt_vis[mask_i] = color

                # 生成图例
                for label_i in sorted(list(present_classes)):
                    color_norm = color_palette[label_i % len(color_palette)] / 255.0
                    class_name = class_names[label_i] if label_i < len(class_names) else f"Class {label_i}"
                    legend_elements.append(
                        Rectangle((0, 0), 1, 1, color=color_norm, label=f"{class_name}")
                    )

            ax[1].imshow(gt_vis)
            ax[1].set_title("GT Segmentation Map", fontsize=15)
            ax[1].axis('off')
            
            # 把图例放在图片下方
            if legend_elements:
                # loc='upper center' 指图例的上边缘对齐锚点
                # bbox_to_anchor=(0.5, -0.05) 指锚点在 x=0.5(中间), y=-0.05(坐标轴下方一点)
                ax[1].legend(
                    handles=legend_elements, 
                    loc='upper center', 
                    bbox_to_anchor=(0.5, -0.02),
                    ncol=min(3, len(legend_elements)), # 最多3列
                    fontsize='small', 
                    frameon=False
                )

            # Subplot 3: Pixel-wise Classification
            ax[2].imshow(pixel_vis)
            ax[2].set_title("DINO Pixel-wise Prediction", fontsize=15)
            ax[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "00_overview.jpg"))
            plt.close()

            # =================================================
            # 2. 逐 Query 绘制
            # =================================================
            for i, q_idx in enumerate(sorted_indices):
                rank = i + 1 
                q_iou = max_ious[q_idx].item()
                q_fg_prob = curr_fg_probs[q_idx].item()
                
                # 过滤逻辑
                if q_fg_prob < 0.2: 
                    continue
                
                # 获取 Query 分类信息
                probs = out_vocab_cls_probs[b, q_idx]
                topk_vals, topk_inds = torch.topk(probs, k=5)
                cls_str = ""
                for val, ind in zip(topk_vals, topk_inds):
                    c_name = class_names[ind] if ind < len(class_names) else "Void"
                    cls_str += f"{c_name}: {val:.2f}\n"
                
                # Mask 可视化
                mask_map = curr_pred_masks[q_idx]
                mask_map = F.interpolate(mask_map.view(1, 1, *mask_map.shape), size=(img_h, img_w), mode='bilinear').squeeze()
                mask_bin = mask_map > 0.5
                
                mask_vis = img_np.copy()
                mask_vis[mask_bin.cpu().numpy()] = mask_vis[mask_bin.cpu().numpy()] * 0.5 + np.array([255, 0, 0]) * 0.5
                
                # Attention Map 可视化
                attn = attn_weights[b, q_idx]
                attn_map = attn.view(feat_h, feat_w)
                attn_map = F.interpolate(attn_map.view(1, 1, feat_h, feat_w), size=(img_h, img_w), mode='bilinear').squeeze()
                attn_map = attn_map.cpu().numpy()
                
                attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
                attn_heatmap = cv2.applyColorMap((attn_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
                attn_vis = cv2.addWeighted(img_np, 0.5, attn_heatmap, 0.5, 0)
                
                # 绘图
                fig, ax = plt.subplots(1, 2, figsize=(16, 8))
                
                # 调整子图布局，为顶部的文本框留出空间
                fig.subplots_adjust(top=0.88)

                # 左图：预测的 Mask
                ax[0].imshow(mask_vis)
                # 将 Rank 和 Top-5 分类信息放在左上角标题
                title_str = f"Rank: {rank}\nTop-5 Classes:\n{cls_str.strip()}"
                ax[0].set_title(title_str, loc='left', fontsize=11, fontweight='bold')
                ax[0].axis('off')
                
                # 右图：Cross Attention Map
                ax[1].imshow(attn_vis[:, :, ::-1])
                ax[1].set_title("Cross Attention Map", fontsize=15)
                ax[1].axis('off')
                
                # --- 修改后的代码：在 Figure 的右上角添加文本框 ---
                # 准备要显示的文本和样式
                info_text = f"IoU: {q_iou:.3f}\nFG Prob: {q_fg_prob:.3f}"
                bbox_props = dict(boxstyle="round,pad=0.4", fc="ivory", ec="black", lw=1, alpha=0.9)

                # 使用 fig.text() 在 Figure 坐标系中添加文本
                # (0,0) 是画布左下角, (1,1) 是画布右上角
                fig.text(0.98, 0.97, info_text,         # 坐标 (x, y)
                         transform=fig.transFigure,     # 指定使用 Figure 坐标系
                         fontsize=10,
                         fontweight='bold',
                         color='black',
                         verticalalignment='top',       # 垂直对齐方式
                         horizontalalignment='right',   # 水平对齐方式
                         bbox=bbox_props)               # 应用文本框样式
                # --- 修改结束 ---
                
                save_name = f"{rank:03d}_iou_{q_iou:.2f}_prob_{q_fg_prob:.2f}_query_{q_idx}.jpg"
                plt.savefig(os.path.join(save_dir, save_name))
                plt.close(fig) # 确保关闭 figure 释放内存
                
        print(f"Visualization saved to {save_root}")

def aggregate_name_to_class_logits(query_names_results, num_templates):
    """
    将包含同义词/模板的 name logits 转化为唯一类别的 class logits。
    
    参数:
        query_names_results (torch.Tensor): 形状为 [bs, N, total_names] 的张量
        num_templates (list[int]): 每个类别包含的名称/模板数量列表，长度为 num_classes
        
    返回:
        query_cls_results (torch.Tensor): 形状为 [bs, N, num_classes] 的张量
    """
    # 使用 torch.split 根据每个类别的模板数量对最后一维进行拆分
    # 这会返回一个包含多个张量的 list，每个张量形状为 [bs, N, num_t]
    name_splits = torch.split(query_names_results, num_templates, dim=-1)
    
    # 对每个分块取最大值 (max pooling over synonyms/templates)
    # s.max(-1).values 得到形状为 [bs, N] 的张量
    cls_logits_list = [s.max(dim=-1).values for s in name_splits]
    
    # 在最后一维堆叠，得到 [bs, N, num_classes]
    query_cls_results = torch.stack(cls_logits_list, dim=-1)
    
    return query_cls_results


def get_gt_labels_from_sem_seg(sem_seg):
    """
    基于 sem_seg_2_gt_masks 逻辑提取当前图像中存在的有效类别 ID。
    """
    # 确保是 2D 张量 (H, W)
    if sem_seg.dim() == 3: 
        sem_seg = sem_seg.squeeze(0)
    
    # 获取唯一类别
    classes = torch.unique(sem_seg, sorted=False, return_inverse=False, return_counts=False)
    
    # 过滤掉背景/忽略类 (通常是 255，或者是 void 类)
    # 注意：这里假设 255 是忽略索引，根据你的数据集调整
    gt_labels = classes[classes != 255]
    
    return gt_labels.cpu().numpy().tolist()

def get_grid_reference_points(spatial_shapes, device):
    """
    根据特征图形状生成归一化的网格坐标。
    spatial_shapes: tensor or list, shape (N_levels, 2) -> [(H, W), ...]
    返回: (1, sum(H*W), 2)
    """
    reference_points_list = []
    for lvl, (H, W) in enumerate(spatial_shapes):
        H, W = int(H), int(W)
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
            torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
            indexing='ij'
        )
        # 归一化到 [0, 1]
        ref_y = ref_y.reshape(-1)[None] / H
        ref_x = ref_x.reshape(-1)[None] / W
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    
    reference_points = torch.cat(reference_points_list, 1) # (1, total_pixels, 2)
    return reference_points

def visualize_segmentation(
    pred_result, 
    gt_result,
    class_names, 
    original_image_tensor, 
    save_path="./show/result.png", 
    fig_size=(20, 10),
    ignore_index=255
):
    """
    可视化分割结果：[原图, 预测图, GT图] 并排显示，并附带图例。
    """
    # --- 数据准备 ---
    if isinstance(pred_result, torch.Tensor):
        pred_result = pred_result.cpu().numpy()
    if isinstance(gt_result, torch.Tensor):
        gt_result = gt_result.cpu().numpy()
    if isinstance(original_image_tensor, torch.Tensor):
        # (C, H, W) -> (H, W, C)
        original_image = original_image_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    else:
        original_image = original_image_tensor

    # 统一尺寸（以原图为准）
    h, w = original_image.shape[:2]
    num_classes = len(class_names)

    # --- 颜色调色板 ---
    np.random.seed(42) # 固定种子
    palette = np.random.randint(0, 255, size=(num_classes, 3))
    # 为 ignore_index (255) 分配灰色
    ignore_color = np.array([128, 128, 128], dtype=np.uint8)

    def mask_to_rgb(mask):
        rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for i in range(num_classes):
            rgb[mask == i] = palette[i]
        rgb[mask == ignore_index] = ignore_color
        return rgb

    # 生成 RGB 掩码图
    pred_rgb = mask_to_rgb(pred_result)
    gt_rgb = mask_to_rgb(gt_result)

    # --- 统计出现的类别用于图例 ---
    # 合并 Pred 和 GT 中出现的类别，以便在图例中全部展示
    present_in_pred = np.unique(pred_result)
    present_in_gt = np.unique(gt_result)
    all_present_classes = np.unique(np.concatenate([present_in_pred, present_in_gt]))
    
    # 过滤掉 ignore_index 和超出范围的索引
    all_present_classes = [c for c in all_present_classes if c < num_classes and c >= 0]
    
    # 按像素占比（在 Pred 中）排序，让图例更整洁
    class_counts = {c: np.sum(pred_result == c) for c in all_present_classes}
    sorted_classes = sorted(all_present_classes, key=lambda x: class_counts[x], reverse=True)

    # --- 绘图 ---
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=fig_size)

    # 1. 原图
    ax[0].imshow(original_image)
    ax[0].set_title("Original Image", fontsize=15)
    ax[0].axis('off')

    # 2. 预测图
    ax[1].imshow(pred_rgb)
    ax[1].set_title("Prediction", fontsize=15)
    ax[1].axis('off')

    # 3. GT 图
    ax[2].imshow(gt_rgb)
    ax[2].set_title("Ground Truth", fontsize=15)
    ax[2].axis('off')

    # --- 图例 ---
    legend_elements = []
    for c_idx in sorted_classes:
        color = palette[c_idx] / 255.0
        name = class_names[c_idx]
        count = class_counts[c_idx]
        legend_elements.append(
            Rectangle((0, 0), 1, 1, color=color, label=f"{name} ({count:,} px)")
        )
    
    # 如果有忽略区域，添加说明
    if np.any(gt_result == ignore_index):
        legend_elements.append(
            Rectangle((0, 0), 1, 1, color=ignore_color/255.0, label="Ignore/Void")
        )

    # 放置图例
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.02),
        ncol=min(5, len(legend_elements)), 
        frameon=True,
        fontsize='small'
    )

    plt.tight_layout(rect=[0, 0.1, 1, 0.95]) # 为底部图例留出空间
    
    try:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"可视化已保存: {save_path}")
    except Exception as e:
        print(f"保存失败: {e}")
    plt.close(fig)

def get_dataname(batched_input):
    """
    从输入数据中提取或推断 dataname (数据集名称)。
    逻辑：
    1. 尝试直接从 meta 或 batched_input 根目录找 'dataname'
    2. 如果找不到，通过 'file_name' 字符串匹配
    """
    # 1. 尝试从 meta 字典或根字典中直接获取
    if "meta" in batched_input and "dataname" in batched_input["meta"]:
        return batched_input["meta"]["dataname"]
    if "dataname" in batched_input:
        return batched_input["dataname"]

    # 2. 备选方案：通过 file_name 推断
    # Detectron2 的 dataset_dict 几乎百分之百包含 file_name
    file_name = batched_input.get("file_name", "")
    
    # 逻辑：目前只管 lvis，也可以顺便兼容 ade
    file_name_lower = file_name.lower()
    
    if "lvis" in file_name_lower:
        return "lvis"
    elif "ade" in file_name_lower:
        return "ade20k"
    
    # 如果都匹配不上，返回一个默认值或抛出警告
    print(f"Warning: Could not infer dataname from {file_name}, using default 'lvis_v1_val'")
    return "lvis_v1_val"

def compute_mask_iou(pred_masks, tgt_masks):
    
    pred_masks = pred_masks.sigmoid()
    
    binarized_pred_masks = (pred_masks >= 0.4).float()
    binarized_tgt_masks = (tgt_masks > 0.5).float()

    intersection = torch.einsum('nc,mc->nm', binarized_pred_masks, binarized_tgt_masks)
    
    pred_area = binarized_pred_masks.sum(dim=-1)  
    tgt_area = binarized_tgt_masks.sum(dim=-1)    
    
    union = pred_area[:, None] + tgt_area[None, :] - intersection
    
    iou_matrix = intersection / (union + 1e-6)
    
    return iou_matrix

def init_query_proj(mlp_module):
    """
    针对 Open-Vocabulary 适配器的初始化策略：
    1. 隐藏层使用 Kaiming 初始化，配合 ReLU。
    2. 最后一层（输出层）使用极小值初始化。
    3. 配合 Residual=True，使得 MLP 在初始阶段输出接近 x + 0，即保留 SAM3 原始特征。
    """
    for i, layer in enumerate(mlp_module.layers):
        if isinstance(layer, nn.Linear):
            # 判断是否为隐藏层 (i < num_layers - 1)
            if i < mlp_module.num_layers - 1:
                # 隐藏层使用 Kaiming Normal，因为后面接了 ReLU
                nn.init.kaiming_normal_(layer.weight, a=0, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            else:
                # 最后一层（输出层）：非常关键
                # 将权重初始化为极小分布（或全0），使得初始状态下 MLP 的增量几乎为 0
                # 这样：Output = x (原始特征) + MLP(x) (几乎为0) ≈ x
                nn.init.normal_(layer.weight, std=0.0001)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    # 如果存在 LayerNorm，初始化为标准状态
    if hasattr(mlp_module, "out_norm") and isinstance(mlp_module.out_norm, nn.LayerNorm):
        nn.init.constant_(mlp_module.out_norm.weight, 1.0)
        nn.init.constant_(mlp_module.out_norm.bias, 0)

def init_score_head(mlp_module, prior_prob=0.01):
    # 1. 遍历 MLP 的所有层
    for i, layer in enumerate(mlp_module.layers):
        if isinstance(layer, nn.Linear):
            # 对于隐藏层，使用 Kaiming 初始化（因为你用了 ReLU）
            if i < len(mlp_module.layers) - 1:
                nn.init.kaiming_normal_(layer.weight, a=0, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            else:
                # 2. 关键点：对最后一层（输出层）进行特殊处理
                # 将权重初始化为非常小的值，让 bias 起主导作用
                nn.init.normal_(layer.weight, std=0.01)
                
                # 设置先验偏置
                bias_value = -math.log((1 - prior_prob) / prior_prob)
                nn.init.constant_(layer.bias, bias_value)

def get_classification_logits(x, text_classifier, logit_scale, num_templates=None):
    # x in shape of [B, *, C]
    # text_classifier in shape of [num_classes, C]
    # logit_scale is a learnable scalar https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/model.py#L201
    # return: [B, *, num_classes]
    x = F.normalize(x, dim=-1)
    logit_scale = torch.clamp(logit_scale.exp(), max=100)
    if len(text_classifier.shape) == 2:
        pred_logits = logit_scale * x @ text_classifier.T # B, *, N + 1
    else:
        pred_logits = logit_scale * x @ text_classifier.permute(0,2,1) # B, *, N + 1
        
    # max ensembel as in OpenSeg/ODISE
    final_pred_logits = []
    cur_idx = 0
    for num_t in num_templates: 
        final_pred_logits.append(pred_logits[:, :, cur_idx: cur_idx + num_t].max(-1).values)
        cur_idx += num_t
    # final_pred_logits.append(pred_logits[:, :, -1]) # the last classifier is for void
    final_pred_logits = torch.stack(final_pred_logits, dim=-1)
    return final_pred_logits


from sam3.model.decoder import (
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoderCrossAttention,
)

def _create_pool_decoder() -> TransformerDecoder:
    """Create transformer decoder with its layer."""
    decoder_layer = TransformerDecoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        cross_attention=nn.MultiheadAttention(
        # cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
        ),
        n_heads=8,
        use_text_cross_attention=True,
    )

    decoder = TransformerDecoder(
        layer=decoder_layer,
        num_layers=3,
        num_queries=200,   
        return_intermediate=True,
        box_refine=True,
        num_o2m_queries=0,    
        dac=False,            
        boxRPB="log",
        d_model=256,
        frozen=False,
        interaction_layer=None,
        dac_use_selfatt_ln=False, # 【建议修改】既然不用 dac，该参数已无意义，设为 False 即可
        resolution=1008,
        stride=14,
        use_act_checkpoint=True,
        presence_token=False,
    ) 

    return decoder

def load_partial_weights(target_module, source_state_dict):
    """
    加载权重并自动处理结构差异，将差异记录到 detectron2 日志中。
    
    Args:
        target_module (nn.Module): 需要加载权重的模型 (例如 self.pooling_decoder)
        source_state_dict (dict): 源权重字典 (例如 self.detector.transformer.decoder.state_dict())
    """
    logger = logging.getLogger("detectron2")
    
    target_state_dict = target_module.state_dict()
    processed_source_dict = {}
    shape_mismatch_keys = []
    
    # 1. 过滤形状不匹配的参数
    # 如果不做这一步，load_state_dict 遇到同名但形状不同的参数会直接报错 Crash
    for k, v in source_state_dict.items():
        if k in target_state_dict:
            if v.shape != target_state_dict[k].shape:
                shape_mismatch_keys.append(
                    f"{k}: source {v.shape} vs target {target_state_dict[k].shape}"
                )
                continue
        processed_source_dict[k] = v

    # 2. 加载权重 (strict=False 允许结构不一致)
    # missing_keys: target 中有但 source 中没有 (需要被初始化但没加载到的)
    # unexpected_keys: source 中有但 target 中没有 (预训练里多余的，比如你删掉的层)
    msg = target_module.load_state_dict(processed_source_dict, strict=False)
    
    missing_keys = msg.missing_keys
    unexpected_keys = msg.unexpected_keys
    
    # 3. 记录日志
    log_info = []
    if len(missing_keys) > 0:
        log_info.append(
            f"[Pooling Decoder Init] MISSING keys (initialized randomly): \n" + 
            "\n".join([f"\t- {k}" for k in missing_keys])
        )
    
    if len(shape_mismatch_keys) > 0:
        log_info.append(
            f"[Pooling Decoder Init] SHAPE MISMATCH keys (skipped & initialized randomly): \n" + 
            "\n".join([f"\t- {k}" for k in shape_mismatch_keys])
        )

    # 对于 Unexpected keys (多余的权重)，由于你是“减少层数”，这部分可能会很多
    # 通常我们只需要知道数量，或者打印出来确认是被移除的层
    if len(unexpected_keys) > 0:
        # 这里选择简略打印数量，如果想看详细列表可以将下方注释打开
        log_info.append(
            f"[Pooling Decoder Init] UNEXPECTED keys in source (ignored, likely removed layers): {len(unexpected_keys)} keys."
        )
        # 详细打印前10个例子
        example_keys = unexpected_keys[:10]
        log_info.append(f"\tExamples: {example_keys} ...")

    if not log_info:
        logger.info("[Pooling Decoder Init] Weights loaded perfectly matches!")
    else:
        logger.warning("\n".join(log_info))

    return msg
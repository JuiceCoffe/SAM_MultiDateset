"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/maskformer_model.py
"""
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher
from .modeling.transformer_decoder.fcclip_transformer_decoder import MaskPooling, get_classification_logits

from .modeling.maft.mask_aware_loss import  MA_Loss
from .modeling.maft.representation_compensation import  Representation_Compensation
from .modeling.maft.content_dependent_transfer import ContentDependentTransfer

from .utils.text_templetes import VILD_PROMPT

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import os
import numpy as np
import torchvision



@META_ARCH_REGISTRY.register()
class MAFT_Plus(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        backbone_name: str,
        # backbone_t,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        train_metadata,
        test_metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        # MAFT
        rc_weights,
        cdt_params,
        # PE
        PE_ENABLED: bool,
    ):

        super().__init__()
        self.backbone = backbone
        self.backbone_name = backbone_name
        # self.backbone_t = backbone_t
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.train_metadata = train_metadata
        self.test_metadata = test_metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        # FC-CLIP args
        # self.mask_pooling = MaskPooling()
        self.train_text_classifier = None
        self.test_text_classifier = None
        self.void_embedding = nn.Embedding(1, backbone.dim_latent) # use this for void

        _, self.train_num_templates, self.train_class_names = self.prepare_class_names_from_metadata(train_metadata, train_metadata)
        # print(f"Test dataset has {len(self.raw_class_names)} classes:\n", self.raw_class_names)

        # self.cdt = ContentDependentTransfer(d_model = cdt_params[0], nhead = cdt_params[1], panoptic_on = panoptic_on)
        # self.ma_loss = MA_Loss()  # BCELoss BCEWithLogitsLoss SmoothL1Loss
        # self.rc_loss = Representation_Compensation()
        self.rc_weights = rc_weights

        self._freeze()
        self.train_dataname = None
        self.test_dataname = None

        self.cache = None # for caching RAW text embeds in inference
        self.PE_ENABLED = PE_ENABLED
        if self.PE_ENABLED:
            print("Using PE ")

    def _freeze(self, ):
        for name, param in self.named_parameters():
            if 'backbone_t' in name:
                param.requires_grad = False

            elif 'backbone' in name:
                if 'clip_model.visual.trunk.stem' in name:
                    param.requires_grad = True
                if 'clip_model.visual.trunk.stages' in name:
                    param.requires_grad = True
                if 'clip_model.visual.trunk.norm_pre' in name:
                    param.requires_grad = True

                if 'clip_model.visual.trunk.head.norm.' in name:
                    param.requires_grad = False
                if 'clip_model.visual.head.mlp.' in name:
                    param.requires_grad = False

        for name, param in self.named_parameters():
            if param.requires_grad == True and 'sem_seg_head' not in name:
                print(name, param.requires_grad)

    def prepare_class_names_from_metadata(self, metadata, train_metadata):
        def split_labels(x):
            res = []
            for x_ in x:
                x_ = x_.replace(', ', ',')
                x_ = x_.split(',') # there can be multiple synonyms for single class
                res.append(x_)
            return res
        # get text classifier
        try:
            class_names = split_labels(metadata.stuff_classes) # it includes both thing and stuff
            train_class_names = split_labels(train_metadata.stuff_classes)
        except:
            # this could be for insseg, where only thing_classes are available
            class_names = split_labels(metadata.thing_classes)
            train_class_names = split_labels(train_metadata.thing_classes)
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
                for template in VILD_PROMPT:
                    res.append(template.format(x))
            return res, len(res) // len(VILD_PROMPT)
       
        num_templates = []
        templated_class_names = []
        # print('class_names: ',len(class_names)) 171
        for x in class_names:
            templated_classes, templated_classes_num = fill_all_templates_ensemble(x)
            templated_class_names += templated_classes
            num_templates.append(templated_classes_num) # how many templates for current classes
        class_names = templated_class_names        
        return category_overlapping_mask, num_templates, class_names

    def set_metadata(self, metadata):
        self.test_metadata = metadata
        self.category_overlapping_mask, self.test_num_templates, self.test_class_names = self.prepare_class_names_from_metadata(metadata, self.train_metadata)
        self.test_text_classifier = None
        return

    def get_text_classifier(self, dataname):
        if self.training:
            if self.train_dataname != dataname:
                text_classifier = []
                # this is needed to avoid oom, which may happen when num of class is large
                bs = 128
                # print("train_class_names len: ",len(self.train_class_names)) 4592
                # print("train_class_names: ",self.train_class_names) 带模板的类别名
                # exit()
                for idx in range(0, len(self.train_class_names), bs):
                    text_classifier.append(self.backbone.get_text_classifier(self.train_class_names[idx:idx+bs], self.device).detach())
                text_classifier = torch.cat(text_classifier, dim=0)

                # average across templates and normalization.
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                text_classifier = text_classifier.reshape(text_classifier.shape[0]//len(VILD_PROMPT), len(VILD_PROMPT), text_classifier.shape[-1]).mean(1)
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                self.train_text_classifier = text_classifier
                self.train_dataname = dataname
                self.class_name_of_classifier = [element for index, element in enumerate(self.train_class_names) if index % len(VILD_PROMPT) == 0]
                # print('train_class_names: ',len(self.class_name_of_classifier))
                # exit()
            return self.train_text_classifier, self.train_num_templates
        else:
            if self.test_dataname != dataname:
                self.category_overlapping_mask, self.test_num_templates, self.test_class_names = self.prepare_class_names_from_metadata(self.test_metadata[dataname], self.train_metadata)
                text_classifier = []
                bs = 128
                for idx in range(0, len(self.test_class_names), bs):
                    text_classifier.append(self.backbone.get_text_classifier(self.test_class_names[idx:idx+bs], self.device).detach())
                text_classifier = torch.cat(text_classifier, dim=0)

                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                print("text_classifier shape before reshape:", text_classifier.shape)
                text_classifier = text_classifier.reshape(text_classifier.shape[0]//len(VILD_PROMPT), len(VILD_PROMPT), text_classifier.shape[-1]).mean(1) 
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                self.test_text_classifier = text_classifier
                self.test_dataname = dataname
                self.class_name_of_classifier = [element for index, element in enumerate(self.train_class_names) if index % len(VILD_PROMPT) == 0]
            return self.test_text_classifier, self.test_num_templates

    
    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg) # cfg.MODEL.BACKBONE.NAME : CLIP
        # backbone_t = build_backbone(cfg)


        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        test_metadata = {i: MetadataCatalog.get(i) for i in cfg.DATASETS.TEST}
        return {
            "backbone": backbone,
            "backbone_name": cfg.MODEL.BACKBONE.NAME,
            # "backbone_t":backbone_t,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "train_metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "test_metadata": test_metadata, # MetadataCatalog.get(cfg.DATASETS.TEST[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "rc_weights": cfg.MODEL.rc_weights,
            "cdt_params": cfg.MODEL.cdt_params,
            "PE_ENABLED": cfg.PE.ENABLED,
        }

    @property
    def device(self):
        return self.pixel_mean.device


    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """

        images = [x["image"].to(self.device) for x in batched_inputs]

        original_image = images[0].clone() # for visualization only

        
        if self.backbone_name == 'PEEncoder':
            images = [image.float().div(127.5).sub(1.0) for image in images]  # scale to [-1, 1]
            images = ImageList.from_tensors(images, self.size_divisibility)
            features = self.backbone.extract_features(images.tensor) 
        else:
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]

            images = ImageList.from_tensors(images, self.size_divisibility)

            features = self.backbone.extract_features(images.tensor) # 多尺度特征图,不包括用于与文本匹配的（该层在self.backbone.visual_prediction_forward中调用）

        file_names = [x["file_name"] for x in batched_inputs] # 可去变量
        file_names = [x.split('/')[-1].split('.')[0] for x in file_names] # 可去变量

        meta = batched_inputs[0]["meta"]
        # text_classifier, num_templates = self.get_text_classifier('openvocab_ade20k_panoptic_val')
        text_classifier, num_templates = self.get_text_classifier(meta['dataname'])
        # print("meta['dataname']:",meta['dataname'])
        if self.backbone_name == 'CLIP':
            text_classifier = torch.cat([text_classifier, F.normalize(self.void_embedding.weight, dim=-1)], dim=0)
        # print("text_classifier:", text_classifier.shape) # text_classifier: torch.Size([329, 768]) 328个类别+1个void
    
        
        for k in features.keys():
            features[k] = features[k].detach()

        features['text_classifier'] = text_classifier
        features['num_templates'] = num_templates

        clip_feature = features['clip_vis_dense'] # torch.Size([1, 1536, 38, 25])
        if self.backbone_name == 'PEEncoder':
            img_feat = clip_feature # PE情况下直接输出图像特征
        elif self.backbone_name == 'CLIP':
            img_feat = self.visual_prediction_forward_convnext_2d(clip_feature) # 输出可以在CLIP空间中直接理解的语义特征图
        elif self.backbone_name == 'DINOv3TXT':
            img_feat = clip_feature
        # print("img_feat shape:", img_feat.shape)

        img_feats = F.normalize(img_feat, dim=1) # B C H W
        print("img_feats shape:", img_feats.shape)
        text_feats = text_classifier# T C 带模板
        # print("text_feats shape:", text_feats.shape)
        logit_scale = torch.clamp(self.backbone.clip_model.logit_scale.exp(), max=100)
        logits = self.backbone.clip_model.logit_scale * torch.einsum('bchw, tc -> bthw', img_feats, text_feats)
        
        # ade20K: seg_logits:[1,150,336,448] --> [1,150,512,683]
        seg_logits = logits

        

        final_seg_logits = []
        cur_idx = 0
        for num_t in num_templates: 
            final_seg_logits.append(seg_logits[:, cur_idx: cur_idx + num_t,:,:].max(1).values)
            cur_idx += num_t
        if self.backbone_name == 'CLIP':
            final_seg_logits.append(seg_logits[:, -1,:,:]) # the last classifier is for void
            
        final_seg_logits = torch.stack(final_seg_logits, dim=1) # B T(+1) H W
        seg_probs = torch.softmax(final_seg_logits, dim=1) # B T(纯类别)(+1) H W

        def get_PseudoMasks(logits):


                   
            B, T, H, W = logits.shape
            threshold = 1.3/(H*W)  # 设置阈值为1/(H*W)
            
            # 应用softmax转换为概率
            logits = F.softmax(logits.view(B, T, -1), dim=2).view(B, T, H, W)
            
            # 二值化：将概率大于阈值的设为1，否则设为0
            binary_masks = (logits > threshold).float()

            pseudo_masks_list = []
            pseudo_points_list = []
            N0 = 5  # 设定聚类数量

            # 获取设备信息，确保新建的tensor在同一个设备上
            device = binary_masks.device 

            for b in range(B):
                for t in range(T):
                    # --- 初始化当前 (b, t) 的临时列表 ---
                    current_masks = []
                    current_points = []
                    
                    # 获取当前类别和batch的二值掩码
                    mask = binary_masks[b, t]
                    
                    # 找到mask中为1的像素坐标
                    coords = torch.nonzero(mask > 0, as_tuple=True)
                    
                    # --- 情况1：完全没有前景像素 ---
                    if len(coords[0]) == 0:
                        # 必须填充 N0 个空数据，以满足 view 的形状要求
                        for _ in range(N0):
                            pseudo_masks_list.append(torch.zeros(H, W, device=device))
                            pseudo_points_list.append([0, 0])
                        continue
                        
                    # 将坐标转换为二维坐标 (2, N)
                    coords = torch.stack(coords, dim=1)
                    coords = coords.float().t() # Shape: (2, N)
                    
                    num_pixels = coords.shape[1]
                    
                    # --- 情况2 & 3：像素数量处理 ---
                    if num_pixels < N0:
                        # 如果像素太少，不进行kmeans，直接把每个像素当作一个聚类中心
                        # 这种情况下，有效聚类数 = 像素数
                        # print(f"Warning: num_pixels ({num_pixels}) < N0 ({N0}). Using pixel coords as cluster centers.")
                        cluster_centers = coords
                        # 标签直接就是 0, 1, 2...
                        cluster_labels = torch.arange(num_pixels, device=device)
                        valid_clusters = num_pixels
                    else:
                        # 正常执行 k-means
                        # 注意：需确保 kmeans 函数能处理 device 问题
                        cluster_centers, cluster_labels = simple_kmeans(coords, num_clusters=N0)
                        valid_clusters = N0
                    
                    # --- 处理有效的聚类区域 ---
                    for i in range(valid_clusters):
                        # 获取当前聚类的像素索引
                        idx = (cluster_labels == i).nonzero(as_tuple=True)[0]
                        
                        # 创建掩码
                        mask_region = torch.zeros(H, W, device=device)
                        
                        if len(idx) > 0:
                            # 填充 mask
                            # 注意 coords 是 (2, N)，即 (y, x) 还是 (x, y) 取决于你的 nonzero 顺序
                            # 通常 nonzero 返回 (y, x)
                            y_coords = coords[0, idx].long()
                            x_coords = coords[1, idx].long()
                            mask_region[y_coords, x_coords] = 1
                        
                        # 概率掩码乘积
                        prob_mask = logits[b, t] * mask_region
                        current_masks.append(prob_mask)
                        
                        # 找到中心点
                        # 这里的逻辑是找离聚类中心最近的真实像素点
                        if len(idx) > 0:
                            dists = torch.norm(coords[:, idx] - cluster_centers[:, i].unsqueeze(1), dim=0)
                            min_dist_idx = torch.argmin(dists)
                            # 还原回在 coords 中的全局索引
                            center_idx = idx[min_dist_idx]
                            current_points.append([coords[0, center_idx].item(), coords[1, center_idx].item()])
                        else:
                            # 理论上 k-means 不会产生空簇，但在特殊初始化下可能发生，做个防守
                            current_points.append([0, 0])

                    # --- 填充 (Padding) ---
                    # 如果有效聚类数小于 N0（即 num_pixels < N0 的情况），补齐剩余的空位
                    for _ in range(N0 - valid_clusters):
                        current_masks.append(torch.zeros(H, W, device=device))
                        current_points.append([0, 0])
                    
                    # --- 将当前 batch/class 的结果存入总列表 ---
                    # 此时 current_masks 长度必然为 5
                    pseudo_masks_list.extend(current_masks)
                    pseudo_points_list.extend(current_points)
            

            # 转换张量
            # 此时列表长度保证为 B * T * N0
            pseudo_masks = torch.stack(pseudo_masks_list).view(B, T, N0, H, W)
            pseudo_points = torch.tensor(pseudo_points_list, device=device).view(B, T, N0, 2)

            return pseudo_masks, pseudo_points
            
        def get_PseudoMasksfromMasks(logits):
                   
            B, T, H, W = logits.shape
            n = 3  # 这里设定你想要的 n 值 (Top-N)

            # 1. 获取 logits 在 dim=1 (类别维度) 上前 n 大的索引
            # topk_values: (B, n, H, W), topk_indices: (B, n, H, W)
            _, topk_indices = torch.topk(logits, k=n, dim=1)

            # 2. 创建一个全 0 的 mask，形状与 logits 相同
            binary_masks = torch.zeros_like(logits)

            # 3. 使用 scatter_ 将对应索引位置的值设为 1
            # dim=1 表示在类别维度操作
            # src=1.0 表示填充的值
            binary_masks.scatter_(1, topk_indices, 1.0)

            pseudo_masks_list = []
            pseudo_points_list = []
            N0 = 5  # 设定聚类数量

            # 获取设备信息，确保新建的tensor在同一个设备上
            device = binary_masks.device 

            for b in range(B):
                for t in range(T):
                    # --- 初始化当前 (b, t) 的临时列表 ---
                    current_masks = []
                    current_points = []
                    
                    # 获取当前类别和batch的二值掩码
                    mask = binary_masks[b, t]
                    
                    # 找到mask中为1的像素坐标
                    coords = torch.nonzero(mask > 0, as_tuple=True)
                    
                    # --- 情况1：完全没有前景像素 ---
                    if len(coords[0]) == 0:
                        # 必须填充 N0 个空数据，以满足 view 的形状要求
                        for _ in range(N0):
                            pseudo_masks_list.append(torch.zeros(H, W, device=device))
                            pseudo_points_list.append([0, 0])
                        continue
                        
                    # 将坐标转换为二维坐标 (2, N)
                    coords = torch.stack(coords, dim=1)
                    coords = coords.float().t() # Shape: (2, N)
                    
                    num_pixels = coords.shape[1]
                    
                    # --- 情况2 & 3：像素数量处理 ---
                    if num_pixels < N0:
                        # 如果像素太少，不进行kmeans，直接把每个像素当作一个聚类中心
                        # 这种情况下，有效聚类数 = 像素数
                        
                        # print(f"Warning: num_pixels ({num_pixels}) < N0 ({N0}). Using pixel coords as cluster centers.")
                        cluster_centers = coords
                        # 标签直接就是 0, 1, 2...
                        cluster_labels = torch.arange(num_pixels, device=device)
                        valid_clusters = num_pixels
                    else:
                        # 正常执行 k-means
                        # 注意：需确保 kmeans 函数能处理 device 问题
                        cluster_centers, cluster_labels = simple_kmeans(coords, num_clusters=N0)
                        valid_clusters = N0
                    
                    # --- 处理有效的聚类区域 ---
                    for i in range(valid_clusters):
                        # 获取当前聚类的像素索引
                        idx = (cluster_labels == i).nonzero(as_tuple=True)[0]
                        
                        # 创建掩码
                        mask_region = torch.zeros(H, W, device=device)
                        
                        if len(idx) > 0:
                            # 填充 mask
                            # 注意 coords 是 (2, N)，即 (y, x) 还是 (x, y) 取决于你的 nonzero 顺序
                            # 通常 nonzero 返回 (y, x)
                            y_coords = coords[0, idx].long()
                            x_coords = coords[1, idx].long()
                            mask_region[y_coords, x_coords] = 1
                        
                        # 概率掩码乘积
                        prob_mask = logits[b, t] * mask_region
                        current_masks.append(prob_mask)
                        
                        # 找到中心点
                        # 这里的逻辑是找离聚类中心最近的真实像素点
                        if len(idx) > 0:
                            dists = torch.norm(coords[:, idx] - cluster_centers[:, i].unsqueeze(1), dim=0)
                            min_dist_idx = torch.argmin(dists)
                            # 还原回在 coords 中的全局索引
                            center_idx = idx[min_dist_idx]
                            current_points.append([coords[0, center_idx].item(), coords[1, center_idx].item()])
                        else:
                            # 理论上 k-means 不会产生空簇，但在特殊初始化下可能发生，做个防守
                            current_points.append([0, 0])

                    # --- 填充 (Padding) ---
                    # 如果有效聚类数小于 N0（即 num_pixels < N0 的情况），补齐剩余的空位
                    for _ in range(N0 - valid_clusters):
                        current_masks.append(torch.zeros(H, W, device=device))
                        current_points.append([0, 0])
                    
                    # --- 将当前 batch/class 的结果存入总列表 ---
                    # 此时 current_masks 长度必然为 5
                    pseudo_masks_list.extend(current_masks)
                    pseudo_points_list.extend(current_points)
            # 转换张量
            # 此时列表长度保证为 B * T * N0
            pseudo_masks = torch.stack(pseudo_masks_list).view(B, T, N0, H, W)
            pseudo_points = torch.tensor(pseudo_points_list, device=device).view(B, T, N0, 2)

            return pseudo_masks, pseudo_points

        pseudo_masks, pseudo_points = get_PseudoMasksfromMasks(final_seg_logits.clone())

        visualize_pseudo_masks_and_points_inGT(
            pseudo_masks = pseudo_masks,
            pseudo_points = pseudo_points,
            class_names = self.vis_class_names, 
            original_image_tensor = original_image, 
            gt_sem_seg= batched_inputs[0]["sem_seg"],
            save_path=f"./pseudo_masks_fromMasks/{file_names[0]}_"
        )

        # pseudo_masks, pseudo_points = get_PseudoMasks(final_seg_logits.clone())

        # visualize_pseudo_masks_and_points_inGT(
        #     pseudo_masks = pseudo_masks,
        #     pseudo_points = pseudo_points,
        #     class_names = self.vis_class_names, 
        #     original_image_tensor = original_image, 
        #     gt_sem_seg= batched_inputs[0]["sem_seg"],
        #     save_path=f"./pseudo_masks/{file_names[0]}_"
        # )

        # visualize_pseudo_masks_and_points(
        #     pseudo_masks = pseudo_masks,
        #     pseudo_points = pseudo_points,
        #     class_names= self.vis_class_names, 
        #     original_image_tensor= original_image, # (B, 3, H, W)
        #     save_path=f"./pseudo_masks/{file_names[0]}_"
        # )       





        def post_process(seg_probs):
        
            area_thd = 28.1 # 当前最佳 8.5 
            if self.backbone_name == 'CLIP':
                corr_prob = seg_probs[:, :-1, :, :].clone()  # B T H W 去除void
            else:
                corr_prob = seg_probs.clone()
            pred_cls = corr_prob.argmax(dim=1) # B H W 最大索引为T-1
            pred_mask = F.one_hot(pred_cls, num_classes=corr_prob.size(1)) # B H W T
            area = pred_mask.sum(dim=(1, 2))  # [B, T]
            valid_area_cls = area > area_thd
            valid_area_mask = torch.einsum('bhwt, bt -> bhwt', pred_mask, valid_area_cls)

            corr_prob = corr_prob * valid_area_mask.permute(0, 3, 1, 2).contiguous() # B T H W
            corr_prob = F.softmax(corr_prob, dim=1)

            # 将corr_prob上采样到原始图像大小
            original_h, original_w = batched_inputs[0]["height"], batched_inputs[0]["width"]
            corr_prob = F.interpolate(corr_prob, size=(original_h, original_w), mode='bilinear', align_corners=False)
            
            max_prob, pred_result = corr_prob.max(dim=1) # B H W 最大索引为T-1

            return pred_result

        pred_result = post_process(seg_probs)
        

        # visualize_segmentation(pred_result, self.vis_class_names+['void'],batched_inputs[0]["image"],f"./show/{file_names[0]}_")



        mask_results = F.one_hot(pred_result, num_classes=seg_probs.shape[1]).permute(0, 3, 1, 2).float() # B T H W
        # mask_results = mask_results[0].detach() # T H W

        if self.training:
            pass
        else:
            original_h, original_w = batched_inputs[0]["height"], batched_inputs[0]["width"]
            # mask_results = F.interpolate(mask_results, size=(original_h, original_w), mode='bilinear', align_corners=False)[0,:-1]     
            # print("mask_results:", mask_results.shape) # mask_results: torch.Size([150, 512, 683]) ade20k
            mask_results = retry_if_cuda_oom(sem_seg_postprocess)(mask_results, images.image_sizes[0], original_h, original_w)
            return [{"sem_seg": mask_results}] # 去除void



    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred, dataname):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()
        num_classes = len(self.test_metadata[dataname].stuff_classes)
        keep = labels.ne(num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.test_metadata[dataname].thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
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
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        # if this is panoptic segmentation
        if self.panoptic_on:
            num_classes = len(self.test_metadata[dataname].stuff_classes)
        else:
            num_classes = len(self.test_metadata[dataname].thing_classes)
        labels = torch.arange(num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // num_classes
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]

        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.test_metadata[dataname].thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result


    #错误的函数
    def visual_prediction_forward_convnext(self, x):
        batch, channel, h, w = x.shape
        x = x.reshape(batch*h*w, channel).unsqueeze(-1).unsqueeze(-1) # fake 2D input
        x = self.backbone.clip_model.visual.trunk.head(x)
        x = self.backbone.clip_model.visual.head(x)
        return x.reshape(batch, h, w, x.shape[-1]).permute(0,3,1,2) 
    
    def visual_prediction_forward_convnext_2d(self, x):
        
        clip_vis_dense = self.backbone.clip_model.visual.trunk.head.norm(x)
        clip_vis_dense = self.backbone.clip_model.visual.trunk.head.drop(clip_vis_dense.permute(0, 2, 3, 1))
        clip_vis_dense = self.backbone.clip_model.visual.head(clip_vis_dense).permute(0, 3, 1, 2)
        
        return clip_vis_dense

    def sem_seg_2_gt_masks(self, sem_seg, height, width):
        classes = torch.unique(sem_seg,sorted=False,return_inverse=False,return_counts=False)
        gt_labels = classes[classes != 255]
        masks = [sem_seg == class_id for class_id in gt_labels]

        if len(masks) == 0:
            gt_masks = torch.zeros((0, sem_seg.shape[-2],
                                            sem_seg.shape[-1])).to(sem_seg)
        else:
            gt_masks = torch.stack(masks).squeeze(1)
            
        num_masks = gt_masks.shape[0]
        total_masks = torch.zeros((num_masks, gt_masks.shape[1], gt_masks.shape[2]), dtype=gt_masks.dtype, device=gt_masks.device)
        labels = torch.zeros((num_masks), device=gt_masks.device)
        
        total_masks[:num_masks] = gt_masks[:num_masks]
        labels[:num_masks] = gt_labels[:num_masks]
        
        return total_masks.float(), labels

def simple_kmeans(coords, num_clusters, max_iter=100, tol=1e-4):
    """
    一个简单的 PyTorch K-Means 实现。
    Args:
        coords: (2, N) or (D, N) 数据点
        num_clusters: 聚类数量
    Returns:
        cluster_centers: (2, K) 聚类中心
        cluster_labels: (N,) 每个点的标签
    """
    D, N = coords.shape
    device = coords.device
    
    # 随机初始化中心
    # 从数据中随机选择 num_clusters 个点作为初始中心
    perm = torch.randperm(N, device=device)
    centers = coords[:, perm[:num_clusters]] # (D, K)
    
    for i in range(max_iter):
        old_centers = centers.clone()
        
        # 计算距离: (D, N, 1) - (D, 1, K) -> (D, N, K) -> norm -> (N, K)
        # 这里为了省显存，手动展开计算
        # dists = torch.norm(coords.unsqueeze(2) - centers.unsqueeze(1), dim=0) 
        # 上面写法显存占用大，改用下式：
        dists = torch.cdist(coords.t(), centers.t()) # (N, K)
        
        # 分配标签
        labels = torch.argmin(dists, dim=1) # (N,)
        
        # 更新中心
        for k in range(num_clusters):
            mask = labels == k
            if mask.sum() > 0:
                centers[:, k] = coords[:, mask].mean(dim=1)
            else:
                # 处理空簇：重新随机选择一个点
                idx = torch.randint(0, N, (1,), device=device)
                centers[:, k] = coords[:, idx].squeeze()
                
        # 检查收敛
        if torch.norm(centers - old_centers) < tol:
            break
            
    return centers, labels

def visualize_segmentation(pred_result, class_names,original_image_tensor, save_path="./show/",fig_size=(10, 10)):
    """
    可视化初步分割结果并将其保存到文件。
    图例会根据每个类别占有的像素数从多到少进行排序。

    Arguments:
        pred_result (torch.Tensor): 模型预测的分割结果，形状为 (H, W)，值为类别索引。
        class_names (list): 一个包含分类器所有类别实际名称的列表。
        save_path (str): 可视化结果的保存路径。
    """
    print("类别数：", len(class_names))

   
    # 确保pred_result在CPU上并且是numpy数组
    if isinstance(pred_result, torch.Tensor):
        pred_result = pred_result.cpu().numpy()

    # 检查是否是批处理的结果，如果是，则只取第一个样本
    if len(pred_result.shape) == 3 and pred_result.shape[0] == 1:
        pred_result = pred_result[0]
    
    height, width = pred_result.shape
    num_classes = len(class_names)

    # 1. 为所有可能的类别生成一个固定的随机颜色调色板
    np.random.seed(0) # 使用固定的种子以确保每次颜色一致
    palette = np.random.randint(0, 255, size=(num_classes, 3))

    # 2. 创建一个彩色的图像（与之前相同）
    color_image = np.zeros((height, width, 3), dtype=np.uint8)
    for class_index in range(num_classes):
        color_image[pred_result == class_index] = palette[class_index]

    # 3. 统计每个类别的像素数
    # np.unique 返回图像中实际出现过的类别索引和它们对应的像素数
    unique_classes, pixel_counts = np.unique(pred_result, return_counts=True)
    
    # 4. 将统计结果与类名结合，并按像素数降序排序
    class_statistics = []
    for i, class_index in enumerate(unique_classes):
        if class_index < num_classes:
            class_statistics.append({
                "index": class_index,
                "name": class_names[class_index],
                "count": pixel_counts[i]
            })
    sorted_class_statistics = sorted(class_statistics, key=lambda x: x['count'], reverse=True)

    # 创建目录
    os.makedirs(os.path.dirname(save_path + "prediction.png"), exist_ok=True)

    # 保存原图
    original_image = original_image_tensor.permute(1, 2, 0).numpy().astype(np.uint8).copy()
    plt.imsave(save_path + 'original_image.png', original_image)

    # 绘制分割结果
    fig, ax = plt.subplots(figsize=fig_size)
    ax.imshow(color_image)
    ax.axis('off')

    # 图例放在底部
    legend_elements = []
    for stats in sorted_class_statistics:
        class_index = stats["index"]
        class_name = stats["name"]
        pixel_count = stats["count"]
        color = palette[class_index] / 255.0
        label = f"{class_name} ({pixel_count:,} px)"
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, color=color, label=label))

    # 使用 fig.legend 放置在底部
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.05),
        ncol=min(4, len(legend_elements)),  # 一行最多显示4个类别
        frameon=False
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # 给图例留空间

    # 保存
    pred_save_path = save_path + "prediction.png"
    try:
        plt.savefig(pred_save_path, bbox_inches='tight')
        print(f"可视化结果已保存至: {pred_save_path}")
    except Exception as e:
        print(f"保存文件时出错: {e}")

    plt.close(fig)


def visualize_pseudo_masks_and_points(pseudo_masks, pseudo_points, class_names, original_image_tensor, save_path , batch_idx=0):
    """
    可视化生成的伪掩码聚类结果和中心点。
    【无过滤版】
    - 不进行二值化阈值过滤。
    - 使用 Alpha 通道展示概率强弱（热力图效果）。
    - 只要有非零数值，就会被画出来。
    """
    
    # ---------------------------------------------------------
    # 1. 图像数据准备
    # ---------------------------------------------------------
    img_tensor = original_image_tensor.cpu()
    
    # 处理 Batch 维度
    if img_tensor.dim() == 4:
        img_tensor = img_tensor[batch_idx] 
    
    # 转换为 Numpy (H, W, 3)
    img = img_tensor.permute(1, 2, 0).numpy()
    
    # 归一化/类型转换检查
    if img.dtype == np.float32 or img.dtype == np.float64:
        if img.max() <= 1.05:
            img = img * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    H_img, W_img = img.shape[:2]

    # ---------------------------------------------------------
    # 2. 掩码与点数据准备 (自动上采样)
    # ---------------------------------------------------------
    masks_tensor = pseudo_masks[batch_idx].detach().float().cpu() 
    points_tensor = pseudo_points[batch_idx].detach().float().cpu()

    T, N0, H_mask, W_mask = masks_tensor.shape
    
    if H_mask != H_img or W_mask != W_img:
        scale_y = H_img / H_mask
        scale_x = W_img / W_mask
        
        # 掩码上采样
        masks_reshaped = masks_tensor.view(1, T * N0, H_mask, W_mask)
        masks_upsampled = F.interpolate(
            masks_reshaped, 
            size=(H_img, W_img), 
            mode='bilinear', 
            align_corners=False
        )
        masks_tensor = masks_upsampled.view(T, N0, H_img, W_img)
        
        # 坐标缩放
        points_tensor[..., 0] *= scale_y 
        points_tensor[..., 1] *= scale_x 

    masks = masks_tensor.numpy()
    points = points_tensor.numpy()

    # ---------------------------------------------------------
    # 3. 可视化循环
    # ---------------------------------------------------------
    os.makedirs(save_path, exist_ok=True)
    cmap = plt.get_cmap('tab10')
    
    saved_count = 0
    print(f"开始处理 {T} 个类别的伪掩码 (无过滤模式)...")

    for t in range(T):
        # 处理同义词列表名称
        raw_name = class_names[t]
        if isinstance(raw_name, list):
            class_name = str(raw_name[0])
        else:
            class_name = str(raw_name)

        class_masks = masks[t]   # (N0, H, W)
        class_points = points[t] # (N0, 2)
        
        # 【唯一保留的过滤】：全零的掩码不画，否则会生成几百张纯原图，毫无意义。
        # 这里判断非常宽松，只要最大值大于0就画。
        if np.max(class_masks) <= 0:
            continue

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"Class: {class_name} (ID: {t}) - Raw output", fontsize=15)
        
        legend_patches = []
        has_content = False 
        
        for n in range(N0):
            mask_region = class_masks[n] # (H, W)
            center_point = class_points[n] # (y, x)
            
            mask_max = np.max(mask_region)
            mask_min = np.min(mask_region)
            
            # 如果最大值是0，说明这个聚类是空的（Padding产生的）
            if mask_max <= 0:
                continue
            
            # ---【核心修改：移除二值化，使用归一化热力图】---
            # 1. 相对归一化：将极小的数值映射到 0~1，以便肉眼可见
            # 如果不这样做，spatial softmax 的值太小，画出来全是透明的
            if mask_max - mask_min > 0:
                norm_mask = (mask_region - mask_min) / (mask_max - mask_min)
            else:
                norm_mask = mask_region # 全是平的
            
            has_content = True
            
            # 2. 绘制：将归一化后的掩码直接赋值给 Alpha 通道
            color_mask = np.zeros((H_img, W_img, 4))
            r, g, b = cmap(n)[:3]
            
            color_mask[..., 0] = r
            color_mask[..., 1] = g
            color_mask[..., 2] = b
            
            # Alpha通道 = 掩码强度 * 0.7 (最大不透明度0.7)
            # 这样你可以看到：哪里概率高（深色），哪里概率低（浅色）
            color_mask[..., 3] = norm_mask * 0.7 
            
            ax.imshow(color_mask)
            
            # 3. 绘制中心点
            pt_y, pt_x = center_point
            
            # 即使是 Padding 的点 (0,0)，只要在这个循环里，我们也画出来（除非越界）
            if 0 <= pt_x < W_img and 0 <= pt_y < H_img:
                # 白底黑圈点
                ax.scatter(pt_x, pt_y, c='white', s=120, edgecolors='black', marker='o', linewidth=2, zorder=10)
                ax.text(pt_x + 5, pt_y + 5, f"{n}", color='white', fontsize=12, fontweight='bold', 
                        bbox=dict(facecolor='black', alpha=0.6, pad=1))
            
            legend_patches.append(mpatches.Patch(color=(r, g, b), label=f'Cluster {n}'))

        if has_content:
            if legend_patches:
                ax.legend(handles=legend_patches, loc='upper right', fontsize=12, framealpha=0.8)
            
            safe_class_name = class_name.replace(" ", "_").replace("/", "-").replace(",", "")
            save_filename = os.path.join(save_path, f"{t:03d}_{safe_class_name}.png")
            
            try:
                plt.tight_layout()
                plt.savefig(save_filename, bbox_inches='tight', dpi=100)
                saved_count += 1
            except Exception as e:
                print(f"Error saving {class_name}: {e}")
        
        plt.close(fig)

    print(f"可视化完成。路径: {save_path}，共保存类别: {saved_count}")

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

def visualize_pseudo_masks_and_points_inGT(pseudo_masks, pseudo_points, class_names, original_image_tensor, gt_sem_seg=None, save_path="./show_pseudo/", batch_idx=0):
    """
    可视化生成的伪掩码聚类结果和中心点。
    【GT 过滤版 + 无阈值显示】
    
    Arguments:
        pseudo_masks (torch.Tensor): (B, T, N0, H_mask, W_mask)
        pseudo_points (torch.Tensor): (B, T, N0, 2)
        class_names (list): 类别名称列表
        original_image_tensor (torch.Tensor): (B, 3, H, W) 或 (3, H, W)
        gt_sem_seg (torch.Tensor, optional): (H, W) 或 (B, H, W) 的语义分割真值标签。
                                            如果提供，只保存该标签中存在的类别。
        save_path (str): 保存路径
        batch_idx (int): 处理 batch 中的哪张图
    """
    
    # ---------------------------------------------------------
    # 0. GT 标签过滤准备
    # ---------------------------------------------------------
    valid_gt_indices = None
    if gt_sem_seg is not None:
        # 处理 Batch 维度，取出当前图片的 GT
        if gt_sem_seg.dim() == 3:
            current_gt = gt_sem_seg[batch_idx]
        else:
            current_gt = gt_sem_seg
            
        # 调用辅助函数获取存在的类别 ID 列表
        valid_gt_indices = get_gt_labels_from_sem_seg(current_gt)
        # print(f"当前图片包含的 GT 类别索引: {valid_gt_indices}")

    # ---------------------------------------------------------
    # 1. 图像数据准备
    # ---------------------------------------------------------
    img_tensor = original_image_tensor.cpu()
    if img_tensor.dim() == 4:
        img_tensor = img_tensor[batch_idx] 
    
    img = img_tensor.permute(1, 2, 0).numpy()
    
    if img.dtype == np.float32 or img.dtype == np.float64:
        if img.max() <= 1.05:
            img = img * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    H_img, W_img = img.shape[:2]

    # ---------------------------------------------------------
    # 2. 掩码与点数据准备 (自动上采样)
    # ---------------------------------------------------------
    masks_tensor = pseudo_masks[batch_idx].detach().float().cpu() 
    points_tensor = pseudo_points[batch_idx].detach().float().cpu()

    T, N0, H_mask, W_mask = masks_tensor.shape
    
    if H_mask != H_img or W_mask != W_img:
        scale_y = H_img / H_mask
        scale_x = W_img / W_mask
        
        masks_reshaped = masks_tensor.view(1, T * N0, H_mask, W_mask)
        masks_upsampled = F.interpolate(
            masks_reshaped, 
            size=(H_img, W_img), 
            mode='bilinear', 
            align_corners=False
        )
        masks_tensor = masks_upsampled.view(T, N0, H_img, W_img)
        points_tensor[..., 0] *= scale_y 
        points_tensor[..., 1] *= scale_x 

    masks = masks_tensor.numpy()
    points = points_tensor.numpy()

    # ---------------------------------------------------------
    # 3. 可视化循环
    # ---------------------------------------------------------
    os.makedirs(save_path, exist_ok=True)
    cmap = plt.get_cmap('tab10')
    
    saved_count = 0
    # print(f"开始可视化... (仅保存 GT 中存在的类别)")

    for t in range(T):
        # ---【新增逻辑：GT 过滤】---
        # 如果提供了 GT，且当前类别 t 不在 GT 列表中，直接跳过
        if valid_gt_indices is not None:
            if t not in valid_gt_indices:
                continue
        
        # 处理类别名称
        raw_name = class_names[t]
        if isinstance(raw_name, list):
            class_name = str(raw_name[0])
        else:
            class_name = str(raw_name)

        class_masks = masks[t]   
        class_points = points[t] 
        
        # 即使在 GT 中，如果模型完全没有预测出值（全是0），也可以跳过，或者选择强制画出来看模型到底有多差
        # 这里为了稳健，还是保留“如果全是0就不画”的底线逻辑
        if np.max(class_masks) <= 0:
            print(f"Warning: Class {class_name} (ID {t}) exists in GT but model prediction is empty.")
            continue

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"Class: {class_name} (ID: {t}) - GT Exists", fontsize=15)
        
        legend_patches = []
        has_content = False 
        
        for n in range(N0):
            mask_region = class_masks[n]
            center_point = class_points[n]
            
            mask_max = np.max(mask_region)
            mask_min = np.min(mask_region)
            
            if mask_max <= 0:
                continue
            
            # 相对归一化 (热力图效果)
            if mask_max - mask_min > 0:
                norm_mask = (mask_region - mask_min) / (mask_max - mask_min)
            else:
                norm_mask = mask_region
            
            has_content = True
            
            # 绘制 (Alpha 通道映射概率)
            color_mask = np.zeros((H_img, W_img, 4))
            r, g, b = cmap(n)[:3]
            color_mask[..., 0] = r
            color_mask[..., 1] = g
            color_mask[..., 2] = b
            color_mask[..., 3] = norm_mask * 0.7 
            
            ax.imshow(color_mask)
            
            # 绘制点
            pt_y, pt_x = center_point
            if 0 <= pt_x < W_img and 0 <= pt_y < H_img:
                ax.scatter(pt_x, pt_y, c='white', s=120, edgecolors='black', marker='o', linewidth=2, zorder=10)
                ax.text(pt_x + 5, pt_y + 5, f"{n}", color='white', fontsize=12, fontweight='bold', 
                        bbox=dict(facecolor='black', alpha=0.6, pad=1))
            
            legend_patches.append(mpatches.Patch(color=(r, g, b), label=f'Cluster {n}'))

        if has_content:
            if legend_patches:
                ax.legend(handles=legend_patches, loc='upper right', fontsize=12, framealpha=0.8)
            
            safe_class_name = class_name.replace(" ", "_").replace("/", "-").replace(",", "")
            # 文件名加上 GT 标记，方便区分
            save_filename = os.path.join(save_path, f"GT_{t:03d}_{safe_class_name}.png")
            
            try:
                plt.tight_layout()
                plt.savefig(save_filename, bbox_inches='tight', dpi=100)
                saved_count += 1
            except Exception as e:
                print(f"Error saving {class_name}: {e}")
        
        plt.close(fig)

    print(f"可视化完成。仅保存了 GT 中存在的 {saved_count} 个类别结果至: {save_path}")
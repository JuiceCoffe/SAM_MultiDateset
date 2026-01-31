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

import os
import numpy as np
import torchvision

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
from .dinov3txt import DINO_PROMPT_TEMPLATES
SIMPLE_TEMPLATES = ["{}"]


from .loss.matcher import HungarianMatcher
from .loss.criterion import SetCriterion
from sam3.model.content_dependent_transfer import ContentDependentTransfer
from sam3.model.box_ops import masks_to_boxes, box_xyxy_to_cxcywh


from maft.modeling.transformer_decoder.fcclip_transformer_decoder import MaskPooling, get_classification_logits

import random

import math

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

@META_ARCH_REGISTRY.register()
class SAM3MC_o365(nn.Module):
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

        if cfg.MODEL.SAM3.USE_VILD_PROMPT:
            self.PROMPT = VILD_PROMPT
        else:
            self.PROMPT = ["{}"]
        # -------------------------------------------------------
        # 新增模块
        # -------------------------------------------------------
        self.new_score_head = cfg.MODEL.SAM3.NEW_SCORE_HEAD
        if self.new_score_head:
            self.score_head = MLP(256, 256, 1, 3)
            init_score_head(self.score_head)

        self.use_pe_text = cfg.MODEL.SAM3.USE_PE_TEXT
        
        self.use_cos_sim = getattr(cfg.MODEL.SAM3, "COS_SIM", False) # 默认为 False


        # 通过 cfg 控制是否启用，硬编码输入输出维度为 256
        self.use_query_proj = cfg.MODEL.SAM3.USE_QUERY_PROJ
        if self.use_query_proj:
            if self.use_pe_text:
                self.query_proj = nn.Linear(256, 1024, bias=False)
            else:
                self.query_proj = MLP(
                    input_dim=256,
                    hidden_dim=2048,
                    output_dim=256,
                    num_layers=2,
                    dropout=0.1,
                    residual=True,
                    out_norm=nn.LayerNorm(256),
                )
                init_query_proj(self.query_proj)
        else:
            self.query_proj = None


        self.num_decoder_layers = 6 # SAM3/DETR 标准层数
        
        self.logit_bias = None

        if not self.new_score_head:
            prior_prob = 0.01
            bias_value = -np.log((1 - prior_prob) / prior_prob)
            self.logit_bias = nn.ParameterList([
                nn.Parameter(torch.ones([]) * bias_value) 
                for _ in range(self.num_decoder_layers)
            ])

        if self.use_cos_sim:
            init_scale_value = np.log(1 / 0.07) # 这是一个经验值
            self.logit_scale = nn.ParameterList([
                nn.Parameter(torch.ones([]) * init_scale_value) 
                for _ in range(self.num_decoder_layers)
            ])

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


        self.text_encoder_cache = {} 


        self.Teacher = cfg.MODEL.TEACHER
        self.Teacher_MaskPool = cfg.MODEL.TEACHER_MASKPOOL and self.Teacher is not None
        if self.Teacher_MaskPool:
            self.mask_pooling = MaskPooling()

        self.use_MaskAdapter = cfg.MODEL.USE_MASKADAPTER
        if self.use_MaskAdapter:
            from .mask_adapter_head import load_mask_adapter_standalone
            self.mask_adapter = load_mask_adapter_standalone(
                weight_path="/data/hmp/MaskAdapterWeight/adapter_stage1.pth",
                clip_model_name="fcclip_convnext_large", # 只要包含 '_large' 即可
                num_channels=768,              
                num_output_maps=16,           
                mask_in_chans=16,           
                use_checkpoint=False         
            )
            self.mask_adapter.eval()
            self.num_output_maps = 16

        
        self.text_classifier2 = None
        if self.Teacher == "DINOv3TXT":
            from .dinov3txt import DINOv3TXT
            self.backbone2 = DINOv3TXT()
        elif self.Teacher == "CONVCLIP":
            from .clip import CLIP
            self.backbone2 = CLIP(cfg)
        elif self.Teacher == "PE":
            from .PEEncoder import PEEncoder
            self.backbone2 = PEEncoder(cfg)
        elif self.Teacher == "SigLIP2":
            from .SigLIP2 import SigLIP2Backbone
            self.backbone2 = SigLIP2Backbone()
        
        # -------------------------------------------------------
        # 训练配置
        # -------------------------------------------------------
        # 你需要检查 sam3_loss 的初始化参数
        self.train_dataname = None
        self.test_dataname = None
        self.test_metadata = {i: MetadataCatalog.get(i) for i in cfg.DATASETS.TEST}
        # 【修改】建立 Metadata 字典，方便后续查找
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
        # -------------------------------------------------------
        # criterion损失函数
        # -------------------------------------------------------

        

        # loss weights
        class_weight = cfg.SOLVER.CLASS_WEIGHT
        dice_weight = cfg.SOLVER.DICE_WEIGHT
        mask_weight = cfg.SOLVER.MASK_WEIGHT
        bbox_weight = cfg.SOLVER.BBOX_WEIGHT
        giou_weight = cfg.SOLVER.GIOU_WEIGHT

        objectness_weight = cfg.SOLVER.OBJECT_WEIGHT


        weight_dict = {}
        criterion_weight_dict = {
            "loss_mask": mask_weight, 
            "loss_dice": dice_weight,
            'loss_bbox':bbox_weight, 
            'loss_giou':giou_weight
        }
        if self.new_score_head:
            criterion_weight_dict["loss_objectness"] = objectness_weight
            criterion_weight_dict["loss_focal"] = class_weight
        else:
            criterion_weight_dict["loss_focal"] = class_weight

        weight_dict.update(criterion_weight_dict)

        if self.use_aux:
            for i in range (5):
                for k in criterion_weight_dict.keys():
                    weight_dict[f"{k}_{i}"] = criterion_weight_dict[k]
        
        self.encoder_loss = cfg.MODEL.SAM3.ENCODER_LOSS or cfg.MODEL.SAM3.DYNAMIC_QUERY
        if self.encoder_loss:
            self.num_encoder_query = cfg.MODEL.SAM3.NUM_ENCODER_QUERY
            encoder_losses = ["labels", "boxes",]
            encoder_weight_dict = {
                "loss_focal": class_weight,  
                'loss_bbox':bbox_weight, 
                'loss_giou':giou_weight
            }

            encoder_matcher = HungarianMatcher(
                cost_class=class_weight,
                cost_mask=mask_weight,
                cost_dice=dice_weight,
                num_points=cfg.SOLVER.TRAIN_NUM_POINTS,
                use_mask = False,
            )

            self.encoder_criterion = SetCriterion(
                matcher=encoder_matcher,
                weight_dict=encoder_weight_dict,
                losses=encoder_losses,
                num_points=cfg.SOLVER.TRAIN_NUM_POINTS,
                oversample_ratio=cfg.SOLVER.OVERSAMPLE_RATIO,
                importance_sample_ratio=cfg.SOLVER.IMPORTANCE_SAMPLE_RATIO,
            )

            
            prior_prob = 0.01
            bias_value = -np.log((1 - prior_prob) / prior_prob)
            self.encoder_logit_bias = nn.Parameter(torch.ones([]) * bias_value)

            if self.use_cos_sim:
                self.encoder_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


        losses = ["labels", "masks", "boxes"]
        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            cost_bbox=bbox_weight, # 新增：用于匹配计算
            cost_giou=giou_weight, # 新增：用于匹配计算
            num_points=cfg.SOLVER.TRAIN_NUM_POINTS,
        )
        

        self.criterion = SetCriterion(
            matcher=matcher,
            weight_dict=weight_dict,
            losses=losses,
            num_points=cfg.SOLVER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.SOLVER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.SOLVER.IMPORTANCE_SAMPLE_RATIO,
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

    def _freeze(self, ):
        for name, param in self.named_parameters():
            if 'backbone' in name:
                param.requires_grad = False
            # elif 'dot_prod_scoring' in name:
            #     param.requires_grad = False
            elif 'geometry_encoder' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        print('='*10,'Parameters to be trained', '='*10)
        for name, param in self.named_parameters():
            if param.requires_grad == True:
                print(name)
        # exit()

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

    def get_teacher_text_classifier(self, dataname, prompt_teacher):
        if self.training:
            if self.train_dataname != dataname:
                text_classifier = []
                # this is needed to avoid oom, which may happen when num of class is large
                bs = 128
                # print("train_class_names len: ",len(self.train_class_names)) 4592
                # print("train_class_names: ",self.train_class_names) 带模板的类别名
                # exit()
                for idx in range(0, len(self.train_class_names), bs):
                    text_classifier.append(self.backbone2.get_text_classifier(self.train_class_names[idx:idx+bs], self.device).detach())
                text_classifier = torch.cat(text_classifier, dim=0)

                # average across templates and normalization.
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                text_classifier = text_classifier.reshape(text_classifier.shape[0]//len(self.PROMPT), len(self.PROMPT), text_classifier.shape[-1]).mean(1)
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                self.text_classifier2 = text_classifier
                self.train_dataname = dataname

            return self.text_classifier2, self.train_num_templates
        else:
            if self.test_dataname != dataname or self.text_classifier2 is None:
                self.category_overlapping_mask_teacher, self.test_num_templates_teacher, self.test_class_names_teacher = self.prepare_class_names_from_metadata(self.test_metadata[dataname], self.train_metadata, prompt_teacher)
                text_classifier = []
                bs = 128
                print("Generating text classifier for", dataname, "with", len(self.test_class_names), "classes.")
                for idx in range(0, len(self.test_class_names_teacher), bs):
                    text_classifier.append(self.backbone2.get_text_classifier(self.test_class_names_teacher[idx:idx+bs], self.device).detach())
                text_classifier = torch.cat(text_classifier, dim=0)
                print("text_classifier shape before normalize:", text_classifier.shape)

                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                print("text_classifier shape before reshape:", text_classifier.shape)
                text_classifier = text_classifier.reshape(text_classifier.shape[0]//len(prompt_teacher), len(prompt_teacher), text_classifier.shape[-1]).mean(1) 
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                self.text_classifier2 = text_classifier
                self.test_dataname = dataname
            return self.text_classifier2, self.test_num_templates_teacher


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
                    print("Generating text classifier for", dataname, "with", len(self.train_class_names), "classes.")
                    for idx in range(0, len(self.train_class_names), bs):
                        state_text = self.detector.backbone.forward_text(self.train_class_names[idx:idx+bs], device=self.device)

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
                    text_feat = text_feat.reshape(text_feat.shape[0]//len(self.PROMPT), len(self.PROMPT), text_feat.shape[-2], text_feat.shape[-1]) # num_names, self.PROMPT, L, D
                    text_feat /= (text_feat.norm(dim=-1, keepdim=True) + 1e-6)
                    text_feat[language_mask.view(text_feat.shape[0],text_feat.shape[1],text_feat.shape[2])] = 0.0 # [num_names, self.PROMPT, L, D] 掩码掉 padding 部分
                    
                    language_features = text_feat.mean(1) # num_names, L, D

                    text_classifier = text_classifier.reshape(text_classifier.shape[0]//len(self.PROMPT), len(self.PROMPT), text_classifier.shape[-2], text_classifier.shape[-1]) # num_names, self.PROMPT, L, D
                    text_classifier /= (text_classifier.norm(dim=-1, keepdim=True) + 1e-6)
                    text_classifier[language_mask.view(text_classifier.shape[0],text_classifier.shape[1],text_classifier.shape[2])] = 0.0 # [num_names, self.PROMPT, L, D] 掩码掉 padding 部分
                    text_classifier = text_classifier.mean(-2) 
                    text_classifier /= (text_classifier.norm(dim=-1, keepdim=True) + 1e-6)
                    text_classifier = text_classifier.mean(1)
                    text_classifier /= (text_classifier.norm(dim=-1, keepdim=True) + 1e-6)
                    
                    self.language_features = language_features.detach() # num_names , L, D
                    self.language_mask = torch.min(language_mask.view(language_features.shape[0],len(self.PROMPT),language_features.shape[1]), dim=1).values# [num_names, L]
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
                text_feat = []
                language_mask = []
                # this is needed to avoid oom, which may happen when num of class is large
                bs = 128
                print("Generating text classifier for", dataname, "with", len(self.test_class_names), "classes.")
                for idx in range(0, len(self.test_class_names), bs):
                    state_text = self.detector.backbone.forward_text(self.test_class_names[idx:idx+bs], device=self.device)

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
                text_feat = text_feat.reshape(text_feat.shape[0]//len(self.PROMPT), len(self.PROMPT), text_feat.shape[-2], text_feat.shape[-1]) # num_names, self.PROMPT, L, D
                text_feat /= (text_feat.norm(dim=-1, keepdim=True) + 1e-6)
                text_feat[language_mask.view(text_feat.shape[0],text_feat.shape[1],text_feat.shape[2])] = 0.0 # [num_names, self.PROMPT, L, D] 掩码掉 padding 部分
                
                language_features = text_feat.mean(1) # num_names, L, D

                text_classifier = text_classifier.reshape(text_classifier.shape[0]//len(self.PROMPT), len(self.PROMPT), text_classifier.shape[-2], text_classifier.shape[-1]) # num_names, self.PROMPT, L, D
                text_classifier /= (text_classifier.norm(dim=-1, keepdim=True) + 1e-6)
                text_classifier[language_mask.view(text_classifier.shape[0],text_classifier.shape[1],text_classifier.shape[2])] = 0.0 # [num_names, self.PROMPT, L, D] 掩码掉 padding 部分
                text_classifier = text_classifier.mean(-2) 
                text_classifier /= (text_classifier.norm(dim=-1, keepdim=True) + 1e-6)
                text_classifier = text_classifier.mean(1)
                text_classifier /= (text_classifier.norm(dim=-1, keepdim=True) + 1e-6)
                
                self.language_features = language_features.detach() # num_names , L, D
                self.language_mask = torch.min(language_mask.view(language_features.shape[0],len(self.PROMPT),language_features.shape[1]), dim=1).values# [num_names, L]
                self.test_text_classifier = text_classifier.detach()
                self.test_dataname = dataname
            return self.test_text_classifier.clone(), self.test_num_templates

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        # print('='*10,'Parameters to be trained', '='*10)
        # for name, param in self.named_parameters():
        #     if param.requires_grad == True:
        #         print(name)
        # exit()


        images = [x["image"].to(self.device) for x in batched_inputs]
        # print("shape of first image:", images[0].shape)
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, 14)
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
                backbone_fpn[k] = backbone_fpn[k].detach() # 72, 144, 288

            # 语言特征
            # text_classifier:[num_names, dim] 
            # language_features:[num_names, num_templates, L, dim] language_mask:[num_names, num_templates, L]
            
            # text_classifier, num_templates = self.get_text_classifier(meta['dataname'])
            text_classifier, num_templates = self.get_text_classifier(dataname)


            # others
            geometric_prompt = self.detector._get_dummy_prompt(bs)
        
        if self.use_cdt:
            # text_classifier = self.cdt(img_feat,text_classifier)
            for layer in self.cdt:
                text_classifier = layer(img_feat,text_classifier) # 逐层通过

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
            for i,num_t in enumerate(num_templates): 
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
            language_features_input = self.language_features.expand(bs, -1, -1, -1) # (bs, num_names, L, dim)
            language_mask_input = self.language_mask.expand(bs, -1, -1) # (bs, num_names, L)

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

        if self.encoder_loss:
            if self.use_cdt:
                encoder_logits = torch.einsum("bld,bcd->blc", fusion_feat, text_classifier)
            else:
                encoder_logits = torch.einsum("bld,cd->blc", fusion_feat, text_classifier)
            
            if self.use_cos_sim:
                e_logit_scale = self.encoder_logit_scale.exp()
                e_logit_scale = torch.clamp(e_logit_scale, max=100.0)
                encoder_logits = encoder_logits * e_logit_scale + self.encoder_logit_bias
            else:
                encoder_logits = encoder_logits / (fusion_feat.shape[-1] ** 0.5) + self.encoder_logit_bias


            encoder_logits = aggregate_name_to_class_logits(encoder_logits, num_templates)

            encoder_score = encoder_logits.max(-1).values # [bs, HW]
            k_selected = self.num_encoder_query
            topk_values, topk_indices = torch.topk(encoder_score, k_selected, dim=1) # [bs, k]
            topk_indices_unsqueezed = topk_indices.unsqueeze(-1).repeat(1, 1, fusion_feat.shape[-1])
            topK_fusion_feat = torch.gather(fusion_feat, 1, topk_indices_unsqueezed) # [bs, k, D]
            num_classes = encoder_logits.shape[-1]
            encoder_cls_logits = torch.gather(
                encoder_logits, 
                1, 
                topk_indices.unsqueeze(-1).expand(-1, -1, num_classes)
            )


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
            if self.DynamicQuery:
                # ---------------- Relative Offset Prediction 修改开始 ----------------
                
                # 1. 获取全图的网格锚点 (Grid Anchors)
                # encoder_out["spatial_shapes"] 包含了特征图的高宽
                # grid_anchors shape: [1, Total_HW, 2] -> (x, y)
                grid_anchors = get_grid_reference_points(
                    encoder_out["spatial_shapes"], 
                    device=self.device
                )
                
                # 2. 扩展到 Batch 维度
                # grid_anchors shape: [bs, Total_HW, 2]
                grid_anchors = grid_anchors.expand(bs, -1, -1)
                
                # 3. 根据 TopK 索引提取对应的锚点
                # topk_indices shape: [bs, k]
                # 我们需要 gather 最后一个维度 (x, y)，所以要扩展 indices
                gather_idx = topk_indices.unsqueeze(-1).repeat(1, 1, 2) # [bs, k, 2]
                
                # topk_anchors_xy shape: [bs, k, 2]
                topk_anchors_xy = torch.gather(grid_anchors, 1, gather_idx)
                
                # 4. 构建完整的 4D 锚点 (x, y, w, h)
                # x, y 来自网格，w, h 初始化为一个较小的先验值 (例如 0.05)
                # 这样 inverse_sigmoid 不会溢出，且符合物体初始尺寸较小的假设
                anchor_wh_prior = torch.full_like(topk_anchors_xy, 0.05)
                topk_anchors = torch.cat([topk_anchors_xy, anchor_wh_prior], dim=-1) # [bs, k, 4]
                
                # 5. 计算偏移量 (Logit Space)
                # encoder_box_head 输出的是相对于锚点的偏移修正量
                delta_box_logits = self.encoder_box_head(topK_fusion_feat)
                
                # 6. 核心公式：Box = Sigmoid( Delta + InverseSigmoid(Anchor) )
                # 将锚点转换到 logit 域，加上偏移量，再转回 [0, 1] 域
                anchor_logits = inverse_sigmoid(topk_anchors)
                encoder_box = (delta_box_logits + anchor_logits).sigmoid() # [bs, k, 4]
                
                # ---------------- Relative Offset Prediction 修改结束 ----------------

                hs, reference_boxes, dec_presence_out, dec_presence_feats = (
                    self.detector.transformer.decoder(

                        tgt=topK_fusion_feat.permute(1,0,2).detach(), # TopK fusion feat

                        memory=out["encoder_hidden_states"],
                        memory_key_padding_mask=encoder_out["padding_mask"],
                        pos=encoder_out["pos_embed"],

                        reference_boxes=encoder_box.permute(1,0,2).detach(),

                        level_start_index=encoder_out["level_start_index"],
                        spatial_shapes=encoder_out["spatial_shapes"],
                        valid_ratios=encoder_out["valid_ratios"],
                        tgt_mask=None,
                        memory_text=prompt,
                        text_attention_mask=prompt_mask,
                        apply_dac=False,
                    )
                )

            else:
                query_embed = self.detector.transformer.decoder.query_embed.weight
                query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

                hs, reference_boxes, dec_presence_out, dec_presence_feats = (
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


        out_masks = outputs["pred_masks"].clone()

        out_masks = out_masks.sigmoid()

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
        # print("pred_boxes:",pred_boxes.shape)
        # print("pred_boxes_xyxy:",pred_boxes_xyxy.shape)
        

        bs, N, H, W = out_masks.shape
        C_ = text_classifier.shape[0] # num_names 

        queries_masks = out_masks # out_probs是通过与池化prompt投影卷积实现的，多类别下失效，直接用原始mask_logits

        queries = outputs["obj_queries"] # 6, bs, N, D
        pixel_embed = outputs["pixel_embed"] # bs, D, H', W'
        instance_embeds = outputs["instance_embeds"] 

        use_aux = self.use_aux and self.training
        aux_outputs = []

        obj_logits = None
        if self.new_score_head:
            obj_logits = self.score_head(queries).squeeze(-1)

        for i in range(6):
            assert queries.shape[0] == 6
            assert queries.shape[2] == N
            if use_aux or i == 5 :
                tp_queries = queries[i,:,:,:].clone() 

                cur_obj_logits = obj_logits[i] if obj_logits is not None else None

                if self.use_query_proj:
                    tp_queries = self.query_proj(tp_queries)

                if self.use_cos_sim:
                    tp_queries = F.normalize(tp_queries, dim=-1, p=2)

                if self.use_cdt:
                    query_names_results = torch.einsum("bnd,bcd->bnc", tp_queries, text_classifier) # bs, N, C
                else:
                    query_names_results = torch.einsum("bnd,cd->bnc", tp_queries, text_classifier) # bs, N, C
                

                if self.use_cos_sim:
                    cur_logit_scale = self.logit_scale[i].exp()
                    cur_logit_scale = torch.clamp(cur_logit_scale, max=100.0)
                    query_names_results = cur_logit_scale * query_names_results

                if self.logit_bias is not None:
                    query_names_results = query_names_results + self.logit_bias[i]

                query_cls_results= []
                cur_idx = 0
                for num_t in num_templates: 
                    query_cls_results.append(query_names_results[:,:, cur_idx: cur_idx + num_t].max(-1).values)
                    cur_idx += num_t
                query_cls_results = torch.stack(query_cls_results, dim=-1) # bs, N, num_classes
                # print(f"aux query_cls_results[{i}] shape:", query_cls_results.shape)
                    

                if i<5:
                    aux_out = {
                        'pred_logits': query_cls_results, 
                        'pred_masks': outputs['aux_outputs'][i]["pred_masks"], 
                        'pred_boxes': outputs['aux_outputs'][i]['pred_boxes'],
                        'pred_boxes_xyxy': outputs['aux_outputs'][i]["pred_boxes_xyxy"],
                    }
                    if cur_obj_logits is not None:
                        aux_out['pred_objectness_logits'] = cur_obj_logits
                    
                    aux_outputs.append(aux_out)
                else:
                    query_cls_results_final = query_cls_results
                    obj_logits_final = cur_obj_logits


        if self.training:
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images, batched_inputs)
            else:
                targets = None

            criterion_pred = {
                'pred_logits': query_cls_results_final,
                'pred_masks': outputs["pred_masks"],
                'pred_boxes': outputs['pred_boxes'],
                'pred_boxes_xyxy': outputs["pred_boxes_xyxy"],
                'aux_outputs': aux_outputs if use_aux is True else None,
            }
            if obj_logits_final is not None:
                criterion_pred['pred_objectness_logits'] = obj_logits_final

            losses = self.criterion(criterion_pred, targets)

            for k in list(losses.keys()):
                # print("loss:", k, losses[k].item())
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            
            if self.encoder_loss:
                encoder_outputs = {'pred_logits': encoder_cls_logits, 'pred_boxes': encoder_box}
                encoder_losses = self.encoder_criterion(encoder_outputs, targets)
                for k in list(encoder_losses.keys()):
                    # print("loss:", k, losses[k].item())
                    if k in self.encoder_criterion.weight_dict:
                        losses[k + '_encoder'] = encoder_losses[k] * self.encoder_criterion.weight_dict[k]

            return losses
        
        else:

            if self.Teacher_MaskPool:
                # ==========================================
                # 1. 图像域转换 (SAM3 -> DINOv3)
                # ==========================================

                imgs_bb2 = images.tensor * self.pixel_std + self.pixel_mean
                h_orig = batched_inputs[0]["height"]
                w_orig = batched_inputs[0]["width"]
                aligned_masks = None

                mean = None
                std = None
                if self.Teacher == "DINOv3TXT":                
                    imgs_bb2 = imgs_bb2 / 255.0  # 转换到 [0, 1]
                    mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
                elif self.Teacher == "CONVCLIP":
                    mean = torch.tensor([122.7709383, 116.7460125, 104.09373615], device=self.device).view(1, 3, 1, 1)
                    std =  torch.tensor([68.5005327, 66.6321579, 70.32316305], device=self.device).view(1, 3, 1, 1)
                    target_l = 896
                    if h_orig > w_orig:
                        new_h, new_w = target_l, int(target_l * w_orig / h_orig + 0.5)
                    else:
                        new_h, new_w = int(target_l * h_orig / w_orig + 0.5), target_l

                    imgs_bb2 = F.interpolate(imgs_bb2, size=(new_h, new_w), mode='bilinear', align_corners=False)

                    aligned_masks = F.interpolate(outputs["pred_masks"], size=(new_h//4, new_w//4), mode='bilinear', align_corners=False)

                elif self.Teacher == "PE":
                    imgs_bb2 = imgs_bb2 / 255.0  # 转换到 [0, 1]
                    mean = torch.tensor([0.5, 0.5, 0.5], device=self.device).view(1, 3, 1, 1)
                    std = torch.tensor([0.5, 0.5, 0.5], device=self.device).view(1, 3, 1, 1)
                
                if mean is not None and std is not None:
                    imgs_bb2 = (imgs_bb2 - mean) / std
                
                # ==========================================
                # 2. 提取特征
                # ==========================================
                # 你的 extract_features 内部已经包含了 resize 到 patch 倍数的逻辑 (encode_image)
                clip_feature = self.backbone2.extract_features(imgs_bb2)["clip_vis_dense"]
                if self.Teacher == "CONVCLIP":
                    clip_vis_dense = self.backbone2.visual_prediction_forward_convnext_2d(clip_feature)
                
                text_classifier2, num_templates_teacher = self.get_teacher_text_classifier(meta['dataname'], VILD_PROMPT)

                if self.use_MaskAdapter:
                    if aligned_masks is not None:
                        binary_masks = aligned_masks > 0
                    else:
                        binary_masks = outputs["pred_masks"] > 0
                    maps_for_pooling = self.mask_adapter(clip_vis_dense, binary_masks)
                    maps_for_pooling = F.interpolate(maps_for_pooling, size=clip_vis_dense.shape[-2:],
                                                mode='bilinear', align_corners=False)
                    N_maps = maps_for_pooling.size(1)
                    num_instances = N_maps // self.num_output_maps
                    maps_for_pooling = F.softmax(F.logsigmoid(maps_for_pooling).view(bs, N_maps,-1), dim=-1)
                    pooled_clip_feature = torch.bmm(maps_for_pooling, clip_feature.view(bs, clip_feature.size(1), -1).permute(0, 2, 1))
                    pooled_clip_feature = self.backbone2.visual_prediction_forward(pooled_clip_feature)
                    pooled_clip_feature = (pooled_clip_feature.reshape(bs,num_instances, self.num_output_maps, -1).mean(dim=-2).contiguous())
                    pooled_img_feat = pooled_clip_feature
                    pooled_img_feat = F.normalize(pooled_img_feat, dim=-1, p=2)
                else:
                    img_feat_for_pool = F.normalize(clip_vis_dense, dim=1, p=2)
                    mask_for_pool = F.interpolate(outputs["pred_masks"], size=img_feat_for_pool.shape[-2:],
                                                        mode='bilinear', align_corners=False)
                    pooled_img_feat = self.mask_pooling(img_feat_for_pool, mask_for_pool)
                    pooled_img_feat = F.normalize(pooled_img_feat, dim=-1, p=2)

                
                maskpool_name_logits = torch.einsum("cd,bnd->bnc", text_classifier2, pooled_img_feat) 
                if self.Teacher == "SigLIP2":
                    maskpool_cls_logits = aggregate_name_to_class_logits(maskpool_name_logits, num_templates_teacher)
                    out_vocab_cls_probs =  torch.sigmoid(maskpool_cls_logits)
                else:
                    maskpool_name_logits = maskpool_name_logits * torch.clamp(self.backbone2.clip_model.logit_scale.exp(), max=100)
                    
                    maskpool_cls_logits = aggregate_name_to_class_logits(maskpool_name_logits, num_templates_teacher)
                    out_vocab_cls_probs = F.softmax(maskpool_cls_logits, dim=-1)


                in_vocab_cls_probs = query_cls_results_final.softmax(dim=-1)
                category_overlapping_mask = self.category_overlapping_mask.to(self.device)
                alpha = 1.0
                beta = 1.0
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

                # 组合概率 (注意这里不是加 log，而是选通概率)
                final_probs = (
                    probs_seen * category_overlapping_mask + 
                    probs_unseen * (1 - category_overlapping_mask)
                )
                
                # 钳位数值防止 logit 计算出现 inf/-inf
                final_probs = torch.clamp(final_probs, min=eps, max=1-eps)
                query_cls_results_final = torch.logit(final_probs)




        # ====================== Oracle 逻辑 ====================== 
            self.OracleSelect_on = False

            if self.OracleSelect_on:
                # 临时构造 outputs 字典，用于传递给 oracle 函数
                temp_outputs = {
                    "pred_masks": outputs["pred_masks"],
                    # pred_logits 仅用于获取形状，其内容不参与计算
                    "pred_logits": query_cls_results_final 
                }
                
                # 获取为每个 prediction 分配了最佳 GT 类别的 oracle logits
                oracle_logits = self.get_oracle_logits_per_prediction(
                    temp_outputs, 
                    batched_inputs, 
                    images
                )
                
                # 直接用 oracle logits 替换模型原来的分类结果
                query_cls_results_final = oracle_logits
        # =======================================================
        
            
            mask_pred_logits = outputs["pred_masks"]  # 保持 Logits 状态
            if obj_logits_final is not None:
                obj_scores = obj_logits_final.sigmoid().unsqueeze(-1) # [B, N, 1]
                cls_probs = query_cls_results_final.sigmoid() # [B, N, C]
                final_scores = cls_probs * obj_scores 
                mask_cls_logits = torch.logit(torch.clamp(final_scores, min=1e-6, max=1-1e-6))

            else:
                mask_cls_logits = query_cls_results_final # 保持 Logits 状态

            results = []
            
            for i in range(bs):
                # 获取单张图数据
                mask_cls_i = mask_cls_logits[i]       # [Q, C]
                mask_pred_i = mask_pred_logits[i]     # [Q, H, W]
                
                # 获取原始图像尺寸
                img_h_orig = batched_inputs[i]["height"]
                img_w_orig = batched_inputs[i]["width"]
                
                # 上采样 Mask 到原始图像尺寸 (非常重要)
                # 使用 bilinear 插值 logits
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
                    mask_cls_prob = mask_cls_i.sigmoid()
                    mask_pred_prob = mask_pred_i.sigmoid()
                    semseg = torch.einsum("qc,qhw->chw", mask_cls_prob, mask_pred_prob)
                    # mask_pred_binary = (mask_pred_i.sigmoid() > 0.5).float() 
                    # semseg = torch.einsum("qc,qhw->chw", mask_cls_prob, mask_pred_binary)
                    res["sem_seg"] = semseg

                    # =========== 修改开始：为可视化准备 Square 数据 ===========
                    
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

                    # 5. 准备 Image (它本身就是 Square 的)
                    img_tensor_square = batched_inputs[i]["image"]

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
                            original_image_tensor=img_tensor_square, # 本身就是 Square
                            save_path=f"./show_{self.Teacher}_semantic/{batched_inputs[i]['file_name'].split('/')[-1].split('.')[0]}.png"
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
        # mask_cls: [Q, K] (Logits)
        # mask_pred: [Q, H, W] (Logits or Probs, depends on input)
        
        # 1. 计算分数 (Sigmoid 而不是 Softmax)
        scores, labels = mask_cls.sigmoid().max(-1) # [Q]
        mask_pred = mask_pred.sigmoid() # [Q, H, W]

        # 2. 过滤掉低分 Query
        keep = scores > self.object_mask_threshold
        
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        
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
        scores = mask_cls.sigmoid() # [Q, K]
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

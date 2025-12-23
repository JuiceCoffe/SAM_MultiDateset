# sam3/modeling_d2.py
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import boxes, ImageList, Instances, BitMasks
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

from maft.utils.text_templetes import VILD_PROMPT
# VILD_PROMPT = ["{}"]

from maft.modeling.matcher import HungarianMatcher
from maft.modeling.criterion import SetCriterion
from maft.modeling.transformer_decoder.fcclip_transformer_decoder import MaskPooling, get_classification_logits

from .NearMatcher import match_every_pred_to_best_gt

class PixelProjector(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None):
        """
        Args:
            input_dim: pixel_embed 的通道数 (通常是 SAM 解码器的输出维度，如 256)
            output_dim: text_classifier 的通道数 (即文本编码器的维度，如 CLIP 的 512/768/1024)
            hidden_dim: MLP 隐藏层维度，默认等于 input_dim
        """
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim
            
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 初始化参数：Xavier 初始化有助于训练稳定
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.net(x)


@META_ARCH_REGISTRY.register()
class SAM3MC_ora(nn.Module):
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
        transformer = _create_sam3_transformer()
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


        # -------------------------------------------------------
        # 新增模块
        # -------------------------------------------------------
        self.mask_pooling = MaskPooling()
        
        self.no_object_embed = nn.Embedding(1, 256)
        # 初始化常数，通常对背景类做一点特殊的初始化有助于收敛
        nn.init.normal_(self.no_object_embed.weight, mean=0, std=0.02)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(10))

        self.sem_seg_projector = PixelProjector(
            input_dim= 256,
            output_dim= 256,
        )

        # -------------------------------------------------------
        # 训练配置
        # -------------------------------------------------------
        # 你需要检查 sam3_loss 的初始化参数

        self.train_dataname = None
        self.test_dataname = None
        self.test_metadata = {i: MetadataCatalog.get(i) for i in cfg.DATASETS.TEST}
        self.train_metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        _, self.train_num_templates, self.train_class_names = self.prepare_class_names_from_metadata(self.train_metadata, self.train_metadata)

        self.use_aux = cfg.SOLVER.USE_AUX

        # -------------------------------------------------------
        # criterion损失函数
        # -------------------------------------------------------
        no_object_weight = cfg.SOLVER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.SOLVER.CLASS_WEIGHT
        dice_weight = cfg.SOLVER.DICE_WEIGHT
        mask_weight = cfg.SOLVER.MASK_WEIGHT

        weight_dict = {}
        criterion_weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}
        weight_dict.update(criterion_weight_dict)

        if self.use_aux:
            for i in range (5):
                for k in criterion_weight_dict.keys():
                    weight_dict[f"{k}_{i}"] = criterion_weight_dict[k]
        
        

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.SOLVER.TRAIN_NUM_POINTS,
        )
        
        losses = ["labels", "masks"]

        self.criterion = SetCriterion(
            171, 
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.SOLVER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.SOLVER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.SOLVER.IMPORTANCE_SAMPLE_RATIO,
        )

        # --------------------------------------------------
        # Oracle
        # --------------------------------------------------

        self.OracleSelect_on = True
        self.oracle_matcher = HungarianMatcher(
            cost_class=0,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.SOLVER.TRAIN_NUM_POINTS,
        )
        # --------------------------------------------------
        # sam3的其他配置
        # --------------------------------------------------

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
                language_mask = []
                # this is needed to avoid oom, which may happen when num of class is large
                bs = 128
                print("Generating text classifier for", dataname, "with", len(self.train_class_names), "classes.")
                for idx in range(0, len(self.train_class_names), bs):
                    state_text = self.detector.backbone.forward_text(self.train_class_names[idx:idx+bs], device=self.device)

                    batch_text_feat = state_text["language_features"].detach()
                    mask = state_text["language_mask"] # bs, L
                    batch_text_feat = batch_text_feat.permute(1,0,2) # -> bs, L, D 
                    text_classifier.append(batch_text_feat)
                    language_mask.append(mask) # bs, L
                text_classifier = torch.cat(text_classifier, dim=0)
                language_mask = torch.cat(language_mask, dim=0) # (num_names * VILD_PROMPT,  L)
                # average across templates and normalization.
                text_classifier = text_classifier.reshape(text_classifier.shape[0]//len(VILD_PROMPT), len(VILD_PROMPT), text_classifier.shape[-2], text_classifier.shape[-1]) # num_names, VILD_PROMPT, L, D
                text_classifier /= (text_classifier.norm(dim=-1, keepdim=True) + 1e-6)

                text_classifier[language_mask.view(text_classifier.shape[0],text_classifier.shape[1],text_classifier.shape[2])] = 0.0 # [num_names, VILD_PROMPT, L, D] 掩码掉 padding 部分
                language_features = text_classifier.mean(1) # num_names, L, D
                text_classifier = text_classifier.mean(-2) 
                text_classifier /= (text_classifier.norm(dim=-1, keepdim=True) + 1e-6)
                text_classifier = text_classifier.mean(1)
                text_classifier /= (text_classifier.norm(dim=-1, keepdim=True) + 1e-6)
                
                self.language_features = language_features.detach() # num_names , L, D
                self.language_mask = torch.min(language_mask.view(language_features.shape[0],len(VILD_PROMPT),language_features.shape[1]), dim=1).values# [num_names, L]
                self.train_text_classifier = text_classifier.detach()
                self.train_dataname = dataname
            return self.train_text_classifier, self.train_num_templates
        else:
            if self.test_dataname != dataname:
                self.category_overlapping_mask, self.test_num_templates, self.test_class_names = self.prepare_class_names_from_metadata(self.test_metadata[dataname], self.train_metadata)
                text_classifier = []
                language_mask = []
                # this is needed to avoid oom, which may happen when num of class is large
                bs = 128
                print("Generating text classifier for", dataname, "with", len(self.test_class_names), "classes.")
                for idx in range(0, len(self.test_class_names), bs):
                    state_text = self.detector.backbone.forward_text(self.test_class_names[idx:idx+bs], device=self.device)

                    batch_text_feat = state_text["language_features"].detach()
                    mask = state_text["language_mask"] # bs, L
                    batch_text_feat = batch_text_feat.permute(1,0,2) # -> bs, L, D 
                    text_classifier.append(batch_text_feat)
                    language_mask.append(mask) # bs, L
                text_classifier = torch.cat(text_classifier, dim=0)
                language_mask = torch.cat(language_mask, dim=0) # (num_names * VILD_PROMPT,  L)
                # average across templates and normalization.
                text_classifier = text_classifier.reshape(text_classifier.shape[0]//len(VILD_PROMPT), len(VILD_PROMPT), text_classifier.shape[-2], text_classifier.shape[-1]) # num_names, VILD_PROMPT, L, D
                text_classifier /= (text_classifier.norm(dim=-1, keepdim=True) + 1e-6)

                text_classifier[language_mask.view(text_classifier.shape[0],text_classifier.shape[1],text_classifier.shape[2])] = 0.0 # [num_names, VILD_PROMPT, L, D] 掩码掉 padding 部分
                language_features = text_classifier.mean(1) # num_names, L, D
                text_classifier = text_classifier.mean(-2) 
                text_classifier /= (text_classifier.norm(dim=-1, keepdim=True) + 1e-6)
                text_classifier = text_classifier.mean(1)
                text_classifier /= (text_classifier.norm(dim=-1, keepdim=True) + 1e-6)
                
                self.language_features = language_features.detach() # num_names , L, D
                self.language_mask = torch.min(language_mask.view(language_features.shape[0],len(VILD_PROMPT),language_features.shape[1]), dim=1).values# [num_names, L]
                self.test_text_classifier = text_classifier.detach()
                self.test_dataname = dataname
            return self.test_text_classifier, self.test_num_templates

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

            meta = batched_inputs[0]["meta"]
            
            # 图形特征
            backbone_out_vision = self.detector.backbone.forward_image(images.tensor)
            img_feat = backbone_out_vision["vision_features"].detach() # bs, C, H', W'
            backbone_fpn = backbone_out_vision["backbone_fpn"]
            for k in range(len(backbone_fpn)):
                backbone_fpn[k] = backbone_fpn[k].detach()


            # 语言特征
            # text_classifier:[num_names, dim] 
            # language_features:[num_names, num_templates, L, dim] language_mask:[num_names, num_templates, L]
            text_classifier, num_templates = self.get_text_classifier(meta['dataname'])
            no_obj_normalized = torch.nn.functional.normalize(self.no_object_embed.weight, p=2, dim=-1)
            text_classifier = torch.cat([text_classifier, no_obj_normalized], dim=0)# 增加一个背景类

            # others
            geometric_prompt = self.detector._get_dummy_prompt(bs)
        
        batch_gt_names_idx = []
        for i in range(bs):
            gt_classes = get_gt_labels_from_sem_seg(batched_inputs[i]["sem_seg"].to(self.device))
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
            out, hs = self.detector._run_decoder(
                memory=out["encoder_hidden_states"],
                pos_embed=encoder_out["pos_embed"],
                src_mask=encoder_out["padding_mask"],
                out=out,
                prompt=prompt,
                prompt_mask=prompt_mask,
                encoder_out=encoder_out,
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

        out_masks = outputs["pred_masks"].clone()

        out_masks = out_masks.sigmoid()


        bs, N, H, W = out_masks.shape
        C_ = text_classifier.shape[0] # num_names + 1 

        # out_logits = outputs["pred_logits"] 
        # out_probs = out_logits.sigmoid() # bs, N, 1

        # queries_masks = torch.mul(out_masks,out_probs.unsqueeze(-1)) # 每个实例的mask乘以对应的概率得分 [bs, N, H, W]

        queries_masks = out_masks # out_probs是通过与池化prompt投影卷积实现的，多类别下失效，直接用原始mask_logits

        queries = outputs["obj_queries"] # 6, bs, N, D

        text_classifier_mask = torch.zeros((C_, bs), dtype=torch.bool, device=text_classifier.device)

        
        self.use_aux = self.use_aux and self.training
        aux_outputs = []

        for i in range(6):
            if self.use_aux or i == 5 :
                tp_queries = queries[i,:,:N,:].clone() # 避免DAC造成的tp_queries翻倍
                tp_queries = F.normalize(tp_queries, dim=-1, p=2)

                # hs=tp_queries.unsqueeze(1)
                # hs = hs.expand(-1, C_, -1, -1).contiguous().view(bs*(C_), N, -1).unsqueeze(0) # 1, bs*(C+1), N, D
                # print("hs shape:", hs.shape)
                # prompt= text_classifier.unsqueeze(0).expand( bs, -1, -1).contiguous().view(1, bs * C_, -1) # 1, bs*(C+1), D
                # print("prompt shape", prompt.shape)

                out_vocab_names_results = self.detector.dot_prod_scoring( # 这里把num names当作batch维度实现并行，算出每个query对所有类别的分数
                    hs= tp_queries.unsqueeze(1).expand(-1, C_, -1, -1).contiguous().view(bs*(C_), N, -1).unsqueeze(0), # 1, bs*(C+1), N, D
                    prompt = text_classifier.unsqueeze(0).expand( bs, -1, -1).contiguous().view(1, bs * C_, -1), # 1, bs*(C+1), D
                    prompt_mask=text_classifier_mask.expand(-1, bs).view(-1, 1), # (C+1) * bs 
                ) 

                out_vocab_names_results = out_vocab_names_results.view(bs, C_, N).permute(0,2,1) # bs, N, C+1
                out_vocab_cls_results= []
                cur_idx = 0
                for num_t in num_templates: 
                    out_vocab_cls_results.append(out_vocab_names_results[:,:, cur_idx: cur_idx + num_t].max(-1).values)
                    cur_idx += num_t
                out_vocab_cls_results = torch.stack(out_vocab_cls_results, dim=-1) # bs, N, num_classes
                out_vocab_cls_results = torch.cat([out_vocab_cls_results, out_vocab_names_results[:,:,-1:]], dim=-1) # bs, N, num_classes + 1 (添加背景类)
                # print(f"aux out_vocab_cls_results[{i}] shape:", out_vocab_cls_results.shape)
                if i<5:
                    aux_outputs.append({'pred_logits': out_vocab_cls_results, 'pred_masks': outputs['aux_outputs'][i]["pred_masks"]})
                else:
                    out_vocab_cls_results_final = out_vocab_cls_results

        if self.OracleSelect_on:
            # print('='*20,"Oracle select开始",'='*20) # ==================================
            targets = []
            input_per_image = batched_inputs[0]
            height = input_per_image.get("height")
            width = input_per_image.get("width")
            sem_seg = input_per_image["sem_seg"].to(self.device)

            image_masks, image_labels = self.sem_seg_2_gt_masks(sem_seg, height, width)

            if image_labels.numel() == 0:
                print('='*20,"image_labels.numel() == 0",'='*20)
                # return self.forward(batched_inputs,no_gt_masks=True)
                return "image_labels.numel() == 0"
            
            target_for_this_image = {
                "labels": image_labels.long(), 
                "masks": image_masks
            }
            targets.append(target_for_this_image)

            mask_cls_results = out_vocab_cls_results_final
            mask_pred_results = outputs["pred_masks"]

            outputs_for_match = {"pred_logits": mask_cls_results, "pred_masks": mask_pred_results}
            # print('='*20,"准备Oracle select匹配",'='*20)
            # oracle_indices = self.oracle_matcher(outputs_for_match, targets)
            oracle_indices = match_every_pred_to_best_gt(self.oracle_matcher, outputs_for_match, targets)
            # print('='*20,"结束Oracle select匹配",'='*20)
            new_mask_cls_results = []
            new_mask_pred_results = []
            new_out_vocab_cls_results = [] 

            for i, (pred_idx, gt_idx) in enumerate(oracle_indices):
                # 1. 提取预测相关的 Mask 和原始分类结果
                selected_cls = mask_cls_results[i, pred_idx]
                selected_pred = mask_pred_results[i, pred_idx]
                
                # 2. 关键修改：获取该图片匹配到的真实类别 ID
                # targets[i]["labels"] 是该图所有 GT 的类别，使用 gt_idx 按匹配顺序重排
                matched_gt_labels = targets[i]["labels"][gt_idx] 
                
                # 3. 将 matched_gt_labels 存入结果（它现在直接就是真实的类别数字）
                # 注意：如果后续需要它跟 mask_cls_results 维度对齐，可能需要根据业务逻辑处理
                selected_out_vocab_cls = matched_gt_labels

                new_mask_cls_results.append(selected_cls)
                new_mask_pred_results.append(selected_pred)
                new_out_vocab_cls_results.append(selected_out_vocab_cls)

            # 堆叠回 Tensor
            mask_cls_results = torch.stack(new_mask_cls_results)
            mask_pred_results = torch.stack(new_mask_pred_results)
            queries_masks = mask_pred_results.sigmoid()
            # 此时 out_vocab_cls_results 里的值就是真实的物体类别 ID
            out_vocab_cls_results = torch.stack(new_out_vocab_cls_results)
            out_vocab_cls_results = F.one_hot(out_vocab_cls_results, num_classes=len(num_templates)).float()

            # print("out_vocab_cls_results:",out_vocab_cls_results.shape)


        if self.training:
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None
            
            criterion_pred = {
                'pred_logits': out_vocab_cls_results_final,
                'pred_masks': outputs["pred_masks"],
                'aux_outputs': aux_outputs if self.use_aux is True else None,
            }

            losses = self.criterion(criterion_pred, targets)

            for k in list(losses.keys()):
                # print("loss:", k, losses[k].item())
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)

            return losses
            
        else:
            out_vocab_cls_results = out_vocab_cls_results.sigmoid() 
            queries_seg_result = torch.einsum("bnc,bnhw->bchw", out_vocab_cls_results, queries_masks) # [bs, num_classes+1, H, W]


        seg_logits = queries_seg_result 

        if USE_GT_NAMES_ONLY:
            pass
            # for b in range(bs):
            #     gtcls_mask = torch.ones_like(seg_logits[b], dtype=torch.bool)
            #     gtcls_mask[batch_gt_names_idx[b],:,:] = False
            #     seg_logits[b][gtcls_mask] = 0.0
        
        final_seg_logits = seg_logits[:, :-1, :, :]
        # print("final_seg_logits shape:", final_seg_logits.shape) # bs,num_classes+1, H, W

        pred_result = final_seg_logits[0].argmax(0)
        # visualize_segmentation(pred_result, self.vis_class_names+['void'],batched_inputs[0]["image"],f"./show_queries/{file_names[0]}_")
        # visualize_segmentation(pred_result, self.vis_class_names+['void'],batched_inputs[0]["image"],f"./show_seg/{file_names[0]}_")

        # =======================================================
        


        results = []
        for i in range(bs):
            orig_size = (batched_inputs[i]["height"], batched_inputs[i]["width"]) # 没经过 resize 到 1008 的大小
            res = sem_seg_postprocess(
                final_seg_logits[i], 
                (img_h, img_w),
                orig_size[0],
                orig_size[1],
            )
            results.append({"sem_seg": res})
        # print("跑通！")
        return results

    def semantic_segmentation_loss(self, pred_logits, batched_inputs, images_tensor_shape):
        """
        Args:
            pred_logits (Tensor): 形状为 [B, NumClasses, H_feat, W_feat] 的预测结果 (通常是 Stride 4).
            batched_inputs (list[dict]): 包含 'sem_seg' 的原始输入列表.
            images_tensor_shape (tuple): images.tensor.shape, 即 [B, 3, H_pad, W_pad].
                                        我们需要知道 Padded 后的总高宽来做正确的上采样.
        """
        # 1. 获取 Padded 的目标尺寸 (即 images.tensor 的 H, W)
        # pred_logits 通常是下采样过的 (如 1/4)，我们需要先把它还原到 images.tensor 的尺度
        _, _, pad_h, pad_w = images_tensor_shape
        
        # 2. 将整个 Batch 的预测上采样到 Padded 尺寸
        # mode='bilinear' align_corners=False 是分割任务的标准做法
        pred_logits_upsampled = F.interpolate(
            pred_logits, 
            size=(pad_h, pad_w), 
            mode="bilinear", 
            align_corners=False
        )
        
        total_loss = 0.0
        valid_samples = 0
        
        # 3. 逐张图片处理：裁剪出有效区域并计算 Loss
        for i, input_per_image in enumerate(batched_inputs):
            # 获取该样本的 GT (Semantic Map)
            # 注意：sem_seg 可能在 CPU 上，需要移到 GPU
            gt_sem_seg = input_per_image["sem_seg"].to(pred_logits.device)
            
            # 获取 GT 的真实尺寸 (未 Padding 的尺寸)
            gt_h, gt_w = gt_sem_seg.shape
            
            # 从上采样后的预测中，裁剪出有效区域 (Top-Left corner)
            # Detectron2 的 padding 默认是在右侧和下侧，所以切片 [:gt_h, :gt_w] 是安全的
            valid_pred = pred_logits_upsampled[i, :, :gt_h, :gt_w]
            
            # 此时 valid_pred 的形状是 [NumClasses, gt_h, gt_w]
            # gt_sem_seg 的形状是 [gt_h, gt_w]
            # 增加一个 Batch 维度以使用 cross_entropy
            valid_pred = valid_pred.unsqueeze(0) # [1, C, H, W]
            gt_sem_seg = gt_sem_seg.unsqueeze(0) # [1, H, W]
            
            # 4. 计算 Loss
            # ignore_index=255 是通用的忽略背景/无效区域的设定，请根据你的数据集调整
            loss = F.cross_entropy(valid_pred, gt_sem_seg.long(), ignore_index=255)
            
            total_loss += loss
            valid_samples += 1

        # 平均 Loss
        return total_loss / max(valid_samples, 1)    


    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

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


def visualize_segmentation(
    pred_result, 
    class_names, 
    original_image_tensor, 
    save_path="./show/result.png", 
    fig_size=(15, 10)
):
    """
    可视化分割结果，将原始图像和分割掩码并排显示在同一张图片中，并保存到文件。
    图例会根据每个类别占有的像素数从多到少进行排序。

    Arguments:
        pred_result (torch.Tensor): 模型预测的分割结果，形状为 (H, W)，值为类别索引。
        class_names (list): 一个包含分类器所有类别实际名称的列表。
        original_image_tensor (torch.Tensor): 原始图像的张量，形状为 (C, H, W)。
        save_path (str): 可视化结果的保存路径及文件名。
        fig_size (tuple): 整张图的大小。
    """
    print(f"类别数: {len(class_names)}")

    # 确保pred_result在CPU上并且是numpy数组
    if isinstance(pred_result, torch.Tensor):
        pred_result = pred_result.cpu().numpy()

    # 检查是否是批处理的结果，如果是，则只取第一个样本
    if len(pred_result.shape) == 3 and pred_result.shape[0] == 1:
        pred_result = pred_result[0]
    
    height, width = pred_result.shape
    num_classes = len(class_names)

    # 1. 为所有可能的类别生成一个固定的随机颜色调色板
    np.random.seed(0)  # 使用固定的种子以确保每次颜色一致
    palette = np.random.randint(0, 255, size=(num_classes, 3))

    # 2. 创建一个彩色的图像用于分割结果
    color_image = np.zeros((height, width, 3), dtype=np.uint8)
    for class_index in range(num_classes):
        color_image[pred_result == class_index] = palette[class_index]

    # 3. 统计每个类别的像素数
    unique_classes, pixel_counts = np.unique(pred_result, return_counts=True)
    
    # 4. 将统计结果与类名结合，并按像素数降序排序
    class_statistics = [
        {
            "index": class_index,
            "name": class_names[class_index],
            "count": pixel_counts[i]
        }
        for i, class_index in enumerate(unique_classes) if class_index < num_classes
    ]
    sorted_class_statistics = sorted(class_statistics, key=lambda x: x['count'], reverse=True)

    # 创建保存目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 准备原始图像用于显示
    original_image = original_image_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

    # 5. 绘制并排的图像
    fig, ax = plt.subplots(1, 2, figsize=fig_size)

    # 显示原始图像
    ax[0].imshow(original_image)
    ax[0].set_title("Original Image")
    ax[0].axis('off')

    # 显示分割结果
    ax[1].imshow(color_image)
    ax[1].set_title("Segmentation Result")
    ax[1].axis('off')

    # 6. 创建并放置图例
    legend_elements = [
        Rectangle((0, 0), 1, 1, color=palette[stats["index"]] / 255.0, 
                  label=f"{stats['name']} ({stats['count']:,} px)")
        for stats in sorted_class_statistics
    ]

    # 使用 fig.legend 将图例放置在整个图的底部
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.01), # 将图例稍微向上移动一点，避免与边界重合
        ncol=min(4, len(legend_elements)),  # 一行最多显示4个类别
        frameon=False,
        fontsize='small'
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1]) # 调整布局为图例留出空间 (left, bottom, right, top)
    
    # 7. 保存最终的图像
    try:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        print(f"可视化结果已保存至: {save_path}")
    except Exception as e:
        print(f"保存文件时出错: {e}")

    plt.close(fig)


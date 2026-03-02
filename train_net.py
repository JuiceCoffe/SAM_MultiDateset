import warnings
import os

warnings.filterwarnings("ignore")

import sys
import types

try:
    # 1. 核心改进：直接“深度导入” detectron2 会用到的具体底层文件
    # 这样可以迫使 Python 提前将这些文件加载进内存。
    # 如果这里报错，说明是环境真的缺库（比如缺 cv2 或 Pillow），而不是路径问题。
    import cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling as eval_pixel
    
    # 顺便把实例分割的评测也导入（如果 detectron2 需要的话）
    try:
        import cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling as eval_instance
    except ImportError:
        eval_instance = None

except ImportError as e:
    # 拦截真实的依赖错误，并提供清晰的报错信息
    print("\n" + "="*60)
    print(f"❌ 导入 cityscapesscripts 核心模块失败！\n真实的错误原因是: {e}")
    print("👉 如果提示 'No module named cityscapesscripts'，请运行: pip install cityscapesscripts")
    print("👉 如果提示缺少 cv2、numpy 等其他库，请 pip install 对应的库。")
    print("="*60 + "\n")
    sys.exit(1)

# 2. 完美的深层别名映射 (模拟 Facebook 内部的完整路径)
# 这样无论是 detectron2 走 try 分支还是 except 走 deeplearning 分支，都会命中缓存，绝不报错
sys.modules['deeplearning'] = types.ModuleType('deeplearning')
sys.modules['deeplearning.projects'] = types.ModuleType('deeplearning.projects')
sys.modules['deeplearning.projects.cityscapesApi'] = types.ModuleType('deeplearning.projects.cityscapesApi')

# 将完整的深度模块精准挂载上去
sys.modules['deeplearning.projects.cityscapesApi.cityscapesscripts'] = sys.modules['cityscapesscripts']
sys.modules['deeplearning.projects.cityscapesApi.cityscapesscripts.evaluation'] = sys.modules['cityscapesscripts.evaluation']
sys.modules['deeplearning.projects.cityscapesApi.cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling'] = eval_pixel

if eval_instance:
    sys.modules['deeplearning.projects.cityscapesApi.cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling'] = eval_instance


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

from sam3.data.dataset_mappers.coco_combine_new_baseline_dataset_mapper_2 import COCOCombineNewBaselineDatasetMapper
from sam3.data.custom_dataset_dataloader import build_custom_train_loader
from sam3.config import add_sam3_config
from sam3.modeling_d2 import SAM3Wrapper # 导入这个类就会自动触发 REGISTER
# from sam3.SAM3MC import SAM3MC
from sam3.SAM3MC_ora import SAM3MC_ora
from sam3.SAM3MC_o365 import SAM3MC_o365
from sam3.SAM3_teacher import SAM3_TEACHER
from sam3.SAM3CLIP import SAM3CLIP
from sam3.DINOSAM import DINOSAM
from sam3.DINOTXTSAM import DINOTXTSAM
from sam3.RADIOSAM import RADIOSAM

from sam3.mask_adapter_head import MASKAdapterHead


# 临时修复 lvis 报错：AttributeError: module 'numpy' has no attribute 'float'.
import numpy as np
try:
    np.float = float
except AttributeError:
    pass  # 如果 numpy 版本较低本身支持，则忽略


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to FCCLIP.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        
        evaluator_list = []
        
        # ---------------------------------------------------------
        # 1. 实例分割 (Instance Segmentation) -> 对应 INSTANCE_ON
        #    通常用于计算 AP (Average Precision)
        # ---------------------------------------------------------
    
        if "lvis" in dataset_name:
            print("="*20,"使用lvis评估器","="*20)
            evaluator_list.append( LVISEvaluator(dataset_name, cfg, True, output_folder))
            return evaluator_list

        if "cityscape" in dataset_name:
            print("="*20,"使用cityscape评估器","="*20)
            evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
            evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))
            return evaluator_list

        if cfg.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))

        # ---------------------------------------------------------
        # 2. 全景分割 (Panoptic Segmentation) -> 对应 PANOPTIC_ON
        #    通常用于计算 PQ (Panoptic Quality)
        # ---------------------------------------------------------
        if cfg.TEST.PANOPTIC_ON:
            evaluator_list.append(
                COCOPanopticEvaluator(dataset_name, output_dir=output_folder)
            )

        # ---------------------------------------------------------
        # 3. 语义分割 (Semantic Segmentation) -> 对应 SEMANTIC_ON
        #    通常用于计算 mIoU
        # ---------------------------------------------------------
        if cfg.TEST.SEMANTIC_ON:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )

        # ---------------------------------------------------------
        # 异常处理与返回
        # ---------------------------------------------------------
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "No Evaluator generated. Please check your cfg.TEST.*_ON settings "
                "or the dataset evaluator_type."
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        
        return DatasetEvaluators(evaluator_list)


    @classmethod
    def build_train_loader(cls, cfg):
        
        if cfg.DATALOADER.SAMPLER_TRAIN == "MultiDatasetSampler":
            mapper = COCOCombineNewBaselineDatasetMapper(cfg, True) 
            data_loader = build_custom_train_loader(cfg, mapper=mapper)   
            return data_loader


        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Panoptic segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
            mapper = MaskFormerPanopticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Instance segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_instance":
            mapper = MaskFormerInstanceDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco instance segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj":
            mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco panoptic segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_panoptic_lsj":
            mapper = COCOPanopticNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco panoptic segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_semantic_lsj":
            mapper = COCOSemanticNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_combine_lsj":
            mapper = COCOCombineNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)        
        else:
            print(f"mapper {cfg.INPUT.DATASET_MAPPER_NAME}不存在！")
            exit()
            mapper = None
            return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] *  cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_sam3_config(cfg)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # cfg.merge_from_list(['SEED', 123])

    cfg.eval_only = args.eval_only
    
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "maft-plus" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="sam3")
    return cfg


def main(args):
    # torch.multiprocessing.set_start_method('spawn')

    # -------- 1. 新增：预占显存逻辑 --------
    # 获取当前进程应使用的 GPU 编号
    # Detectron2 会在 launch 时自动设置好当前进程的设备环境
    current_device = torch.cuda.current_device()
    reserve_gb = 16  # 你想要固定的显存大小
    
    print(f"==> 进程 {comm.get_rank()} 正在 GPU:{current_device} 上预分配 {reserve_gb}GB 显存...")
    try:
        # 预分配 22GB 的空张量
        # 1024**3 字节 = 1GB
        temp_tensor = torch.empty(int(reserve_gb * 1024**3), dtype=torch.uint8, device=f'cuda:{current_device}')
        
        # 销毁变量，但不要执行 torch.cuda.empty_cache()
        # 这样显存就会被保留在 PyTorch 的缓存池中，别人抢不走
        del temp_tensor
        print(f"==> 进程 {comm.get_rank()} 预分配成功，已占坑。")
    except RuntimeError as e:
        print(f"==> 预分配失败 (可能是显存不足以分配 {reserve_gb}GB): {e}")
    # ------------------------------------
        
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )

        # model = build_sam3_model(cfg)
        res = Trainer.test(cfg, model)
        # if cfg.TEST.AUG.ENABLED:
        #     res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

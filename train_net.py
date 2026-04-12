import warnings
import os
import gc

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
import contextlib
import itertools
import logging
import os
import time
import weakref
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,4,6'

from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import (
    AMPTrainer,
    DefaultTrainer,
    SimpleTrainer,
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
from detectron2.engine.defaults import create_ddp_model
from detectron2.utils.events import get_event_storage
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

from detectron2.engine import HookBase
class PromptMonitorHook(HookBase):
    def __init__(self, check_period=100):
        self.check_period = check_period
        self.initial_weights = None

    def before_train(self):
        # 提取 DDP 模型或普通模型
        self.model = self.trainer.model.module if hasattr(self.trainer.model, "module") else self.trainer.model
        # 记录初始权重，用于比对
        self.initial_weights = self.model.detector.backbone.vision_backbone.trunk.student.model.blocks.tuner.prompts.clone().detach()

    def after_step(self):
        if self.trainer.iter % self.check_period == 0:
            prompts = self.model.detector.backbone.vision_backbone.trunk.student.model.blocks.tuner.prompts
            
            # 1. 检查梯度
            if prompts.grad is not None:
                grad_norm = prompts.grad.norm().item()
            else:
                grad_norm = "None (No Gradient!)"
            
            # 2. 检查权重是否发生实质性改变
            weight_diff = torch.abs(prompts.data - self.initial_weights).sum().item()
            
            print(f"[Iter {self.trainer.iter}] Prompt Grad Norm: {grad_norm} | Weight Diff from Start: {weight_diff:.6f}")


class EvalMemoryCleanupHook(HookBase):
    def __init__(self, eval_period=0):
        self.eval_period = eval_period

    def _cleanup(self):
        model = self.trainer.model.module if hasattr(self.trainer.model, "module") else self.trainer.model
        if hasattr(model, "clear_inference_cache"):
            model.clear_inference_cache()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def after_step(self):
        if self.eval_period <= 0:
            return
        next_iter = self.trainer.iter + 1
        if next_iter % self.eval_period == 0 and next_iter != self.trainer.max_iter:
            self._cleanup()

    def after_train(self):
        self._cleanup()



def infer_gradient_accumulation(cfg, world_size):
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")

    total_batch_size = int(cfg.SOLVER.IMS_PER_BATCH)
    if total_batch_size <= 0:
        raise ValueError(f"SOLVER.IMS_PER_BATCH must be positive, got {total_batch_size}")
    if total_batch_size % world_size != 0:
        raise ValueError(
            f"SOLVER.IMS_PER_BATCH={total_batch_size} must be divisible by world_size={world_size}"
        )

    target_batch_per_gpu = total_batch_size // world_size
    use_diff_bs_size = bool(getattr(cfg.DATALOADER, "USE_DIFF_BS_SIZE", False))
    dataset_bs = [int(bs) for bs in getattr(cfg.DATALOADER, "DATASET_BS", [])]

    if cfg.DATALOADER.SAMPLER_TRAIN == "MultiDatasetSampler" and use_diff_bs_size:
        if not dataset_bs:
            raise ValueError(
                "DATALOADER.DATASET_BS must be non-empty when USE_DIFF_BS_SIZE=True"
            )
        unique_bs = sorted(set(dataset_bs))
        if any(bs <= 0 for bs in unique_bs):
            raise ValueError(f"DATALOADER.DATASET_BS must all be positive, got {dataset_bs}")
        if len(unique_bs) != 1:
            raise ValueError(
                "Gradient accumulation with USE_DIFF_BS_SIZE=True currently requires all "
                f"DATASET_BS values to be identical so that IMS_PER_BATCH maps to an exact "
                f"effective batch size. Got {dataset_bs}"
            )
        micro_batch_size = unique_bs[0]
        micro_batch_source = "DATALOADER.DATASET_BS"
    else:
        micro_batch_size = target_batch_per_gpu
        micro_batch_source = "SOLVER.IMS_PER_BATCH/world_size"

    if micro_batch_size > target_batch_per_gpu:
        raise ValueError(
            f"Per-GPU micro-batch {micro_batch_size} exceeds target per-GPU effective batch "
            f"{target_batch_per_gpu}. Increase SOLVER.IMS_PER_BATCH or reduce DATASET_BS."
        )
    if target_batch_per_gpu % micro_batch_size != 0:
        raise ValueError(
            f"Target per-GPU effective batch {target_batch_per_gpu} must be divisible by the "
            f"per-GPU micro-batch {micro_batch_size} inferred from {micro_batch_source}."
        )

    accumulation_steps = target_batch_per_gpu // micro_batch_size
    return {
        "world_size": world_size,
        "total_batch_size": total_batch_size,
        "target_batch_per_gpu": target_batch_per_gpu,
        "micro_batch_size": micro_batch_size,
        "accumulation_steps": accumulation_steps,
        "effective_batch_size": micro_batch_size * world_size * accumulation_steps,
        "micro_batch_source": micro_batch_source,
    }


class _GradientAccumulationMixin:
    def __init__(self, *args, accumulation_steps=1, **kwargs):
        self.accumulation_steps = int(accumulation_steps)
        if self.accumulation_steps <= 0:
            raise ValueError(
                f"accumulation_steps must be >= 1, got {self.accumulation_steps}"
            )
        super().__init__(*args, **kwargs)

    @staticmethod
    def _get_micro_batch_size(data):
        if isinstance(data, (list, tuple)):
            return len(data)
        try:
            return len(data)
        except TypeError:
            return 1

    def _no_sync_context(self, is_last_micro_step):
        if is_last_micro_step or not hasattr(self.model, "no_sync"):
            return contextlib.nullcontext()
        return self.model.no_sync()

    def _write_aggregated_metrics(self, loss_sums, total_samples, data_time):
        normalizer = max(total_samples, 1)
        reduced_loss_dict = {
            key: value / normalizer for key, value in loss_sums.items()
        }

        if getattr(self, "async_write_metrics", False):
            self.concurrent_executor.submit(
                self._write_metrics, reduced_loss_dict, data_time, iter=self.iter
            )
        else:
            self._write_metrics(reduced_loss_dict, data_time)


class GradientAccumulationSimpleTrainer(_GradientAccumulationMixin, SimpleTrainer):
    def run_step(self):
        assert self.model.training, "[GradientAccumulationSimpleTrainer] model was changed to eval mode!"

        self.optimizer.zero_grad()
        total_data_time = 0.0
        total_samples = 0
        loss_sums = {}

        for micro_step in range(self.accumulation_steps):
            start = time.perf_counter()
            data = next(self._data_loader_iter)
            total_data_time += time.perf_counter() - start

            batch_size = self._get_micro_batch_size(data)
            total_samples += batch_size
            is_last_micro_step = micro_step == self.accumulation_steps - 1

            with self._no_sync_context(is_last_micro_step):
                loss_dict = self.model(data)
                if isinstance(loss_dict, torch.Tensor):
                    losses = loss_dict
                    loss_dict = {"total_loss": loss_dict}
                else:
                    losses = sum(loss_dict.values())

                scaled_losses = losses / self.accumulation_steps
                scaled_losses.backward()

            for key, value in loss_dict.items():
                detached_value = value.detach()
                loss_sums[key] = loss_sums.get(key, detached_value.new_zeros(())) + detached_value * batch_size

        if hasattr(self, "after_backward"):
            self.after_backward()
        self._write_aggregated_metrics(loss_sums, total_samples, total_data_time)
        self.optimizer.step()


class GradientAccumulationAMPTrainer(_GradientAccumulationMixin, AMPTrainer):
    def run_step(self):
        assert self.model.training, "[GradientAccumulationAMPTrainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[GradientAccumulationAMPTrainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        self.optimizer.zero_grad()
        total_data_time = 0.0
        total_samples = 0
        loss_sums = {}

        for micro_step in range(self.accumulation_steps):
            start = time.perf_counter()
            data = next(self._data_loader_iter)
            total_data_time += time.perf_counter() - start

            batch_size = self._get_micro_batch_size(data)
            total_samples += batch_size
            is_last_micro_step = micro_step == self.accumulation_steps - 1

            with self._no_sync_context(is_last_micro_step):
                with autocast():
                    loss_dict = self.model(data)
                    if isinstance(loss_dict, torch.Tensor):
                        losses = loss_dict
                        loss_dict = {"total_loss": loss_dict}
                    else:
                        losses = sum(loss_dict.values())

                scaled_losses = losses / self.accumulation_steps
                self.grad_scaler.scale(scaled_losses).backward()

            for key, value in loss_dict.items():
                detached_value = value.detach()
                loss_sums[key] = loss_sums.get(key, detached_value.new_zeros(())) + detached_value * batch_size

        if getattr(self, "log_grad_scaler", False):
            storage = get_event_storage()
            storage.put_scalar("[metric]grad_scaler", self.grad_scaler.get_scale())

        if hasattr(self, "after_backward"):
            self.after_backward()
        self._write_aggregated_metrics(loss_sums, total_samples, total_data_time)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to FCCLIP.
    """

    def __init__(self, cfg):
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):
            setup_logger()

        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        accumulation_info = infer_gradient_accumulation(cfg, comm.get_world_size())

        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False)
        trainer_cls = (
            GradientAccumulationAMPTrainer if cfg.SOLVER.AMP.ENABLED else GradientAccumulationSimpleTrainer
        )
        self._trainer = trainer_cls(
            model,
            data_loader,
            optimizer,
            accumulation_steps=accumulation_info["accumulation_steps"],
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg
        self.accumulation_info = accumulation_info

        logging.getLogger("detectron2").info(
            "Gradient accumulation enabled: per_gpu_micro_batch=%d, world_size=%d, "
            "accumulation_steps=%d, effective_batch_size=%d (SOLVER.IMS_PER_BATCH=%d)",
            accumulation_info["micro_batch_size"],
            accumulation_info["world_size"],
            accumulation_info["accumulation_steps"],
            accumulation_info["effective_batch_size"],
            accumulation_info["total_batch_size"],
        )

        self.register_hooks(self.build_hooks())


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

    trainer.register_hooks([PromptMonitorHook(check_period=200), EvalMemoryCleanupHook(cfg.TEST.EVAL_PERIOD)])

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

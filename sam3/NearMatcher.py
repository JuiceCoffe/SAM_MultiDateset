import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.cuda.amp import autocast

from detectron2.projects.point_rend.point_features import point_sample


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)  # type: torch.jit.ScriptModule


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw


batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule



def match_every_pred_to_best_gt(matcher, outputs, targets):
    """
    修改版的匹配逻辑：
    不进行 1-to-1 匹配，而是强制给每一个 Prediction 找到一个 Cost 最小的 GT。
    结果会导致多个 Prediction 匹配到同一个 GT。
    """
    bs, num_queries = outputs["pred_logits"].shape[:2]
    indices = []

    # Iterate through batch size (此处假设 BS=1，循环一次)
    for b in range(bs):
        # ================== 1. 计算 Cost Matrix (逻辑与原版完全一致) ==================
        out_prob = outputs["pred_logits"][b].softmax(-1)
        tgt_ids = targets[b]["labels"]
        
        # Cost Class
        cost_class = -out_prob[:, tgt_ids]

        out_mask = outputs["pred_masks"][b]
        tgt_mask = targets[b]["masks"].to(out_mask)

        out_mask = out_mask[:, None]
        tgt_mask = tgt_mask[:, None]
        
        # Point Sample 采样以节省显存
        point_coords = torch.rand(1, matcher.num_points, 2, device=out_mask.device)
        tgt_mask = point_sample(
            tgt_mask,
            point_coords.repeat(tgt_mask.shape[0], 1, 1),
            align_corners=False,
        ).squeeze(1)

        out_mask = point_sample(
            out_mask,
            point_coords.repeat(out_mask.shape[0], 1, 1),
            align_corners=False,
        ).squeeze(1)

        with autocast(enabled=False):
            out_mask = out_mask.float()
            tgt_mask = tgt_mask.float()
            # Cost Mask & Dice
            cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)
            cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)

        # 最终的 Cost Matrix: [num_queries, num_gt]
        C = (
            matcher.cost_mask * cost_mask
            + matcher.cost_class * cost_class
            + matcher.cost_dice * cost_dice
        )
        
        # ================== 2. 核心修改：改为取 Argmin ==================
        
        # 形状: C is [num_queries, num_gt]
        # 我们要为每个 query (行) 找最小值的 gt (列)
        # min_cost_values: 每个 query 对应的最小 cost 值
        # best_gt_indices: 每个 query 对应的最佳 GT 的索引 (范围 0 ~ num_gt-1)
        min_cost_values, best_gt_indices = C.min(dim=1) 

        # 构造返回值
        # pred_idx: 就是所有的 queries (0, 1, 2, ..., 99)
        pred_idx = torch.arange(num_queries, dtype=torch.int64, device=C.device)
        # gt_idx: 对应的最佳 GT 索引
        gt_idx = best_gt_indices.to(torch.int64)
        
        indices.append((pred_idx, gt_idx))

    return indices
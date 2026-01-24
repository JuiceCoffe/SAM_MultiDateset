        for i in range(bs):
            # 获取单张图数据
            mask_cls_i = mask_cls_logits[i]       # [Q, C] (Logits)
            mask_pred_i = mask_pred_logits[i]     # [Q, H, W] (Logits)
            
            img_h_orig = batched_inputs[i]["height"]
            img_w_orig = batched_inputs[i]["width"]

            # -----------------------------------------------------------------
            # [Fix 2] 重写语义分割推理逻辑 (Semantic Inference)
            # 解决背景噪声累积问题的关键
            # -----------------------------------------------------------------
            res = {}
            if self.semantic_on:
                # 1. 准备概率
                mask_cls_prob = mask_cls_i.sigmoid()    # [Q, C]
                mask_pred_prob = mask_pred_i.sigmoid()  # [Q, H, W]
                
                # 2. 上采样 Mask 到原图尺寸 (为了更好的边缘)
                # 显存优化：如果不追求极致边缘，可以先处理再上采样
                mask_pred_prob = F.interpolate(
                    mask_pred_prob.unsqueeze(0), 
                    size=(img_h_orig, img_w_orig), 
                    mode="bilinear", 
                    align_corners=False
                ).squeeze(0) # [Q, H, W]

                # 3. 计算每个 Query 对每个像素的贡献分数
                # Score[q, h, w] = MaskProb[q, h, w] * MaxClassProb[q]
                # 这里我们先找出每个 Query 最可能的类别和分数
                scores_per_query, topk_classes = mask_cls_prob.max(dim=1) # [Q], [Q]
                
                # 4. 像素级竞争 (Pixel-wise Competition)
                # 计算全图 Mask 分数: [Q, H, W]
                # 这里的逻辑是：某个像素到底属于哪个 Query？看谁的 (Mask * ClassScore) 最大
                weighted_mask_scores = mask_pred_prob * scores_per_query[:, None, None] # [Q, H, W]
                
                # 5. 取最大值
                # final_mask_scores: [H, W], best_query_idx: [H, W]
                final_mask_scores, best_query_idx = weighted_mask_scores.max(dim=0)
                
                # 6. 映射回类别 ID
                # best_query_idx 指向 Query ID，我们需要它对应的 Class ID
                pred_class_map = topk_classes[best_query_idx] # [H, W]
                
                # 7. 背景处理 (Background Filtering)
                # 如果最大分数都很低，说明是背景
                # 这里的阈值可以设低一点，因为已经乘过了 mask_prob
                is_background = final_mask_scores < self.object_mask_threshold # e.g. 0.01 or 0.05
                
                # 构建最终语义图 [C, H, W] 或者直接输出 [H, W] 的 label map
                # Detectron2 要求输出 [C, H, W] 的 sem_seg (Logits or Probs)
                # 我们这里构造一个 One-Hot 近似
                
                # 为了兼容 visualize_segmentation 和 eval，我们生成 label map
                semseg_result = pred_class_map.clone()
                # 假设 semantic segmentation 的背景类 index 通常是很大的值或 -1，
                # 但 evaluation 时通常只需要预测出的 mask。
                # 如果必须输出 [C, H, W] 概率图供 evaluator 使用:
                
                # 更高效的方法：只保留 Top Query 进行 einsum (类似 Mask2Former 官方实现)
                # 过滤掉得分为 0 的 Query 以加速
                keep_queries = scores_per_query > self.object_mask_threshold
                if keep_queries.sum() > 0:
                    mask_cls_prob_filtered = mask_cls_prob[keep_queries]
                    mask_pred_prob_filtered = mask_pred_prob[keep_queries]
                    
                    # 归一化 (关键步骤：模拟 Softmax 的竞争效果)
                    # 在 Sigmoid 模式下，必须归一化，否则叠加后数值 > 1
                    semseg = torch.einsum("qc,qhw->chw", mask_cls_prob_filtered, mask_pred_prob_filtered)
                    
                    # [Fix] 这里的归一化并不是简单的除法，而是类似 Softmax
                    # 但简单起见，我们直接由 Logits 推导或者不做归一化(如果evaluator只看argmax)
                    # 建议：既然你之前的效果差是因为求和，我们这里直接返回 semseg
                    # 但要确保 mask_pred_prob 是 "exclusive" 的
                    
                    # 更好的替代方案：
                    res["sem_seg"] = semseg
                else:
                    # 如果没有检测到物体
                    res["sem_seg"] = torch.zeros((mask_cls_prob.shape[1], img_h_orig, img_w_orig), device=self.device)

                # --- 可视化部分修正 ---
                # 你的可视化代码里需要 Square 的图
                # ... (保留你的可视化逻辑，但记得应用同样的 Max 逻辑来生成 pred_result_square)
                
                # 简单的可视化逻辑修正示例：
                tensor_h, tensor_w = batched_inputs[i]["image"].shape[-2:]
                mask_pred_i_square = F.interpolate(
                        mask_pred_logits[i].unsqueeze(0), 
                        size=(tensor_h, tensor_w), 
                        mode="bilinear", 
                        align_corners=False
                    ).squeeze(0).sigmoid()
                
                # 使用 Max Logic 生成可视化结果
                sq_scores, sq_classes = mask_cls_prob.max(dim=1)
                sq_weighted = mask_pred_i_square * sq_scores[:, None, None]
                sq_val, sq_idx = sq_weighted.max(dim=0)
                pred_result_square = sq_classes[sq_idx]
                pred_result_square[sq_val < self.object_mask_threshold] = 255 # 设为背景
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

                visualize_segmentation(
                    pred_result=pred_result_square,       # 修改点：传入 Square 预测
                    gt_result= batched_inputs[i]["sem_seg"].to(self.device),           # 本身就是 Square
                    class_names=current_class_names + ['background'],
                    original_image_tensor=batched_inputs[i]["image"], # 本身就是 Square
                    save_path=f"./show_{self.Teacher}_semantic/{batched_inputs[i]['file_name'].split('/')[-1].split('.')[0]}.png"
                )

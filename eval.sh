#!/bin/bash

python train_net.py \
    --config-file configs/RADIOSAM_attnpool.yaml \
    --num-gpus 1 \
    --eval-only \
    OUTPUT_DIR ./out/test/copa_attn_cdt/4ep/ \
    MODEL.WEIGHTS /root/hmp/SAMweight/copa_attn_cdt/model_0029999.pth


# python train_net.py \
#     --config-file configs/RADIOSAM_attnpool.yaml \
#     --num-gpus 1 \
#     --eval-only \
#     OUTPUT_DIR ./out/test/copa_attn_cdt/3ep/ \
#     MODEL.WEIGHTS /root/hmp/SAMweight/copa_attn_cdt/model_0022499.pth

# python train_net.py \
#     --config-file configs/RADIOSAM_attnpool.yaml \
#     --num-gpus 1 \
#     --eval-only \
#     OUTPUT_DIR ./out/test/copa_attn_cdt/2ep/ \
#     MODEL.WEIGHTS /root/hmp/SAMweight/copa_attn_cdt/model_0014999.pth

# python train_net.py \
#     --config-file configs/RADIOSAM_attnpool.yaml \
#     --num-gpus 1 \
#     --eval-only \
#     OUTPUT_DIR ./out/test/copa_attn_cdt/1ep/ \
#     MODEL.WEIGHTS /root/hmp/SAMweight/copa_attn_cdt/model_0007499.pth

echo "All tasks done."